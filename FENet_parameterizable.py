from pyexpat import features
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from decoder import PLS_Model
from os.path import join as path_join, basename
from math import floor
from typing import Optional
import numpy as np


# notes
# - data sample rate is ~30kHz
# - bucketing samples into 900 sample chunks -> ~30ms resolution
# - an action potential takes ~48 samples

# constants
from configs import ATTEMPT_GPU

# runtime variables
device = torch.device('cuda:0' if ATTEMPT_GPU and torch.cuda.is_available() else 'cpu')

# model modules

class BitshiftApproxAveragePool(nn.Module):
    """
    Average pool, but instead of dividing by the length of the input, it
    divides by the nearest power of two (rounded up to encourage smaller vals)
    """
    def __init__(self):
        super(BitshiftApproxAveragePool, self).__init__()

    def forward(self, x, return_poolDivisor=False):
        """
        Takes (N, C, L) or (C, L) and returns (N, C, 1) or (C, 1)
        """
        from math import log2, ceil
        len_to_div = x.shape[-1]
        divisor = 2**round(log2(len_to_div))
        if(return_poolDivisor):
            return torch.sum(x, dim=len(x.shape)-1) / divisor, divisor
        else:
            return torch.sum(x, dim=len(x.shape)-1) / divisor

class WaveletConvolution(nn.Module):
    """
    A wrapper around two Conv1d layers, one (feat / hpf)which returns the
    intermediate value, and the other (feat / lpf) whose output is to be feated
    onto the next WaveletConvolution.

    `feat_out_channels` = number of convolutional channels the feature
                          convolution should output
    `features`          = number of wavelet transform features this layer
                          should produce, from AdaptiveAvgPool-ing the output
                          of the feat_l conv layer
    """
    def __init__(self, in_channels, features,
                       feat_out_channels, feat_kernel_size, feat_stride,
                       pass_out_channels, pass_kernel_size, pass_stride,
                       feat_kwargs={}, pass_kwargs={},
                       dropout=0.2, activation_fn=nn.LeakyReLU(-1)):
        """
        Use the `feat_kwargs` and `pass_kwargs` arguments to forward optional
        arguments to the underlying nn.Conv1d layers.

        Expects an element-wise `activation_fn`, eg. not softmax.
        Uses LeakyReLU with negative slope -1 to simulate absolute value by
        default, to make this more analogous to wavelet transform.
        """
        super(WaveletConvolution, self).__init__()
        from math import ceil


        self.feat_l = nn.Conv1d(in_channels, feat_out_channels, feat_kernel_size, feat_stride, **feat_kwargs)
        self.pass_l = nn.Conv1d(in_channels, pass_out_channels, pass_kernel_size, pass_stride, **pass_kwargs)
        self.feat_pad = nn.ConstantPad1d((ceil(max((feat_kernel_size - feat_stride), 0)), feat_kernel_size - 1), 0)
        self.pass_pad = nn.ConstantPad1d((ceil(max((pass_kernel_size - pass_stride), 0)), pass_kernel_size - 1), 0)

        if feat_out_channels > 1:
            raise NotImplementedError("FENet reimplementation cannot currently handle more than one conv out filter, since that would mix neural channels and convolutional channels/filters. see tag todo-multiconvchannel")
        else:
            self.features = features
            self.pool = nn.AdaptiveAvgPool1d(features)  # todo-multiconvchannel: to enable feat_out_channels > 1, use AdaptiveAvgPool2d
            #self.pool = BitshiftApproxAveragePool()    # divide by nearest power of 2 instead, bc hardware
        self.dropout = dropout
        self.activation_fn = activation_fn

    def forward(self, x):
        """
        Expects shape = (batch_size * neural_channels, in_channels, n_samples)
        Returns shape = (batch_size, 1, features*feat_out_channels), (batch_size, pass_out_channels, convolved_len)
        """
        batch_size, n_channels, n_samples = x.shape

        feat_x = self.feat_pad(x)
        pass_x = self.pass_pad(x)

        feat_x = self.feat_l(feat_x)
        pass_x = self.pass_l(pass_x)
        feat_x = self.activation_fn(feat_x)


        feat_x = self.pool(feat_x)
        feat_x = feat_x.view(batch_size, self.features) # flatten feat_x into 1d array per batch-element*neural-channel

        pass_x = F.dropout(pass_x, p=self.dropout, training=self.training)

        return feat_x, pass_x

class FENet(nn.Module):
    def __init__(self,
                 features_by_layer=[1]*8,
                 kernel_by_layer=[40]*7,
                 stride_by_layer=[2]*7,
                 relu_by_layer=[0]*7,
                 checkpoint_name=None,
                 pls=2,
                 dropout=0.2):
        """
        `features_by_layer`: an array of how many features each layer should return. The last element is the number of features of the output of the full convolutional stack.
        """
        super(FENet, self).__init__()

        if len(features_by_layer)-1 != len(kernel_by_layer) or len(features_by_layer)-1 != len(stride_by_layer) or len(features_by_layer)-1 != len(relu_by_layer):
            print(features_by_layer, kernel_by_layer, stride_by_layer, relu_by_layer)
            raise ValueError("`features_by_layer`[:-1], `sizes_by_layer`, and `strides_by_layer`, and 'relu_by_layer' must be same len")

        # todo-experiment: allow different kernel sizes and strides for feat_l and pass_l

        jank_serialize = lambda int_list: '-'.join(str(x) for x in int_list)
        self.checkpoint_name = checkpoint_name or f"training_{jank_serialize(features_by_layer)}_{jank_serialize(kernel_by_layer)}_{jank_serialize(stride_by_layer)}" # used to identify models when logging
        self.pls = pls  # TODO: PLS should probably be a part of decoder.

        self.features_by_layer = features_by_layer
        self.kernel_by_layer = kernel_by_layer
        self.stride_by_layer = stride_by_layer
        self.relu_by_layer = relu_by_layer
        self.activation_fn = [nn.LeakyReLU(-1/(2**int(power))) for power in relu_by_layer]  ## TODO: shouldbe corrected
        self.poolDivisor = [0]*len(kernel_by_layer)
        self.layers = nn.ModuleList([
            WaveletConvolution(
                in_channels=1, features=feats,
                feat_out_channels=1, feat_kernel_size=kernel, feat_stride=stride, feat_kwargs={ 'bias': False },
                pass_out_channels=1, pass_kernel_size=kernel, pass_stride=stride, pass_kwargs={ 'bias': False },
                dropout=dropout, activation_fn=activation_fn,
            )
            for feats, kernel, stride, activation_fn in zip(features_by_layer[:-1], kernel_by_layer, stride_by_layer, self.activation_fn) ])

        self.pool = BitshiftApproxAveragePool() # TODO: adaptive average pool?

    def forward(self, x):
        """
        Expects a tensor of electrode streams, shape = (batch_size * n_channels=192, in_channels, n_samples=900)
        Returns a tensor of electrode features, shape = (batch_size * n_channels=192, sum(features_by_layer))
        """
        features_list = []  # todo-optm: preallocate zeros, then copy feat_x output into the ndarray
        pass_x = x

        # feed `x` through FENet, storing intermediate `feat_x`s along the way
        for wvlt_cnn_layer in self.layers:
            feat_x, pass_x = wvlt_cnn_layer(pass_x)
            features_list.append(feat_x)
            del(feat_x)
            torch.cuda.empty_cache()


        # end case: non-linear + adaptive_avgpool the output of the
        # WaveletConvolution stack to create the final feature
        final_feat = self.activation_fn[-1](pass_x)
        final_feat = self.pool(final_feat)
        final_feat = final_feat.view(-1, self.features_by_layer[-1]) # flatten feat_x into 1d array per batch-element*neural-channel
        features_list.append(final_feat)

        # concatenate the features from each layer for the final output
        x_total_feat = torch.cat(features_list, dim=1)

        return x_total_feat


def make_daubechies_wavelet_initialization(fe_net):
    """
    Takes in a FENet, and returns a state dict of the Daubechies wavelet coefficients.

    USE ME LIKE
    ```python
    from FENet_parameterizable import FENet, make_daubechies_wavelet_initialization
    fe_net = FENet()
    fe_net.load_state_dict(make_daubechies_wavelet_initialization(fe_net))
    # good to go! start training, or just evaluate to simulate wavelet decomposition
    ```

    Assumes the WaveletConvolutions in the FENet have even kernel_size ∈ [2, 40] and
    in_channels == 1 and out_channels == 1 (bc. we only have 1 thing to init with, and
    if we initialized every filter to the same daubechies coeffs, then they would never
    diverge, and thus the extra filters would do nothing. todo-multiconvchannel)
    """
    assert isinstance(fe_net, FENet)
    import pywt # PyWavelets

    def get_wavelet_name_from_size(kernel_size, dbg_key='unknown'):
        # there could eventually be an initialiation interface that takes a
        # list of these functions and tries to initialize WaveletConvolution
        # layers using the functions
        if kernel_size % 2 != 0:
            raise ValueError(f"Can only initialize daubechies wavelet with even-sized kernel, got {kernel_size} for {dbg_key}")
        if kernel_size <= 0:
            raise ValueError(f"Can only initialize daubechies wavelet with positive kernel size, got {kernel_size} for {dbg_key}")
        if kernel_size > 40:
            raise ValueError(f"PyWavelets only provides daubechies filters up to 40, got {kernel_size} for {dbg_key}")

        return f"db{kernel_size//2}"

    # create a new state_dict and populate it with keys from the og state dict and values from either og or PyWavelets
    new_weights_dict = {}
    for key, mat in fe_net.state_dict().items():
        out_channels, in_channels, kernel_size = mat.shape

        if out_channels != 1 or in_channels != 1:
            raise NotImplementedError('todo-multiconvchannel how would you initialize multiple filters for the same channel? they need to be diff, else. they will get stuck doing the same thing as the other convolutional filter with the same wavelet initialization')

        wavelet = pywt.Wavelet(get_wavelet_name_from_size(kernel_size, dbg_key=key))
        # figure out which coefficients of the Wavelet we want for this layer (either feat_l or pass_l)
        if 'feat_l' in key:
            weights = torch.Tensor([[wavelet.dec_hi]]).float()
            weights = torch.flip(weights,[2])
        elif 'pass_l' in key:
            weights = torch.Tensor([[wavelet.dec_lo]]).float()
            weights = torch.flip(weights,[2])
        else: # didn't find 'feat_l' or 'pass_l' in the layer name, so its porbs not a WaveletConvolution weight mat
            weights = mat
            print("daubechies initialization: skipping layer {key} because it doesn't seem to have feat_l or pass_l weight matrices")
        new_weights_dict[key] = weights

    return new_weights_dict


def cross_validated_eval(decoder, outputs: torch.Tensor, labels: torch.Tensor, folds: int=10):
    """
    expects outputs shape (n_chunks, n_channels*n_feats) and labels shape (n_chunks, 2)
    generates `folds` cross-validation folds from `outputs` and `labels`
    calls decoder.train() on each fold
    returns cat of the eval on the validation set

    obj = type('MyClass', (object,), {'content':{}})()
    >>> decoder = type('Decoder', (object,), {    'train': lambda _self, inp, lab: None, 'forward': lambda _self, inp: inp[:, :2]    })()
    >>> outputs = torch.tensor([ [0,0,0],[1,1,1],[2,2,2],[3,3,3] ])
    >>> labels  = torch.tensor([ [0,0],[1,1],[2,2],[3,3] ])
    >>> cross_validated_eval(decoder, outputs, labels, folds=2)
    tensor([[0., 0.],
            [1., 1.],
            [2., 2.],
            [3., 3.]])
    """
    from utils import KFoldsGenerator
    assert outputs.size()[0] == labels.size()[0]
    k_folds_manager = KFoldsGenerator(zip(outputs, labels), folds)

    tot_dev_decoder_preds = torch.zeros(labels.shape)
    rows_filled_counter = 0
    for train_dl, dev_dl in k_folds_manager.make_folds():
        # train
        train_inp, train_lab = zip(*train_dl)
        decoder.train(torch.vstack(train_inp), torch.vstack(train_lab).cpu().detach().numpy())

        # validate
        dev_inp, dev_lab = zip(*dev_dl)
        dev_inp = torch.vstack(dev_inp)
        tot_dev_decoder_preds[rows_filled_counter:rows_filled_counter+len(dev_inp), :] = decoder.forward(dev_inp)
        rows_filled_counter += len(dev_inp)
    return tot_dev_decoder_preds


def inference_batch(device, net: FENet, dim_red, decoder, inputs, labels, batch_size=None, decoder_crossvalidate=False):
    """
    expects inputs shape (n_chunks, n_channels, n_samples)
    """

    net.eval()
    n_chunks, n_channels, n_samples = inputs.shape
    inputs = inputs.reshape(n_chunks * n_channels, 1, n_samples) # train on each sample seprately. the middle dimension is 1 is n_conv_channel=1
    net = net.to(device)
    with torch.no_grad():

        if batch_size == None:
            inputs = inputs.to(device)
            outputs = net(inputs)
        else:
            outputs = []
            for batch in torch.split(inputs, batch_size):
                batch = batch.to(device)
                outputs.append(net(batch))
            outputs = torch.cat(outputs)

        outputs = outputs.reshape(n_chunks, n_channels, len(net.features_by_layer))    # (batch_size * n_channels, n_samples) -> (batch_size * n_channels, n_features)

        # if isinstance(labels, torch.Tensor):
        #     labels = labels.cpu().detach().numpy()

        if(net.pls != None and net.pls > 0):
            #if(not dim_red.trained):
            #    dim_red.train(outputs, labels.cpu().detach().numpy())
            dim_red.train(outputs, labels.cpu().detach().numpy())
            outputs = dim_red.forward(outputs)
        outputs = outputs.reshape(n_chunks, n_channels*dim_red.n_out_dims)  # TODO: should dim_red.n_out_dims possibly be sum(net.features_by_layer) when pls_dims=0?

        #if(not decoder.trained):
        #    decoder.train(outputs, labels.cpu().detach().numpy())
        if decoder_crossvalidate:
            return cross_validated_eval(decoder, outputs, torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels)   # CLEAN: get rid of numpy
        else:
            decoder.train(outputs, labels)
            return decoder.forward(outputs)



def train_batch(device, net: FENet, dim_red, decoder, optimizer, scheduler, criterion, inputs, labels, batch_size=None):
    n_chunks, n_channels, n_samples = inputs.shape

    # combine n_channels into the batch dimension, to treat each channel as
    # a seprate training example in the batch, because we want to train a
    # general FENet which extracts latent features from an arbitrary
    # electrode, rather than a FENet optimized to this placement of elecdrodes
    inputs = inputs.reshape(n_chunks * n_channels, 1, n_samples)
    #print("chunks: ", n_chunks, " samples: ", n_samples, " channels: ", n_channels)
    #print("\n\ninputs shape", inputs.shape)

    optimizer.zero_grad()

    net =  net.to(device)
    labels = labels.to(device)
    labels_np = labels.cpu().detach().numpy()

    net.train()

    if batch_size == None:
        inputs = inputs.to(device)
        outputs = net(inputs)
    else:
        outputs = []
        # for batch in torch.split(inputs, batch_size):
        for batch in tqdm(torch.split(inputs, batch_size), desc="batches", leave=False):
            batch = batch.to(device)
            outputs.append(net(batch))
        outputs = torch.cat(outputs)
    outputs = outputs.reshape(n_chunks, n_channels, len(net.features_by_layer))    # (batch_size * n_channels, n_samples) -> (batch_size * n_channels, n_features)

    if(net.pls != None and net.pls > 0):
        if(not dim_red.trained):
            dim_red.train(outputs, labels_np)
        outputs = dim_red.forward(outputs)
    outputs = outputs.reshape(n_chunks, n_channels*dim_red.n_out_dims)
    if(not decoder.trained):
        decoder.train(outputs, labels_np)
    predictions = decoder.forward(outputs)
    loss = criterion(predictions, labels)   # TODO: expensive


    # loss, *_ = decoder_loss(reduced_feats, labels, pls_mode=None, using_gpu='cuda' in device)
    loss.backward()
    optimizer.step()
    if(not decoder.trained):
        try:
            decoder.step()
        finally:
            pass

    if scheduler: scheduler.step()

    return loss, outputs

def read_checkpoint(checkpoint):
    try:
        config, fe_net_state, optimizer_state, scheduler_state = torch.load(checkpoint)
        return config, fe_net_state, optimizer_state, scheduler_state
    except Exception as e:
        raise ValueError(f"Couldn't load config from checkpoint {checkpoint}, got error: {e}")

def write_checkpoint(save_path, config, fe_net_state, optimizer_state=None, scheduler_state=None):
    torch.save([config, fe_net_state, optimizer_state, scheduler_state], save_path)

def make_fenet_from_checkpoint(checkpoint, device, override_shape=None, pls_dims=None):
    """
    must specify override_shape to pass to FENet constructor if config is not included in the checkpoint
    must specify pls_dims if config is not included in the checkpoint, or if config does not include pls_dims
    """
    # grossness to deal with different versions of config and checkpoint saving
    if override_shape is None:
        config, fe_net_state, optimizer_state, scheduler_state = read_checkpoint(checkpoint)
        model_config = [[1]*config['n_feat_layers'],
                        [config[f"kernel{i}"] for i in range(1, config['n_feat_layers'])],
                        [config[f"stride{i}"] for i in range(1, config['n_feat_layers'])],
                        [config[f"relu{i}"]   for i in range(1, config['n_feat_layers'])]]
        if pls_dims is None and 'pls_dims' not in config:
            print("\n\n\n\nCOULDN'T FIND `pls_dims` IN `config`!! USING DEFAULT VALUE OF `pls_dims=0`\n\n\n\n")
            pls_dims = 0
        elif pls_dims is None:
            pls_dims = config['pls_dims']
    else:
        if pls_dims is None:
            raise ValueError("must specify pls_dims if override_shape is specified")
        fe_net_state, optimizer_state, scheduler_state = torch.load(checkpoint)
        model_config = override_shape
    from os.path import basename
    fe_net = FENet(*model_config, checkpoint_name=basename(checkpoint), pls=pls_dims)
    fe_net.load_state_dict(fe_net_state)
    fe_net.to(device)
    return fe_net


if __name__ == '__main__':
    import doctest
    doctest.testmod()
