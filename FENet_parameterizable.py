from pyexpat import features
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join as path_join, basename
from math import floor
from typing import Optional
import numpy as np
from data_parser import standard_scalar_normalize


# notes
# - data sample rate is ~30kHz
# - bucketing samples into 900 sample chunks -> ~30ms resolution
# - an action potential takes ~48 samples

# constants
from configs import ATTEMPT_GPU
from configs import QUANTIZED_WORD_LENGTH

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
                       dropout=0.2, activation_fn=nn.LeakyReLU(-1),
                       cache_intermediate_outputs=False,
                       num_to_cache=None):
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
        self._cache = cache_intermediate_outputs
        self.num_to_cache = num_to_cache
        self.output_cache = [None]*2

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

        num_to_cache = batch_size if self.num_to_cache is None else self.num_to_cache

        if self._cache: self.output_cache[0] = self.activation_fn(pass_x[0:num_to_cache, :, :])
        if self._cache: self.output_cache[1] = feat_x[0:num_to_cache, :, :]

        feat_x = self.pool(feat_x)
        feat_x = feat_x.view(batch_size, self.features) # flatten feat_x into 1d array per batch-element*neural-channel

        pass_x = F.dropout(pass_x, p=self.dropout, training=self.training)

        return feat_x, pass_x

# TODO: quantize before pooling

class FENet(nn.Module):
    def __init__(self,
                 features_by_layer=[1]*8,
                  kernel_by_layer=[40]*7,
                   stride_by_layer=[2]*7,
                   relu_by_layer=[0]*7,
                    checkpoint_name=None,
                           pls_dims=None,
                             dropout=0.2,
                  normalize_at_end=False,

                     cache_intermediate_outputs=False,
                     num_to_cache=None,

                    annealing_alpha=0.01,
                     thermal_sigma=0.001,
                    anneal_eval_window=8,
                            anneal=False,
                     ):
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
        self.pls = pls_dims  # TODO: Create a FENet Pipeline class that handles different PLS and Decoder things

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
                cache_intermediate_outputs=cache_intermediate_outputs, num_to_cache=num_to_cache
            )
            for feats, kernel, stride, activation_fn in zip(features_by_layer[:-1], kernel_by_layer, stride_by_layer, self.activation_fn) ])

        # self.anneal_noise = [[(torch.randn(weights_pass.size(), device=weights_pass.device) * annealing_alpha,
        #                        torch.randn(weights_feat.size(), device=weights_feat.device) * annealing_alpha)
        #     for weights_pass, weights_feat in zip(layer.pass_l.parameters(), layer.feat_l.parameters())]
        #     for layer in self.layers]
        # self.pool = BitshiftApproxAveragePool()
        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.bn = nn.BatchNorm1d(sum(self.features_by_layer), affine = False, track_running_stats = False) if normalize_at_end else None
        self.normalize_at_end = normalize_at_end  # FIXME: actually take in n_channels and construct the batchnorm


        self.annealing_alpha = annealing_alpha
        self.thermal_sigma = thermal_sigma
        self.running_annealed_loss = 0
        self.running_non_annealed_loss = 0
        self.loss_recieved_counter = 0
        self.anneal_eval_window = anneal_eval_window
        self.anneal=anneal

    def forward(self, x, use_annealed_weights=False):
        """
        Expects a tensor of electrode streams, shape = (batch_size, n_channels=192, n_samples=900)
        Returns a tensor of electrode features, shape = (batch_size, n_channels=192, sum(features_by_layer))
        """
        n_chunks, n_channels, n_samples = x.shape
        x = x.view(-1, 1, n_samples)
        features_list = []  # todo-optm: preallocate zeros, then copy feat_x output into the ndarray
        pass_x = x

        # feed `x` through FENet, storing intermediate `feat_x`s along the way
        # if(self.anneal and use_annealed_weights):
            # annealed_layers = self.layers
            # for anneal_noise, wvlt_cnn_layer in zip(self.anneal_noise, annealed_layers):
                # for pass_noise, feat_noise, weights_pass, weights_feat in zip(anneal_noise[0][0],
                                                                            #   anneal_noise[0][1],
                                                        #    wvlt_cnn_layer.pass_l.parameters(),
                                                        #    wvlt_cnn_layer.feat_l.parameters()):
                    # weights_pass.add_(pass_noise.to(weights_pass.device))
                    # weights_feat.add_(feat_noise.to(weights_feat.device))

        # for wvlt_cnn_layer in tqdm(annealed_layers if (use_annealed_weights and self.anneal) else self.layers, desc="fenet layer", leave=False):
        # for wvlt_cnn_layer in (annealed_layers if (use_annealed_weights and self.anneal) else self.layers):
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
        x_total_feat = x_total_feat.view(-1, n_channels * sum(self.features_by_layer))
        if self.normalize_at_end:
            bn = nn.BatchNorm1d(sum(self.features_by_layer) * n_channels, affine=False, track_running_stats=False)
            x_total_feat = bn(x_total_feat)

        return x_total_feat

    # def anneal_weights(self):
    #     self.anneal_noise = [[(torch.randn(weights_pass.size(), device=weights_pass.device) * self.annealing_alpha,
    #                            torch.randn(weights_feat.size(), device=weights_feat.device) * self.annealing_alpha)
    #         for weights_pass, weights_feat in zip(layer.pass_l.parameters(), layer.feat_l.parameters())]
    #         for layer in self.layers]

    # def save_annealed_weights(self, losses=None):

    #     if(losses !=None):
    #         #store a running sum of each of the losses
    #         self.running_annealed_loss += losses[0]
    #         self.running_non_annealed_loss += losses[1]
    #         #when a full evaluation windown of losses is obtained, compute the average
    #         if(self.anneal_eval_window > self.loss_recieved_counter):
    #             self.anneal_eval_window += 1
    #             return False
    #         else:
    #             self.running_annealed_loss = 0
    #             self.running_non_annealed_loss = 0
    #             self.loss_recieved_counter = 0
    #             avg_annealed_loss = self.running_annealed_loss/self.anneal_eval_window
    #             avg_non_annealed_loss = self.running_non_annealed_loss/self.anneal_eval_window
    #             #if on average the annealed weight is better, keep it and continue with saving
    #             #the new weight, otherwise, overwrite with a new weight set an return before
    #             #saving
    #             if(avg_non_annealed_loss < avg_annealed_loss):
    #                 self.anneal_weights()
    #                 return False

    #     for anneal_noise, wvlt_cnn_layer in zip(self.anneal_noise, self.layers):
    #         for pass_noise, feat_noise, weights_pass, weights_feat in zip(anneal_noise[0][0],
    #                                                                       anneal_noise[0][1],
    #                                                    wvlt_cnn_layer.pass_l.parameters(),
    #                                                    wvlt_cnn_layer.feat_l.parameters()):
    #                 weights_pass.data = torch.add(weights_pass.data, pass_noise.to(weights_pass.device))
    #                 weights_pass.feat = torch.add(weights_feat.data, feat_noise.to(weights_feat.device))
    #     self.annealing_alpha = self.annealing_alpha*self.thermal_sigma
    #     return True


def make_truncate_quantizer(wl, fl):
    # just use fxpmath.Fxp if this gets scuffed—supports truncation and wrap
    def truncate_quantize(mat):
        mat = torch.fmod(mat, 2**(wl-1-fl))
        # print('clamped', mat)
        mat = torch.trunc(mat * 2**fl)
        mat = mat / 2**fl
        # print('trunced', mat)
        return mat
    return truncate_quantize

class QuantizedFENet(FENet):
    """
    FENet with quantization. See FENet class for more docs.
    OPTM: allow using different quantization for weights and values
    """
    def __init__(   self,
                    wl,
                    fl,
                    features_by_layer,
                    kernel_by_layer,
                    stride_by_layer,
                    relu_by_layer,
                    checkpoint_name,
                    pls,
                    cache_intermediate_outputs=True,
                    num_to_cache=None,
                    dropout=0.2):
        super(QuantizedFENet, self).__init__(
            features_by_layer,
            kernel_by_layer,
            stride_by_layer,
            relu_by_layer,
            annealing_alpha=0.01,
            thermal_sigma=0.001,
            anneal_eval_window=8,
            anneal=False,
            checkpoint_name=checkpoint_name,
            pls=pls,
            dropout=dropout,
            cache_intermediate_outputs=cache_intermediate_outputs,
            num_to_cache=num_to_cache)
        from qtorch import FixedPoint
        from qtorch.quant import Quantizer
        from numpy import ndarray
        self.wl = wl
        self.fl = fl
        self.quantize = Quantizer(forward_number=FixedPoint(wl=wl, fl=fl, clamp=True, symmetric=True), forward_rounding='nearest')
        #self.quantize = make_truncate_quantizer(wl, fl)
        self._cache = cache_intermediate_outputs
        self.num_to_cache = num_to_cache
        self.cache_formatter = lambda x : x 
        self.pass_layers = [None]*(len(features_by_layer)-2)
        self.feat_layers = [None]*(len(features_by_layer))

    def forward(self, x):
        """
        Same as the normal FENet.forward() except with quantization
        """
        from numpy import concatenate
        features_list = []  # todo-optm: preallocate zeros, then copy feat_x output into the ndarray
        # if self.cache: self.values_hist_data[0] += x.flatten().tolist()
        pass_x = self.quantize(x)

        # feed `x` through FENet, storing intermediate `feat_x`s along the way
        for i, wvlt_cnn_layer in enumerate(self.layers):

            feat_x, pass_x, self.poolDivisor[i] = wvlt_cnn_layer(pass_x, return_poolDivisor=True)

            #Quantize the intermediates/features. if this is the last layer, save a copy for final
            #processing
            feat_x = self.quantize(feat_x)
            if(i == len(self.kernel_by_layer)-1):
                pre_quantized_pass_layer = pass_x
            pass_x = self.quantize(pass_x)

            #Cahce results
            features_list.append(feat_x)

            if self._cache:
                if(i < len(self.kernel_by_layer)-1):
                    if self.pass_layers[i] == None:
                        self.pass_layers[i] = self.cache_formatter(pass_x if self.num_to_cache is None else pass_x[0:self.num_to_cache, :, :])
                    else:
                        self.pass_layers[i].append(self.cache_formatter(pass_x if self.num_to_cache is None else pass_x[0:self.num_to_cache, :, :]))

                if self.feat_layers[i] == None:
                    self.feat_layers[i] = wvlt_cnn_layer.output_cache[1]
                else:
                    self.feat_layers[i].append(wvlt_cnn_layer.output_cache[1])

        # end case: non-linear + adaptive_avgpool the output of the
        # WaveletConvolution stack to create the final feature
        if self.feat_layers[-1] == None:
            self.feat_layers[-1] = wvlt_cnn_layer.output_cache[0]
        else:
            self.feat_layers[-1].append(wvlt_cnn_layer.output_cache[0])
        final_feat = self.activation_fn[-1](pre_quantized_pass_layer)
        final_feat, self.poolDivisor[-1] = self.pool(final_feat, return_poolDivisor=True)
        final_feat = self.quantize(final_feat)
        final_feat = final_feat.view(-1, self.features_by_layer[-1]) # flatten feat_x into 1d array per batch-element*neural-channel
        features_list.append(final_feat)

        # concatenate the features from each layer for the final output
        x_total_feat = torch.cat(features_list, dim=1)

        return x_total_feat

    def set_cache_format(self, cache_format, total_length=QUANTIZED_WORD_LENGTH):
        from numpy import vectorize
        from functools import partial
        from export import convert_to_hex, convert_to_oct, convert_to_bin

        enforce_np = lambda x : x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

        match cache_format:
            case 'Float':
                formatter = lambda x : x
            case 'Hex':         formatter = partial(convert_to_hex, sign_mag=True,  wl=self.wl, fl=self.fl)
            case 'Oct':         formatter = partial(convert_to_oct, sign_mag=True,  wl=self.wl, fl=self.fl)
            case 'Bin':         formatter = partial(convert_to_bin, sign_mag=True,  wl=self.wl, fl=self.fl)
            case 'Hex_2s_comp': formatter = partial(convert_to_hex, sign_mag=False, wl=self.wl, fl=self.fl)
            case 'Oct_2s_comp': formatter = partial(convert_to_oct, sign_mag=False, wl=self.wl, fl=self.fl)
            case 'Bin_2s_comp': formatter = partial(convert_to_bin, sign_mag=False, wl=self.wl, fl=self.fl)
            case other:
                raise ValueError(f"unknown FENet cache format type `{other}`")

        if cache_format != 'Float':
            formatter = vectorize(partial(formatter, wl=self.wl, fl=self.fl, total_length=total_length))

        self.cache_formatter = lambda x : formatter(enforce_np(x))
    def clear_cache(self):
        self.pass_layers = [None]*(len(self.features_by_layer)-1)
        for layer in self.layers:
            layer.output_cache = torch.tensor([])



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

def quantize_state_dict(wl, fl, state_dict):
    """
    Quantizes the weights in a state dict using the provided num format

    Use like:
    ```python
    fe_net = FENet(...)
    fe_net.load_state_dict(quantize_weights(wl, fl, torch.load('path')))
    ```
    """
    from qtorch.quant import fixed_point_quantize

    new_state_dict = {}

    for key, mat in state_dict.items():
        new_state_dict[key] = fixed_point_quantize(mat, wl=wl, fl=fl, rounding='nearest')

    return new_state_dict



def cross_validated_eval(decoder, dim_red, outputs: torch.Tensor, labels: torch.Tensor, folds: int=10):
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

    from scipy.stats import pearsonr
    pr2s = []
    tot_dev_decoder_preds = torch.zeros(labels.shape)
    rows_filled_counter = 0
    for train_dl, dev_dl in k_folds_manager.make_folds():
        # train 
        train_inp, train_lab = [torch.vstack(dl) for dl in zip(*train_dl)]
        dev_inp, dev_lab = [torch.vstack(dl) for dl in zip(*dev_dl)]

        if (dim_red != None):
            train_plsed, test_plsed = dim_red.fit_transform(train_inp, train_lab, dev_inp)
            # outputs = outputs.view(n_chunks, n_channels*dim_red.n_out_dims)  # TODO: should dim_red.n_out_dims possibly be sum(net.features_by_layer) when pls_dims=0?

        # decoder.train(torch.vstack(train_inp), torch.vstack(train_lab))
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(train_plsed.cpu().detach().numpy(), train_lab.cpu().detach().numpy())

        # validate
        g = reg.predict(test_plsed.cpu().detach().numpy())
        
        rx, p = pearsonr(g[:, 0], dev_lab[:, 0])
        ry, p = pearsonr(g[:, 1], dev_lab[:, 1])
        pr2s.append([rx**2, ry**2, np.sqrt((rx**4+ry**4)/2)])
        tot_dev_decoder_preds[rows_filled_counter:rows_filled_counter+len(test_plsed), :] = torch.tensor(g)
        rows_filled_counter += len(test_plsed)
    print(pr2s)
    print(torch.tensor(pr2s).mean(axis=0))
    import pdb; pdb.set_trace()
    return tot_dev_decoder_preds


def inference_batch(device, net: FENet, dim_red, decoder, inputs, labels, quantization=None, batch_size=None, decoder_crossvalidate=False):
    """
    expects inputs shape (n_chunks, n_channels, n_samples)
    """
    
    net.eval()
    n_chunks, n_channels, n_samples = inputs.shape
    # inputs = inputs.reshape(n_chunks * n_channels, 1, n_samples) # train on each sample seprately. the middle dimension is 1 is n_conv_channel=1
    net = net.to(device)
    with torch.no_grad():

        # run the model. (batch_size * n_channels, n_samples) -> (batch_size, n_channels * n_features)
        if batch_size == None:
            inputs = inputs.to(device)
            outputs = net(inputs)
        else:
            outputs = []
            for batch in torch.split(inputs, batch_size):
                batch = batch.to(device)
                outputs.append(net(batch))
            outputs = torch.cat(outputs)

        # FIX: MOVE PLSR TO BE CROSSVALIDATED
        # if (net.pls != None and net.pls > 0):
        #     outputs = dim_red.fit_transform(outputs, labels.cpu().detach().numpy())
        #     outputs = outputs.view(n_chunks, n_channels*dim_red.n_out_dims)  # TODO: should dim_red.n_out_dims possibly be sum(net.features_by_layer) when pls_dims=0?
        
        # import pdb; pdb.set_trace()

        # FIX: REMOVE STANDARD SCALAR BECASUE PLSR SHOULD ROUGHLY NORMALIZE ANYWAYS, AND ITS ANNOYING TO CROSS VALIDATE
        # outputs = torch.from_numpy(standard_scalar_normalize(outputs)).to(device)    # additional renormalization for inference time only
        
        # decoder expcets (n_chunks, n_channels * feats_per_channel)
        if decoder_crossvalidate:
            return cross_validated_eval(decoder, dim_red, outputs, torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels)   # CLEAN: get rid of numpy 
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

    # if(net.anneal):
    #     net.eval()
    #     with torch.no_grad():
    #         test_criterion = criterion
    #         annealed_test_criterion = criterion

    #         if batch_size == None:
    #             inputs = inputs.to(device)
    #             outputs = net(inputs)
    #         else:
    #             outputs = []
    #             for batch in torch.split(inputs, batch_size):
    #                 batch = batch.to(device)
    #                 outputs.append(net(batch))
    #             outputs = torch.cat(outputs)
    #         outputs = outputs.reshape(n_chunks, n_channels, len(net.features_by_layer))    # (batch_size * n_channels, n_samples) -> (batch_size * n_channels, n_features); TODO: should len(features_by_layer) be sum(features_by_layer)

    #         if(net.pls != None and net.pls > 0):
    #             dim_red.train(outputs, labels_np)
    #         #print("\n\noutputs shape", outputs.shape)
    #         if(net.pls != None and net.pls > 0):
    #             if(not dim_red.trained):
    #                 dim_red.train(outputs, labels_np)
    #             outputs = dim_red.forward(outputs)
    #         outputs = outputs.reshape(n_chunks, n_channels*dim_red.n_out_dims)
    #         if(not decoder.trained):
    #             decoder.train(outputs, labels_np)
    #         predictions = decoder.forward(outputs)
    #         loss = test_criterion(predictions.float(), labels.float())   # TODO: expensive


    #         if batch_size == None:
    #             inputs = inputs.to(device)
    #             outputs = net(inputs)
    #         else:
    #             outputs = []
    #             for batch in torch.split(inputs, batch_size):
    #                 batch = batch.to(device)
    #                 outputs.append(net(batch, use_annealed_weights=True))
    #             outputs = torch.cat(outputs)
    #         outputs = outputs.reshape(n_chunks, n_channels, len(net.features_by_layer))    # (batch_size * n_channels, n_samples) -> (batch_size * n_channels, n_features)

    #         if(net.pls != None and net.pls > 0):
    #             outputs = dim_red.forward(outputs)
    #         outputs = outputs.reshape(n_chunks, n_channels*dim_red.n_out_dims)
    #         if(not decoder.trained):
    #             decoder.train(outputs, labels_np)
    #         predictions = decoder.forward(outputs)
    #         annealed_loss = annealed_test_criterion(predictions, labels)   # TODO: expensive

    #     if(net.save_annealed_weights(losses=(annealed_loss, loss))): net.anneal_weights()

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
    outputs = outputs.reshape(n_chunks, n_channels * sum(net.features_by_layer))

    if(net.pls != None and net.pls > 0):
        outputs = dim_red.fit_transform(outputs, labels.cpu().detach().numpy())
        outputs = outputs.reshape(n_chunks, n_channels*dim_red.n_out_dims)  # TODO: should dim_red.n_out_dims possibly be sum(net.features_by_layer) when pls_dims=0?

    # decoder expcets (n_chunks, n_channels * feats_per_channel)
    decoder.train(outputs, labels)
    predictions = decoder.forward(outputs)
    loss = criterion(predictions, labels)   # TODO: expensive


    # loss, *_ = decoder_loss(reduced_feats, labels, pls_mode=None, using_gpu='cuda' in device)
    loss.backward()
    optimizer.step()

    # if(not decoder.trained):  # this is for recurrent networks only right? sbulfer?
    #     try:
    #         decoder.step()
    #     finally:
    #         pass

    if scheduler: scheduler.step()

    return loss, outputs

def make_qfenet_from_quantized_statedict(data_dir, device='cpu', cache_intermediate_outputs=False, num_to_cache=None):
    state_dict = torch.load(path_join(data_dir, 'qfe_net.pt'), map_location=torch.device(device))
    kernel_by_layer = [v.shape[2] for k, v in state_dict.items() if 'feat_l' in k]
    n_layers = len(kernel_by_layer) + 1
    stride_by_layer = state_dict["stride_by_layer"]
    relu_by_layer = state_dict["relu_by_layer"]
    quantization = state_dict["quantization"]
    qfe_net = QuantizedFENet(   quantization[0],
                                quantization[1],
                                [1]*n_layers,
                                kernel_by_layer,
                                stride_by_layer,
                                relu_by_layer,
                                checkpoint_name=basename(data_dir),
                                pls=0,
                                dropout=0.0,
                                cache_intermediate_outputs=cache_intermediate_outputs,
                                num_to_cache=num_to_cache)
    del(state_dict["stride_by_layer"])
    del(state_dict["relu_by_layer"])
    del(state_dict["quantization"])
    qfe_net.load_state_dict(state_dict)
    qfe_net.to(device)
    return qfe_net

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
    fe_net = FENet(*model_config,
                   checkpoint_name=basename(checkpoint),
                    **{ k: config[k] for k in ['pls_dims', 'normalize_at_end'] + ['annealing_alpha', 'thermal_sigma', 'anneal'] if k in config }   # pass additional config kwargs if they are in the config
                   )
    fe_net.load_state_dict(fe_net_state)
    fe_net.to(device)
    return fe_net

def make_QFENet_from_FENet(wl: int, fl: int, fe_net: FENet, device, quantize_weights=True, cache_intermediate_outputs=False, num_to_cache=None):
    ret = QuantizedFENet(   wl,
                            fl,
                            fe_net.features_by_layer,
                            fe_net.kernel_by_layer,
                            fe_net.stride_by_layer,
                            fe_net.relu_by_layer,
                            fe_net.checkpoint_name + f"_wl{wl}_fl{fl}",
                            pls=fe_net.pls,
                            cache_intermediate_outputs=cache_intermediate_outputs,
                            num_to_cache=num_to_cache)
    if quantize_weights:
        ret.load_state_dict(quantize_state_dict(wl, fl, fe_net.state_dict()))
    else:
        ret.load_state_dict(fe_net.state_dict())
    ret.to(device)
    return ret


if __name__ == '__main__':
    import doctest
    doctest.testmod()
