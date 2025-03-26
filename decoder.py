import torch
from torch import nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from multiprocessing import set_start_method
from multiprocessing import Pool
from multiprocessing import get_context
from multiprocessing import shared_memory
import sys

from typing import Tuple, Optional

from configs import NOT_IMPLEMENTED
from configs import PLS_TRAIN_TEST_RATIO
from configs import MAX_POOL_WORKERS
from configs import USE_MULTITHREADED_PLS_TRAINING
from configs import THREAD_CONTEXT

def train_pls(outputs: np.ndarray, labels: np.ndarray, indx, n_feat):
    #Expects outputs shape (n_samples, 1) and lables (n_samples, 2) since we are training on each
    #channel individually. n_feat is the number of output features after dimensionality reduction.
    #train size is the number of samples the pls model will be trained on before pls is applied to the
    #entire model. 
    pls = PLSRegression(n_components=n_feat, copy=False)
    pls = pls.fit(outputs[:, indx, :], labels)

    mean = np.asarray(pls._x_mean)
    std_dev = np.asarray(pls._x_std)
    pls_weights = np.asarray(pls.x_rotations_)

    return mean, std_dev, pls_weights

class SimplePLS():
    """PLS_Model without training state, that gets retrained each time"""
    def __init__(self, n_channels, n_in_dims, n_out_dims):
        self.n_channels = n_channels
        self.n_in_dims = n_in_dims
        self.n_out_dims = n_out_dims

    def fit_transform(self, x, labels, test_x=None, n_channels=None, n_in_dims=None, n_out_dims=None):
        """expects x.shape = (n_chunks, n_channels*n_in_dims) and labels.shape = (n_chunks, n_channels*n_pls=2)"""
        if n_channels is None: n_channels = self.n_channels
        if n_in_dims is None: n_in_dims = self.n_in_dims
        if n_out_dims is None: n_out_dims = self.n_out_dims
        def PLS_Generation(X, y, x_test, num,wt_feature_num = 8, pls_features_num = 2):
            X_numpy = X.cpu().detach().numpy() if isinstance(X, torch.Tensor) else X
            y_numpy = y.cpu().detach().numpy() if isinstance(y, torch.Tensor) else y
            my_flag = False
            for i in range(num):
                pls = PLSRegression(n_components = pls_features_num)
                pls.fit(X_numpy[:,i*wt_feature_num:(i+1)*wt_feature_num],y_numpy)
                self.pls_weights = pls.x_rotations_
                self.pls_weights = torch.Tensor(self.pls_weights).type(torch.FloatTensor).to(X.device)

                if my_flag == False:
                    train_x_temp = torch.matmul(X[:,i*wt_feature_num:(i+1)*wt_feature_num], self.pls_weights)
                    if test_x is not None: test_x_temp = torch.matmul(test_x[:,i*wt_feature_num:(i+1)*wt_feature_num], self.pls_weights)
                    my_flag = True
                else:
                    train_x_temp = torch.cat((train_x_temp, torch.matmul(X[:,i*wt_feature_num:(i+1)*wt_feature_num], self.pls_weights)) , dim = 1)
                    if test_x is not None: test_x_temp = torch.cat((test_x_temp, torch.matmul(test_x[:,i*wt_feature_num:(i+1)*wt_feature_num], self.pls_weights)) , dim = 1)

            if test_x is not None:
                return train_x_temp, test_x_temp
            return train_x_temp
        return PLS_Generation(x, labels, test_x, n_channels, n_in_dims, n_out_dims)



class PLS_Model():
    def __init__(self, n_channels, n_in_dims, n_out_dims, train_batch_size, device):
        self.mean = torch.zeros((n_channels, n_in_dims))
        self.std_dev = torch.zeros((n_channels, n_in_dims))
        self.pls_weights = torch.zeros((n_channels, n_in_dims, n_out_dims))
        self.n_in_dims = n_in_dims
        self.n_out_dims = n_out_dims
        self.n_channels = n_channels
        self.train_batch_size = train_batch_size
        self.device = device
        self.trained=False

    def forward(self, x):
        try:
            if(self.Trained):
                x.to(self.device)
                n_samples, n_channels, n_features = x.shape
                if(n_channels != self.n_channels or n_features != self.n_in_dims):
                    tb = sys.exc_info()[2]
                    raise TypeError().with_traceback(tb)

                mean = self.mean.to(self.device)
                std_dev = self.std_dev.to(self.device)
                pls_weights = self.pls_weights.to(self.device)

                x = (x - mean)
                del mean
                torch.cuda.empty_cache()
                x = x / std_dev.unsqueeze(0)
                del std_dev
                torch.cuda.empty_cache()
                x = torch.matmul(x.transpose(0,1), pls_weights)
                del pls_weights
                torch.cuda.empty_cache()

                return x.transpose(0,1).reshape(n_samples,-1).float()
            else:
                tb = sys.exc_info()[2]
                raise ReferenceError().with_traceback(tb)
        except ReferenceError:
            print("PLS dimension reduction used before it was trained")
        except TypeError:
            raise TypeError("Number of features invalid for n_channels: ", self.n_channels, " and n_in_dims: ", self.n_in_dims)

    def train(self, outputs: torch.Tensor, labels_np: np.ndarray):
        #Model to reduct the total number of dimensions of the outpus using PLS. This function
        #utilizes multithreading to run separate pls training on each channel as is decided by ben
        #expects outputs of shape (n_samples, nchannels*n_features_per_channel), labels of shape
        #(n_samples, n_lables_per_sample), n_feat is the number of features expected after dimensionality
        #reduction, and train ratio is the ratio of pls training samples to the total size of the batch
        n_samples, n_channels, n_feats = outputs.shape

        outputs_np = outputs[0:self.train_batch_size, :, :].cpu().detach().numpy()
        if(USE_MULTITHREADED_PLS_TRAINING):

            try:
                shm0 = shared_memory.SharedMemory(create=True, name='shared_labels', size=labels_np[0:self.train_batch_size, :].nbytes)
                shm1 = shared_memory.SharedMemory(create=True, name='shared_outputs', size=outputs_np.nbytes)
                shared_labels = np.ndarray(labels_np[0:self.train_batch_size, :].shape, dtype=labels_np[0:self.train_batch_size, :].dtype, buffer=shm0.buf)
                shared_outputs = np.ndarray(outputs_np.shape, dtype=outputs_np.dtype, buffer=shm1.buf)
                shared_labels[:] = labels_np[0:self.train_batch_size, :]
                shared_outputs[:] = outputs_np
                pls_args = [(shared_outputs, shared_labels, i, self.n_out_dims) for i in range(n_channels)]
                with get_context(THREAD_CONTEXT).Pool(min(MAX_POOL_WORKERS, len(pls_args))) as pool:
                    pls_model = pool.starmap(train_pls, pls_args)
            finally:
                shm0.close()
                shm0.unlink()
                shm1.close()
                shm1.unlink()
        else:
            pls_model = [train_pls(outputs_np, labels_np[0:self.train_batch_size, :], i, self.n_out_dims) for i in range(self.n_channels)]

        mean = []
        std_dev = []
        pls_weights = []
        for mdl in pls_model:
            mean.append(mdl[0])
            std_dev.append(mdl[1])
            pls_weights.append(mdl[2])
        self.mean = torch.tensor(np.asarray(mean), requires_grad=False)
        self.std_dev = torch.tensor(np.asarray(std_dev), requires_grad=False)
        self.pls_weights = torch.tensor(np.asarray(pls_weights), requires_grad=False)
        self.Trained = True

class Linear_Decoder():
    def __init__(self, train_batch_size, device, quantization=None):
        self.weights = torch.tensor([])
        self.biases = torch.tensor([])
        self.device = device
        if quantization is not None:
            #print("about to import quantization")
            from qtorch import FixedPoint
            from qtorch.quant import Quantizer
            quantize = Quantizer(
                                    forward_number=FixedPoint(wl=quantization[0],
                                                              fl=quantization[1],
                                                                           clamp=True,
                                                                       symmetric=True),
                                 forward_rounding='nearest')
            #print("Loaded Quantizer")
        else:
            quantize = lambda x: x

        self.quantize = quantize
        self.train_batch_size = train_batch_size
        self.trained=False

    def forward(self, x):
        try:
            if(self.trained):
                x = self.quantize(x)
                x.to(self.device)
                weights, biases = self.weights.to(self.device), self.biases.to(self.device)
                mul = torch.matmul(x, torch.transpose(weights, 0, 1))
                return mul + biases
            else:
                tb = sys.exc_info()[2]
                raise ReferenceError().with_traceback(tb)
        except ReferenceError:
            print("PLS dimension reduction used before it was trained")

    def train(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Fits Linear Regression to `outputs`, keeping data in tensors when possible.
        expects outputs shape (n_samples, n_feats) and labels shape (n_samples, 2)?
        """
        # regress
        labels_np = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
        outputs = self.quantize(outputs)
        reg = LinearRegression().fit(outputs[0:self.train_batch_size, :].cpu().detach().numpy(), labels_np[0:self.train_batch_size])

        weights = torch.tensor(reg.coef_, requires_grad=False)
        biases = torch.tensor(reg.intercept_, requires_grad=False)
        weights, biases = self.quantize(weights), self.quantize(biases)
        self.weights = weights
        self.biases = biases
        self.trained = True
        
class Color_Decoder():
    def __init__(self, arrayMap, pixel_size=100, pixX=10, pixY=10, numColors=3, numArrays=2, colorDialation=8, colorRes=256):
        self.pixX = pixX
        self.pixY = pixY
        self.numColors = numColors
        self.videoStack = [None]
        self.arrayMap=arrayMap
        self.pixel_size=pixel_size
        self.numArrays=numArrays
        self.dialation = colorDialation
        self.colorRes = colorRes
        self.trained=True

    def forward(self, x):

        num_chunks, num_features = x.shape

        if(isinstance(x, torch.Tensor)):
            x = x.reshape(num_chunks, self.numArrays, self.pixX*self.pixY-4, self.numColors).cpu().detach().numpy()
        frame = np.zeros([self.pixX, self.pixY, self.numColors], dtype=np.uint8)
        frame_stack = np.empty([self.numArrays, num_chunks, self.pixX*self.pixel_size, self.pixY*self.pixel_size, self.numColors], dtype=np.uint8)

        for s, sample in enumerate(x):
            for e, elecArray in enumerate(sample):
                for i, numChan in enumerate(self.arrayMap['ChanNum'].flatten()):
                    frame[self.arrayMap['Row'].flatten()[i],self.arrayMap['Column'].flatten()[i], :] = np.uint8(np.clip((elecArray[numChan-1, :] + self.dialation/2)*self.dialation, 0, self.colorRes-1))
                npimgframe = np.repeat(frame, self.pixel_size, axis=0)
                npimgframe = np.repeat(npimgframe, self.pixel_size, axis=1)
                frame_stack[e,s,:] = npimgframe
        return frame_stack
    def train(self, outputs: torch.Tensor, labels_np: np.ndarray):
        return

class RNN_decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_dims,
                 n_layers,
                 output_size,
                 training_epochs,
                 learning_rate,
                 optim_eps,
                 weight_decay,
                 device,
                 quantization=None):
        from torch import optim
        from torch.nn import RNN
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        self.output_size = output_size
        self.training_epochs = training_epochs
        self.lr = learning_rate
        self.eps = optim_eps
        self.wd = weight_decay
        if quantization is not None:
            #print("about to import quantization")
            from qtorch import FixedPoint
            from qtorch.quant import Quantizer
            quantize = Quantizer(
                                    forward_number=FixedPoint(wl=quantization[0],
                                                              fl=quantization[1],
                                                                           clamp=True,
                                                                       symmetric=True),
                                 forward_rounding='nearest')
            #print("Loaded Quantizer")
        else:
            quantize = lambda x: x

        self.quantize = quantize
        self.device = device
        self.hidden_layers = torch.zeros(self.n_layers, self.input_size, self.hidden_dims)
        self.rnn = RNN(input_size, hidden_dims, output_size, n_layers )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.wd)
        self.trained=False
        self.train_iters_completed = 0

    def forward(self, x):
        try:
            if(self.trained):
                results = []
                for element in x:
                    out, hidden = nn.rnn(element, self.hidden)
                    results.append(out)
                    self.hidden = hidden
                outs = torch.cat(outs, dim=1)
                return outs
            else:
                tb = sys.exc_info()[2]
                raise ReferenceError.with_traceback(tb)
        except ReferenceError:
            print("PLS dimension reduction used before it was trained")

    def train(self, outputs: torch.Tensor):
        if(self.train_iters_completed == 0):
            self.model.train()
            self.optimizer.zero_grad()
        results = []
        for element in outputs:
            out, hidden = nn.rnn(element, self.hidden)
            results.append(out)
            self.hidden = hidden
        outs = torch.cat(outs, dim=1)
        if(self.train_iters_completed >= self.training_epochs):
            self.model.eval()
            self.trained = True
        return outs

    def step(self):
        self.optimizer.step()

def compute_linear_decoder_loss_preds(outputs: torch.Tensor, labels: torch.Tensor, device, criterion=None, quantization: Optional[Tuple[int, int]]=None):

    if(criterion == None):
        criterion = nn.MSELoss(reduction='mean')

    if quantization is not None:
        #print("about to import quantization")
        from qtorch import FixedPoint
        from qtorch.quant import Quantizer
        quantize = Quantizer(forward_number=FixedPoint(wl=quantization[0], fl=quantization[1], clamp=True, symmetric=True), forward_rounding='nearest')
        #print("Loaded Quantizer")
    else:
        quantize = lambda x: x
    outputs = quantize(outputs)

    """
    Fits Linear Regression to `outputs`, keeping data in tensors when possible.
    """
    # regress
    reg = LinearRegression().fit(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())

    weights, biases = torch.Tensor(reg.coef_), torch.Tensor(reg.intercept_)
    weights, biases = weights.to(device), biases.to(device)
    weights, biases = quantize(weights), quantize(biases)

    #print("linear regression weights shape:", weights.shape)

    # predict for backprop
    def apply_linear_regression_coefficients(weights, biases, X):
        mul = torch.matmul(X, torch.transpose(weights, 0, 1))
        return mul + biases
    pred = apply_linear_regression_coefficients(weights, biases, outputs)

    loss = criterion(pred, labels)
    return loss, pred, weights, biases

def linear_decoder_predict_day(outputs: torch.Tensor, labels: torch.Tensor, device, quantization: Optional[Tuple[int, int]]=None):
    if quantization is not None:
        #print("about to import quantization")
        from qtorch import FixedPoint
        from qtorch.quant import Quantizer
        quantize = Quantizer(forward_number=FixedPoint(wl=quantization[0], fl=quantization[1], clamp=True, symmetric=True), forward_rounding='nearest')
        #print("Loaded Quantizer")
    else:
        quantize = lambda x: x
    outputs = quantize(outputs)

    """
    Fits Linear Regression to `outputs`, keeping data in tensors when possible.
    """
    # regress
    reg = LinearRegression().fit(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())

    weights, biases = torch.Tensor(reg.coef_), torch.Tensor(reg.intercept_)
    weights, biases = weights.to(device), biases.to(device)
    weights, biases = quantize(weights), quantize(biases)

    #print("linear regression weights shape:", weights.shape)

    # predict for backprop
    def apply_linear_regression_coefficients(weights, biases, X):
        mul = torch.matmul(X, torch.transpose(weights, 0, 1))
        return mul + biases
    pred = apply_linear_regression_coefficients(weights, biases, outputs)

    return pred


def r2_score(labels, preds):
    assert preds.shape == labels.shape

    if isinstance(preds, torch.Tensor):
        if isinstance(labels, np.ndarray):
            labels = torch.Tensor(labels).to(preds.device)
        if len(preds.shape) == 1:
            preds = preds.unsqueeze(1)
            labels = labels.unsqueeze(1)
        mean = torch.mean(labels, dim=0)
        SSE = torch.sum(torch.linalg.vector_norm(preds - labels, dim=1)**2)
        SSyy = torch.sum(torch.linalg.vector_norm(labels - mean, dim=1)**2)
        return (1 - (SSE / SSyy)).cpu().detach().numpy()
    if isinstance(preds, np.ndarray):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()
        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, 1)
            labels = np.expand_dims(labels, 1)
        mean = np.mean(labels, axis=0)
        SSE = np.sum(np.linalg.norm(preds - labels, axis=1)**2)
        SSyy = np.sum(np.linalg.norm(labels - mean, axis=1) ** 2)
        return 1 - (SSE / SSyy)

    raise NotImplementedError(f"cannot compute R2 for type {type(preds)}, expected torch.Tensor or np.ndarray")

