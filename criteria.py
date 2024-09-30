import torch
import numpy as np
from tqdm import tqdm

from FENet_parameterizable import inference_batch, FENet
from abc import ABC, abstractmethod
import pandas as pd

from configs import FENET_MEMLIMIT_SERIAL_BATCH_SIZE

def devicify(device, *args):
    return (x.to(device) for x in args)

def pearson_r_squared_criterion(preds: np.ndarray, labels: np.ndarray):
    from scipy.stats import pearsonr
    pr2x = pearsonr(preds[:, 0], labels[:, 0]).statistic **2
    pr2y = pearsonr(preds[:, 1], labels[:, 1]).statistic **2

    return {
        'pearsonr2-x': pr2x,
        'pearsonr2-y': pr2y,
        'pearsonr2-xy-normed': np.sqrt((pr2x**2 + pr2y**2) / 2)
    }

def mean_squared_error_criterion(preds, labels):
    if not isinstance(preds, torch.Tensor): preds = torch.tensor(preds, requires_grad=False)
    if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels, requires_grad=False)

    return {
        "decoder-MSE": torch.nn.functional.mse_loss(preds, labels).item()
    }



class FENetCriterion(ABC):
    """
    An abstract criterion class that might cache downstream-tuned models
    (eg. linear decoders) used to evaluate the model output.
    """
    @abstractmethod
    def eval_batch(self, outputs, labels) -> torch.Tensor:
        """
        Evaluate the quality of a single batch of model output. Acts with
        normal torch evaluation semantics.
        """
        pass

    @abstractmethod
    def evaluate(self, outputs_dl, labels_dl) -> dict:
        """
        Evaluate the quality of each model output in the dataloader,
        possibly by tuning a downstream model which can be expensive.
        """
        pass

    @abstractmethod
    def evaluate_cheaply(self, outputs_dl, labels_dl) -> dict:
        """
        Evaluate the quality of each model output in the dataloader,
        using cached training parameters if applicable and possible.
        """
        pass

@DeprecationWarning
class CriterionConverter(FENetCriterion):
    """
    Wraps a standard pytorch criterion to use the FENetCriterion interface

    ```python
    fe_net_criterion = CriterionConverter(nn.MSELoss)
    # ... (inside train loop)
        loss = fe_net_criterion.evaluate(outputs, labels)
        loss.backward()
    ```
    """
    def __init__(self, name, functional_criterion):
        """Expects a functional criterion, such as torch.nn.CrossEntropyLoss"""
        self.criterion = functional_criterion
        self.name = name

    def eval_batch(self, outputs, labels):
        return self.criterion(outputs, labels)

    def evaluate(self, outputs_dl, labels_dl):
        tot = len(outputs_dl)
        met_sum = sum(self.eval_batch(outputs, labels) for outputs, labels in zip(outputs_dl, labels_dl))
        return { self.name: met_sum / tot }

    def evaluate_cheaply(self, outputs_dl, labels_dl):
        return self.evaluate(outputs_dl, labels_dl)

class EfficiencyCriterion(FENetCriterion):
    # FIXME: convert to just a pure function of a FENet
    def __init__(self, fe_net: FENet):
        """
        Figure out how many calculations are needed for one inference
        """
        self.fe_net = fe_net

    def evaluate(self, input_seq_len):
        seq_len = input_seq_len
        cost = 0
        for stride, kernel in zip(self.fe_net.stride_by_layer, self.fe_net.kernel_by_layer):
            seq_len = (seq_len + max(kernel - stride, 0) - 1)//stride   # size of the output for this layer
            cost += 2 * seq_len * kernel                                # cost is porportional to size of output
                                                                        # because each output = one application
                                                                        # of the kernel
        return { 'efficiency/operations-per-eval': cost }

    def eval_batch(self, outputs, labels) -> torch.Tensor:
        raise NotImplementedError("you shouldn't be using EfficiencyCriterion as a loss metric!")

    def evaluate_cheaply(self, input_seq_len) -> dict:
        return self.evaluate(input_seq_len)

def evaluate_with_criteria(net, dim_red, decoder, test_dl, criteria, device, preds_dl=None, quantization=None, select_n=None, silent=False):
    """
    Runs `net` on `test_dl`, then takes the output and labels and passes it to `criteria` to get an output.

    Uses FENet_Parameterizable.inference_batch(), which compresses network dimension using the FENet.pls parameter. Set FENet.pls=0 to disable.

    `criteria` will be passed (outputs_dl, labels_dl) if it takes two arguments that end in "_dl",
               and (inputs_dl, outputs_dl, lables_dl) if it takes three.
                    some metrics that need the inputs_dl and others that don't. could just make taking all three the default.
    `criteria` should return a dictionary of the form { 'metric-name': 'wandb_loggable' }
    """
    if preds_dl is not None: raise NotImplementedError("you must not provide your own preds_dl, as evaluate_with_criteria will call inference_batch and grab metrics directly")

    if select_n and select_n < len(test_dl): test_dl = test_dl[:select_n]   # FIXME: should use itertools slice in case test_dl is not indexable

    with torch.no_grad():
        preds_dl = [
            inference_batch(
                device,
                net,
                dim_red,
                decoder,
                inputs,
                labels,
                batch_size=FENET_MEMLIMIT_SERIAL_BATCH_SIZE,
                decoder_crossvalidate=True,
                crit_fns = criteria
            )
            for inputs, labels in tqdm(test_dl, desc="evaluation: running model", leave=False, disable=silent)
        ]
    
    return pd.DataFrame(preds_dl).mean().to_dict()




