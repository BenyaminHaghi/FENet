import torch
from torch import nn
from sklearn.linear_model import LinearRegression
import numpy as np
import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# from FENet_Training import Loss_Function as decoder_loss
# from FENet_Training import Loss_Function_Test as decoder_loss_test
# from FENet_Training import PLS_Generation
from FENet_parameterizable import inference_batch, FENet
from decoder import compute_linear_decoder_loss_preds, r2_score
from utils import convert_to_buf
from abc import ABC, abstractmethod

from math import sqrt

from configs import NOT_IMPLEMENTED # todo-clean: this is used as a placeholder input to arguments that aren't used in depended-on functions. Those argument should either be documented and used, or removed.
from configs import FENET_MEMLIMIT_SERIAL_BATCH_SIZE

def devicify(device, *args):
    return (x.to(device) for x in args)

def linear_decoder_loss(outputs, labels, device):
    outputs, labels = devicify(device, outputs, labels)
    loss, *_ = compute_linear_decoder_loss_preds(outputs, labels, device, quantization=None)
    return loss

def R2_avg_criterion(preds_dl, labels_dl, device, quantization=None):
    tot_r2 = 0
    for preds, labels in zip(preds_dl, labels_dl):
        tot_r2 += r2_score(labels, preds)
    return { 'timely/avg-decoder-R2': tot_r2 / len(preds_dl) }

def R2_hist_criterion(preds_dl, labels_dl, device, quantization=None):
    r2s = []
    for preds, labels in zip(preds_dl, labels_dl):
        r2 = r2_score(labels, preds)
        r2s.append(r2)
    wandb.log({ 'histogramhistogramhistogram': wandb.Histogram(np.array(r2s).flatten()) }, commit=False)
    return { 'timely/decoder-R2': wandb.Histogram(np.array(r2s).flatten()) }

def directional_R2_criterion(preds_dl, labels_dl, device, quantization=None):

    #labels_dl = [labels.cpu().detach().numpy() for labels in labels_dl]

    xcomp_dl = [(preds[:, 0], labels[:, 0]) for preds, labels in zip(preds_dl, labels_dl)]
    ycomp_dl = [(preds[:, 1], labels[:, 1]) for preds, labels in zip(preds_dl, labels_dl)]

    xcomp_r2s = [r2_score(labels, preds) for preds, labels in xcomp_dl]
    ycomp_r2s = [r2_score(labels, preds) for preds, labels in ycomp_dl]

    return {
        'timely/decoder-x-R2': sum(xcomp_r2s) / len(xcomp_r2s),
        'timely/decoder-y-R2': sum(ycomp_r2s) / len(ycomp_r2s),
        'timely/decoder-xy-avg-R2': (sum(xcomp_r2s) + sum(ycomp_r2s)) / len(xcomp_r2s) / 2,
        'timely/decoder-xy-norm-R2': sqrt(((sum(xcomp_r2s)/len(xcomp_r2s))**2 + (sum(ycomp_r2s)/len(ycomp_r2s))**2) / 2)
    }

def mean_squared_error_criterion(preds_dl, labels_dl, device):
    MSEs = [torch.nn.functional.mse_loss(
        torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds,
        torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels)
        for preds, labels in zip(preds_dl, labels_dl)]
    return {
        "decoder-MSE": sum(MSEs)/len(MSEs)
    }


def axes_plot_criterion(preds_dl, labels_dl, device, quantization=None, day_names=None, compatibility='wandb'):
    import matplotlib
    if compatibility == 'wandb': matplotlib.use('agg')   # see tag https://stackoverflow.com/a/7821917/10372825
    from matplotlib import pyplot as plt
    labels_dl = list(labels_dl)
    for i, (preds, labels) in enumerate(zip(preds_dl, labels_dl)):
        if isinstance(preds, torch.Tensor):
            preds_dl[i] = preds.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels_dl[i] = labels.cpu().detach().numpy()

    xcomp_dl = [(preds[:, 0], labels[:, 0]) for preds, labels in zip(preds_dl, labels_dl)]
    ycomp_dl = [(preds[:, 1], labels[:, 1]) for preds, labels in zip(preds_dl, labels_dl)]

    xcomp_r2s = [r2_score(labels, preds) for preds, labels in xcomp_dl]
    ycomp_r2s = [r2_score(labels, preds) for preds, labels in ycomp_dl]

    PLOT_START_INDEX = 0
    PLOT_FIRST_N_POINTS = 500
    if day_names is None: day_names = list(range(len(preds_dl)))
    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex='all', sharey='all', squeeze=True)
    for ax, ax_name, preds_labs_in_dim, r2s_in_dim in zip(axs, ['x', 'y'], [xcomp_dl, ycomp_dl], [xcomp_r2s, ycomp_r2s]):
        preds_dl, labels_dl = zip(*preds_labs_in_dim)
        for day, r2, preds in zip(day_names, r2s_in_dim, preds_dl):
            ax.plot(preds[PLOT_START_INDEX:PLOT_START_INDEX+PLOT_FIRST_N_POINTS], label=f"Day {day}, mean R²={r2:.4f}")
        ax.set_prop_cycle(None)    # reset the color cycle to align colors, https://stackoverflow.com/a/24283087/10372825
        for day, r2, labels in zip(day_names, r2s_in_dim, labels_dl):
            ax.plot(labels[PLOT_START_INDEX:PLOT_START_INDEX+PLOT_FIRST_N_POINTS], ":", label=f"Day {day} ground truth")
        ax.set_ylabel(ax_name + ' (normalized)')
        ax.set_ylim(-1.5, 1.5)
        ax.legend()

    axs[-1].set_xlabel("Time (steps)")
    np_fig = convert_to_buf(fig)
    # clear the figure
    plt.close('all')

    match compatibility:

        case 'wandb':
            return { 'timely/decoder-preds-chart': wandb.Image(np_fig) }
        case 'matplot':
            return { 'timely/decoder-preds-chart': np_fig }

def color_scatter_criterion(preds_dl, labels_dl, device, pls=None, quantization=None):
    """
    Create a scatter plot where each point has the HSV color of its label,
    to hopefully show clumps of colored points converging
    """
    for preds, labels in zip(preds_dl, labels_dl):
        preds = preds.cpu().detach().numpy()
        preds_dl.append(preds)

    fig, axes = plt.subplots(1, len(preds_dl), figsize=(5 * len(preds_dl), 5), sharex='all', sharey='all', squeeze=True)
    for ax, preds, labels in zip(axes, preds_dl, labels_dl):
        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        angles = np.arctan2(*labels.transpose())

        # rgb cycle from https://stackoverflow.com/a/59578106/10372825
        rgb_cycle = np.vstack((            # Three sinusoids
            .5*(1.+np.cos(angles          )), # scaled to [0,1]
            .5*(1.+np.cos(angles+2*np.pi/3)), # 120° phase shifted.
            .5*(1.+np.cos(angles-2*np.pi/3)))).T # Shape = (60,3)

        ax.scatter(*labels.transpose(), marker='2', c=rgb_cycle, alpha=0.01)
        ax.scatter(*preds.transpose(), c=rgb_cycle, alpha=0.1)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("x velocity")
        ax.set_ylabel("y velocity")

    np_fig = convert_to_buf(fig)
    plt.close('all')
    return { 'timely/decoder-color-scatter': wandb.Image(np_fig) }

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

    Compresses network dimension using the FENet.pls parameter. Set FENet.pls=0 to disable.

    `criteria` will be passed (outputs_dl, labels_dl) if it takes two arguments that end in "_dl",
               and (inputs_dl, outputs_dl, lables_dl) if it takes three.
            todo-refactor: this is super jank, but useful for some metrics that need the inputs_dl and others that don't. could just make taking all three the default.
    `criteria` should return a dictionary of the form { 'metric-name': 'wandb_loggable' }
    """
    from inspect import signature
    inputs_dl, labels_dl = list(zip(*test_dl))
    if preds_dl == None:

        if select_n and select_n < len(test_dl): test_dl = test_dl[:select_n]

        # fix: test_dl should be coerced to multitime-iterable, eg. if passed a zip w/o this then it will consume it then be sad the second time around
        if type(test_dl) != list and type(test_dl) != DataLoader: test_dl = list(test_dl)

        with torch.no_grad():
            preds_dl = [inference_batch(device, net, dim_red, decoder, inputs, labels, batch_size=FENET_MEMLIMIT_SERIAL_BATCH_SIZE, decoder_crossvalidate=True)\
                for inputs, labels in tqdm(test_dl, desc="evaluation: running model", leave=False, disable=silent)]
        if isinstance(preds_dl[0], torch.Tensor):
            preds_dl = [preds.cpu().detach().numpy() for preds in preds_dl]

    eval = {}
    for crit_fn in tqdm(criteria, desc="evaluating on criteria", leave=False, disable=silent):
        # inspect the signature of this criteria_function to see if it should take inputs_dl or just outpus_dl and labels_dl
        sig = signature(crit_fn)
        num_dl_args = len([x for x in sig.parameters if x.endswith('_dl')])
        if num_dl_args == 3:
            print("\n\n\nbeginning quantization eval!!")
            got = crit_fn(inputs_dl, preds_dl, labels_dl)
        elif num_dl_args == 2:
            got = crit_fn(preds_dl, labels_dl, device=device)
        else: raise NotImplementedError(f"expected each `criteria` fn to take 2 or 3 _dl arguments; instead, it appears to take {num_dl_args}. ({sig.parameters})")
        # now that we got the eval outputs, add them to the dictionary
        eval = { **eval, **got }

    eval = { 'eval/'+k: v for k, v in eval.items() }

    return eval
