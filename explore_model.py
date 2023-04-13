import torch
from qtorch.auto_low import lower, sequential_lower
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import wandb
import numpy as np
import seaborn as sns
import pandas as pd

from copy import deepcopy
from os.path import join as path_join

from FENet_parameterizable import FENet, QuantizedFENet, make_fenet_from_checkpoint
from data_parser import make_total_training_data
from criteria import LinearDecoderCriterion, QuantizationCriterion, evaluate_with_criteria, directional_R2_criterion
from main_sweeps import DATA_DIR
from utils import make_outputs_directory
from multiprocessing import set_start_method


# NO CONFIG
# MODEL_PATH = "F:\\Albert\\FENet\\wandb_saves\\genial-sweep-57_step-1160_perf-0.6615"; MODEL_SHAPE = [[1]*8, [10, 10, 2, 4, 32, 34, 26], [3, 2, 30, 13, 23, 25, 27]]
# MODEL_PATH, MODEL_SHAPE = "F:\\Albert\\FENet\\wandb_saves\\desert-sweep-21_step-1390_perf-0.6526", [[1]*7, [14, 20, 38, 26, 22, 10], [6, 4, 11, 18, 3, 8]]

MODEL_PATHS = [
    "F:\\Albert\\FENet\\wandb_saves_with_config\\genial-sweep-57_step-1160_perf-0.6615",
    # "F:\\Albert\\FENet\\wandb_saves_with_config\\neat-sweep-60_step-570_perf-0.6405",
    # "F:\\Albert\\FENet\\wandb_saves_with_config\\flowing-sweep-86_step-310_perf-0.6512",
    # "F:\\Albert\\FENet\\wandb_saves_with_config\\flowing-sweep-86_step-570_perf-0.6504",
    # "F:\\Albert\\FENet\\wandb_saves_with_config\\toasty-sweep-173_step-360_perf-0.6486",
]

QUANTIZE_WEIGHTS = True
USING_WANDB = True
MAKE_CHARTS = True

SAMPLE_SIZE = 50
# SAMPLE_SIZE = 1
# WLFL_PAIRS = [(8, 4), (8, 5)]
# WLFL_PAIRS = [[(8, 4), (8, 5), (8, 6), (8, 7)], [, (9, 4), (9, 5), (9, 6), (9, 7)]]

def make_model_weights_histogram(model, savepath):
    weights = []
    for mat in model.state_dict().values():
        weights += mat.flatten().tolist()
    fig, ax = plt.subplots()
    ax.hist(weights, bins=50)
    ax.set_xlabel('Weight size')
    ax.set_ylabel('# of weights')
    ax.set_title(f"Histogram of weights in model, min={min(weights):.4f}, max={max(weights):.4f}")
    fig.savefig(savepath)

def draw_values_histogram_from_QFENet(ax, fe_net: QuantizedFENet, title_suffix=None):
    def flatten_once(it):
        return [y for x in it for y in x]
    # def plot_histogram_sequence(ax, sequences):
    #     x = flatten_once([[i] * len(seq) for i, seq in enumerate(sequences)])
    #     y = flatten_once(sequences)
    #     return ax.hist2d(x, y, bins=40)

    def plot_histogram_sequence_norm_over_layers(ax, sequences):
        """
        Plots a series of histograms vertically, but normalizes them all so that each column is readable
        So that a very large number of points in one histogram doesn't wash out the lower density in others
        """
        import seaborn as sns
        from utils import get_hist_bin_sizes
        min_v, max_v = min(min(x) for x in sequences), max(max(x) for x in sequences)
        n_bins = 15
        heights = [get_hist_bin_sizes(seq, min_v=min_v, max_v=max_v, bins=n_bins) for seq in sequences]
        sns.heatmap(list(zip(*heights)), yticklabels=[f"{min_v * (1 - i/n_bins) + max_v * (i/n_bins):.3f}" for i in range(n_bins)], ax=ax, cbar=False)

    # got = plot_histogram_sequence(ax, fe_net.values_hist_data)
    got = plot_histogram_sequence_norm_over_layers(ax, fe_net.values_hist_data)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weight size")
    # fig.colorbar(got[3], ax=ax)
    ax.set_title(f"min={min(flatten_once(fe_net.values_hist_data)):.3f}, max={max(flatten_once(fe_net.values_hist_data)):.3f}{title_suffix if title_suffix else ''}")

def draw_quantization_performance_line_charts_with_error(ax, wlfl_pairs, fe_net, val_dl, sample_size, device, silent=False, key_format="wl={wl} fl={fl}", xlabel='Quantization', title=None):
    crit = LinearDecoderCriterion(device)
    quantization_crit = QuantizationCriterion(fe_net, wlfl_pairs, device, key_fmt=key_format, quantize_decoder=True)
    eval_outputs = []
    for i in trange(sample_size, desc="sampling quantization", leave=False, disable=silent):
        inputs_dl, labels_dl = zip(*val_dl)
        to_append = quantization_crit.evaluate(inputs_dl, None, labels_dl)
        eval_outputs.append(to_append)

    df = pd.DataFrame(eval_outputs)
    df = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(df)
    count, mean, std, mn, p05, p25, p50, p75, p95, mx = [list(x) for i, x in df.iterrows()]
    xs = [key_format.format(wl=wl, fl=fl) for wl, fl in wlfl_pairs]

    ax.plot(xs, p50, color="darkorchid", label="median")
    ax.fill_between(xs, p25, p75, alpha=0.3, color="darkorchid", label="25% - 75%")
    ax.fill_between(xs, p05, p95, alpha=0.1, color="darkorchid", label="5% - 95%")
    if title: ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Combined R²')
    ax.legend()
    return df

def make_quantization_decoder_histograms(fe_net, val_dl, device, savepath):
    from FENet_parameterizable import inference_batch
    decoder_crit = LinearDecoderCriterion(device)
    outputs_list = []
    preds_list = []
    for inputs, labels in val_dl:
        outputs = inference_batch(device, fe_net, inputs, labels)
        outputs_list += outputs.flatten().tolist()

        _, preds = decoder_crit.make_loss_preds_batch(outputs, labels, force_recalc=True, quantization=None)
        preds_list += preds.flatten().tolist()

    fig, ax = plt.subplots()
    ax.hist(outputs_list, label="Decoder inputs", bins=50, alpha=0.3, color="darkgoldenrod")
    ax2 = ax.twinx()
    ax2.hist(preds_list * 5, label="Decoder outputs", bins=50, alpha=0.3, color="darkorchid")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.set_xlabel("Weight size")
    ax.set_ylabel("# of inputs")
    ax2.set_ylabel("# of outputs")

    # lines, labels = zip(*[x.get_legend_handles_labels() for x in [ax, ax2]])
    # print(lines, labels)
    # lines = ax.get_lines() + ax2.get_lines()
    # labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    # ax.legend(sum(lines), sum(labels), loc=0)
    fig.savefig(savepath, dpi=150)

def make_quantization_stochastic_error_charts(wls, fls, fe_net, val_dl, sample_size, device, output_dir, silent=False):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    median_by_sl_fl = np.zeros((len(wls), len(fls)))
    for i, (wl, ax) in enumerate(zip(wls, axs.flatten())):
        ax.axhline(y=baseline_r2, label="Baseline (no quantization)")
        fig.savefig(path_join(output_dir, 'quantization_performance.png'), dpi=150)
        got = draw_quantization_performance_line_charts_with_error(ax, [(wl, fl) for fl in fls], fe_net, val_dl, sample_size, device,
                                                                   key_format="{fl}", xlabel="Fractional Length", title=f"Quantization Performance with word length {wl}")
        median_by_sl_fl[i] = list(got.loc['50%', :])
    fig.savefig(path_join(output_dir, 'quantization_performance.png'), dpi=150)

    print(median_by_sl_fl)
    fig, ax = plt.subplots()
    xticklabels = fls
    yticklabels = wls
    sns.heatmap(median_by_sl_fl, ax=ax, vmin=0, vmax=0.7, xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_xlabel("Fractional Length")
    ax.set_ylabel("Word Length")
    fig.savefig(path_join(output_dir, 'quantization_performance_heatmap.png'), dpi=150)

if __name__ == '__main__':
    device = 'cuda'
    # crit = LinearDecoderCriterion(device)
    set_start_method("spawn")
    _, val_dl, _ = make_total_training_data(DATA_DIR)
    print("got the data")

    for checkpoint in MODEL_PATHS:
        fe_net = make_fenet_from_checkpoint(checkpoint, device, pls_dims=0)
        baseline_r2 = evaluate_with_criteria(fe_net, val_dl, criteria=[directional_R2_criterion], device=device)['eval/timely/decoder-xy-norm-R2']

        if MAKE_CHARTS:
            output_dir = make_outputs_directory(checkpoint)
            # make_model_weights_histogram(fe_net, savepath=path_join(output_dir, "weights_histogram.png"))
            make_quantization_decoder_histograms(fe_net, val_dl, device, savepath=path_join(output_dir, "decoder_values_histogram.png"))
            # make_quantization_stochastic_error_charts([8, 9, 10, 11], list(range(4, 8+1)), fe_net, val_dl, SAMPLE_SIZE, device, output_dir)






    # fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # fig2, values_axs = plt.subplots(2, 3, figsize=(16, 8))

    # for fl, (ax, values_ax) in enumerate(zip(tqdm(axs.flatten(), desc="frac lens"), values_axs.flatten()), start=3):
    #     xs = list(range(fl, fl+7))
    #     ys = []
    #     for wl in tqdm(xs, desc="word lens", leave=False):

    #         evaled_lp = get_model_performance_on_precision(wl, fl, fe_net, val_dl, device, QUANTIZE_WEIGHTS, USING_WANDB)
    #         ys.append(evaled_lp['eval/decoder-retrain/R²'])
    #         if wl == 9: draw_values_histogram_from_QFENet(values_ax, net, title_suffix=f" wl={wl}_fl={fl}")
    #         suffix = '___with_q_weights' if QUANTIZE_WEIGHTS else ''
    #         if MAKE_CHARTS: fig2.savefig(f"out/explore_model_values_hist{suffix}.png", dpi=100)

    #     ax.plot(xs, ys, label=f"fl={fl}")
    #     ax.axhline(y=evaled['eval/decoder-retrain/R²'])
    #     ax.set_ylim(0, 0.7)
    #     ax.set_ylabel("validation R²")
    #     ax.set_xlabel("wl")
    #     ax.legend()
    #     if MAKE_CHARTS: fig.savefig(f"out/explore_model_by_precision_neo{suffix}.png", dpi=100)

    # suffix = '___with_q_weights' if QUANTIZE_WEIGHTS else ''
    # if MAKE_CHARTS: fig.savefig(f"out/explore_model_by_precision_neo{suffix}.png", dpi=100)
    # if MAKE_CHARTS: fig2.savefig(f"out/explore_model_values_hist{suffix}.png", dpi=100)

    # # # print("qtized eval", evaled_lp)
