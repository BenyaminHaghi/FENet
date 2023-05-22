import wandb
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd
import seaborn as sns
import numpy as np
from rich import print
from scipy import stats as st
from tqdm import tqdm

from operator import itemgetter
from datetime import datetime
import json

N = 8
WANDB_ENTITY = "mics-fenet"
WANDB_PROJECT = "20230324_fenet_sweeps_ben"
METRIC = 'cross-validation-avg-r2'
PERF_METRIC = 'efficiency/operations-per-eval'
params_to_plot = [f"kernel{i}" for i in range(1, N)] + [f"stride{i}" for i in range(1, N)] + ['n_feat_layers']
print(params_to_plot)

def calculate_avg_best_r2(run):
    # df = pd.DataFrame(run.history)
    pass

def make_hyperparameter_impact_plot(df, key, fname=None, xlabel=None, ylabel=None):
    fname = fname or f"out_{key}.png"
    xlabel = xlabel or key
    ylabel = ylabel or "Cross-validated R$^2$"
    overlay_xlegend = "Mean R$^2$"

    # TODO: scuffed
    if 'kernel' in key or 'stride' in key:
        layer = int(key.replace('kernel', '').replace('stride', ''))
        df = df[df['n_feat_layers'] > layer]    # > not >= because # of layers = n_feats (ie. n_feat_layers) -1

    grouped = df[[key, METRIC]].groupby(key)
    agg = grouped.aggregate(['mean', 'count', st.sem])

    #####################################
    # base sns.FacetGrid for histograms #
    #####################################
    # TO DO apparently sns.relplot is better?
    grid = sns.FacetGrid(df, col=key, height=4, aspect=0.2, gridspec_kws={ 'wspace': 0.1 })
    grid.map_dataframe(sns.histplot, y=METRIC)
    grid.set_ylabels(ylabel)
    # remove lines
    for ax in grid.axes.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    for ax in grid.axes.flat[1:]:
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis='y', left=False)
        ax.set_xlabel(None)
        ax.set_xticks([])
    leftmost_ax = grid.axes.flat[0]
    leftmost_ax.set_xlabel("# Sweeps", fontsize=8)
    leftmost_ax.xaxis.set_label_position('top')
    # leftmost_ax.xaxis.set_ticks(10**np.arange(1, np.log10(agg[METRIC, 'count'].max())))
    max_hist_height = np.floor(np.log10(agg[METRIC, 'count'].max())); leftmost_ax.xaxis.set_ticks(10**np.array([max(max_hist_height-1, 1), max_hist_height]))
    leftmost_ax.spines["top"].set_visible(True)
    leftmost_ax.tick_params(axis='x', top=True, labeltop=True, labelsize=9)

    grid.set_titles(col_template="{col_name}", y=-0.075, x=0.15, fontsize=12)
    grid.figure.supxlabel(f"{key}")


    #####################################
    # overlay plot with mean line chart #
    #####################################
    x_vals = np.array(list(grouped.groups.keys()))

    # Add a new axis on top of the grid
    bboxes = [grid.axes.flat[0].get_position(), grid.axes.flat[-1].get_position()]  # assume that the first and last axes are the furthest corners; you could also just get_position every single one
    containing_bbox = Bbox.union(bboxes)
    overlay_ax = grid.figure.add_axes(containing_bbox)
    # possible_values = np.sort(df[key].unique())
    overlay_ax.set_xlim(x_vals[0], x_vals[-1] + (x_vals[-1]-x_vals[-2])*0.9)
    overlay_ax.set_ylim(grid.axes.flat[0].get_ybound())

    # make transparent
    sns.despine(ax=overlay_ax, left=True, bottom=True)
    overlay_ax.set_xticks([])
    overlay_ax.set_yticks([])
    overlay_ax.patch.set_alpha(0)

    # calculate overlayplot data and plot it
    mean_yvals = agg[METRIC, 'mean']
    confidence_interval_95 = np.array([
        st.t.interval(
            confidence=0.95,
            df=c-1,
            loc=m,
            scale=sem
        )
        for _, (m, c, sem) in agg.iterrows() ]).T

    overlay_ax.plot(x_vals, mean_yvals, label=overlay_xlegend)
    overlay_ax.fill_between(x_vals, confidence_interval_95[0], confidence_interval_95[1], alpha=0.2, label='95% Confidence')
    overlay_ax.legend(loc='lower right')

    plt.savefig('out/' + fname, dpi=300)


def make_hyperparameter_impact_boxplot(df, key, fname=None, xlabel=None, ylabel=None):
    fname = fname or f"boxplot_out_{key}.png"
    xlabel = xlabel or key
    ylabel = ylabel or "Cross-validated R$^2$"

    # TODO: scuffed
    if 'kernel' in key or 'stride' in key:
        layer = int(key.replace('kernel', '').replace('stride', ''))
        df = df[df['n_feat_layers'] > layer]    # > not >= because # of layers = n_feats (ie. n_feat_layers) -1

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=key, y=METRIC, ax=ax, color='tab:blue')
    # sns.stripplot(data=df, x=key, y=METRIC, ax=ax, c='black', alpha=0.3)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_yticklabels(['{:.2f}'.format(y) for y in ax.get_yticks()], weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')

    plt.savefig('out/' + fname, dpi=300)

def make_hyperparameter_impact_striplineplot(df, key, plotter=sns.swarmplot, fname=None, xlabel=None, ylabel=None):
    fname = fname or f"hyperparam_stripline_out_{key}.png"
    xlabel = xlabel or key
    ylabel = ylabel or "Cross-validated R$^2$"

    # TODO: scuffed
    if 'kernel' in key or 'stride' in key:
        layer = int(key.replace('kernel', '').replace('stride', ''))
        df = df[df['n_feat_layers'] > layer]    # > not >= because # of layers = n_feats (ie. n_feat_layers) -1

    fig, ax = plt.subplots()
    plotter(data=df, x=key, y=METRIC, hue='actual_layers', ax=ax, alpha=0.5, native_scale=True)
    quartiles = df.groupby(key).describe()[METRIC][['25%', '75%']]
    ax.plot(quartiles.index, quartiles['75%'], color='tab:blue', alpha=0.5, label='75%')
    sns.lineplot(data=df, x=key, y=METRIC, label='Mean')
    ax.plot(quartiles.index, quartiles['25%'], color='tab:blue', alpha=0.5, label='25%')
    # sns.lineplot(data=quartiles, hue=['tab:blue'], alpha=0.6, ax=ax)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_yticklabels(['{:.2f}'.format(y) for y in ax.get_yticks()], weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.legend()

    plt.savefig('out/' + fname, dpi=300)


def make_perf_by_cost_plot(df, key=PERF_METRIC, fname=None, xlabel=None, ylabel=None, ogmarker=[85360, 0.757]):
    fname = fname or f"perf_by_cost.png"
    xlabel = xlabel or key
    ylabel = ylabel or "Cross-validated R$^2$"


    fig, ax = plt.subplots()
    # shuffle to ensure even scatterplot overlap
    df = df.sample(frac=1)
    sns.scatterplot(data=df, x=key, y=METRIC, hue='actual_layers', legend='full', ax=ax, zorder=1)
    df = df.sort_values(PERF_METRIC)
    df['maxperf'] = df[METRIC].cummax()
    ax.plot(df[key], df['maxperf'], label='R$^2$ by Cost', zorder=2)
    ax.scatter(x=[ogmarker[0]], y=[ogmarker[1]], label='db20 Arch.', marker='X', sizes=[70], zorder=3)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_yticklabels(['{:.2f}'.format(y) for y in ax.get_yticks()], weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.set_ylim(0.6, None)
    leg = ax.legend(title='# Layers')
    leg._legend_box.align = "left"  # https://stackoverflow.com/a/44620643/10372825

    plt.savefig('out/' + fname, dpi=300)


if __name__ == '__main__':
    # plt.rcParams["font.family"] = "Times New Roman"

    # df = pd.read_csv('cached_runs_2023-05-09 18:30:10.638142.tsv', sep="	")  # all 1076 finished runs in the project before mics-fenet/20230324_fenet_sweeps_ben/wzh23dwp
    # df = pd.read_csv('cached_runs_2023-05-11 16:24:36.204057.tsv', sep="	")  # runs from 5eh4gf58 and ounwhc0k
    df = pd.read_csv('cached_runs_2023-05-18 15:45:34.270258.tsv', sep="	")  # runs from 5eh4gf58, ounwhc0k, and 6mflazgd

    df = df.dropna()
    df['actual_layers'] = df['n_feat_layers'] - 1
    # df[METRIC] *= 2 / np.sqrt(2)

    # key = 'stride7'
    # # make_hyperparameter_impact_plot(df, key)
    # # make_hyperparameter_impact_boxplot(df, key)
    # make_hyperparameter_impact_striplineplot(df, key, plotter=sns.swarmplot)
    # plt.show()
    # exit()

    make_perf_by_cost_plot(df, xlabel="Computation Cost (Operations per Eval)")

    for key in tqdm(params_to_plot):
        # make_hyperparameter_impact_plot(df, key)
        make_hyperparameter_impact_boxplot(df, key)
        make_hyperparameter_impact_striplineplot(df, key, plotter=sns.stripplot)
        plt.close()


