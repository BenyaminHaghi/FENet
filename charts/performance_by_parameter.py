import wandb
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd
import seaborn as sns
import numpy as np
from rich import print
from scipy import stats as st

from operator import itemgetter
from datetime import datetime
import json

N = 8
WANDB_ENTITY = "mics-fenet"
WANDB_PROJECT = "20230324_fenet_sweeps_ben"
METRIC = 'cross-validation-avg-r2'
params_to_plot = [f"kernel{i}" for i in range(1, N)] + [f"stride{i}" for i in range(1, N)]
print(params_to_plot)

def calculate_avg_best_r2(run):
    # df = pd.DataFrame(run.history)
    pass

if __name__ == '__main__':
    api = wandb.Api({ 'entity': WANDB_ENTITY, 'project': WANDB_PROJECT })
    runs = api.runs(filters={ 'state': 'finished' })
    # api = wandb.Api()
    # runs = api.runs(path="mics-fenet/20230324_fenet_sweeps_ben", filters={ 'state': 'finished' })




    if False:
        def key_getter(run):
            cfg = json.loads(run.json_config)
            if 'cross-validation-avg-r2' not in run.summary:
                run.summary['cross-validation-avg-r2'] = calculate_avg_best_r2(run)
                run.summary.update()
            return { **{ k: cfg[k]['value'] for k in params_to_plot }, METRIC: run.summary[METRIC] }

        df = pd.DataFrame.from_records(map(key_getter, runs))
        print(df)

        df.to_csv(f"cached_runs_{datetime.now()}.tsv", sep="	")
    else:
        df = pd.read_csv('cached_runs_2023-05-09 18:30:10.638142.tsv', sep="	")

    df = df.dropna()

    key = 'stride1'

    grid = sns.FacetGrid(df, col=key, height=4, aspect=0.2, gridspec_kws={ 'wspace': 0.1 })
    grid.map_dataframe(sns.histplot, y=METRIC)

    # remove lines
    for ax in grid.axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
    grid.axes.flat[0].spines['left'].set_visible(True)
    for ax in grid.axes.flat[1:]:
        ax.tick_params(axis='y', left=False)
    grid.set(xlabel=None, xticklabels=[])

    grid.set_titles(col_template="{col_name}", y=-0.05, x=0.15)
    grid.figure.supxlabel(f"{key}")



    grouped = df[[key, METRIC]].groupby(key)
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
    agg = grouped.aggregate(['mean', 'count', st.sem])
    mean_yvals = agg[METRIC, 'mean']
    confidence_interval_95 = np.array([
        st.t.interval(
            confidence=0.95,
            df=c-1,
            loc=m,
            scale=sem
        )
        for _, (m, c, sem) in agg.iterrows() ]).T

    overlay_ax.plot(x_vals, mean_yvals, label='Mean')
    overlay_ax.fill_between(x_vals, confidence_interval_95[0], confidence_interval_95[1], alpha=0.2, label='95% Confidence')
    overlay_ax.legend()

    plt.savefig(f'out_{key}.png', dpi=300)
    plt.show()


    # fig, axs = plt.subplots(nrows=1, ncols=len(grouped), figsize=(8, 5), sharey=True)
    # for (x, gdf), ax in zip(grouped, axs):
    #     sns.histplot(ax=ax, data=gdf, y=METRIC, bins=30)
    # plt.show()

