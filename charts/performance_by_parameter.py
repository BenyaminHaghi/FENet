import wandb
from matplotlib import pyplot as plt
import pandas as pd

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
    # api = wandb.Api({ 'entity': WANDB_ENTITY, 'project': WANDB_PROJECT })
    api = wandb.Api()
    runs = api.runs(path="mics-fenet/20230324_fenet_sweeps_ben", filters={ 'state': 'finished' })






    if :
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

    print(df.describe())
    print(df.isna().sum())

    # print([x for x in runs[10].summary.keys() if 'avg' in x])
    # print(runs[0].state)
    # runs = filter(lambda r: r.is)
    # df = pd.DataFrame([itemgetter('id')])

