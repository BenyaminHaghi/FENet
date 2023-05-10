import wandb
from matplotlib import pyplot as plt
import pandas as pd

from operator import itemgetter
import json

N = 8
WANDB_ENTITY = "mics-fenet"
WANDB_PROJECT = "20230324_fenet_sweeps_ben"
params_to_plot = [f"kernel{i}" for i in range(1, N)] + [f"stride{i}" for i in range(1, N)]
print(params_to_plot)

if __name__ == '__main__':
    api = wandb.Api({ 'entity': WANDB_ENTITY, 'project': WANDB_PROJECT })
    runs = api.runs(filters={ 'state': 'finished' })



    # print(runs[0].json_config['kernel1'])




    def key_getter(run):
        cfg = json.loads(run.json_config)
        return { **{ k: cfg[k]['value'] for k in params_to_plot }, 'cross-validation-avg-r2': run.summary['cross-validation-avg-r2'] }

    df = pd.DataFrame(map(key_getter, runs))
    print(df)

    # print([x for x in runs[10].summary.keys() if 'avg' in x])
    # print(runs[0].state)
    # runs = filter(lambda r: r.is)
    # df = pd.DataFrame([itemgetter('id')])

