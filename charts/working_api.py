import wandb
from matplotlib import pyplot as plt
import pandas as pd

from operator import itemgetter
from datetime import datetime
import json

from performance_by_parameter import METRIC, PERF_METRIC, N, WANDB_ENTITY, WANDB_PROJECT, params_to_plot

print(params_to_plot)

# if __name__ == '__main__':
api = wandb.Api({ 'entity': WANDB_ENTITY, 'project': WANDB_PROJECT })
runs = api.runs(filters={ "$and": [
    { "state": 'finished', },
    { "sweep": { "$in": [ '5eh4gf58', 'ounwhc0k', '6mflazgd' ] } }
                                   ]
 })

def key_getter(run):
    cfg = json.loads(run.json_config)
    # if 'cross-validation-avg-r2' not in run.summary:
    #     run.summary['cross-validation-avg-r2'] = calculate_avg_best_r2(run)
    #     run.summary.update()
    # print(list(run.summary.keys()))
    return {
            **{ k: cfg[k]['value'] for k in params_to_plot },
            METRIC: run.summary[METRIC] if METRIC in run.summary else None,
            PERF_METRIC: run.summary[PERF_METRIC],
            'id': run.id }

df = pd.DataFrame.from_records(map(key_getter, runs))
print(df.describe())

df.to_csv(f"cached_runs_{datetime.now()}.tsv", sep="	")


print(f"Retrieved {len(runs)} runs")


# print(runs[0].json_config['kernel1'])




# def key_getter(run):
#     cfg = json.loads(run.json_config)
#     return { **{ k: cfg[k]['value'] for k in params_to_plot }, 'cross-validation-avg-r2': run.summary['cross-validation-avg-r2'] }
#
# df = pd.DataFrame(map(key_getter, runs))
# print(df)

    # print([x for x in runs[10].summary.keys() if 'avg' in x])
    # print(runs[0].state)
    # runs = filter(lambda r: r.is)
    # df = pd.DataFrame([itemgetter('id')])

