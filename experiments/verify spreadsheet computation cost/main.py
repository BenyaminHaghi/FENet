import pandas as pd
import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt

from operator import itemgetter 

import sys
sys.path.append("C:\\Users\\ahuang3\\bmi\\FENet_Model")
from FENet_parameterizable import FENet
from criteria import EfficiencyCriterion

FILEPATH = "C:\\Users\\ahuang3\\Downloads\\computation_cost.tsv"
N_FEAT_LAYERS = 7+1
INPUT_SEQ_LEN = 900

if __name__ == '__main__':
    df = pd.read_csv(FILEPATH, sep='\t')
    print(df.columns)

    python_calculated_costs = []
    config_keys = [f"kernel{i}" for i in range(1, N_FEAT_LAYERS)] + [f"stride{i}" for i in range(1, N_FEAT_LAYERS)]
    for i, (compute_cost, *model_config) in df[['Computation Cost', *config_keys]].iterrows():

        model_config = [int(v) for v in model_config]

        fe_net = FENet([1]*N_FEAT_LAYERS, model_config[:N_FEAT_LAYERS-1], model_config[N_FEAT_LAYERS-1:])
        crit = EfficiencyCriterion(fe_net)
        val = crit.evaluate(INPUT_SEQ_LEN)['efficiency/operations-per-eval']
        python_calculated_costs.append(val)

    df['efficiency/operations-per-eval'] = python_calculated_costs

    fig, ax = plt.subplots()
    ax.scatter(df['Computation Cost'], df['efficiency/operations-per-eval'])
    ax.set_xlabel('computation cost (spreadsheet)')
    ax.set_ylabel('computation cost (python)')
    fig.savefig('computation_cost_comparison.png')

# if __name__ == '__main__':  # update all the runs
#     api = wandb.Api()

#     runs = api.runs('mics-bmi/FENet_Model')
    
#     for run in (pbar := tqdm(runs)):
#         pbar.set_description(run.name)
#         config = { k: v for k, v in run.config.items() if not k.startswith('_') }
#         if 'kernel1' not in config: continue
#         n_feat_layers = config['n_feat_layers'] if 'n_feat_layers' in config else 8
#         config_keys = [f"kernel{i}" for i in range(1, n_feat_layers)] + [f"stride{i}" for i in range(1, n_feat_layers)]
#         config = itemgetter(*config_keys)(config)

#         # update efficiency computation cost metric
#         # fe_net = FENet([1]*n_feat_layers, config[:n_feat_layers-1], config[n_feat_layers-1:])
#         # crit = EfficiencyCriterion(fe_net)
#         # val = crit.evaluate(INPUT_SEQ_LEN)['efficiency/operations-per-eval']
#         # run.summary["efficiency/operations-per-eval"] = val

#         # update run n_feat_layers
#         if 'n_feat_layers' not in config or (config['n_feat_layers'] != 6 and config['n_feat_layers'] != 7):
#             run.config['n_feat_layers'] = n_feat_layers

#             run.update()