MODEL_PATH = "F:\\Albert\\FENet\\wandb_saves\\genial-sweep-57_step-1160_perf-0.6615"; MODEL_SHAPE = [[1]*8, [10, 10, 2, 4, 32, 34, 26], [3, 2, 30, 13, 23, 25, 27]]
SAVEPATH = "F:\\Albert\\FENet\\wandb_saves_with_config\\genial-sweep-57_step-1160_perf-0.6615"

import wandb
import torch
api = wandb.Api()
run = api.run("mics-bmi/FENet_Model/n7pygxkc")

elapsed_steps, fe_net, optimizer, scheduler = torch.load(MODEL_PATH)

config = { 'pls_dims': 0, **run.config }
print(config)

torch.save((config, elapsed_steps, fe_net, optimizer, scheduler), SAVEPATH)

