# FENet: a lightweight, hardware-ready threshhold crossing replacement for extracting single-channel dynamics 


## Getting Started

1. Create and activate a conda environment with the correct dependencies
```sh
conda env create -f environment.yml # creates a new environment on your machine called mics-fenet
conda env activate mics-fenet
```

2. Log in to [Weights & Biases](https://wandb.ai)
Run `wandb login` in the terminal.

## Running Sample Checkpoints

Launch the example inference script with `python Test_Load_FENet.py`. This script will load a FENet checkpoint, load the other pipeline elements (PLS regression and a linear decoder), then load evaluation data and evaluate the performance.

## Fine-tuning on Custom Data
### Training Configuration
Set the following environment variables:
1. Set how many processes are used for data loading  with `FENET_MODEL_MAX_POOL_WORKERS`. `set FENET_MODEL_MAX_POOL_WORKERS=1` is a good default to disable multiprocessing. 
2. Set the `FENET_MODEL_DATA_DIR` to the directory with the `FennData_YYYYMMDD_Reshaped_30ms_cell_new.mat` files. 
3. Set the `FENET_MODEL_MODEL_SAVE_DIR`, the relative path to the directory where checkpoints are saved. This repo implements k-folds, and the ten best models from each training fold will be saved.

Check what else you'd like to configure in `config.py`

### Data Caching 
If `SAVE_LOCAL_DATA_CACHE` is `True` in `configs.py`, then a pickle file of the preprocessed and normalized training data will be created in the directory where the agent is running. If `LOAD_LOCAL_DATA_CACHE` is enabled, then that data will be used for all new runs, including when the agent command is re-launched, or a new sweep is created. If you need to change the data, either set `LOAD_LOCAL_DATA_CACHE` to `False` or delete the pickle file to trigger reprocessing.

### Configure Sweep through [Weights & Biases](https://wandb.ai)
1. Create an W&B account and project.
2. Run `wandb login` in the terminal where your sweeps will be running.
2. Create a sweep (step 2 in [this tutorial](https://docs.wandb.ai/guides/sweeps/existing-project#2-create-a-sweep)) but use the FENet [sweep_config.yml](./sweep_config.yml).
3. Run the `wandb agent` command in the root of this repository to start running sweeps.

