# FENet: a lightweight, hardware-ready threshhold crossing replacement for extracting single-channel dynamics 

FENet is a wavelets-inspired network that convolves single channel data in the time domain. It is designed to extract neural features from spike data to local field potentials from a single channel, agnostic of anatomical placement or channel hardware. It is also optimized for ASIC implementation, the details for which will be published soon. 

## Getting Started

To load and evaluate the full FENet pipeline (including a pretrained checkpoint, PLS regression, and a linear decoder) on sample data:

1. Create and activate a conda environment with the correct dependencies
```sh
conda env create -f environment.yml # creates a new environment on your machine called mics-fenet
conda env activate mics-fenet
```
2. Launch the example inference script with `python Test_Load_FENet.py`. 
3. View the results on [Weights & Biases](https://wandb.ai). Check the console for output for a link to view the results.


## Fine-tuning on Custom Data

To fine-tune on custom training data, you must prepare the recorded data as described in the paper, then configure training and view sweeps on [Weights & Biases](https://wandb.ai):

### Training Configuration
Set the following environment variables:
1. Set how many processes are used for data loading  with `FENET_MODEL_MAX_POOL_WORKERS`. `set FENET_MODEL_MAX_POOL_WORKERS=1` is a good default to disable multiprocessing. 
2. Set the `FENET_MODEL_DATA_DIR` to the directory with the `FennData_YYYYMMDD_Reshaped_30ms_cell_new.mat` files. 
3. Set the `FENET_MODEL_MODEL_SAVE_DIR`, the relative path to the directory where checkpoints are saved. This repo implements k-folds, and the ten best models from each training fold will be saved.

You can persist these variables in your conda environment with `conda env config vars set MY_VARIABLE="VALUE" FENET_MODEL_DATA_DIR="/path/to/data"`, etc. 

Check what else you'd like to configure in `config.py`

### Data Caching 
The data preprocessing takes a while, but the result can be cached. If `SAVE_LOCAL_DATA_CACHE` is `True` in `configs.py`, then a pickle file of the preprocessed and normalized training data will be created in the directory where the agent is running. If `LOAD_LOCAL_DATA_CACHE` is enabled, then that data will be used for all new runs, including when the agent command is re-launched, or a new sweep is created. If you need to change the data, either set `LOAD_LOCAL_DATA_CACHE` to `False` or delete the pickle file to trigger reprocessing.

### Configure Sweep through [Weights & Biases](https://wandb.ai)
1. Create an W&B account and project.
2. Run `wandb login` in the terminal where your sweeps will be running.
2. Create a sweep (step 2 in [this tutorial](https://docs.wandb.ai/guides/sweeps/existing-project#2-create-a-sweep)) but use the FENet [sweep_config.yml](./sweep_config.yml).
3. Run the `wandb agent` command in the root of this repository to start running sweeps.

### Select and Use Best Model

To use the best model found during the sweeps, we provide a script that re-evaluates all trained models on withheld data (the sweep-validation split). This script finds the best checkpoint which can be saved for later usage.

1. In [Select_Best_Architecture.py](./Select_Best_Architecture.py), change the `RUNS_TO_CHECK` configuration variable to match the sweep checkpoints in your `$FENET_MODEL_MODEL_SAVE_DIR`.
2. Run `python Select_Best_Architecture.py` to load the validation data and find the highest-performance checkpoint. The selection is based on mean-squared error, but this can be changed by providing different `criteria`.   TODO: why are there two criteria provided
3. To test the checkpoint found, update `CHECKPOINT` in [Test_Load_FENet.py](./Test_Load_FENet.py), and run that file. 

The current selection is based on the norm of the Pearson's R^2 correlation between the decoded and ground-truth signals. In our dataset, this is the decoded cursor velocity versus the intended cursor velocity in an open-loop center-out task.

You can change the model selection criteria by changing the `CRITERIA_FNS` and `SELECTION_CRITERIA_KEY` variables. See the available criteria functions in [criteria.py](./criteria.py), or implement your own there, and set `SELECTION_CRITERIA_KEY` to one of the keys that is provided by the `CRITERIA_FNS`.
