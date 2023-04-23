# FENet_Model

## TODO:
- [ ] export environment.yml
- [ ] update the File Structure mermaid diagram below
- [ ] remove dead code

## Getting Started
Set the following environment variables:
1. Set the default worker pool size with `FENET_MODEL_MAX_POOL_WORKERS`. `set FENET_MODEL_MAX_POOL_WORKERS=1` is a good default to disable multiprocessing. 
2. Set the `FENET_MODEL_DATA_DIR` to the directory with the `FennData_YYYYMMDD_Reshaped_30ms_cell_new.mat` files. 
3. Set the `FENET_MODEL_MODEL_SAVE_DIR`, the relative path to the directory where checkpoints are saved. Downstream experiments (eg. quantization sweeps, video generation) look for the BEST_MODEL in this directory.
4. There are additional configuration environment variables that can be left blank for training.
	1. Set the `FENET_MODEL_BEST_MODEL` to the **name** of the best model that will be used as the base for downstream exports (eg. hardware). Can be left blank for training.
4. Check what else you'd like to configure in `config.py`

TODO: activate the conda environment: 
```sh
conda env create -f environment.yml # creates a new environment on your machine called mics-fenet
conda env activate mics-fenet
```
Then, launch a run with `python3 main_sweeps.py`, installing Python packages as necessary.


## File Structure

This diagram details how the repository is layed out.

```mermaid
classDiagram

class FENet_Training {
    PLS_Generation()
}

class FENet_parameterizable {
    FENet
    QuantizedFENet

    make_fenet_from_checkpoint()
    make_QFENet_from_FENet()
    inference_batch()
}

class criteria {
    evaluate_with_criteria﴾﴿
    EfficiencyCriterion
    QuantizationCriterion
    
    R2_avg_criterion()
    R2_hist_criterion()
    axes_plot_criterion()
    directional_R2_criterion()
}

class data_parser {
    make_total_training_data()
    pickle_memoize()
}

class decoder {
    compute_linear_decoder_loss_preds()
}

class explore_data {
    show_precision_distribution()
    show_heatmap()
}

class explore_model {
    make_quantization_stochastic_error_charts()
    make_quantization_decoder_histograms()
    make_model_weights_histogram()
}

class export {
    export_for_ndt()
    export_weights_for_hardware()
    export_data_for_hardware()
}

class utils {
    KFoldsGenerator
    BestEpochSaver
    seed_everything()
    make_outputs_directory()
}

class main_sweeps {
    DATA_DIR
    reset_wandb_env()
    initialize()
    train_batch()
    kfolds_train_worker()
}

FENet_Training <-- FENet_parameterizable 

FENet_parameterizable <-- main_sweeps
main_sweeps ..> data_parser
criteria <-- main_sweeps
main_sweeps ..> utils

FENet_parameterizable <-- export
export ..> data_parser
criteria <-- export
export ..> utils

FENet_parameterizable <-- explore_model
explore_model ..> data_parser
criteria <-- explore_model
explore_model ..> utils

FENet_parameterizable <-- criteria
decoder <-- criteria
data_parser <-- explore_data 

data_parser <-- parameter_estimation
data_parser <-- emperical_best_channels
main_sweeps <-- emperical_best_channels
```
