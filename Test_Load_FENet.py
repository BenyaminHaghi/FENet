import wandb
from main_sweeps import DATA_DIR, initialize
from data_parser import make_total_training_data, pickle_memoize
from FENet_parameterizable import make_fenet_from_checkpoint, read_checkpoint
from criteria import evaluate_with_criteria
from criteria import mean_squared_error_criterion, pearson_r_squared_criterion

# you can choose which checkpoint to evaluate here. these are the files that would be saved to FENET_MODEL_MODEL_SAVE_DIR
path = '.\\checkpoints\\db20_architechture'
# path = '.\\checkpoints\\db20-alternative-iqk20aq0'
# path = '.\\checkpoints\\efficient-model-volcanic486'

if __name__ == '__main__':

    # load data and cache it as a pickle
    def test_data_maker():
        withheld_test_dl, = list(make_total_training_data(DATA_DIR, splits = ['test'], n_filtered_channels=None, make_data_from_day_kwargs={ 'normalize_inputs': False }))
        print(type(withheld_test_dl))
        return withheld_test_dl
    withheld_test_dl = pickle_memoize('test_data.pkl', test_data_maker)

    # load pipeline components
    config, *_ = read_checkpoint(path)
    config['n_channels'] = withheld_test_dl[0][0].shape[1]
    device, _, pls_model, decoder, _, _, _, _, _  = initialize(config=config)   # get the pls and decoder models based on how this fenet was configured
    fenet = make_fenet_from_checkpoint(path, device)    # load custom fenet that we are evaluating
    fenet.normalize_at_end = True

    # evaluate model
    with wandb.init(project="fenet_publishing_testbed", config=config) as run:
        evals = evaluate_with_criteria(
            fenet, pls_model, decoder,      # the pipeline we are evaluating
            withheld_test_dl,               # eval data
            [ mean_squared_error_criterion, pearson_r_squared_criterion ],   # the metrics to evaluate with
            device
        )
        wandb.log(evals)
        print(evals)

