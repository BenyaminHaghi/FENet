path = './FENet_Max'
from FENet_parameterizable import make_fenet_from_checkpoint, read_checkpoint
from criteria import evaluate_with_criteria
from criteria import directional_R2_criterion
from decoder import PLS_Model
from decoder import Linear_Decoder
from criteria import mean_squared_error_criterion, directional_R2_criterion
from data_parser import make_total_training_data
from main_sweeps import DATA_DIR, initialize
from configs import LOAD_LOCAL_DATA_CACHE, SAVE_LOCAL_DATA_CACHE, FILTERING_MIN_R2
from data_parser import pickle_memoize
import wandb
import pickle



def test_data_maker():
    _, _, withheld_test_dl = make_total_training_data(DATA_DIR, splits = ['test'], n_filtered_channels=None)
    return withheld_test_dl
if (LOAD_LOCAL_DATA_CACHE):
    data_pickle_name = f'total_training_data_minR2-{FILTERING_MIN_R2}_nchan-{FILTERING_MIN_R2}_WITHHELD_TEST.pkl'
    withheld_test_dl = pickle_memoize(data_pickle_name, test_data_maker)
else:
    withheld_test_dl = test_data_maker()

def get_model_evals(path, test_dl):
    config, *_ = read_checkpoint(path)
    device, _, pls_model, decoder, _, _, _, _, _  = initialize(run=run, config=config)
    fenet = make_fenet_from_checkpoint(path, device)    # don't use the fenet from initialize() because that one is not from the checkpoint
    evals = evaluate_with_criteria(fenet, pls_model, decoder, test_dl, [
        mean_squared_error_criterion,
        directional_R2_criterion
    ], device)
    return evals, fenet

with wandb.init(project="publishing_evaluation_ben", config=config) as run:
    evals, _ = get_model_evals(path, withheld_test_dl)
    print(evals)

