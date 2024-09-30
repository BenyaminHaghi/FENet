from Test_Load_FENet import DATA_DIR

from operator import itemgetter
from glob import glob
from data_parser import make_total_training_data, pickle_memoize

from main_sweeps import initialize
from FENet_parameterizable import read_checkpoint, make_fenet_from_checkpoint
from criteria import evaluate_with_criteria, mean_squared_error_criterion, pearson_r_squared_criterion


RUNS_TO_CHECK = "/Path/To/Saved/Checkpoints/*" # used with a glob later
CRITERIA_FNS = [mean_squared_error_criterion, pearson_r_squared_criterion]
SELECTION_CRITERIA_KEY = 'pearsonr2-xy-normed'
device = 'cuda'

def get_model_evals(path, test_dl):
    config, *_ = read_checkpoint(path)
    config = { 'annealing_alpha': 0, 'anneal': False, 'thermal_sigma': 0, 'decoder': 0, 'pls_dims': 2, 'random_seed': 0, **config }
    device, _, pls_model, decoder, _, _, _, _, _  = initialize(config=config)
    fenet = make_fenet_from_checkpoint(path, device)    # don't use the fenet from initialize() because that one is not from the checkpoint
    evals = evaluate_with_criteria(fenet, pls_model, decoder, test_dl, CRITERIA_FNS, device)
    return evals, fenet

if __name__ == "__main__":
    gen = lambda: make_total_training_data(DATA_DIR, n_filtered_channels=192, splits=['dev-sweep-selection'])
    withheld_model_selection_val_set = next(pickle_memoize('dev-sweep-selection-data.pkl', gen))
    performances = []
    for savefile in glob(RUNS_TO_CHECK):
        performances.append(get_model_evals(savefile, withheld_model_selection_val_set) + (savefile,))

    performances = sorted(performances, reverse=True, key=lambda evals, *_: evals[0][SELECTION_CRITERIA_KEY])
    print(f"best model:\n{performances[0][2]}\nwith performance\n{performances[0][0]}.")
