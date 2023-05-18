from Test_Load_FENet import DATA_DIR, get_model_evals

from operator import itemgetter
from glob import glob
from data_parser import make_total_training_data

from decoder import PLS_Model

RUN_NAME = "F:/Ben/fenet_sweeps_save/absurd-sweep-4" # used with a glob later
device = 'cuda'

if __name__ == "__main__":
    withheld_model_selection_val_set = make_total_training_data(DATA_DIR, n_filtered_channels=192, splits=['dev-sweep-selection'])
    performances = []
    for savefile in glob(RUN_NAME + "*"):
        performances.append(get_model_evals(savefile, withheld_model_selection_val_set) + (savefile,))

    performances = sorted(performances, reverse=True)
    print(f"best model: {performances[0][2]} with performance {performances[0][0]:.3f}.")
