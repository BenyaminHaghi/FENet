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
from configs import DECODER_TRAIN_BATCH_SIZE as train_batch_size
from data_parser import pickle_memoize
import wandb
import pickle
from rich.console import Console; from rich import print
import pandas as pd


from datetime import datetime


class TimeTracker:
    def __init__(self):
        self.console = Console()
        self.last_time = datetime.now()

    def log(self, message):
        self.console.print(f"TIME: {message} after", datetime.now() - self.last_time, style='purple')
        self.reset()

    def reset(self):
        self.last_time = datetime.now()

tracker = TimeTracker()
from OriginalModelLoadingExample import cnn_model
print(cnn_model)
tracker.log('Loaded Bencode/og model')


def get_model_evals(path, test_dl):
    config, *_ = read_checkpoint(path)
    config = { 'annealing_alpha': 0, 'anneal': False, 'thermal_sigma': 0, 'decoder': 0, 'pls_dims': 2, 'random_seed': 0, **config }
    device, _, pls_model, decoder, _, _, _, _, _  = initialize(run=run, config=config)
    fenet = make_fenet_from_checkpoint(path, device)    # don't use the fenet from initialize() because that one is not from the checkpoint
    evals = evaluate_with_criteria(fenet, pls_model, decoder, test_dl, [
        mean_squared_error_criterion,
        directional_R2_criterion
    ], device)
    return evals, fenet

if __name__ == '__main__':
    def test_data_maker():
        _, _, withheld_test_dl = make_total_training_data(DATA_DIR, splits = ['test'], n_filtered_channels=None, make_data_from_day_kwargs={ 'normalize_inputs': True })
        return withheld_test_dl
    if (LOAD_LOCAL_DATA_CACHE):
        print("loading data pickle!")
        data_pickle_name = f'total_training_data_minR2-{FILTERING_MIN_R2}_nchan-{FILTERING_MIN_R2}_WITHHELD_TEST.pkl'
        withheld_test_dl = pickle_memoize(data_pickle_name, test_data_maker)
    else:
        withheld_test_dl = test_data_maker()
    tracker.log("loaded test data")




    config, *_ = read_checkpoint(path)
    tracker.log('loaded new checkpoint of old model')
    with wandb.init(project="publishing_evaluation_ben", config=config, tags="oldcode-model") as run:
        evals, _ = get_model_evals(path, withheld_test_dl)
        # print(evals)
    
    tracker.log('new code evaled')





    # device, _, pls_model, decoder, _, _, _, _, _  = initialize(run=run, config=config)  # TODO: what run and config?
    device = 'cuda'
    pls_model = PLS_Model(192, 8, 2, train_batch_size, device)
    decoder = Linear_Decoder(train_batch_size=train_batch_size, device=device, quantization=None)
    cnn_model.features_by_layer = [1]*8; cnn_model.pls = 2
    tracker.log('re-initialized for old code')
    og_code_evals = evaluate_with_criteria(cnn_model, pls_model, decoder, withheld_test_dl, [
        mean_squared_error_criterion,
        directional_R2_criterion
    ], device)
    # print(og_code_evals)
    tracker.log('old code evaled')


    print(pd.DataFrame.from_dict([evals, og_code_evals]))

    new_r2s = { 'new_x': evals['TEMP-RAW_X-R2'], 'new_y': evals['TEMP-RAW_Y-R2'] }
    old_r2s = { 'old_x': og_code_evals['TEMP-RAW_X-R2'], 'old_y': og_code_evals['TEMP-RAW_Y-R2'] }
    df = pd.DataFrame({ **new_r2s, **old_r2s })
    print(df)

    df['x_diff'] = df['new_x'] - df['old_x']
    df['y_diff'] = df['new_y'] - df['old_y']

    print(df)

    # with wandb.init(project="publishing_evaluation_ben") as run:
    #     fenet = make_fenet_from_checkpoint('F:/Ben/wandb_saves')

