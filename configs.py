from os import getenv, environ
HYPER_PARAM_CONFIG = {
        "kernel1": 40,
        "kernel2": 40,
        "kernel3": 40,
        "kernel4": 40,
        "kernel5": 40,
        "kernel6": 40,
        "kernel7": 40,
        "n_feat_layers": 8,
        "optim.adamw_eps": 1e-7,
        "optim.adamw_wd": 0.02,
        "optim.lr": 0.00078,
        "pls_dims": 2,
        "annealing_alpha": 1,
        "thermal_sigma": 0.99,
        "anneal": False,
        "decoder": 0,
        "random_seed": 1335,
        "stride1": 2,
        "stride2": 2,
        "stride3": 2,
        "stride4": 2,
        "stride5": 2,
        "stride6": 2,
        "stride7": 2,
        "relu1": 0,
        "relu2": 0,
        "relu3": 0,
        "relu4": 0,
        "relu5": 0,
        "relu6": 0,
        "relu7": 0,
}

EVAL_STEPS = 8
MAX_EPOCHS = 27
EFFICIENCY_METRIC_INPUT_LEN = 900
COMPUTE_COST_DIVISOR = 30000

LOAD_LOCAL_DATA_CACHE = True
SAVE_LOCAL_DATA_CACHE = True

USE_MULTITHREADED_WORKERS = False
USE_MULTITHREADED_PLS_TRAINING = False
USE_MULTITHREADED_DATA_LOADING = False
MAX_POOL_WORKERS = int(getenv('FENET_MODEL_MAX_POOL_WORKERS') or 1)
THREAD_CONTEXT = "forkserver"

NOT_IMPLEMENTED = None
DECODER_TRAIN_BATCH_SIZE = 1600
FENET_MEMLIMIT_SERIAL_BATCH_SIZE = None  # doesn't change batch size like normal because we must batch by day; instead this artifically serializes and repacks batches before the pls/decoder step
PLS_TRAIN_TEST_RATIO = 0.7

ATTEMPT_GPU = True

# https://docs.wandb.ai/guides/track/advanced/environment-variables
# offline - save run data locally
# disabled - turn wandb off entirely
# online - run as normal
environ["WANDB_MODE"] = "online"
DATA_DIR = getenv('FENET_MODEL_DATA_DIR')
MODEL_SAVE_DIR = getenv('FENET_MODEL_MODEL_SAVE_DIR')
QUANTIZED_MODEL_DIR = getenv('FENET_MODEL_QUANTIZED_MODEL_DIR')
QUANTIZED_DATA_DIR = getenv('FENET_MODEL_QUANTIZED_DATA_DIR')
BEST_MODEL = getenv('FENET_MODEL_BEST_MODEL')
if (MODEL_SAVE_DIR is not None and BEST_MODEL is not None):
        UNQUANTIZED_MODEL_DIR = MODEL_SAVE_DIR + BEST_MODEL
else:
        UNQUANTIZED_MODEL_DIR = None
FILTERING_MIN_R2 = 0
TRAIN_FILTERING_N_TOP_CHANNELS = 40
FILE_FORMAT = "{}.mat"
DAY_SPLITS = {
       'train': ['filename1', 'filename2', 'etc...'],
       'dev-sweep-selection': ['filename3', 'filename4', '...'],
       'test':  ['filename5', 'filename6', 'etc...']
    }
N_FOLDS = 9

REDO_QUANTIZE = True
GENERATE_WEIGHT_FILE= True
QUANTIZED_FILE_WRITE_MODE='w+'
COMPARE_WITH_UNQUANTIZED = True
EVAL_WITH_QUANTIZATION = False
EXPORT_TARGET = 'hardware'
EVAL_WLFL_PAIRS = ((9, 6),)# (10, 7))
QUANTIZED_WORD_LENGTH = 9
POOL_REG_BIT_WIDTH = 22
ACCUM_REG_BIT_WIDTH = 16
DATA_VALID_DELAY = 0
NUM_FENET_BUILT = 64
NUM_FEM_BUILT = 7
MAX_WEIGHT_DEPTH = 256
MAX_NUM_CYCLES = 900
EXPORT_MODEL_STRIDES = [2,2,2,2,2,2]

WANDB_FIX_TAGS = []

