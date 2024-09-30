from os import getenv, environ



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
FILE_FORMAT = "{}.mat"  # expect files of the form DATA_DIR/filename1.mat
DAY_SPLITS = {
       'train': ['filename1', 'filename2', 'etc...'],
       'dev-sweep-selection': ['filename3', 'filename4', 'etc...'],
       'test':  ['filename5', 'filename6', 'etc...']
    }
N_FOLDS = 9

# https://docs.wandb.ai/guides/track/advanced/environment-variables
# offline - save run data locally
# disabled - turn wandb off entirely
# online - run as normal
environ["WANDB_MODE"] = "online"
WANDB_FIX_TAGS = []

# training
EVAL_STEPS = 8
MAX_EPOCHS = 27
EFFICIENCY_METRIC_INPUT_LEN = 900
COMPUTE_COST_DIVISOR = 30000
DECODER_TRAIN_BATCH_SIZE = 1600
FENET_MEMLIMIT_SERIAL_BATCH_SIZE = None  # doesn't change batch size like normal because we must batch by day; instead this artifically serializes and repacks batches before the pls/decoder step

# data caching
LOAD_LOCAL_DATA_CACHE = True
SAVE_LOCAL_DATA_CACHE = True

# multiprocessing
USE_MULTITHREADED_WORKERS = False
USE_MULTITHREADED_PLS_TRAINING = False
USE_MULTITHREADED_DATA_LOADING = False
MAX_POOL_WORKERS = int(getenv('FENET_MODEL_MAX_POOL_WORKERS') or 1)
THREAD_CONTEXT = "forkserver"


ATTEMPT_GPU = True


# quantization
EVAL_WITH_QUANTIZATION = False
EXPORT_TARGET = 'hardware'
EVAL_WLFL_PAIRS = ((9, 6),)# (10, 7))
QUANTIZED_WORD_LENGTH = 9


