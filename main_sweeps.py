from os import getenv
from re import I
from time import thread_time
import math
################################ data constants
from configs import DATA_DIR, EVAL_WITH_QUANTIZATION, EVAL_WLFL_PAIRS, FENET_MEMLIMIT_SERIAL_BATCH_SIZE
from configs import MODEL_SAVE_DIR
from configs import FILTERING_MIN_R2
from configs import TRAIN_FILTERING_N_TOP_CHANNELS

from configs import HYPER_PARAM_CONFIG as CONFIG

from configs import N_FOLDS
from configs import EVAL_STEPS
from configs import MAX_EPOCHS

from configs import LOAD_LOCAL_DATA_CACHE
from configs import SAVE_LOCAL_DATA_CACHE

from configs import USE_MULTITHREADED_WORKERS
from configs import MAX_POOL_WORKERS
from configs import THREAD_CONTEXT

from configs import EFFICIENCY_METRIC_INPUT_LEN, COMPUTE_COST_DIVISOR
from FENet_parameterizable import train_batch
from data_parser import pickle_memoize

#print('importing modules...')
from torch import optim
from torch.cuda import is_available as cuda_is_available
from pandas import DataFrame

import wandb

# from sweep_worker import kfolds_train_worker, WANDB_FIX_TAGS, MAX_EPOCHS, EVAL_STEPS
from utils import BestEpochSaver, KFoldsGenerator

from torch.multiprocessing import Lock, SimpleQueue, get_context
import pickle
from os import environ
import traceback


def reset_wandb_env():
    # from https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-cross-validation/train-cross-validation.py
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_DIR",
        "WANDB_REQUIRE_SERVICE",    # keep wandb.require('service') env vars
        "WANDB_SERVICE",
    }
    for k, v in environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del environ[k]

def initialize(run = None, config=None):

    from torch import optim
    from torch.cuda import is_available as cuda_is_available

    import wandb
    from functools import partial

    from FENet_parameterizable import FENet, make_daubechies_wavelet_initialization, make_fenet_from_checkpoint
    from decoder import Linear_Decoder, RNN_decoder
    from decoder import SimplePLS
    from utils import seed_everything
    from configs import DECODER_TRAIN_BATCH_SIZE as train_batch_size
    from torch.nn import MSELoss

    if run is None: run = wandb.init(config=config)
    if config is None: config = wandb.config



    seed_everything(config['random_seed'])

    device = 'cuda:0' if cuda_is_available() else 'cpu'

    # train_dl, dev_dl, test_dl = make_total_training_data(DATA_DIR, FILTERING_MIN_R2, FILTERING_N_TOP_CHANNELS)
    N = config['n_feat_layers']
    fe_net = FENet([1]*N,   [config[f'kernel{i}'] for i in range(1, N)],
                            [config[f'stride{i}'] for i in range(1, N)],
                            [config[f'relu{i}']   for i in range(1, N)],
                            **{ k: config[k] for k in ['pls_dims', 'normalize_at_end'] + ['annealing_alpha', 'thermal_sigma', 'anneal'] if k in config }   # pass additional config kwargs if they are in the config
                )

    fe_net.load_state_dict(make_daubechies_wavelet_initialization(fe_net))
    fe_net.to(device)


    if('pls_dims' in config and config['pls_dims'] > 0 and config['pls_dims'] != None):
        # pls_mdl = PLS_Model(config['n_channels'], N, config['pls_dims'], train_batch_size, device)
        pls_mdl = SimplePLS(config['n_channels'], N, config['pls_dims'])
    else:
        pls_mdl = None

    if config['decoder'] == 0:
            decoder = Linear_Decoder(train_batch_size=train_batch_size, device=device, quantization=None)

    loss_fn = MSELoss(reduction='mean')

    wandb.watch(fe_net, log_freq=10, log='all', log_graph=True)

    optimizer = optim.AdamW(fe_net.parameters(), lr=config['optim.lr'], eps=config['optim.adamw_eps'], weight_decay=config['optim.adamw_wd'])
    scheduler = None

    return device, fe_net, pls_mdl, decoder, loss_fn, optimizer, scheduler, run, config



# train with just wandb
if __name__ == '__main__':
    wandb.require("service")    # fix a wandb multiprocessing error as per https://github.com/wandb/wandb/issues/1994#issuecomment-1075436252

    from data_parser import make_total_training_data
    if(LOAD_LOCAL_DATA_CACHE or SAVE_LOCAL_DATA_CACHE):
        data_pickle_name = f'total_training_data_minR2-{FILTERING_MIN_R2}_nchan-{FILTERING_MIN_R2}.pkl'
    if(LOAD_LOCAL_DATA_CACHE):
        print("Loading local cache...\n")
        train_dl, dev_dl = pickle_memoize(data_pickle_name,
                lambda: make_total_training_data(DATA_DIR, FILTERING_MIN_R2, TRAIN_FILTERING_N_TOP_CHANNELS, load_test_dl=False),
                writefile=SAVE_LOCAL_DATA_CACHE)
    else:
        train_dl, dev_dl = make_total_training_data()
        if(SAVE_LOCAL_DATA_CACHE):
            #Test data_set is not touched during training
            with open(data_pickle_name, 'wb') as wf: pickle.dump([train_dl, dev_dl], wf); print("pick? led.")

    device = 'cuda:0' if cuda_is_available() else 'cpu'

    k_folds_manager = KFoldsGenerator(train_dl + dev_dl, n_folds=N_FOLDS)
    del(train_dl)
    del(dev_dl)


    best_performance_by_fold = []

    with wandb.init(job_type='sweep run', dir = 'F:/Ben/copy_wandb_for_N1/') as run:
        config_to_pass = { k: v for k, v in wandb.config.items() if not k.startswith('_') }

        run.config['n_channels'] = TRAIN_FILTERING_N_TOP_CHANNELS

        for fold, (train_dl, dev_dl) in enumerate(k_folds_manager.make_folds()):
            # jankily log each fold sequentially. better the alternative which requires multiprocessing
            run.config.update({'fold': fold}, allow_val_change=True)
            saver = BestEpochSaver(MODEL_SAVE_DIR, n_saves=10)


            import torch

            from tqdm import tqdm, trange
            import wandb

            from criteria import EfficiencyCriterion
            from criteria import pearson_r_squared_criterion
            from criteria import evaluate_with_criteria
            # from utils import seed_everything, BestEpochSaver, KFoldsGenerator

            from functools import partial
            import os
            from configs import TRAIN_FILTERING_N_TOP_CHANNELS
            from utils import filter_dict
            import traceback

            device, fe_net, pls_mdl, decoder, loss_fn, optimizer, scheduler, run, config = initialize(run)

            print("STAT: model and data initailized\r")

            # calculate and log compute cost for this run
            efficiency_crit = EfficiencyCriterion(fe_net)
            compute_cost = efficiency_crit.evaluate(EFFICIENCY_METRIC_INPUT_LEN)

            #print("Cost computed")
            run.log(compute_cost, commit=False)
            run.log({ 'compute_cost_divisor': COMPUTE_COST_DIVISOR }, commit=False)
            compute_cost = compute_cost['efficiency/operations-per-eval']

            elapsed_steps = 0

            print(run.name)
            print("STAT: beginning training...")

            # train loop!
            for epoch_n in trange(MAX_EPOCHS, desc=f"{run.name if run else 'unnamed_run'} | epochs", leave=False):
                for i, (inputs, labels) in enumerate(pbar := tqdm(train_dl, desc="batches", leave=False)):
                    pbar.set_description(f"tot_step: {elapsed_steps}. batches")

                    n_chunks, n_channels, n_samples = inputs.shape
                    decoder.train_batch_size = n_chunks
                    loss, _ = train_batch(  device,
                                            fe_net,
                                            pls_mdl,
                                            decoder,
                                            optimizer,
                                            scheduler,
                                            loss_fn,
                                            inputs,
                                            labels,
                                            batch_size=FENET_MEMLIMIT_SERIAL_BATCH_SIZE)

                    quantization = EVAL_WLFL_PAIRS if EVAL_WITH_QUANTIZATION else None

                    if elapsed_steps % EVAL_STEPS == 0:

                        eval_res = evaluate_with_criteria(fe_net, pls_mdl, decoder, dev_dl, [
                            pearson_r_squared_criterion
                        ], device, quantization=quantization)

                        #save model dictionary if good
                        states_to_save = [
                                            dict(config),
                                            fe_net.state_dict(),
                                            optimizer.state_dict(),
                                            scheduler.state_dict() if scheduler else "no scheduler??"   # clean
                                            ]
                        file_label = f"{run.id}-{run.name}-fold{fold}-step{elapsed_steps}"
                        saver.save_if_good(
                            eval_res['pearsonr2-xy-normed'],
                            elapsed_steps,
                            states_to_save,
                            label=file_label)
                        additional = {
                        }
                        

                        # perf to cost ratio with quantized... maybe? or is this implemented wrong?
                        for k, v in eval_res.items():
                            if k.endswith('/R²') and 'quantized' in k:
                                additional['eval/perf-cost-ratio/' + k.split('/')[2] + '/R²'] = v

                        run.log(eval_res, commit=False)
                        run.log(additional, commit=False)

                    run.log({"fold": fold}, commit=False)
                    run.log({ 'loss': loss.item() })

                    elapsed_steps += 1

            # save the best performance from this fold, for reporting the average best r2 accross folds
            best_performance_in_this_fold = 0
            while not saver.saved_objects.empty():
                best_performance_in_this_fold = max(best_performance_in_this_fold, saver.saved_objects.get().metric)
            best_performance_by_fold.append(best_performance_in_this_fold)
        run.log({ 'cross-validation-avg-r2': sum(best_performance_by_fold) / len(best_performance_by_fold) })
