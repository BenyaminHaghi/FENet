from multiprocessing import reduction
from FENet_parameterizable import FENet, train_batch
WANDB_FIX_TAGS = [ "fix-r2-1", "fix-r2-2-normdiv2", "fix-linux-fullrefresh-1", "fix-pls-two" ]


from configs import EVAL_STEPS
from configs import MAX_EPOCHS
from configs import EFFICIENCY_METRIC_INPUT_LEN
from configs import COMPUTE_COST_DIVISOR
from configs import EVAL_WITH_QUANTIZATION
from configs import EVAL_WLFL_PAIRS
from configs import USE_MULTITHREADED_WORKERS
from configs import UNQUANTIZED_MODEL_DIR as BEST_CURRENT_MODEL
from configs import FENET_BATCH_SIZE



def kfolds_train_worker(queue, saver, run, lock, fold, train_dl, val_dl):
    """
    creates a new wandb run and trains a model using `config` on the given data
    returns tuples of (training_performance, saver.save_if_good) so that the parent thread can save good models across folds
    """
    #print("kfold worker started :", fold)
    import torch

    from tqdm import tqdm, trange
    import wandb

    from criteria import EfficiencyCriterion
    from criteria import R2_avg_criterion, R2_hist_criterion, axes_plot_criterion, directional_R2_criterion
    from criteria import evaluate_with_criteria
    # from utils import seed_everything, BestEpochSaver, KFoldsGenerator

    from functools import partial
    import os
    from configs import FILTERING_N_TOP_CHANNELS
    from utils import filter_dict
    import traceback

    def do_everything_scope(run):
        #print("Everything Scope Began")
        os.environ["WANDB_SILENT"] = "true" # silence wandb in workers to avoid messing up folds progress bar

        print("\n"*8, run.tags, "\n"*8)

        device, fe_net, pls_mdl, decoder, loss_fn, optimizer, scheduler, run, config = initialize(run)

        print("STAT: model and data initailized")

        # calculate and log compute cost for this run
        efficiency_crit = EfficiencyCriterion(fe_net)
        #print("Evaluating efficiency criterion")
        compute_cost = efficiency_crit.evaluate(EFFICIENCY_METRIC_INPUT_LEN)
        #print("Cost computed")
        run.log(compute_cost, commit=False)
        run.log({ 'compute_cost_divisor': COMPUTE_COST_DIVISOR }, commit=False)
        compute_cost = compute_cost['efficiency/operations-per-eval']

        #print("about to try step")
        try:
            elapsed_steps = 0

            yield [
                0,     # performance
                f"{run.name}-fold{fold}-step{elapsed_steps}",   # label
                (                                               # the actual object to save
                    { k: v for k, v in run.config.items() if not k.startswith('_') },
                    elapsed_steps,
                    fold,
                ),
                {                                               # numeric metrics to turn into histograms
                    'decoder-r2': 0,
                    'decoder-x-R2': 0,
                    'decoder-y-R2': 0,
                }
            ]

            print(run.name)
            print("STAT: beginning training...")

            # train loop!
            for epoch_n in trange(MAX_EPOCHS, desc=f"{run.name if run else 'unnamed_run'} | epochs", leave=False):
                #print("trange not freezing")
                for i, (inputs, labels) in enumerate(pbar := tqdm(train_dl, desc="batches", leave=False)):
                    # print("STAT: pbar should be exist")
                    #print("train loop entered")
                    pbar.set_description(f"tot_step: {elapsed_steps}. batches")

                    #print("about to train")
                    #Clear pls model before each batch
                    #fe_net.pls_mdl = None
                    n_chunks, n_channels, n_samples = inputs.shape
                    decoder.train_batch_size = n_chunks
                    pls_mdl.train_batch_size = n_chunks
                    pls_mdl.trained = False
                    loss, _ = train_batch(  device,
                                            fe_net,
                                            pls_mdl,
                                            decoder,
                                            optimizer,
                                            scheduler,
                                            loss_fn,
                                            inputs,
                                            labels,
                                            batch_size=FENET_BATCH_SIZE)
                    #print("choochoo it's trianed!")

                    quantization = EVAL_WLFL_PAIRS if EVAL_WITH_QUANTIZATION else None
                    pls_mdl.trained = False

                    if elapsed_steps % EVAL_STEPS == 0:

                        #if(elapsed_steps != 0):
                            #clear pls training
                            #pls_mdl.trained = False

                        eval_res = evaluate_with_criteria(fe_net, pls_mdl, decoder, val_dl, [
                            partial(R2_avg_criterion, device=device, quantization=quantization),
                            partial(R2_hist_criterion, device=device, quantization=quantization),
                            partial(directional_R2_criterion, device=device, quantization=quantization),
                            partial(axes_plot_criterion, device=device, quantization=quantization, day_names=[fold]),
                        ], device, quantization=quantization)

                        #save model dictionary if good
                        states_to_save = [
                                            dict(config),
                                            fe_net.state_dict(),
                                            optimizer.state_dict(),
                                            scheduler.state_dict() if scheduler else "no scheduler??"   # clean
                                            ]

                        file_label = f"{run.name}-fold{fold}-step{elapsed_steps}"

                        if USE_MULTITHREADED_WORKERS: lock.acquire()
                        saver.save_if_good(
                            eval_res['eval/timely/decoder-xy-norm-R2'],
                            elapsed_steps,
                            states_to_save,
                            label=file_label)
                        if USE_MULTITHREADED_WORKERS: lock.release()

                        # print(list(eval_res.keys()))

                        # jankily add additional metrics
                        additional = {
                            'decoder-r2': eval_res['eval/timely/decoder-xy-norm-R2'],
                            # 'compute-cost': compute_cost,
                            # 'unquantized-R², (8, 5)-over-cost': eval_res['eval/decoder-retrain/R²'] / compute_cost,
                            'unquantized-perf-cost-ratio': eval_res['eval/timely/avg-decoder-R2'] / compute_cost,
                            'unquantized-perf-cost-combo': eval_res['eval/timely/avg-decoder-R2'] - compute_cost / COMPUTE_COST_DIVISOR,
                        }

                        # perf to cost ratio with quantized... maybe? or is this implemented wrong?
                        for k, v in eval_res.items():
                            if k.endswith('/R²') and 'quantized' in k:
                                additional['eval/perf-cost-ratio/' + k.split('/')[2] + '/R²'] = v

                        yield [
                            eval_res['eval/timely/decoder-xy-norm-R2'],     # performance
                            file_label,                                     # label
                            (
                                { k: v for k, v in run.config.items() if not k.startswith('_') },
                                  elapsed_steps,
                                  fold
                            ),
                            {                                               # numeric metrics to turn into histograms
                                'decoder-r2': eval_res['eval/timely/decoder-xy-norm-R2'],
                                'decoder-x-R2': eval_res['eval/timely/decoder-x-R2'],
                                'decoder-y-R2': eval_res['eval/timely/decoder-y-R2'],
                            }
                        ]

                        if USE_MULTITHREADED_WORKERS: lock.acquire()
                        run.log(eval_res, commit=False)
                        run.log(additional, commit=False)
                        if USE_MULTITHREADED_WORKERS: lock.release()

                    if USE_MULTITHREADED_WORKERS: lock.acquire()
                    run.log({ 'loss': loss.item() })
                    if USE_MULTITHREADED_WORKERS: lock.release()

                    elapsed_steps += 1

        except Exception as e:
            traceback.print_exc()
            print(e)
            print("STAT: training failed... adding tags")
            run.tags = ('automated/training-failed',) + run.tags
            print("STAT: tags added")
            yield [None, {}]
            return

        # for k in wandb.summary.keys():
        #     print(k, type(wandb.summary[k]), type(wandb.summary[k]) == int or type(wandb.summary[k]) == float)

        # print('\n' * 10)

        filters = [
            (lambda key, value: key.startswith('_'), 'exclude'),
            (lambda key, value: type(value) == float, 'include'),
            (lambda key, value: type(value) == int, 'include'),
            ]

        #get the summary
        wandb_summary = wandb.summary
        #turn summary into dictionary
        wandb_summary = {key: wandb_summary[key] for key in wandb_summary.keys()}
        #filter and append the summary and add the fold identifier key value pair to the summary
        # summary_to_return = filter_dict(wandb_summary, filters)
        # print(summary_to_return)
        # yield [None, summary_to_return]

    for yielded in do_everything_scope(run):
        # print("STAT: outer yielding", yielded)
        queue.put(yielded)
    # release memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
