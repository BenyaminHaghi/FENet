import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from qtorch import FixedPoint
from qtorch.quant import Quantizer
from FENet_parameterizable import QuantizedFENet
from FENet_parameterizable import make_QFENet_from_FENet
from FENet_parameterizable import make_qfenet_from_quantized_statedict
from FENet_parameterizable import make_fenet_from_checkpoint
from FENet_parameterizable import inference_batch
from data_parser import make_data_from_day
from export import export_weights_for_hardware
from criteria import evaluate_with_criteria
from criteria import directional_R2_criterion, axes_plot_criterion
from decoder import compute_linear_decoder_loss_preds, r2_score, PLS_Model, Linear_Decoder
from criteria import devicify
import matplotlib
from matplotlib import pyplot as plt

from itertools import islice
import json
from os.path import join as path_join, basename
from os import getenv
from functools import partial
from multiprocessing import set_start_method

from configs import DATA_DIR
from configs import UNQUANTIZED_MODEL_DIR
from configs import QUANTIZED_MODEL_DIR
from configs import FILTERING_MIN_R2 as MIN_R2
from configs import TRAIN_FILTERING_N_TOP_CHANNELS as N_CHANNELS
from configs import EXPORT_MODEL_STRIDES as STRIDES

from configs import QUANTIZED_WORD_LENGTH as TOT_LEN
from configs import EVAL_WLFL_PAIRS
from configs import COMPARE_WITH_UNQUANTIZED
from configs import FENET_BATCH_SIZE
WL_RANGE = (4,17)
MIN_INTG = 1
TRAIN_LEN = None

if __name__ == '__main__':

    try: set_start_method("spawn")
    except: pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    silent = False

    #create unquantized fe_net solver
    fe_net = make_fenet_from_checkpoint(UNQUANTIZED_MODEL_DIR, device=device)
    fe_net.pls = 2
    fe_net.to(device)
    fe_net.eval()
    pls_mdl = PLS_Model(N_CHANNELS, len(fe_net.features_by_layer), fe_net.pls, TRAIN_LEN, device)
    decoder = Linear_Decoder(TRAIN_LEN, device)
    
    days = ['20190125', '20190215', '20190507', '20190625', '20190723', '20190806', '20190820', '20191008', '20191115']
    num_days = len(days)
    wordLen = np.zeros((num_days, WL_RANGE[1], WL_RANGE[1] - MIN_INTG))
    fracLen = np.zeros((num_days, WL_RANGE[1], WL_RANGE[1] - MIN_INTG))
    r2 = np.zeros((num_days, WL_RANGE[1], WL_RANGE[1] - MIN_INTG))
    r2q = np.zeros((num_days, WL_RANGE[1], WL_RANGE[1] - MIN_INTG))
    unqcriteria = [partial(directional_R2_criterion, device=device)]

    for day_indx, day in tqdm(enumerate(days)):
        dls = make_data_from_day(DATA_DIR, day, MIN_R2, N_CHANNELS, None, None)
        inputs = dls[0][0]
        labels = dls[0][1]
        n_chunks, n_channels, n_samples = inputs.shape
        pls_mdl = PLS_Model(N_CHANNELS, len(fe_net.features_by_layer), fe_net.pls, TRAIN_LEN, device)
        pls_mdl.trained = False
        decoder.trained = False
        preds_dl = [inference_batch(device, fe_net, pls_mdl, decoder, inputs, labels, batch_size=FENET_BATCH_SIZE).cpu().detach().numpy()]
        unquantR2 = evaluate_with_criteria(fe_net, pls_mdl, decoder, dls, unqcriteria, device, preds_dl=preds_dl)['eval/timely/decoder-xy-avg-R2']

        for wl in tqdm(range(WL_RANGE[0], WL_RANGE[1])):
            for fl in tqdm(range(1,wl-MIN_INTG)):

                qfe_net = make_QFENet_from_FENet(wl, fl, fe_net, device, quantize_weights=True)
                qfe_net.pls = 2
                qfe_net.to(device)
                qfe_net.eval()

                pls_mdl.trained = False
                decoder.trained = False
                qpreds_dl = [inference_batch(device, qfe_net, pls_mdl, decoder, inputs, labels, batch_size=FENET_BATCH_SIZE).cpu().detach().numpy()]

                criteria = [partial(directional_R2_criterion, device=device, quantization=(wl+1,fl+1))]

                wordLen[day_indx, wl,fl] = wl
                fracLen[day_indx, wl,fl] = fl
                r2q[day_indx, wl,fl] = evaluate_with_criteria(qfe_net, pls_mdl, decoder, dls, criteria, device, preds_dl=qpreds_dl)['eval/timely/decoder-xy-avg-R2']
                #print("day: ", day, " wl: ", wl+1, " fl: ", fl, "r2q :", r2q[day_indx, wl,fl])
                r2[day_indx, wl,fl] = unquantR2
        del(inputs)
        del(labels)
        del(preds_dl)
        del(qpreds_dl)
        del(qfe_net)
        del(pls_mdl)
    df = pd.DataFrame({'WordLen':wordLen.flatten(), 'fracLen':fracLen.flatten(), 'unQuantized_R2':r2.flatten(),'Quantized_R2':r2q.flatten()})
    df.to_csv("r2.csv")
