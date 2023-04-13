import torch
from tqdm import tqdm
import numpy as np

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
from configs import FILTERING_N_TOP_CHANNELS as N_CHANNELS
from configs import EXPORT_MODEL_STRIDES as STRIDES

from configs import QUANTIZED_WORD_LENGTH as TOT_LEN
from configs import EVAL_WLFL_PAIRS
WL, FL = EVAL_WLFL_PAIRS[0]
from configs import COMPARE_WITH_UNQUANTIZED
from configs import FENET_BATCH_SIZE

if __name__ == '__main__':

    try: set_start_method("spawn")
    except: pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    silent = False

    #create unquantized fe_net solver
    if COMPARE_WITH_UNQUANTIZED:
        fe_net = make_fenet_from_checkpoint(UNQUANTIZED_MODEL_DIR, device=device)
        fe_net.pls = 2
        fe_net.to(device)
        fe_net.eval()

        qfe_net = make_QFENet_from_FENet(WL, FL, fe_net, device, quantize_weights=True)
    else:
        #create quantized fe_net solver
        qfe_net = make_qfenet_from_quantized_statedict(QUANTIZED_MODEL_DIR, cache_intermediate_outputs=False)
        qfe_net.set_cache_format('Oct')
        qfe_net.pls = 2
        qfe_net.to(device)
        qfe_net.eval()

    pls_mdl = PLS_Model(N_CHANNELS, len(qfe_net.features_by_layer), qfe_net.pls, 1000, device)
    decoder = Linear_Decoder(1000, device)
    
    days = ['20190125']#['20190125', '20190215', '20190507', '20190625', '20190723', '20190806', '20190820', '20191008', '20191115']
    for day in days:
        dls = make_data_from_day(DATA_DIR, day, MIN_R2, N_CHANNELS, None, None)
        inputs = dls[0][0]
        labels = dls[0][1]
        n_chunks, n_channels, n_samples = inputs.shape
        pls_mdl.train_batch_size = n_chunks
        decoder.train_batch_size = n_chunks

        qpreds_dl = [inference_batch(device, qfe_net, pls_mdl, decoder, inputs, labels, batch_size=FENET_BATCH_SIZE).cpu().detach().numpy()]
        preds_dl = [inference_batch(device, fe_net, pls_mdl, decoder, inputs, labels, batch_size=FENET_BATCH_SIZE).cpu().detach().numpy()]
        del(inputs)
        del(labels)

        print("day: ", day )
        criteria = [partial(directional_R2_criterion, device=device, quantization=(WL,FL)), partial(axes_plot_criterion, device=device, quantization=(WL,FL), compatibility='matplot')]

        r2_quant = evaluate_with_criteria(qfe_net, pls_mdl, decoder, dls, criteria, device, preds_dl=qpreds_dl)
        if COMPARE_WITH_UNQUANTIZED: r2_unquant = evaluate_with_criteria(fe_net, pls_mdl, decoder, dls, criteria, device, preds_dl=preds_dl)
        del(dls)
        del(preds_dl)
        del(qpreds_dl)

        print("average quantized R2:", r2_quant['eval/timely/decoder-xy-avg-R2'])
        if COMPARE_WITH_UNQUANTIZED: print("average unquantized R2:", r2_unquant['eval/timely/decoder-xy-avg-R2'])

        q_fig = plt.figure()
        plt.imshow(r2_quant['eval/timely/decoder-preds-chart'], interpolation='nearest')
        plt.show()

        if COMPARE_WITH_UNQUANTIZED:
            fig = plt.figure()
            plt.imshow(r2_unquant['eval/timely/decoder-preds-chart'], interpolation='nearest')
            plt.show()
        
        #wait for user to advance to next day
        input()
        plt.close(q_fig)
        if COMPARE_WITH_UNQUANTIZED: plt.close(fig)

