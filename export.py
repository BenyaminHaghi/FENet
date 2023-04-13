from re import I
import torch
from tqdm import tqdm, trange
import numpy as np
from fxpmath import Fxp     # fixed point conversion/export library
from qtorch import FixedPoint
from qtorch.quant import Quantizer
import pandas as pd

from FENet_parameterizable import make_QFENet_from_FENet, make_fenet_from_checkpoint, inference_batch, FENet, QuantizedFENet, make_qfenet_from_quantized_statedict
from data_parser import make_total_training_data
from utils import make_outputs_directory, send_dl_to_device
from criteria import evaluate_with_criteria, directional_R2_criterion

from itertools import chain as iter_chain
from os.path import join as path_join
from os import getenv
from multiprocessing import set_start_method, get_start_method

# EXPORT_TARGET = 'ndt'
from configs import EXPORT_TARGET

from configs import FILTERING_MIN_R2 as MIN_R2
from configs import FILTERING_N_TOP_CHANNELS as N_CHANNELS
from configs import DATA_DIR
from configs import UNQUANTIZED_MODEL_DIR
from configs import EXPORT_MODEL_STRIDES as STRIDES
from configs import EVAL_WLFL_PAIRS
WL, FL = EVAL_WLFL_PAIRS[0]

from configs import REDO_QUANTIZE

from configs import QUANTIZED_MODEL_DIR as MODEL_DIR
from configs import QUANTIZED_DATA_DIR as OUTPUT_DIRECTORY_BASEPATH
MODEL_DIR = "/home/lost/Research/Sinbad/FENet/model"
OUTPUT_DIRECTORY_BASEPATH = "/home/lost/Research/Sinbad/FENet/data"
from configs import QUANTIZED_DATA_FOLDER as DAY   # new sweep with full impl (pls, etc)

FIRST_N = 4

def export_for_NDT(device, fe_net, dls, checkpoint):
    # uses old interface with MODEL_SHAPE
    # CHECKPOINT_PATH = "F:\\Albert\\FENet\\wandb_saves\\genial-sweep-57_step-1160_perf-0.6615"; MODEL_SHAPE = [[1]*8, [10, 10, 2, 4, 32, 34, 26], [3, 2, 30, 13, 23, 25, 27]]; PLS_DIMS = 0
    # fe_net = make_fenet_from_checkpoint(CHECKPOINT_PATH, override_shape=MODEL_SHAPE, device=device, pls_dims=PLS_DIMS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dls = [z for y in make_total_training_data(DATA_DIR, n_filtered_channels=N_CHANNELS) for z in y]
    output_dir = make_outputs_directory(checkpoint, basepath="C:\\Users\\ahuang3\\Box\\inference-outputs-for-ndt\\pls_2-resiliant-47")
    for i, (inputs, labels) in enumerate(tqdm(dls)):
        out = inference_batch(device, fe_net, inputs, labels)
        print("inputs, labels shape", inputs.shape, labels.shape)
        print("got output shape", out.shape)
        np.save(f"{output_dir}/gitignoreme_outputs_day{i}.npy", out.detach().cpu().numpy())
        np.save(f"{output_dir}/gitignoreme_labels_day{i}.npy", labels.numpy())

    # output_dir = "./out/inference-outputs__genial-sweep-57_step-1160_perf-0.6615"
    for i in range(11):
        outputs, labels = np.load(f"{output_dir}/gitignoreme_outputs_day{i}.npy"), np.load(f"{output_dir}/gitignoreme_labels_day{i}.npy")
        print(outputs.shape, labels.shape)

def recursive_flat_float_iter(inp):
    from collections.abc import Iterable
    if isinstance(inp, Iterable):
        try:
            for item in inp:
                yield from recursive_flat_float_iter(item)
        except TypeError:   # non-iterable 0-d torch.Tensors are apparently Iterables
            yield float(inp.item())
    else:
        yield float(inp)

def convert_to_hex(v, wl, fl, total_length=None, signed=True, sign_mag=False):

    total_length = wl if total_length == None else total_length
    if sign_mag and signed:
        sign = v < 0
        v = np.abs(v)
        v = Fxp(v, signed=False, n_word=(wl-1), n_frac=fl)
        v.resize(signed=False, n_word=total_length, n_frac=fl + total_length - wl)
        bin_str = v.bin()
        bin_str = ('1' if sign else '0') + bin_str[1:]
        return hex(int(bin_str, 2))[2:].zfill(int(np.ceil(total_length/4))).upper()
    elif signed:
        v = Fxp(v, signed=True, n_word=wl, n_frac=fl)
        v.resize(signed=True, n_word=total_length, n_frac=fl + total_length - wl)
        return v.hex()[2:]
    else:
        v = Fxp(v, signed=False, n_word=(wl-1), n_frac=fl)
        v.resize(signed=False, n_word=total_length, n_frac=fl + total_length - wl)
        return v.hex()[2:]

def convert_to_oct(v, wl, fl, total_length=None, signed=True, sign_mag=True):
    total_length = wl if total_length == None else total_length
    hex_val = convert_to_hex(v, wl, fl, total_length, signed=signed, sign_mag=sign_mag)
    return oct(int(hex_val, 16))[2:].zfill(int(np.ceil(total_length/3))).upper()

def convert_to_bin(v, wl, fl, total_length=None, signed=True, sign_mag=True):
    total_length = wl if total_length == None else total_length
    hex_val = convert_to_hex(v, wl, fl, total_length, signed=signed, sign_mag=sign_mag)
    return bin(int(hex_val, 16))[2:].zfill(int(np.ceil(total_length))).upper()

def write_data_file(path, data, formatter=None, buf_n=65536, filemode='a+', separator='\n', silent=True):
    from more_itertools import chunked
    from os.path import basename
    from io import StringIO

    if formatter is None:
        formatter = lambda x : x

    with open(path, filemode) as wf, StringIO() as str_buff:
        data = np.char.add(formatter(data), separator).flatten()
        for chunk in data:
            str_buff.write(chunk)
        wf.write(str_buff.getvalue())

def export_weights_for_hardware(qfe_net: QuantizedFENet, output_dir: str):
    """
    Export fixed-point binary weights for hardware use. Will export
    weights in total_length bits, but only populate the first wl bits.
    """
    # clear the output directories because we're going to append
    with open(path_join(output_dir, 'feat_weights.txt'), 'w+') as wf: wf.write('')
    with open(path_join(output_dir, 'pass_weights.txt'), 'w+') as wf: wf.write('')

    for name, mat in qfe_net.state_dict().items():
        if 'feat' in name:
            write_data_file(path_join(output_dir, 'feat_weights.txt'), mat, formatter=qfe_net.cache_formatter)
        if 'pass' in name:
            write_data_file(path_join(output_dir, 'pass_weights.txt'), mat, formatter=qfe_net.cache_formatter)

    with open(f"{output_dir}/model_info.txt", 'w+') as wf:
        wf.write(f"checkpoint: {qfe_net.checkpoint_name}\npls: {qfe_net.pls}\nfeatures: {qfe_net.features_by_layer}\nkernels: {qfe_net.kernel_by_layer}\nstrides: {qfe_net.stride_by_layer}\n")

def bit_stringify(val, num_bits):
    bit_str = bin(val)[2:]
    if(len(bit_str) > num_bits):
        print("Value Too Large.. Truncating: ", bit_str, " to :", bit_str[:num_bits])
        return bit_str[:num_bits]
    else:
        return bit_str.zfill(num_bits)[::-1]

def export_configs(qfe_net: QuantizedFENet, output_dir: str, num_cycles=450, write_files=True):
    """
    Export configurations into a text format that will be written to hardware.
    """
    from math import log
    # clear the output directories because we're going to append
    with open(path_join(output_dir, 'configs.txt'), 'w+') as wf: wf.write('')
    from configs import POOL_REG_BIT_WIDTH
    from configs import ACCUM_REG_BIT_WIDTH
    from configs import NUM_FENET_BUILT
    from configs import NUM_FEM_BUILT
    from configs import MAX_WEIGHT_DEPTH
    from configs import MAX_NUM_CYCLES
    from configs import FILTERING_N_TOP_CHANNELS
    from configs import DATA_VALID_DELAY
    configs_bit_str = ''

    # data_valid_delay
    configs_bit_str = configs_bit_str + bit_stringify(DATA_VALID_DELAY, WL-1)

    # num_chan_en
    configs_bit_str = configs_bit_str + bit_stringify(FILTERING_N_TOP_CHANNELS, int(np.ceil(log(NUM_FENET_BUILT,2)+1)))

    # num_fem_en
    configs_bit_str = configs_bit_str + bit_stringify(len(qfe_net.kernel_by_layer), int(np.ceil(log(NUM_FEM_BUILT,2)+1)))

    # num_cycles
    configs_bit_str = configs_bit_str + bit_stringify(num_cycles, int(np.ceil(log(MAX_NUM_CYCLES,2)+1)))

    # relu_powers
    for i in range(NUM_FEM_BUILT):
        if(len(qfe_net.relu_by_layer) > i):
            configs_bit_str = configs_bit_str + bit_stringify(qfe_net.relu_by_layer[i], int(np.ceil(log(ACCUM_REG_BIT_WIDTH,2))))
        else:
            configs_bit_str = configs_bit_str + bit_stringify(qfe_net.relu_by_layer[-1], int(np.ceil(log(ACCUM_REG_BIT_WIDTH,2))))

    # div_powers
    for i in range(NUM_FEM_BUILT):
        if(len(qfe_net.poolDivisor) > i):
            configs_bit_str = configs_bit_str + bit_stringify(int(np.ceil(log(qfe_net.poolDivisor[i],2))), int(np.ceil(log(POOL_REG_BIT_WIDTH,2))))
        else:
            configs_bit_str = configs_bit_str + bit_stringify(int(np.ceil(log(qfe_net.poolDivisor[-1],2))), int(np.ceil(log(POOL_REG_BIT_WIDTH,2))))

    # strides
    for i in range(NUM_FEM_BUILT):
        if(len(qfe_net.stride_by_layer) > i):
            configs_bit_str = configs_bit_str + bit_stringify(qfe_net.stride_by_layer[i], int(np.ceil(log(MAX_WEIGHT_DEPTH,2))))
        else:
            configs_bit_str = configs_bit_str + bit_stringify(qfe_net.stride_by_layer[-1], int(np.ceil(log(MAX_WEIGHT_DEPTH,2))))

    # kernel address offsets
    address = 0
    for i in range(NUM_FEM_BUILT):
        
        if(len(qfe_net.kernel_by_layer) > i):
            address += qfe_net.kernel_by_layer[i]

        configs_bit_str = configs_bit_str + bit_stringify(address, int(np.ceil(log(MAX_WEIGHT_DEPTH,2))))


    configs_bit_str = np.asarray([*configs_bit_str])
    if(write_files):
        write_data_file(path_join(output_dir, f"configs.txt"), configs_bit_str, filemode="w+")


def make_qfenet_for_export(fe_net: FENet, output_dir: str, total_length: int, wl: int, fl: int, sample_size=10):
    """
    Make `sample_size=10` quantized FENets, then select the best one (on the validation set) for export.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, val_dl, _ = make_total_training_data(DATA_DIR)

    best_perf = 0
    best_qfe_net = None
    for _ in trange(sample_size, desc="evaluating qfenets"):
        import gc; gc.collect()
        qfenet = make_QFENet_from_FENet(wl, fl, fe_net, device, quantize_weights=True)
        perf = evaluate_with_criteria(qfenet, val_dl, [directional_R2_criterion], device)['eval/timely/decoder-xy-norm-R2']
        if perf > best_perf:
            best_perf = perf
            best_qfe_net = qfenet

    # qfenets = [make_QFENet_from_FENet(wl, fl, fe_net, device, quantize_weights=True) for _ in range(sample_size)]
    # print(list(evaluate_with_criteria(fe_net, val_dl, [decoder_crit.R2], device).keys()))
    # perf = [evaluate_with_criteria(qfe_net, val_dl, [decoder_crit.R2], device)['eval/decoder-retrain/R²'] for qfe_net in tqdm(qfenets, desc="evaluating qfenets")]
    # print(pd.DataFrame({'performance': perf}).describe())

    # the_one = qfenets[np.argmax(perf)]
    export_weights_for_hardware(best_qfe_net, output_dir, total_length)
    return best_qfe_net

def export_data_for_hardware(qfe_net, dls, write_files=True, first_n=None, quantizer=None, interm_layers=True):

    from configs import QUANTIZED_FILE_WRITE_MODE as wm
    from configs import POOL_REG_BIT_WIDTH

    if write_files:
        with open(path_join(output_dir, 'labels_info.txt'), wm) as wf:
            wf.write('\n\n\n\n' + str(qfe_net.state_dict()) + '\n\n\n\n')
    for i, (inputs, labels) in enumerate(tqdm(dls, desc="exporting data by day")):
        import json
        #inputs = torch.transpose(inputs, 1, 2)
        print("inputs, labels shape", inputs.shape, labels.shape)
        outputs = []
        enforce_np = lambda x : x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        qfe_net.eval()
        with torch.no_grad():

            n_chunks, n_channels, n_samples = inputs.shape
            if first_n is not None:
                qfe_net.num_to_cache = first_n*n_channels
                n_chunks = first_n
                inputs = inputs[:first_n, :, :]
                labels = labels[:first_n, :]
            else:
                qfe_net.num_to_cache = n_chunks*n_channels

            inputs = inputs.reshape(n_chunks * n_channels, 1, n_samples)
            outputs.append(qfe_net(inputs).reshape(n_chunks, n_channels, len(qfe_net.features_by_layer)))

        outputs = torch.cat(outputs)

        #Reshape data so it is written in the order it will be needed in hardware
        inputs = inputs.reshape(n_chunks, n_channels, n_samples).transpose(2,1)
        for j, layer in enumerate(qfe_net.pass_layers):
            qfe_net.pass_layers[j] = np.transpose(layer.reshape(n_chunks, n_channels, -1), [0,2,1])
        for j, layer in enumerate(qfe_net.feat_layers):
            layer = enforce_np(layer)
            for k, chunk in enumerate(layer):
                layer[k] = np.cumsum(chunk)
            qfe_net.feat_layers[j] = np.transpose(layer.reshape(n_chunks, n_channels, -1), [0,2,1])

        print("got output shape", outputs.shape)
        
        print("writing inputs file with shape: ", inputs.shape)
        print("writing outputs file with shape: ", outputs.shape)
        print("writing labels file with shape: ", labels.shape)
        print(f"beginning of outputs for day {i}")

        if interm_layers:
            pass_layer_files = [ f"pass_layer_{k}.txt" for k in range(len(qfe_net.pass_layers))]
            feat_layer_files = [ f"feat_layer_{k}.txt" for k in range(len(qfe_net.feat_layers))]

        if write_files:
            formatter_fn = qfe_net.cache_formatter
            feat_layer_formatter_fn = lambda x : convert_to_hex(enforce_np(x), POOL_REG_BIT_WIDTH, 2*FL, total_length=POOL_REG_BIT_WIDTH, signed=False, sign_mag=False)
            feat_layer_formatter_fn = np.vectorize(feat_layer_formatter_fn)
            if quantizer is not None:
                inputs = quantizer(inputs)
            write_data_file(path_join(output_dir, f"inputs.txt"), inputs, formatter=formatter_fn, filemode=wm)
            write_data_file(path_join(output_dir, f"outputs.txt"), outputs, formatter=formatter_fn, filemode=wm)
            write_data_file(path_join(output_dir, f"labels.txt"), labels, formatter=formatter_fn, filemode=wm)
            if interm_layers:
                no_change_fn = lambda x : x
                for layer_indx, file_name in enumerate(pass_layer_files):
                    write_data_file(path_join(output_dir, file_name), qfe_net.pass_layers[layer_indx], filemode=wm)
                for layer_indx, file_name in enumerate(feat_layer_files):
                    write_data_file(path_join(output_dir, file_name), qfe_net.feat_layers[layer_indx], formatter=feat_layer_formatter_fn, filemode=wm)
            with open(path_join(output_dir, 'labels_info.txt'), 'a+') as wf:
                wf.write(f"day: {i}\ninputs shape: {inputs.shape}\nlabels shape: {labels.shape}\noutputs shape: {outputs.shape}\n\n")


def test_hardware_with_first_input(data_dl, fe_net, device):
    first_input = data_dl[0][0][0, 0, :]
    print(first_input.shape)
    first_input = first_input.to(device)

    got = fe_net(torch.unsqueeze(torch.unsqueeze(first_input, 0), 0))
    # got = fe_net(first_input)
    #print(got)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    quantizer = Quantizer(forward_number=FixedPoint(wl=WL, fl=FL, clamp=True, symmetric=True), forward_rounding='nearest')
    
    try:
        set_start_method("spawn") #prevent child process from sharing the lock state of the file and deadlocking
    except RuntimeError:
        pass
    
    dls = [z for y in make_total_training_data(DATA_DIR, n_filtered_channels=N_CHANNELS, min_R2=MIN_R2, days=['20190125']) for z in y]
    send_dl_to_device(dls, device)
    output_dir = make_outputs_directory(DAY, basepath=OUTPUT_DIRECTORY_BASEPATH)
    num_to_cache = FIRST_N*N_CHANNELS

    if REDO_QUANTIZE:
        fe_net = make_fenet_from_checkpoint(UNQUANTIZED_MODEL_DIR, device=device)
        qfe_net = make_QFENet_from_FENet(WL, FL, fe_net, device, quantize_weights=True, cache_intermediate_outputs=True, num_to_cache=num_to_cache)
        state_dict = qfe_net.state_dict()
        state_dict["stride_by_layer"] = qfe_net.stride_by_layer
        state_dict["relu_by_layer"] = qfe_net.relu_by_layer
        state_dict["quantization"] = (qfe_net.wl, qfe_net.fl)
        torch.save(state_dict, path_join(MODEL_DIR, "qfe_net.pt"))
    else:
        qfe_net = make_qfenet_from_quantized_statedict(MODEL_DIR, device=device, cache_intermediate_outputs=True, num_to_cache=num_to_cache)
    qfe_net.pls = 0
    #test_hardware_with_first_input(dls, qfe_net, device)

    qfe_net.set_cache_format('Oct', total_length=WL)
    export_data_for_hardware(qfe_net, dls, write_files=True, first_n=FIRST_N, quantizer=quantizer)
    export_configs(qfe_net, MODEL_DIR)
    export_weights_for_hardware(qfe_net, MODEL_DIR)
