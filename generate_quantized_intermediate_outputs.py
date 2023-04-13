import torch
from tqdm import tqdm
import numpy as np
from fxpmath import Fxp

from FENet_parameterizable import QuantizedFENet
from FENet_parameterizable import make_qfenet_from_quantized_statedict
from export import export_weights_for_hardware

from itertools import islice
import json
from os.path import join as path_join, basename

#QUANTIZED_DATA_DIR = "../data/exportpy_week5out/skilled-sweep-53_step-350_perf-0.6515"
from configs import QUANTIZED_DATA_DIR
from configs import QUANTIZED_MODEL_DIR
from configs import QUANTIZED_WORD_LENGTH as TOT_LEN
from configs import EVAL_WLFL_PAIRS
WL, FL = EVAL_WLFL_PAIRS[0]
from configs import GENERATE_WEIGHT_FILE

def print_octal(arr, n=10):
    # TODO is this correct and done
    for x in arr:
        print(Fxp(x, signed=True, n_word=WL, n_frac=FL).base_repr(8, frac_dot=False), end=' ')

def read_quantized_inputs_from_exported_file(path, wl, fl, first_n_by_shape):
    """
    get product(first_n_by_shape) elements and reshape it to that shape
    """
    values = []
    with open(path, 'r') as rf:
        for line in tqdm(islice(rf, np.prod(first_n_by_shape)), total=np.prod(first_n_by_shape)):
            v = Fxp(0, signed=True, n_word=wl, n_frac=fl)
            v('0x' + line)
            values.append(v.astype(float))

    values = np.array(values).reshape(first_n_by_shape)
    values = values.transpose(1, 0, 2)  # TODO: why is this the correct way of reading to agree with hardware
    values = torch.from_numpy(values).to(dtype=torch.float)
    return values

if __name__ == '__main__':
    qfe_net = make_qfenet_from_quantized_statedict(QUANTIZED_MODEL_DIR, [6, 2, 5, 2, 19, 3])
    if(GENERATE_WEIGHT_FILE):
        with open(path_join(QUANTIZED_DATA_DIR, 'state_dict.txt'), 'w+') as wf: json.dump({ k: v.tolist() for k, v in qfe_net.state_dict().items() }, wf)
        export_weights_for_hardware(qfe_net, QUANTIZED_DATA_DIR, TOT_LEN)
    print('\n'*5)
    q_inputs = read_quantized_inputs_from_exported_file(path_join(QUANTIZED_DATA_DIR, 'gitignoreme_quantized_inputs_day0.txt'), 16, 13, [1, 25, 900])

    n, n_samples, n_channels = q_inputs.shape
    q_inputs = q_inputs.reshape(n*n_samples, 1, n_channels)

    print_octal(q_inputs[:10, 0, 0].tolist())
    print(q_inputs.shape, q_inputs[:10, 0, 0])

    outputs = qfe_net(q_inputs)

    for i, layer in enumerate(qfe_net.values_hist_data):
        # print(len(layer))
        print(i, layer[:20])
