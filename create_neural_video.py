import torch
import numpy as np
import cv2

from FENet_parameterizable import make_fenet_from_checkpoint
from FENet_parameterizable import inference_batch
from data_parser import make_data_from_day
from decoder import PLS_Model, Color_Decoder
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt

from os import getenv
from os.path import join
from functools import partial
from multiprocessing import set_start_method

from configs import DATA_DIR
from configs import UNQUANTIZED_MODEL_DIR
MIN_R2 = None
N_CHANNELS = None

from configs import EVAL_WLFL_PAIRS
WL, FL = EVAL_WLFL_PAIRS[0]
from configs import FENET_BATCH_SIZE
import numpy as np

arrayMap = 'arrayMap.npy'
videoOutDir = "Neural_Videos/"

if __name__ == '__main__':

    try: set_start_method("spawn")
    except: pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    silent = False

    fe_net = make_fenet_from_checkpoint(UNQUANTIZED_MODEL_DIR, device=device)
    fe_net.pls = 3
    fe_net.to(device)
    fe_net.eval()

    arMap = np.load(arrayMap, allow_pickle=True)[()]
    n_chan = len(arMap['ChanNum'].flatten())
    colorDecoder = Color_Decoder(arMap, pixel_size=16)


    pls_mdl = PLS_Model(n_chan*colorDecoder.numArrays, len(fe_net.features_by_layer), fe_net.pls, 1000, device)

    days = ['20190820']
    for day in days:
        dls = make_data_from_day(DATA_DIR, day, MIN_R2, N_CHANNELS, None, None)
        inputs = dls[0][0]
        labels = dls[0][1]
        n_chunks, n_channels, n_samples = inputs.shape
        pls_mdl.train_batch_size = n_chunks

        np_video_stack = inference_batch(device, fe_net, pls_mdl, colorDecoder, inputs, labels, batch_size=8000)
        del(dls)
        del(inputs)
        del(labels)
        width = colorDecoder.pixX*colorDecoder.pixel_size
        height = colorDecoder.pixY*colorDecoder.pixel_size
        del(fe_net)
        del(pls_mdl)
        del(colorDecoder)
        for e, elecArray in enumerate(np_video_stack):
            vid = cv2.VideoWriter(join(videoOutDir, f"day_{day}_array_{e}.avi"), 0, 33.333, (height, width))
            for frame in elecArray:
                vid.write(frame)
            cv2.destroyAllWindows()
            vid.release()

        print("day: ", day )

        # fig = plt.figure()
        # plt.imshow(r2_unquant['eval/timely/decoder-preds-chart'], interpolation='nearest')
        # plt.show()
        # plt.close(fig)

