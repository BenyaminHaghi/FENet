from inspect import trace
from re import I
import torch
import numpy as np
from tqdm import tqdm

from pathlib import Path
from threading import Lock

def seed_everything(the_seed):
    # from https://pytorch.org/docs/stable/notes/randomness.html
    from random import seed; seed(the_seed)
    from numpy.random import seed; seed(the_seed)
    from torch import manual_seed; manual_seed(the_seed)

from dataclasses import dataclass, field
@dataclass(order=True)
class SavedObjectHandle:
    metric: float
    path: Path=field(compare=False)

class BestEpochSaver:
    """
    Saves the top n_saves models according to metric
    """

    def __init__(self, save_dir: str, n_saves=10):
        self.dirname = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.max_n_saves = n_saves
        from queue import PriorityQueue
        self.saved_objects = PriorityQueue(maxsize=n_saves+1)   # stores the top n_saves performing models, lowest performance at the top

    def save_if_good(self, metric: float, step: int, obj, label: str=None):
        if (self.saved_objects.qsize() == self.max_n_saves and  # if queue is full and
            self.saved_objects.queue[0].metric > metric):       # this is worse than the worst
            return False # then we won't be saving, so just return.

        # create save file and remember the filename
        from os import path, remove as os_rm
        path = Path(self.dirname, f"{label if label else 'saved-model'}_step-{step}_perf-{metric:.4f}")
        torch.save(obj, path.absolute())

        handle = SavedObjectHandle(metric, path)
        self.saved_objects.put(handle, block=False)

        if self.saved_objects.qsize() > self.max_n_saves:   # if we are over the limit, remove the worst one
            to_del = self.saved_objects.get()
            to_del.path.unlink()

        return True

    def items(self):
        return self.saved_objects.queue

def convert_to_buf(fig):
    # save the plot to a numpy array so we can upload it to wandb
    # from https://stackoverflow.com/a/7821917/10372825
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    np_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return np_fig


def get_hist_bins(seq, bins=40, min_v=None, max_v=None):
    # two-pointer to find the histogram bins in n log n
    seq = sorted(seq)
    if min_v is None: min_v = min(seq)
    if max_v is None: max_v = max(seq)
    # print(min_v, max_v, seq)
    bin_starts = []
    ptr = 0   # inc, exc

    for i in range(bins):
        # print(f"    i={i}, ptr={ptr}, here={seq[ptr]}, need={(min_v * (1 - i/bins)) + (max_v * i/bins)}")
        # make the ptr catch up to where we need to add a bin
        while ptr < len(seq) and seq[ptr] < (min_v * (1 - i/bins)) + (max_v * i/bins):
            ptr += 1
        # now that it's caught up, this must be where the next bin starts
        # print("        adding!!!!")
        bin_starts.append(ptr)

    # print(bin_starts)
    return bin_starts

def get_hist_bin_sizes(seq, bins=40, min_v=None, max_v=None):
    bins = get_hist_bins(seq, bins, min_v, max_v)
    for i in range(0, len(bins)-1, 1):
        bins[i] = bins[i+1] - bins[i]
    bins[-1] = len(seq) - bins[-1]
    return [ x / len(seq) for x in bins ]
    # return [ x / max(bins) for x in bins ]

def make_outputs_directory(checkpoint_path, basepath=None, prefix=None):
    from os.path import basename, join
    from os import makedirs

    base_path = join(basepath or "C:\\Users\\ahuang3\\Box\\linked_out", (prefix + '_' if prefix is not None else '') + basename(checkpoint_path))
    try:
        makedirs(base_path)
    except FileExistsError:
        print(f"output dir for {base_path} already exists.")
    return base_path

def send_dl_to_device(dl, device):
    for i, day in enumerate(dl):
        dl[i] = (day[0].to(device), day[1].to(device))

def filter_dict(dict_to_filter, filters):
    keys_to_delete = []
    for key, value in dict_to_filter.items():
        include = False
        filter_types = []
        for filter_fn, filter_type in filters:
            filter_types.append(filter_type)
            if filter_fn(key, value):
                if(filter_type == 'exclude'):
                    keys_to_delete.append(key)
                    continue
                if(filter_type == 'include'):
                    include = True
                    continue
        if (not include) and ('include' in filter_types):
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del(dict_to_filter[key])
    return dict_to_filter

class KFoldsGenerator:
    def __init__(self, total_dls, n_folds, silent=False, seed=None, shuffle=False):
        from sklearn.model_selection import KFold
        self.dls = list(total_dls)
        self.n_folds = n_folds
        self.folder = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
        self.silent = silent

    def make_folds(self):
        for train_ids, val_ids in tqdm(self.folder.split(self.dls), desc="k-folds", disable=self.silent, total=self.n_folds):
            yield [self.dls[i] for i in train_ids], [self.dls[i] for i in val_ids]


def import_old_model_code():
    from ben_loading_code import cnn_model
    from FENet_parameterizable import FENet
    fenet = FENet(
        features_by_layer=[1]*8,
        kernel_by_layer=[40]*7,
        stride_by_layer=[2]*7,
        relu_by_layer=[0]*7,
        pls_dims=2,
        normalize_at_end=True,
    )
    for i, layer in enumerate(fenet.children()):
        print(i, layer)
    print(list(cnn_model.state_dict().keys()))

if __name__ == '__main__':
    import_old_model_code()