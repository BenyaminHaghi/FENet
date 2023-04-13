# most of the data processing logic was copied from FENet_Training.py

from pickle import UnpicklingError
import torch
from torch.utils.data import TensorDataset
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from cachetools import cached
from cachetools.keys import hashkey

from collections import defaultdict
from functools import cache
from multiprocessing import Pool
from multiprocessing import set_start_method #https://pythonspeed.com/articles/python-multiprocessing/
from multiprocessing import get_context

from os import getenv
from os.path import join as path_join

#data
from configs import DAY_SPLITS
from configs import MAX_POOL_WORKERS
from configs import USE_MULTITHREADED_DATA_LOADING
from configs import THREAD_CONTEXT

# data getters

def target_normalization_centralization(target):
    # normalize and center the targets
    target = target - target.min(axis=0)
    for j in range(target.shape[1]):
        if target[:,j].max(axis=0) != 0:
            target[:,j] = 2*(target[:,j]/target[:,j].max(axis=0) - 0.5)
    return target

def load_hdf5_data(fname, verbose=False):
    if verbose: print(f"loading data from '{fname}'...")
    matlab_data = h5py.File(fname, 'r')
    neural_cell = matlab_data['neural_cell'][()]
    targets = np.transpose(matlab_data['targets'][:][:]).astype(np.float32)
    targets = target_normalization_centralization(targets)
    R2 = np.transpose(matlab_data['CVR2'][:][:])
    return matlab_data, neural_cell, targets, R2

def standard_scalar_normalize(ndarray):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(copy=False)
    arr = sc.fit_transform(ndarray)
    return arr

def filter_sort_channels_by_R2(bb, R2s, min_R2=0, n_filtered_channels=40):
    """
    filters data by each channel by two parameters. The first is minimum R2
    and the second is the number of filtered channels. Number of filtered
    channels find the minimum r2 which keeps the top n channels. If the min_R2
    constraint is stricter than the n_channels constraint, only the channels with
    an R2 larger than min_R2 will be kept. Othewise, if the top n channels constraint
    is stricter, those channels will be taken.
    """
    if(n_filtered_channels != None or min_R2 != None):
        #find the max r2 for each channel betwen x and y
        stream_len, n_channels, n_samples = bb.shape
        max_xy_R2 = np.where(R2s[:,0]>R2s[:,1], R2s[:,0], R2s[:,1])
    else:
        #if there are no constraints, exit out of the function
        return bb

    if(n_filtered_channels != None):
        #Return the R2 of the channel with the Nth best channel
        nth_R2 = np.sort(max_xy_R2)[-n_filtered_channels]
    else:
        #If there is no n_filtered_channels constraint, set the nth_r2 constraint
        #to the min_R2 constraint so it doesn't affect result
        nth_R2 = min_R2
    
    if(min_R2 != None):
        #Choose the tightest R2 constraint
        min_R2 = max(nth_R2, min_R2)
        #Develope a channel mask to be used to filter bb data
        channel_mask = max_xy_R2 >= min_R2
    else: 
        channel_mask = [1] * len(max_xy_R2)

    return bb[ :, channel_mask, :]

def make_augmented_data_from_file(fname):
    """
    Return lists of broadband data and targets, split by session, augmented by
    selecting (using FENet_Training.Channel_Selection_FixedNum_3D) the top
    `n_filtered_channels` channels by R^2, with minimum R^2 `min_R2`,
    then normalized using StandardScalar.
    """

    matlab_data, neural_cell, targets, R2 = load_hdf5_data(fname)
    bb_data = []

    for recording_session_idx in range(neural_cell.shape[1]):
        session_data = np.transpose(np.asarray(matlab_data[neural_cell[0][recording_session_idx]]))
        n_segments, n_samples, n_channels = session_data.shape
        session_data = session_data.reshape(n_segments * n_samples, n_channels)  # collapse first two dims, to prep normalize each channel vector over (n_samples, aka time)
        session_data = standard_scalar_normalize(session_data)
        session_data = session_data.reshape(n_segments, n_samples, n_channels)   # after normalization, change the shape back
        session_data = session_data.transpose(0, 2, 1)    # transpose into (n_segments, n_channels, n_samples)
        bb_data.append(session_data)

    # use stored session_lengths to partition the labels list
    tot_seen_len = 0    # the start index of the next block
    labels_list = []    # list to store the partitioned labels
    for sess in bb_data:
        sess_len = sess.shape[0]    # get the length of the session
        labels_list.append(targets[tot_seen_len : tot_seen_len+sess_len])   # remember the corresponding slice of targets
        tot_seen_len += sess_len

    return bb_data, labels_list, R2

@cached(cache={}, key=lambda fname, creation_callback, verbose=False: hashkey(fname))
def pickle_memoize(fname, creation_callback, verbose=False):
    """
    Try to read data from the pickle at `fname`, and save the output of
    `creation_callback` to `fname` as a pickle if `fname` doesn't exist.
    """
    from pickle import load, dump
    if verbose: print(f"looking for pickle file '{fname}'...")
    try:
        with open(fname, 'rb') as rf:
            if verbose: print(f"    found pickle file '{fname}'! :)) loading it...")
            return load(rf)
    except (FileNotFoundError, UnpicklingError):
        if verbose: print(f"    did not find pickle file '{fname}' or it was corrupted :( making it...")
    # except UnpicklingError:
    #     if verbose: print(f"    pickle file was corrupted! remaking...")
        got = creation_callback()
        try:
            with open(fname, 'wb') as wf:
                dump(got, wf)
            if verbose: print(f"    successfully made pickle file '{fname}'! :)")
        except TypeError as err:
            from sys import stderr
            if verbose: print("couldn't pickle the object! :(", err, file=stderr)
        return got

def make_data_from_day(data_dir, day_name, min_R2=0, n_filtered_channels=40, channel_mask=None, pbar=None):
    """
    get the broadband data and labels for the given day, using channel_mask and falling back to R2 filtering
    """
    if pbar is not None: pbar.set_description(f"loading data day {day_name}")
    else: print("making data for day", day_name)
    data_fname = path_join(data_dir, f"FennData_{day_name}_Reshaped_30ms_cell_new.mat")
    pickle_fname = data_fname + f".pkl"

    bb_list, targets_list, R2_by_channel = pickle_memoize(pickle_fname,
            creation_callback=lambda: make_augmented_data_from_file(data_fname))

    if pbar is not None: pbar.set_description(f"loading data day {day_name} (filtering channels...)")
    else: print("filtering channels for day", day_name)

    # filter out bad channels, either by R2 or mask
    if channel_mask is None:
        bb_list = [ filter_sort_channels_by_R2(bb, R2_by_channel, min_R2, n_filtered_channels) for bb in bb_list ]
    else:
        bb_list = [ bb[:, :, channel_mask] for bb in bb_list ]

    if pbar is not None: pbar.set_description(f"loading data day {day_name} (torchifying...)")
    else: print("torchifying channels for day", day_name)
    torched_bb = [torch.from_numpy(x).to(torch.float32) for x in bb_list]
    torched_targets = [torch.from_numpy(x).to(torch.float32) for x in targets_list]
    del bb_list
    del targets_list

    if pbar is not None: pbar.update(1)
    return list(zip(torched_bb, torched_targets))

def make_total_training_data(data_dir, min_R2=0, n_filtered_channels=40, days=None, splits=None, channel_mask={}, load_test_dl=True):
    #channel mask expects a dict with the key as the day lablel or index and the value is an array of bools the total
    #lenght of the number of channels with each value indicating if the channel is enabled or not
    #Note, DO NOT have duplicate days in different sets. you shouldn't anyway
    day_splits = DAY_SPLITS
    if(not load_test_dl):
        del(day_splits['test'])

    days_to_get = [ day
        for split, split_v in day_splits.items()    if splits is None or split in splits
        for day in split_v                          if days   is None or day   in days ]
    #remove duplicates in the days set so data isn't retrieved twice. I am doing it this way
    #instead of fixing it above because I do not know how the above code is working.
    #days_to_get = [*set(days_to_get)]

    channel_mask_for_day = lambda rolling_day_id, day_label: (channel_mask[day_label] if day_label in channel_mask else ( channel_mask[str(rolling_day_id)] if str(rolling_day_id) in channel_mask else None ))

    make_data_args = [ (data_dir, day_label, min_R2, n_filtered_channels, channel_mask_for_day(i, day_label), None)
        for i, day_label in enumerate(days_to_get) ]

    print(days_to_get)

    #len(days_to_get)
    if(USE_MULTITHREADED_DATA_LOADING):
        with get_context(THREAD_CONTEXT).Pool(min(MAX_POOL_WORKERS, len(make_data_args))) as pool:
            data_by_day = dict(zip(days_to_get, pool.starmap(make_data_from_day, make_data_args)))
    else:
        data_by_day = dict(zip(days_to_get, [ make_data_from_day(*args) for args in make_data_args]))

    for k, v in data_by_day.items():
        print(k, len(v), v[0][0].shape)

    split_data = defaultdict(list)

    for split, day_labels in day_splits.items():
        if splits is not None and split not in splits: continue
        for day_label in day_labels:
            if days is not None and day_label not in days: continue
            split_data[split] += data_by_day[day_label]

    # split_data = { split: sum(data_by_day[day_name] for day_name in split_days if days is None or day_name in days)
    #         for split, split_days in day_splits.items() if splits is None or split in splits }
    if(load_test_dl):
        return split_data['train'], split_data['dev'], split_data['test']
    else:
        return split_data['train'], split_data['dev']

if __name__ == '__main__':
    set_start_method("spawn")
    train, dev, test = make_total_training_data('/shared/BB_Data/')
    print(train)
    print(dev)
    print(test)

