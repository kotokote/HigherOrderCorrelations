import numpy as np
import h5py
import networkx as nx
import torch
import pickle
from tqdm import tqdm
from scipy.io import loadmat
from correlations import fast_gaussian_filter, compute_2d_correlations
from shuffling import shuffle
from ripser import ripser

ALL_FILES = [
    "2950_spike_mat_or_rand", #26
    "2953_spike_mat_or_rand", #Tal paper
    "2957_spike_mat_or_rand",
    "5116_spike_mat_or_rand",
]

def process(fn, lag_window):
    try:
        f = loadmat(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).T.astype(np.float32)
    except:
        f = h5py.File(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).astype(np.float32)
    br = fast_gaussian_filter(torch.tensor(ar).cuda(), 50.0)
    corr, _ = compute_2d_correlations(br, lag_window)
    
    r = ripser(1.0 - corr, maxdim=3, distance_matrix=True)
    print([len(x) for x in r["dgms"]])
    with open(f"processed/{fn}_lag_window_{lag_window}.pkl", "wb") as f:
        pickle.dump({
            "corr": corr,
            "ripser": r,
        }, f)
    

for lag_window in [0, 10, 20]:
    for file_name in ALL_FILES:
        process(file_name, lag_window)
