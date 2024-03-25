import numpy as np
import h5py
import networkx as nx
import torch
import pickle
from numba import njit, prange
from tqdm import tqdm
from scipy.io import loadmat
from correlations import fast_gaussian_filter, compute_2d_correlations
from shuffling import shuffle
from fast_hole_analysis import fast_hole_analysis
from ripser import ripser

ALL_FILES = [
    "2950_spike_mat_or_rand", #26
    "2953_spike_mat_or_rand", #Tal paper
    "2957_spike_mat_or_rand",
    "5116_spike_mat_or_rand",
]

def process(fn, lag_window, use_shuffle):
    try:
        f = loadmat(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).T.astype(np.float32)
    except:
        f = h5py.File(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).astype(np.float32)
    if use_shuffle:
        ar = shuffle(ar)
    br = fast_gaussian_filter(torch.tensor(ar).cuda(), 50.0)
    corr, _ = compute_2d_correlations(br, lag_window)

    fh = fast_hole_analysis(corr, 6)
    fh = [[(b, d) for c, b, d in fh if c == i] for i in range(7)]
    print([len(a) for a in fh])
    
    r = ripser(1.0 - corr, maxdim=3, distance_matrix=True)
    print([len(x) for x in r["dgms"]])
    with open(f"processed/{fn}_lag_window_{lag_window}{'_shuffle' if use_shuffle else ''}.pkl", "wb") as f:
        pickle.dump({
            "corr": corr,
            "ripser": r,
            "holes": fh,
        }, f)
    

for lag_window in [0, 10, 20]:
    for file_name in ALL_FILES:
        for use_shuffle in [False, True]:
            process(file_name, lag_window, use_shuffle)
