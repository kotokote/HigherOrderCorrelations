import numpy as np
import h5py
import networkx as nx
import torch
import pickle
from numba import njit, prange
from tqdm import tqdm
from scipy.io import loadmat
from correlations import fast_gaussian_filter, compute_2d_correlations, graph_from_correlations
from shuffling import shuffle
from fast_hole_analysis import fast_hole_analysis, connected_components_analysis
from ripser import ripser
from pathlib import Path

ALL_FILES = [
    # "2950_spike_mat_or_rand", #26
    # "2953_spike_mat_or_rand", #Tal paper
    # "2957_spike_mat_or_rand",
    # "5116_spike_mat_or_rand",
    "M1S1_t_spk_mat_sorted",
    "M1S2_t_spk_mat_sorted",
    "M2S1_t_spk_mat_sorted",
    "M2S2_t_spk_mat_sorted",
    "M3S1_t_spk_mat_sorted",
    "M3S2_t_spk_mat_sorted",
    "O5_t_spk_mat_sorted",
    "O6_t_spk_mat_sorted",
]

def process(fn, lag_window, use_shuffle):
    out_fn = f"processed/{fn}_lag_window_{lag_window}{'_shuffle' if use_shuffle else ''}.pkl"
    if Path(out_fn).exists():
        return
    try:
        f = loadmat(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).T.astype(np.float32)
    except:
        f = h5py.File(f"{fn}.mat")
        ar = np.array(f['t_spk_mat']).astype(np.float32)
    if use_shuffle:
        ar = shuffle(ar)

    slices = []
    slice_len = ar.shape[1] // 10
    print("time", ar.shape[1] / 60000.0, "n", ar.shape[0])
    # assert slice_len == 180000
    for i in range(10):
        br = fast_gaussian_filter(torch.tensor(ar[:, i * slice_len : (i + 1) * slice_len]).cuda(), 50.0)
        corr, corr_idx_data = compute_2d_correlations(br, lag_window)
        G, C = graph_from_correlations(corr, 0.5)
        slices.append({
            "corr": corr,
            "corr_idx_data": corr_idx_data,
            "G": G,
            "C": C,
        })
    
    br = fast_gaussian_filter(torch.tensor(ar).cuda(), 50.0)
    corr, corr_idx_data = compute_2d_correlations(br, lag_window)

    G, C = graph_from_correlations(corr, 0.5)

    n = br.shape[0]
    cnt = 6 if n <= 150 else 3
    fh = fast_hole_analysis(corr, cnt)
    fh = [[(b, d) for c, b, d in fh if c == i] for i in range(cnt + 1)]
    print([len(a) for a in fh])
    cc_counts = connected_components_analysis(corr)
    print(cc_counts)
    
    r = ripser(1.0 - corr, maxdim=3 if n <= 150 else 2, distance_matrix=True)
    print([len(x) for x in r["dgms"]])
    with open(out_fn, "wb") as f:
        pickle.dump({
            "all": {
                "corr": corr,
                "corr_idx_data": corr_idx_data,
                "G": G,
                "C": C,
                "ripser": r,
                "holes": fh,
                "cc_counts": cc_counts,
            },
            "slices": slices,
        }, f)
    

for lag_window in [0, 10, 20]:
    for file_name in ALL_FILES:
        for use_shuffle in [False, True]:
            process(file_name, lag_window, use_shuffle)
