import torch
from functools import cache
from scipy.signal import gaussian
from torch.nn import functional as F
import networkx as nx
import numpy as np

@cache
def get_smoothing_kernel(kernel_width):
    return torch.as_tensor(gaussian(20 * kernel_width, kernel_width)).float().cuda()

def fast_gaussian_filter(ar, kernel_width=50):
    return F.conv1d(ar[:, None], get_smoothing_kernel(kernel_width)[None, None], padding='same')[:, 0]

def compute_2d_correlations(br, lag_window):
    torch.cuda.empty_cache()
    var = torch.einsum('it,it->i', br, br)

    inv_var = 1.0 / torch.maximum(var, torch.as_tensor(1e-8)).sqrt()

    corrs = []
    N = br.shape[0]
    SLICES = min(100, N)
    for slice in range(SLICES):
        torch.cuda.empty_cache()
        br_slice = br[N * slice // SLICES:N * (slice + 1) // SLICES]
        inv_var_slice = inv_var[N * slice // SLICES:N * (slice + 1) // SLICES]
        shifts = torch.stack([torch.roll(br_slice, shift, 1) for shift in range(-lag_window, lag_window + 1)])
        corr = torch.einsum('it,sjt,i,j->ijs', br, shifts, inv_var, inv_var_slice)
        corrs.append(corr)
    corr = torch.cat(corrs, 1)
    corr, corr_idx = torch.max(corr, -1)
    corr_idx -= lag_window

    return corr.cpu().numpy(), corr_idx.cpu().numpy()

def compute_3d_correlations(br, corr_idx):
    from scipy.signal import correlate
    from tqdm import tqdm
    import itertools
    pairs = [[] for _ in range(br.shape[0])]
    for i in range(br.shape[0]):
        for j in range(br.shape[0]):
            pairs[i].append(br[i] * np.roll(br[j], corr_idx[i, j]))

    varp = np.einsum('ijt,ijt->ij', pairs, pairs)
    # Three-way correlation calculation
    corr3 = np.zeros((br.shape[0], br.shape[0], br.shape[0]))
    for i, j in tqdm(list(itertools.product(range(br.shape[0]), range(br.shape[0])))):
        for k in range(br.shape[0]):
            corr3[i, j, k] = np.max(correlate(pairs[i][j], br[k])) / np.maximum(np.sqrt(varp[i, j] * var[k]), 1e-8)

def graph_from_correlations(corr, X):
    THRESHOLD = np.quantile(corr[np.triu_indices_from(corr, 1)], X)
    a = np.where(corr > THRESHOLD, corr - np.eye(corr.shape[0]), 0.0)
    C = np.einsum('ij,ik,jk->i', a, a, a) / np.maximum(np.einsum('ij,ik->i', a, a), 1.0)
    
    # Constructing the graph
    G = nx.Graph()
    G.add_nodes_from(range(corr.shape[0]))
    for i in range(corr.shape[0]):
        for j in range(corr.shape[0]):
            if i != j and corr[i, j] > THRESHOLD:
                G.add_edge(i, j)
    return G, C