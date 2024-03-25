import torch
from functools import cache
from scipy.signal import gaussian
from torch.nn import functional as F

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
    SLICES = 20
    N = br.shape[0]
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
