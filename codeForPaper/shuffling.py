import numpy as np

def swap(ar, idxs):
    idx0 = np.random.randint(len(idxs[0]))
    idx1 = np.random.randint(len(idxs[0]))
    i0, j0 = idxs[0][idx0], idxs[1][idx0]
    i1, j1 = idxs[0][idx1], idxs[1][idx1]
    if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
        return False
    ar[i0, j0] = ar[i1, j1] = 0.0
    ar[i0, j1] = ar[i1, j0] = 1.0
    idxs[0][idx0], idxs[1][idx0] = i0, j1
    idxs[0][idx1], idxs[1][idx1] = i1, j0
    return True

def shuffle(ar, n_iters=100000):
    ar = ar.copy()
    idxs = np.where(ar == 1.0)
    if len(idxs[0]) == 0:
        print("No edges to shuffle")
        return ar
    iters = 0
    cnt_swap = 0
    for _ in range(n_iters):
        iters += 1
        if swap(ar, idxs):
            cnt_swap += 1
    return ar