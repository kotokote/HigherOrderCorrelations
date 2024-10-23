import numpy as np
import h5py
import networkx as nx
import torch
import pickle
from tqdm import tqdm
from scipy.io import loadmat
from correlations import fast_gaussian_filter, compute_2d_correlations, compute_3d_correlations, graph_from_correlations
from shuffling import shuffle
from fast_hole_analysis import fast_hole_analysis, connected_components_analysis, connected_components_analysis_range
from ripser import ripser
from pathlib import Path
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
run_ripserer = Main.include("dataprocessing.jl")

ALL_FILES = [
    # "2950_spike_mat_or_rand", #26
    # "2953_spike_mat_or_rand", #Tal paper
    # "2957_spike_mat_or_rand",
    # "5116_spike_mat_or_rand",
    "2953_t_spk_mat_sorted",
    "2957_t_spk_mat_sorted",
    "5116_t_spk_mat_sorted",
    # "M1S1_t_spk_mat_sorted",
    # "M1S2_t_spk_mat_sorted",
    # "M2S1_t_spk_mat_sorted",
    # "M2S2_t_spk_mat_sorted",
    # "M3S1_t_spk_mat_sorted",
    # "M3S2_t_spk_mat_sorted",
    "O5_t_spk_mat_sorted",
    "O6_t_spk_mat_sorted",
    # "UCSC_mouse_Pasca_23129",
    # "UCSC_mouse_Pasca_23149",
    # "UCSC_mouse_Pasca_23179",
    # "result_32",
    # "result_33",
    # "UCSC_mouse/240410/23129/t_spk_mat_sorted",
    # "UCSC_mouse/240410/23179/t_spk_mat_sorted",
    # "UCSC_mouse/240517/23124/t_spk_mat_sorted",
    # "UCSC_mouse/240521/23137/t_spk_mat_sorted",
    # "UCSC_mouse/240524/23150/t_spk_mat_sorted",
    # "UCSC_mouse/240524/23178/t_spk_mat_sorted",
    # "UCSC_mouse/240611/23141/t_spk_mat_sorted",
    # "UCSC_mouse/240618/23150/t_spk_mat_sorted",
    # "UCSC_mouse/240628/23120/t_spk_mat_sorted",
    # "ABL90minpostrec-gaba10uL+rcpp20ul+4uLnmqxper2ml-018",
    # "PREABL90minpostrec_gaba10uL+rcpp20ul+4uLnmqxper2ml_017_Cycle00001",
    # "CTRL_5_converted",
    # "stim_6again_converted",
    # "orgA-5run-predrug",
    # "ctrl_6",
    # "ctrl_6001",
    # "ctrl_6002",
    # "stim_6again",
    # "stim_6again002",
    # "stim_6again003",
    # "OrgB-09032024-1207-RCPP-predrug-035",
    # "OrgB-TSeries-09032024-1207-rcpp5ul-inRA-036",
    # "OrgA-3run-TSeries10min4uLRCPP-08282024-1010-020",
    # "OrgA-3run-TSeries10min20uLRCPP-08282024-1010-023",
    # "OrgA-3run-TSeries10minpredrug-08282024-1010-019",
    # "OrgB-TSeries-09032024-1207-rcppin10min-037",
    # "OrgB-TSeries-09032024-1207-rcppin20min-039",
    # "OrgB-TSeries-09032024-1207-rcppin30min-040",
    # "OrgA-3run-RA-TSeries10min20uLRCPP-08282024-1010-021",
    # "OrgA-TSeries-08222024-1448-top3min-prenmbqx-022",
    # "OrgA-TSeries-08222024-1448-topmbqx-120ul-gabaablin-032",
    # "Org1-TSeries-05152024-1414-predrug-025",
    # "Org1-TSeries-05152024-1414-alldrugsin-plane3top-5 min_later-028",
    # "Org2-TSeries-05152024-1414-alldrugsin-top-predrug-032",
    # "Org2-TSeries-05152024-1414-alldrugsin-top-drugsinrightafter10min-035",
    # "Org3-TSeries-05152024-1414-org3-039-predrug",
    # "Org3-TSeries-05152024-1414-org3-rightafterdru20min-044",
    # "OrgA-2run-TSeries-08262024-1011-ABL90minpostrec-gaba10uL+rcpp20ul+4uLnmqxper2ml-016",
    # "OrgA-2run-TSeries-08262024-1011-gaba10uL+rcpp20ul+4uLnmqxper2ml-011",
    # "OrgA-2run-TSeries-08262024-1011-gaba10uLper2ml-008",
    # "OrgA-2run-TSeries-08262024-1011-predrug-10min-007",
    # "OrgA-2run-TSeries-08262024-1011rec-gaba10uL+rcpp20ul+4uLnmqxper2ml-013",
    # "orgA-4run-gabaseries-TSeries-08292024-1113-predrug-024",
    # "OrgA-4run-TSeries-08292024-1113-20uLgaba10min-026",
    # "OrgA-4run-TSeries-08292024-1113-20uLgaba20min-027",
    # "OrgA-4run-TSeries-08292024-1113-20uLgabarafter-025",
    # "OrgA-5run-TSeries-08302024-1020-20uLgaba15min-030",
    # "OrgA-5run-TSeries-08302024-1020-20uLgaba80min-031",
    # "OrgA-5run-TSeries-08302024-1020-20uLgabarafter-029",
    # "OrgA-5run-TSeries-08302024-1020-predrug-028",
    # "Org3-all-drugs-in-TSeries-05152024-1414-org3-039-predrug",
    # "Org3-all-drugs-in-TSeries-05152024-1414-org3-rightafterdru20min-044",
    # "Org3-all-drugs-in-TSeries-05152024-1414-org3-rightafterdru30min-046",
    # "kilosort/000048",
    # "kilosort/000049",
    # "kilosort/000050",
    # "TSeries-10012024-1105-NBQX10min2-051",
    # "TSeries-10012024-1105-NBQX10min-050",
    # "TSeries-10012024-1105-NBQX-predrug10min-049",
    # "TSeries-10012024-11052-NMQX-predrug10minB-054",
    # "TSeries-10012024-11052NBQX10min2B-056",
    # "TSeries-10012024-11052NBQX10minB-055",
]

# import pathlib
# for dir in ["kilosort"]:
#     for path in pathlib.Path(dir).glob('*.mat'):
#         ALL_FILES.append(str(path).replace(".mat", ""))
# print(ALL_FILES)

def process(fn, lag_window, use_shuffle):
    out_fn = f"processed/{fn}_lag_window_{lag_window}{'_shuffle' if use_shuffle else ''}.pkl"
    Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
    # if Path(out_fn).exists():
    #     return
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
        if "result" in fn:
            br = torch.tensor(ar[:, i * slice_len : (i + 1) * slice_len]).cuda()
        else:
            br = fast_gaussian_filter(torch.tensor(ar[:, i * slice_len : (i + 1) * slice_len]).cuda(), 50.0)
        corr, corr_idx_data = compute_2d_correlations(br, lag_window)
        G, C = graph_from_correlations(corr, 0.5)
        slices.append({
            "corr": corr,
            "corr_idx_data": corr_idx_data,
            "G": G,
            "C": C,
        })
    
    if "result" in fn:
        br = torch.tensor(ar).cuda()
    else:
        br = fast_gaussian_filter(torch.tensor(ar).cuda(), 50.0)
    corr, corr_idx_data = compute_2d_correlations(br, lag_window)
    # corr3 = compute_3d_correlations(br.cpu().numpy(), corr_idx_data)
    # print("here", corr3.shape)

    G, C = graph_from_correlations(corr, 0.5)

    n = br.shape[0]
    cnt = 6 if n <= 150 else 3
    fh = fast_hole_analysis(corr, cnt)
    fh = [[(b, d) for c, b, d in fh if c == i] for i in range(cnt + 1)]
    print([len(a) for a in fh])
    cc_counts = connected_components_analysis(corr)
    print(cc_counts)
    
    # assert np.max(np.abs(corr - corr.T)) < 1e-5
    corr = (corr + corr.T) / 2
    ripserer = run_ripserer(1.0 - corr, maxdim=3)
    r = ripser(1.0 - corr, maxdim=3 if n <= 150 else 2, distance_matrix=True)
    # r = ripser(1.0 - corr, maxdim=2, distance_matrix=True, do_cocycles=True)
    print("barcode sizes", [len(x) for x in r["dgms"]], [len(x) for x in ripserer])
    range_analysis = {}
    for r_min, r_max in [(0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        corr_narrow = np.where((corr >= r_min) & (corr < r_max), 0.0, 1.0)
        r_narrow = ripser(corr_narrow, maxdim=3 if n <= 150 else 2, distance_matrix=True)
        print(r_min, r_max, [len(x) for x in r_narrow["dgms"]])
        cc_counts = connected_components_analysis_range(corr, r_min, r_max)
        print(r_min, r_max, cc_counts)
        range_analysis[(r_min, r_max)] = {
            "ripser": r_narrow,
            "cc_counts": cc_counts,
        }
    with open(out_fn, "wb") as f:
        pickle.dump({
            "all": {
                "corr": corr,
                "corr_idx_data": corr_idx_data,
                "G": G,
                "C": C,
                "ripser": r,
                "ripserer": ripserer,
                "holes": fh,
                "cc_counts": cc_counts,
                "range_analysis": range_analysis,
            },
            "slices": slices,
        }, f)
    

for lag_window in [0, 10, 20, 100, 200]:
    for file_name in ALL_FILES:
        for use_shuffle in [False, True]:
            if use_shuffle and ("result" in file_name):
                continue
            process(file_name, lag_window, use_shuffle)
