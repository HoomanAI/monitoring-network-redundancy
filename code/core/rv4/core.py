"""Multi-state reliability core on top of the v3 fallback tables (Ridge NSE for every subset of a capped pool).
State of a series on a day: G2 if NSE[inputs up] >= GOOD, G1 if >= OK, else G0.  Real joint outage states of 2005-2016 (missing value = down)."""
import pickle, itertools
import numpy as np, pandas as pd, networkx as nx
import rv4
from rv4 import config as K
from rv3 import common as cm
from de_reliability import Net
from f_resilience import episodes

class MSNet(Net):
    def __init__(self, designs_file="designs_ms_raw.pkl", extra=None):
        super().__init__()
        self.stations_all = list(self.d.stations)
        self.designs_loaded = {}
        if designs_file:
            D = pickle.load(open(rv4.CACHE / designs_file, "rb")); tag = f"{K.GOOD}_{K.OK}"
            for (t_, lev, pk), v in D.items():
                if t_ == tag: self.design[f"k{lev}_{pk}"] = {k: list(v["inputs"][k]) for k in self.keys}; self.designs_loaded[f"k{lev}_{pk}"] = v
        if extra: self.design.update(extra)
        self.nattain = {}

    # ---- states -------------------------------------------------------------------------------------------------------------------------
    def nse_matrix(self, design, U=None, series=None, force=None):
        U = self.U if U is None else U; ks = self.keys if series is None else series
        T = next(iter(U.values())).shape[0]; out = np.zeros((T, len(ks)), np.float32)
        for j, k in enumerate(ks):
            t, v = k; mm = self.mem[k]["members"]; ins = set(self.design[design][k]); mask = np.zeros(T, np.int64)
            for i, s in enumerate(mm):
                if s not in ins: continue
                u = U[v][:, self.stn[s]]
                if force is not None and force[0] == s: u = np.full_like(u, force[1])
                mask |= (u.astype(np.int64) << i)
            out[:, j] = self.T[k][mask]
        return out

    @staticmethod
    def to_state(nse, th):
        good, ok = th; return (nse >= ok).astype(np.int8) + (nse >= good).astype(np.int8)

    def full_nse(self, design, k):
        return float(self.T[k][self.maskof(k, self.design[design][k])])

    def attainable(self, design, th):
        """series whose all-up NSE reaches G1 for this design"""
        return [k for k in self.keys if self.full_nse(design, k) >= th[1]]

def score_of(states):
    return np.where(states == 2, K.SCORE[2], np.where(states == 1, K.SCORE[1], K.SCORE[0]))

def pmf(states):
    return np.stack([(states == g).mean(0) for g in (2, 1, 0)], axis=1)       # columns: P(G2), P(G1), P(G0)

def block_ci(x, rng, B=300, blocklen=30):
    """x: (T, n) array; CI of the mean over days and series of x, and of its per-day all() fraction, via 30-day block bootstrap."""
    T = x.shape[0]; blocks = np.array_split(np.arange(T), T // blocklen); m = np.zeros(B)
    for b in range(B):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]); m[b] = x[idx].mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def min_cut(net, design, k, th):
    ins = net.design[design][k]; n = len(ins); full = net.full_nse(design, k)
    if full < th[1]: return None, []
    for r in range(1, n + 1):
        for C_ in itertools.combinations(ins, r):
            if net.T[k][net.maskof(k, [x for x in ins if x not in C_])] < th[1]: return r, list(C_)
    return None, []

def perf_removed(net, design, removed, th, keys=None):
    keys = keys or net.keys; sc = []; ok = 0
    for k in keys:
        v = net.T[k][net.maskof(k, [s for s in net.design[design][k] if s not in removed])]; st = 2 * (v >= th[0]) + 0 if False else (int(v >= th[1]) + int(v >= th[0]))
        sc.append(K.SCORE[st]); ok += st >= 1
    return ok / len(keys), float(np.mean(sc))
