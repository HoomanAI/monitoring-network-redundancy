"""Discrete entropy / MI on log-discharge (10 equal-frequency bins, edges from TRAIN only),
Chow-Liu joint-entropy approximation, total correlation, mRMR-style score."""
import numpy as np
from . import config as C

K = C.N_BINS


def discretise(d, train_mask):
    """-> int array (T, n_stations), -1 where missing. Edges: equal-frequency quantiles of train log1p(Q)."""
    Z = np.full((d.T, len(d.stations)), -1, dtype=np.int16)
    for j, s in enumerate(d.stations):
        x = d.mod_arr["Q"][s]
        tr = x[train_mask & np.isfinite(x)]
        edges = np.unique(np.quantile(tr, np.linspace(0, 1, K + 1)[1:-1]))
        ok = np.isfinite(x)
        Z[ok, j] = np.digitize(x[ok], edges)
    return Z


def _ent(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def marginal_entropy(Z, train_mask):
    H = np.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        z = Z[train_mask, j]; z = z[z >= 0]
        H[j] = _ent(np.bincount(z, minlength=K))
    return H


def mi_matrix(Z, train_mask):
    """Plug-in pairwise MI (bits) over days where both stations are observed in TRAIN."""
    Zt = Z[train_mask]; n = Z.shape[1]
    M = np.zeros((n, n)); N = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            ok = (Zt[:, i] >= 0) & (Zt[:, j] >= 0)
            N[i, j] = N[j, i] = ok.sum()
            if ok.sum() < C.MIN_OVERLAP:
                continue
            a, b = Zt[ok, i], Zt[ok, j]
            c = np.bincount(a * K + b, minlength=K * K).astype(float)
            pxy = c / c.sum(); px = pxy.reshape(K, K).sum(1); py = pxy.reshape(K, K).sum(0)
            nz = pxy > 0
            mi = (pxy[nz] * np.log2(pxy[nz] / np.outer(px, py).reshape(-1)[nz])).sum()
            M[i, j] = M[j, i] = max(mi, 0.0)
    return M, N


def cond_mi(Z, train_mask, x, y, zs):
    """Discrete I(X;Y|Z) with Z a list of conditioning columns (use <=1-2 for sample-size reasons)."""
    Zt = Z[train_mask]
    cols = [x, y] + list(zs)
    ok = np.all(Zt[:, cols] >= 0, axis=1)
    if ok.sum() < C.MIN_OVERLAP:
        return np.nan
    a, b = Zt[ok, x], Zt[ok, y]
    zc = np.zeros(ok.sum(), int)
    for z in zs:
        zc = zc * K + Zt[ok, z]
    _, zc = np.unique(zc, return_inverse=True)
    nz = zc.max() + 1
    def H(*arrs):
        key = np.zeros(len(arrs[0]), np.int64)
        for a_ in arrs:
            key = key * (K * 1000) + a_
        _, cnt = np.unique(key, return_counts=True)
        return _ent(cnt.astype(float))
    return max(H(a, zc) + H(b, zc) - H(a, b, zc) - H(zc), 0.0)


def gauss_cmi(d, train_mask, x, y, zs, var="Q"):
    """Gaussian-copula (rank-normal) conditional MI in bits: -0.5*log2(1 - rho_partial^2)."""
    from scipy.stats import rankdata, norm
    names = [x, y] + list(zs)
    A = np.column_stack([d.mod_arr[var][s][train_mask] for s in names])
    ok = np.isfinite(A).all(1)
    if ok.sum() < C.MIN_OVERLAP:
        return np.nan
    A = A[ok]
    A = np.column_stack([norm.ppf((rankdata(A[:, k]) - 0.5) / len(A)) for k in range(A.shape[1])])
    P = np.linalg.pinv(np.cov(A.T))
    rho = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    return float(-0.5 * np.log2(max(1 - rho ** 2, 1e-12)))


def chow_liu(M, idx):
    """Max-spanning-tree (Prim) over stations `idx`; returns sum of MI on tree edges."""
    idx = np.asarray(idx)
    n = len(idx)
    if n <= 1:
        return 0.0
    S = M[np.ix_(idx, idx)]
    in_t = np.zeros(n, bool); in_t[0] = True
    best = S[0].copy(); tot = 0.0
    best[0] = -1
    for _ in range(n - 1):
        b = np.where(in_t, -1.0, best)
        j = int(np.argmax(b)); tot += b[j]; in_t[j] = True
        best = np.maximum(best, S[j])
    return tot


def joint_entropy_cl(Hm, M, idx):
    """Chow-Liu approximation: sum H_i - sum MI(MST)."""
    idx = np.asarray(idx)
    return float(Hm[idx].sum() - chow_liu(M, idx))


def total_correlation(Hm, M, idx):
    return float(chow_liu(M, idx))


def mrmr_score(pools_inputs, M, Hm, active_mask, tidx, sidx):
    """mean over targets of [ mean relevance MI(x,t) - mean pairwise redundancy among x in S_t ],
    S_t = pool_t ∩ active. pools_inputs: dict t -> list of station names."""
    vals = []
    for t, ins in pools_inputs.items():
        S = [sidx[s] for s in ins if active_mask[sidx[s]]]
        if not S:
            vals.append(0.0); continue
        rel = np.mean([M[i, tidx[t]] for i in S])
        red = 0.0
        if len(S) > 1:
            sub = M[np.ix_(S, S)]; red = (sub.sum() / (len(S) * (len(S) - 1)))
        vals.append(rel - red)
    return float(np.mean(vals))
