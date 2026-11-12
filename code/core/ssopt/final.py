"""Multi-model evaluation workers (validation re-check, blocked-CV check, final test)."""
import numpy as np
import pandas as pd
from . import config as C
from .data import Data
from .models import build_X, fit_predict, nse

_D = None
_CFG = None
_MODE = None


def init(include_test, cfgs, mode):
    global _D, _CFG, _MODE
    _D = Data.load(include_test=include_test)
    _CFG = cfgs; _MODE = mode


def masks(mode):
    d = _D
    if mode == "val":
        return d.fold_masks(2)
    if mode == "cv0":
        return d.fold_masks(0)
    if mode == "cv1":
        return d.fold_masks(1)
    if mode == "test":
        return d.split_masks("test")
    raise KeyError(mode)


def flood_metrics(o, p, yrs, thr):
    hi = o > thr
    out = dict(hf_rmse=float(np.sqrt(np.mean((o[hi] - p[hi]) ** 2))) if hi.sum() > 5 else np.nan,
               hf_hit=float(np.mean(p[hi] > thr)) if hi.sum() > 5 else np.nan, n_hf=int(hi.sum()))
    amax, vol = [], []
    for y in np.unique(yrs):
        m = yrs == y
        if m.sum() < 300:
            continue
        amax.append(abs(p[m].max() - o[m].max()) / o[m].max() * 100)
        vol.append(abs(p[m].sum() - o[m].sum()) / o[m].sum() * 100)
    out["annmax_mae_pct"] = float(np.mean(amax)) if amax else np.nan
    out["annvol_mae_pct"] = float(np.mean(vol)) if vol else np.nan
    return out


def job(args):
    t, v, model = args
    d = _D
    tr, ev = masks(_MODE)
    ym = d.mod_arr[v][t]; yr = d.raw_arr[v][t]
    evo = ev & np.isfinite(yr)
    fitmask = tr & np.isfinite(ym)
    thr = np.nanpercentile(yr[fitmask], 95) if v == "Q" else np.nan
    cache = {}; rows = []
    for cfg, tv in _CFG.items():
        ins = tuple(sorted(tv[(t, v)]))
        if ins not in cache:
            X = build_X(d, list(ins), v)
            p = fit_predict(model, X, ym, tr, evo)
            if v == "Q":
                p = np.clip(np.expm1(p), 0, None)
            o = yr[evo]
            r = dict(nse=nse(o, p), rmse=float(np.sqrt(np.mean((o - p) ** 2))), n_eval=int(evo.sum()))
            if v == "Q":
                r.update(flood_metrics(o, p, d.yr[evo], thr))
            cache[ins] = r
        rows.append(dict(config=cfg, target=t, var=v, model=model, n_inputs=len(ins), inputs=",".join(ins), **cache[ins]))
    return rows
