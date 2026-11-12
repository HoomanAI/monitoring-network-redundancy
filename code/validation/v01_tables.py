# -*- coding: utf-8 -*-
"""Rebuild the subset-NSE fallback tables scored on each blocked CV window.

Fold f in {0,1,2} uses the project's Data.fold_masks(f): the model is fitted on the
complementary years inside 2005-2016 and every input subset is scored on the fold
window.  The candidate pool (members) is held fixed at the one already built, so the
only thing that changes between folds is the window the scores come from.
"""
import sys, pickle, time, os
from pathlib import Path
R = '/home/claude/ws/station_subset_optimization'
sys.path.insert(0, R); sys.path.insert(0, R + '/ress_v3'); sys.path.insert(0, R + '/ress_v4')
import numpy as np
import rv3
from ssopt.data import Data
from ssopt.models import build_X, ridge_fast

OUT = Path('/home/claude/work/val')
_D = None; _FOLD = None

def init(fold):
    global _D, _FOLD
    _D = Data.load(include_test=False); _FOLD = fold

def nse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - np.mean(y[m])) ** 2))

def job(args):
    t, v, members = args
    d = _D; tr, ev = d.fold_masks(_FOLD)
    ym = d.mod_arr[v][t]; yr = d.raw_arr[v][t]; evo = ev & np.isfinite(yr)
    n = len(members); out = np.full(2 ** n, np.nan, np.float32)
    for m in range(2 ** n):
        ins = [members[i] for i in range(n) if (m >> i) & 1]
        p = ridge_fast(build_X(d, ins, v), ym, tr, evo)
        if v == 'Q': p = np.clip(np.expm1(p), 0, None)
        out[m] = nse(yr[evo], p)
    return (t, v), out

def main(fold):
    import multiprocessing as mp
    Z = pickle.load(open(rv3.CACHE / 'fallback_tables.pkl', 'rb'))
    mem, dropped, cap = Z['members'], Z['dropped'], Z['cap']
    jobs = sorted([(t, v, mem[(t, v)]['members']) for (t, v) in mem], key=lambda j: -len(j[2]))
    t0 = time.time(); T = {}
    with mp.Pool(2, initializer=init, initargs=(fold,)) as pool:
        for i, (k, arr) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            T[k] = arr
            if i % 20 == 0: print('fold %d  %3d/%d  %.0fs' % (fold, i + 1, len(jobs), time.time() - t0), flush=True)
    pickle.dump(dict(tables=T, members=mem, dropped=dropped, cap=cap, fold=fold),
                open(OUT / ('tables_f%d.pkl' % fold), 'wb'))
    print('fold %d done in %.0fs' % (fold, time.time() - t0))

if __name__ == '__main__':
    main(int(sys.argv[1]))
