"""Cached skill: NSE of (target, variable, sorted input set, model, fold).
A target's skill depends only on its own inputs, so the cache is keyed on exactly that."""
import numpy as np
from .models import build_X, fit_predict, nse


class Skill:
    def __init__(self, d, model="RidgeFast"):
        self.d = d
        self.model = model
        self.cache = {}
        self.hits = self.misses = 0

    def target_arrays(self, t, v):
        return self.d.mod_arr[v][t], self.d.raw_arr[v][t]

    def predict(self, t, v, inputs, tr, ev, model=None):
        """Predictions on original scale for rows ev with an observed target; returns (obs, pred, idx)."""
        d = self.d
        ym, yr = self.target_arrays(t, v)
        X = build_X(d, list(inputs), v)
        evo = ev & np.isfinite(yr)
        p = fit_predict(model or self.model, X, ym, tr, evo)
        if v == "Q":
            p = np.clip(np.expm1(p), 0, None)
        return yr[evo], p, np.flatnonzero(evo)

    def nse(self, t, v, inputs, fold):
        key = (t, v, tuple(sorted(inputs)), fold)
        r = self.cache.get(key)
        if r is not None:
            self.hits += 1
            return r
        self.misses += 1
        tr, ev = self.d.fold_masks(fold)
        o, p, _ = self.predict(t, v, key[2], tr, ev)
        r = nse(o, p) if len(o) > 30 and np.isfinite(p).all() else np.nan
        self.cache[key] = r
        return r
