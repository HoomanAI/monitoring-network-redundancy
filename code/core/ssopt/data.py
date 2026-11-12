"""Data loading with leakage-by-construction.

Everything before the final test step loads the series TRUNCATED at 2016-12-31, so
test-period values are not even in memory (`Data.load(include_test=False)`).
Only scripts/09_final_test.py calls `Data.load(include_test=True)`.
"""
import hashlib
import numpy as np
import pandas as pd
from . import config as C


def verify_manifest():
    """SHA-256 + size check of every file listed in the package manifest."""
    man = pd.read_csv(C.F_MANIFEST)
    bad = []
    for _, r in man.iterrows():
        f = C.PKG / r.relative_path
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != r.SHA256.lower() or f.stat().st_size != r.file_size_bytes:
            bad.append(r.relative_path)
    return len(man), bad


def load_meta():
    m = pd.read_csv(C.F_META, dtype={"station_number": str})
    return m


def load_links():
    return pd.read_csv(C.F_LINKS, dtype=str)[["target_station", "upstream_station"]]


class Data:
    def __init__(self, W, include_test):
        self.include_test = include_test
        self.meta = load_meta()
        self.links = load_links()
        self.stations = list(self.meta.station_number)
        self.role = dict(zip(self.meta.station_number, self.meta.station_role))
        self.targets = sorted(self.meta[self.meta.station_role.isin(
            ["TARGET_ONLY", "TARGET_AND_UPSTREAM"])].station_number)
        self.upstream_only = sorted(self.meta[self.meta.station_role == "UPSTREAM_ONLY"].station_number)
        self.index = W.index
        self.T = len(W)
        self.raw = {v: W[[f"{v}_{s}" for s in self.stations]].set_axis(self.stations, axis=1)
                    for v in ("H", "Q")}
        # modelling scale: H raw, Q -> log1p
        self.mod = {"H": self.raw["H"], "Q": np.log1p(self.raw["Q"].clip(lower=0))}
        self.mod_arr = {v: {s: self.mod[v][s].to_numpy(float) for s in self.stations} for v in self.mod}
        self.raw_arr = {v: {s: self.raw[v][s].to_numpy(float) for s in self.stations} for v in self.raw}
        # lag blocks (T,4) per station/variable
        self.lagblk = {}
        for v in ("H", "Q"):
            for s in self.stations:
                x = self.mod_arr[v][s]
                B = np.full((self.T, len(C.LAGS)), np.nan)
                for j, l in enumerate(C.LAGS):
                    B[l:, j] = x[: self.T - l] if l else x
                self.lagblk[(s, v)] = B
        doy = self.index.dayofyear.to_numpy()
        self.seas = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
        self.yr = self.index.year.to_numpy()

    @classmethod
    def load(cls, include_test=False):
        W = pd.read_csv(C.F_WIDE, parse_dates=["date"]).set_index("date")
        W = W.apply(pd.to_numeric, errors="coerce")
        if not include_test:
            W = W.loc[: C.DEV[1]]
            assert W.index.max() <= pd.Timestamp(C.DEV[1]), "test-period rows present before final test"
        return cls(W, include_test)

    # ---- masks -------------------------------------------------------------
    def period(self, rng):
        lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
        m = (self.index >= lo) & (self.index <= hi)
        if not self.include_test:
            assert hi <= pd.Timestamp(C.DEV[1]), "LEAKAGE: test dates requested before final test step"
        return np.asarray(m)

    def fold_masks(self, k):
        """Blocked CV fold k in {0,1,2}: (train mask, eval mask) inside 2005-2016.
        Fold 2 is exactly the standard train(2005-2012)/validation(2013-2016) split."""
        a, b = C.CV_FOLDS[k]
        ev = (self.yr >= a) & (self.yr <= b)
        tr = (self.yr >= 2005) & (self.yr <= 2016) & ~ev
        if k == 2:
            tr = (self.yr >= 2005) & (self.yr <= 2012)
        return tr, ev

    def split_masks(self, name):
        if name == "val":      # fit 2005-2012, score 2013-2016
            return self.fold_masks(2)
        if name == "test":     # fit 2005-2016, score 2017-2020
            assert self.include_test
            return self.period(C.DEV), self.period(C.TEST)
        raise KeyError(name)
