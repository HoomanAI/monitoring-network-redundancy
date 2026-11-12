"""Four model families. `ridge_fast` is a numpy re-implementation of the sklearn pipeline
SimpleImputer(median, add_indicator=True) -> StandardScaler -> Ridge(alpha); it is used
inside the NSGA-II loop and verified against sklearn in scripts/00_verify_ridge.py."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from . import config as C

MODELS = ("Ridge", "KNN", "RF", "HGB")


def build_X(d, inputs, var):
    blocks = [d.lagblk[(s, var)] for s in inputs]
    return np.hstack(blocks + [d.seas]) if blocks else d.seas.copy()


def nse(o, p):
    o = np.asarray(o, float); p = np.asarray(p, float)
    sst = np.sum((o - o.mean()) ** 2)
    return np.nan if sst <= 0 else 1.0 - np.sum((o - p) ** 2) / sst


def ridge_fast(X, y, tr, ev, alpha=C.RIDGE_ALPHA):
    ok = tr & np.isfinite(y)
    Xt = X[ok]; yt = y[ok]
    if len(yt) < 50:
        return np.full(int(ev.sum()), np.nan)
    med = np.nanmedian(Xt, axis=0)
    keep = np.isfinite(med)                      # sklearn drops all-NaN training columns
    Xt = Xt[:, keep]; med = med[keep]; Xe = X[ev][:, keep]
    nanc = np.isnan(Xt).any(axis=0)              # missing-only indicators
    It = np.isnan(Xt[:, nanc]).astype(float); Ie = np.isnan(Xe[:, nanc]).astype(float)
    Xt = np.where(np.isnan(Xt), med, Xt); Xe = np.where(np.isnan(Xe), med, Xe)
    Xt = np.hstack([Xt, It]); Xe = np.hstack([Xe, Ie])
    mu = Xt.mean(0); sd = Xt.std(0); sd[sd == 0] = 1.0
    Xt = (Xt - mu) / sd; Xe = (Xe - mu) / sd
    ym = yt.mean()
    A = Xt.T @ Xt + alpha * np.eye(Xt.shape[1])
    w = np.linalg.solve(A, Xt.T @ (yt - ym))
    return Xe @ w + ym


def _pipe(kind):
    if kind == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             Ridge(alpha=C.RIDGE_ALPHA))
    if kind == "KNN":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             KNeighborsRegressor(n_neighbors=15, weights="distance"))
    if kind == "RF":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5,
                                                   n_jobs=1, random_state=0))
    if kind == "HGB":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                             random_state=0)
    raise KeyError(kind)


def fit_predict(kind, X, y, tr, ev):
    """Prediction on rows `ev` (modelling scale)."""
    if kind == "RidgeFast":
        return ridge_fast(X, y, tr, ev)
    ok = tr & np.isfinite(y)
    m = _pipe(kind).fit(X[ok], y[ok])
    return m.predict(X[ev])
