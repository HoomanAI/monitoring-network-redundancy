"""1.3 fault detection by ANALYTICAL REDUNDANCY: one model per input station (+ leave-one-out models); detect when the separate estimates disagree.
Threshold per series = 99th percentile (1% false alarms) of the statistic on 2005-2012. Scenarios on 2013-2016 (6 onsets x 5 fault types x each input).
Methods: A = v3 baseline (single all-input model residual), B = max separate-model residual (target vs each single-input estimate),
         C = max pairwise disagreement of the separate estimates (needs k>=2; no target observation).  Primary: C for k>=2, B for k=1.
Isolation (k>=3): leave-one-out agreement (= majority vote): the input whose exclusion restores agreement.  02GD011 example: exploratory (test data)."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import rv4
from rv4 import config as K
from rv3 import common as cm
from g_faults import fault_series, lagblock, make_X, WIN, DUR
from ssopt import config as C
from ssopt.data import Data

class Ridge:
    """numpy replica of the ssopt ridge pipeline with a reusable predictor (median impute + missing indicators + scaling, alpha = 1)."""
    def __init__(self, X, y, tr, alpha=C.RIDGE_ALPHA):
        ok = tr & np.isfinite(y); Xt = X[ok]; yt = y[ok]; med = np.nanmedian(Xt, axis=0); self.keep = np.isfinite(med); Xt = Xt[:, self.keep]; self.med = med[self.keep]
        self.nanc = np.isnan(Xt).any(axis=0); It = np.isnan(Xt[:, self.nanc]).astype(float); Xt = np.where(np.isnan(Xt), self.med, Xt); Xt = np.hstack([Xt, It])
        self.mu = Xt.mean(0); self.sd = Xt.std(0); self.sd[self.sd == 0] = 1.0; Z = (Xt - self.mu) / self.sd; self.ym = yt.mean()
        self.w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (yt - self.ym))
    def predict(self, X):
        X = X[:, self.keep]; I = np.isnan(X[:, self.nanc]).astype(float); X = np.where(np.isnan(X), self.med, X); X = np.hstack([X, I]); return ((X - self.mu) / self.sd) @ self.w + self.ym

def roll(x): return pd.Series(x).rolling(WIN, min_periods=3).mean().to_numpy()

class Series:
    """all models of one target series fitted on the clean training window"""
    def __init__(self, d, t, v, ins, tr):
        self.d, self.t, self.v, self.ins = d, t, v, list(ins); self.y = d.mod_arr[v][t]; self.tr = tr; k = len(ins)
        self.all = Ridge(make_X(d, ins, v), self.y, tr)
        self.single = {s: Ridge(make_X(d, [s], v), self.y, tr) for s in ins}
        self.loo = {s: Ridge(make_X(d, [x for x in ins if x != s], v), self.y, tr) for s in ins} if k >= 3 else {}
        self.stat_stats()
    def est(self, over=None):
        d, v, ins = self.d, self.v, self.ins
        e = {"all": self.all.predict(make_X(d, ins, v, over))}
        for s in ins: e[s] = self.single[s].predict(make_X(d, [s], v, over if over and over[0] == s else None))
        for s in self.loo: e[("loo", s)] = self.loo[s].predict(make_X(d, [x for x in ins if x != s], v, over if over and over[0] != s else None))
        return e
    def channels(self, e):
        ch = {"A": [self.y - e["all"]], "B": [self.y - e[s] for s in self.ins], "C": [e[a] - e[b] for a, b in itertools.combinations(self.ins, 2)]}
        return {m: [roll(np.where(np.isfinite(c), c, np.nan)) for c in cs] for m, cs in ch.items()}
    def stat_stats(self):
        ch = self.channels(self.est()); self.mu, self.sd, self.thr = {}, {}, {}
        for m, cs in ch.items():
            if not cs: continue
            self.mu[m] = [np.nanmean(c[self.tr]) for c in cs]; self.sd[m] = [np.nanstd(c[self.tr]) + 1e-12 for c in cs]
            stat = self.stat(ch, m); self.thr[m] = np.nanpercentile(stat[self.tr], 100 * (1 - K.FAULT_FAR))
        self.clean = ch
    def stat(self, ch, m): return np.nanmax(np.abs(np.column_stack([(c - mu) / sd for c, mu, sd in zip(ch[m], self.mu[m], self.sd[m])])), axis=1)

def isolate(sr, e, day):
    """leave-one-out agreement at the alarm day: input whose exclusion gives the smallest pairwise disagreement among the rest"""
    best, bs = None, np.inf
    for s in sr.ins:
        rest = [x for x in sr.ins if x != s]
        if len(rest) < 2: continue
        dis = max(abs(roll(e[a] - e[b])[day] / (sr.sd["C"][list(itertools.combinations(sr.ins, 2)).index(tuple(sorted((a, b), key=sr.ins.index)))])) for a, b in itertools.combinations(rest, 2))
        if dis < bs: bs, best = dis, s
    return best

def main():
    d = Data.load(include_test=False); P1 = cm.pools("pool"); tr, ev = d.fold_masks(2); vidx = np.flatnonzero(ev); rng = np.random.default_rng(K.SEED)
    keys = [(t, v) for t in d.targets for v in ("H", "Q")]; scen = [("step", 0.1), ("step", 0.25), ("step", 0.5), ("drift", 0.25), ("stuck", 0.0)]
    rows, far = [], []
    for (t, v) in keys:
        ins = list(P1[(t, v)]["direct"]); k = len(ins)
        if k == 0: continue
        sr = Series(d, t, v, ins, tr); grp = "k=1" if k == 1 else ("k=2" if k == 2 else "k>=3")
        for m in ("A", "B", "C"):
            if m in sr.thr: far.append(dict(target=t, var=v, k=k, group=grp, method=m, threshold=sr.thr[m], far_validation=float(np.nanmean(sr.stat(sr.clean, m)[ev] > sr.thr[m]))))
        for s in ins:
            for kind, mag in scen:
                for rep in range(6):
                    onset = int(rng.choice(vidx[: -(DUR + 5)])); fx = fault_series(d, s, v, kind, mag, onset); over = (s, lagblock(fx)); e = sr.est(over); ch = sr.channels(e); w = slice(onset, min(onset + DUR, d.T))
                    for m in ("A", "B", "C"):
                        if m not in sr.thr: continue
                        alarm = sr.stat(ch, m)[w] > sr.thr[m]; det = bool(alarm.any()); delay = int(np.argmax(alarm)) if det else np.nan; iso = np.nan; iso2 = np.nan
                        if det and m == "C" and k >= 3: iso = float(isolate(sr, e, onset + delay) == s)
                        if det and m == "C" and k == 2:             # k = 2: tie-break with the target observation
                            r = [abs(np.nanmean((sr.y - e[x])[onset + delay: onset + delay + WIN])) / sr.sd["B"][sr.ins.index(x)] for x in sr.ins]; iso2 = float(sr.ins[int(np.argmax(r))] == s)
                        rows.append(dict(target=t, var=v, k=k, group=grp, station=s, kind=kind, magnitude=mag, method=m, detected=det, delay_days=delay, isolated_loo=iso, isolated_k2_with_target=iso2))
    R = pd.DataFrame(rows); F = pd.DataFrame(far); R.to_csv(rv4.RES / "fault_ar_scenarios.csv", index=False); F.to_csv(rv4.RES / "fault_ar_false_alarm.csv", index=False)
    S = R.groupby(["group", "method", "kind", "magnitude"]).agg(n=("detected", "size"), detection_rate=("detected", "mean"), median_delay_days=("delay_days", "median"), isolation_rate_loo=("isolated_loo", "mean"),
                                                             isolation_rate_k2_with_target=("isolated_k2_with_target", "mean")).reset_index()
    fa = F.groupby(["group", "method"]).far_validation.mean().rename("false_alarm_rate_validation").reset_index(); S = S.merge(fa, on=["group", "method"], how="left"); S.to_csv(rv4.RES / "fault_ar_by_k.csv", index=False)
    pd.set_option("display.width", 220); print(S[S.method.isin(["A", "B", "C"])].round(3).query("kind=='step'").to_string())
    # primary summary: C for k>=2, B for k=1 (and A for reference)
    prim = pd.concat([S[(S.group == "k=1") & (S.method == "B")], S[(S.group != "k=1") & (S.method == "C")]]); prim.to_csv(rv4.RES / "fault_ar_primary.csv", index=False)
    print(R.groupby(["group", "method"]).detected.mean().unstack().round(3))
    # ---- 02GD011 H step (2017) EXPLORATORY: test data ---------------------------------------------------------------------------------------------------------------
    dt = Data.load(include_test=True); trd = dt.period(C.DEV); tgt = [t for (t, v) in keys if v == "H" and "02GD011" in P1[(t, "H")]["direct"]]; out = []; z_rows = []
    for t in tgt:
        ins = list(P1[(t, "H")]["direct"]); extra = [s for s in P1[(t, "H")]["inputs"] if s not in ins][:2]
        for lab, inputs in ((f"current inputs (k={len(ins)})", ins), (f"+ redundant inputs (k={len(ins) + len(extra)})", ins + extra)):
            sr = Series(dt, t, "H", inputs, trd); e = sr.est(); ch = sr.channels(e); idx = dt.index
            for m in ("A", "B", "C"):
                if m not in sr.thr: continue
                z = sr.stat(ch, m) / sr.thr[m]; first = np.flatnonzero((z > 1) & (idx >= "2017-01-01")); mk = (idx >= "2016-07-01") & (idx <= "2018-06-30")
                out.append(dict(target=t, design=lab, method=m, first_alarm_after_2017_01_01=str(idx[first[0]].date()) if len(first) else "not detected", false_alarm_rate_dev=float(np.nanmean(z[trd] > 1)), EXPLORATORY=True))
                for dd, zz in zip(idx[mk], z[mk]): z_rows.append(dict(target=t, design=lab, method=m, date=str(dd.date()), statistic_over_threshold=float(zz)))
    pd.DataFrame(out).to_csv(rv4.RES / "fault_ar_02GD011_example_EXPLORATORY.csv", index=False); pd.DataFrame(z_rows).to_csv(rv4.RES / "fault_ar_02GD011_series_EXPLORATORY.csv", index=False); print(pd.DataFrame(out).to_string())

if __name__ == "__main__":
    main()
