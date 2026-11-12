"""Part 2: new reliability analyses 1-11 (outage records 2005-2016 + fallback tables).  usage: s4_reliability_new.py [a|b|c]
 a = items 1-3 (Weibull, availability/MTBF/MTTR with CIs, monthly availability)
 b = items 4-5 (reliability block diagrams, fault trees), 9-10 (sensitivity, Monte Carlo convergence)
 c = item 8 (cost-reliability frontier, multi-state NSGA-II); items 6, 7, 11 are assembled from s1 outputs in s5_figdata.py"""
import sys, json, itertools, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
import rv4
from rv4 import config as K
from rv3 import common as cm
from ssopt.data import Data
from b_reliability import runs

def part_a():
    d = Data.load(include_test=False); T = d.T; rng = np.random.default_rng(K.SEED); years = d.index.year.to_numpy()
    down = {v: pd.DataFrame({s: ~np.isfinite(d.raw_arr[v][s]) for s in d.stations}, index=d.index) for v in ("H", "Q")}; down["either"] = down["H"] | down["Q"]
    wrows, avrows, mrows, pp = [], [], [], []; tbf_all = {}
    ustat = {}
    for v in ("either", "Q", "H"):
        for s in d.stations:
            x = down[v][s].to_numpy(); rs = runs(x); n = len(rs); lens = np.array([l for _, l in rs], float)
            ups = np.array([rs[i + 1][0] - (rs[i][0] + rs[i][1]) for i in range(n - 1)], float); trs = np.array([l for a, l in rs if a > 0 and a + l < T], float)
            tbf_all[(s, v)] = (ups[ups > 0], trs[trs > 0])
            for qty, data in (("time_between_failures_days", ups[ups > 0]), ("repair_time_days", trs[trs > 0])):
                n_ = len(data); row = dict(station=s, variable=v, quantity=qty, n=n_)
                if n_ >= 5:
                    c, loc, sc = stats.weibull_min.fit(data, floc=0); ll = stats.weibull_min.logpdf(data, c, 0, sc).sum(); lam = data.mean(); lle = stats.expon.logpdf(data, 0, lam).sum()
                    bs = [stats.weibull_min.fit(rng.choice(data, n_), floc=0) for _ in range(200)]; sh = np.array([b[0] for b in bs]); scs = np.array([b[2] for b in bs]); ks = stats.kstest(data, lambda z: stats.weibull_min.cdf(z, c, 0, sc))
                    row.update(shape=c, shape_ci_lo=np.percentile(sh, 2.5), shape_ci_hi=np.percentile(sh, 97.5), scale_days=sc, scale_ci_lo=np.percentile(scs, 2.5), scale_ci_hi=np.percentile(scs, 97.5), mean_days=data.mean(),
                               KS_stat=ks.statistic, KS_p=ks.pvalue, AIC_weibull=4 - 2 * ll, AIC_exponential=2 - 2 * lle, weibull_better_by_AIC=(2 - 2 * lle) - (4 - 2 * ll))
                wrows.append(row)
            # availability, MTBF, MTTR with CIs: bootstrap over calendar years
            def stats_(arr):
                rs_ = runs(arr); nn = len(rs_); dd = int(arr.sum()); upd = len(arr) - dd
                return 1 - dd / len(arr), (upd / nn if nn else np.nan), (np.mean([l for _, l in rs_]) if nn else np.nan), nn
            a0, mb0, mr0, ne = stats_(x); B = []
            yrs = np.unique(years)
            for _ in range(300):
                samp = rng.choice(yrs, len(yrs)); xx = np.concatenate([x[years == y] for y in samp]); B.append(stats_(xx)[:3])
            B = np.array(B, float)
            avrows.append(dict(station=s, variable=v, role=cm.roles()[s], n_events=ne, availability=a0, availability_ci_lo=np.percentile(B[:, 0], 2.5), availability_ci_hi=np.percentile(B[:, 0], 97.5),
                               MTBF_days=mb0, MTBF_ci_lo=np.nanpercentile(B[:, 1], 2.5) if ne else np.nan, MTBF_ci_hi=np.nanpercentile(B[:, 1], 97.5) if ne else np.nan,
                               MTTR_days=mr0, MTTR_ci_lo=np.nanpercentile(B[:, 2], 2.5) if ne else np.nan, MTTR_ci_hi=np.nanpercentile(B[:, 2], 97.5) if ne else np.nan, lambda_per_year=ne / (T / 365.25)))
    W = pd.DataFrame(wrows); A = pd.DataFrame(avrows); W.to_csv(rv4.RES / "p2_weibull_fits_station.csv", index=False); A.to_csv(rv4.RES / "p2_availability_mtbf_mttr.csv", index=False)
    # probability-plot data for 6 representative stations (2 low, 2 median, 2 high availability among stations with >= 8 TBF samples, 'either')
    el = A[(A.variable == "either")].set_index("station"); eligible = [s for s in d.stations if len(tbf_all[(s, "either")][0]) >= 8]; srt = el.loc[eligible].sort_values("availability")
    mid = len(srt) // 2; chosen = [(srt.index[0], "low"), (srt.index[1], "low"), (srt.index[mid - 1], "median"), (srt.index[mid], "median"), (srt.index[-2], "high"), (srt.index[-1], "high")]
    for s, grp in chosen:
        for qty, idx in (("time_between_failures_days", 0), ("repair_time_days", 1)):
            data = np.sort(tbf_all[(s, "either")][idx]); n = len(data)
            if n < 3: continue
            c, _, sc = stats.weibull_min.fit(data, floc=0); F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
            for t_, f_ in zip(data, F): pp.append(dict(station=s, availability_group=grp, availability=float(el.loc[s, "availability"]), quantity=qty, n=n, t_days=t_, F_empirical=f_, x_ln_t=np.log(t_), y_ln_neg_ln_1mF=np.log(-np.log(1 - f_)),
                                                         y_fitted=c * (np.log(t_) - np.log(sc)), shape=c, scale_days=sc))
    pd.DataFrame(pp).to_csv(rv4.RES / "p2_weibull_probplot_data.csv", index=False)
    # monthly availability, 2005-2016
    ym = d.index.to_period("M"); recs = []
    for v in ("either", "Q", "H"):
        up = (~down[v]).astype(float); g = up.groupby(ym).mean()
        for p in g.index:
            for s in d.stations: recs.append(dict(station=s, variable=v, year=p.year, month=p.month, availability=float(g.loc[p, s])))
    M = pd.DataFrame(recs); M.to_csv(rv4.RES / "p2_monthly_availability_long.csv", index=False)
    M[M.variable == "either"].pivot_table(index="station", columns="month", values="availability").to_csv(rv4.RES / "p2_monthly_availability_month_of_year.csv")
    print(W[(W.variable == "either") & (W.quantity == "time_between_failures_days")].describe().round(3).to_string()); print(A[A.variable == "either"].describe().round(3).T.head(8).to_string())

def structure(net, k, inputs):
    """minimal path sets (inputs whose being up gives >= G1) and minimal cut sets (failed sets giving G0) from the fallback table; exact for the table"""
    n = len(inputs); T = net.T[k]; ok = K.OK; mk = {s: net.maskof(k, [s]) for s in inputs}; full = net.maskof(k, inputs)
    states = {}
    for r in range(n + 1):
        for sub in itertools.combinations(inputs, r): states[frozenset(sub)] = T[net.maskof(k, list(sub))] >= ok
    paths = [set(s) for s, w in states.items() if w and not any(w2 and set(s2) < set(s) for s2, w2 in states.items())]
    cuts = []
    for r in range(1, n + 1):
        for C_ in itertools.combinations(inputs, r):
            if not states[frozenset(set(inputs) - set(C_))] and not any(set(c) <= set(C_) for c in cuts): cuts.append(C_)
    return states, paths, cuts

def part_b(which=("rbd", "sens")):
    from rv4.core import MSNet, pmf
    import pickle
    net = MSNet(designs_file="designs_ms_final.pkl" if (rv4.CACHE / "designs_ms_final.pkl").exists() else "designs_ms_raw.pkl"); d = net.d; keys = net.keys; th = (K.GOOD, K.OK); rng = np.random.default_rng(K.SEED)
    A = {v: net.U[v].mean(0) for v in ("H", "Q")}
    # ---- RBD and fault trees for two real target series --------------------------------------------------------------------------------------------------
    cand = []
    for k in keys:
        if net.full_nse("current", k) < K.OK: continue
        ins = net.design["current"][k]; qs = [1 - float(A[k[1]][net.stn[s]]) for s in ins]
        cand.append((k, len(ins), sum(qs), min(qs)))
    k1 = max([c for c in cand if c[1] == 1], key=lambda c: c[2])[0]; k2 = max([c for c in cand if c[1] >= 2], key=lambda c: c[3])[0]    # most informative: k=1 with least-available input; k>=2 with the largest minimum unavailability
    targets = [k1, k2]; out = {}; print("RBD/fault-tree targets:", targets)
    for (t, v) in targets:
        k = (t, v); res = dict(target=t, variable=v, ok_threshold_NSE=K.OK, good_threshold_NSE=K.GOOD)
        for dn in [x for x in ("current", "k2_P2") if x in net.design]:
            ins = list(net.design[dn][k]); states, paths, cuts = structure(net, k, ins); q = {s: 1 - float(A[v][net.stn[s]]) for s in ins}
            # exact top-event probability under independent basic events
            p_top = 0.0
            for sub, works in states.items():
                p = 1.0
                for s in ins: p *= (1 - q[s]) if s in sub else q[s]
                if not works: p_top += p
            # Monte Carlo: independent (N=200000) and real joint days
            Ui = {s: rng.random(200000) < q[s] for s in ins}; fail = np.zeros(200000, bool)
            upm = np.zeros(200000, np.int64)
            for s in ins: upm |= ((~Ui[s]).astype(np.int64) << [i for i, x in enumerate(net.mem[k]["members"]) if x == s][0])
            fail = net.T[k][upm] < K.OK; mc_i = fail.mean(); se = np.sqrt(mc_i * (1 - mc_i) / 200000)
            real = (net.nse_matrix(dn, series=[k])[:, 0] < K.OK).mean()
            # structure-function agreement of the series-parallel RBD with the table-defined state
            agree = np.mean([bool(any(p_.issubset(sub) for p_ in paths)) == w for sub, w in states.items()])
            res[dn] = dict(components=[dict(station=s, availability=1 - q[s], unavailability=q[s]) for s in ins], full_set_nse=float(net.T[k][net.maskof(k, ins)]),
                           rbd=dict(type="parallel of series blocks (minimal path sets): system works iff all components of at least one path set are up", path_sets=[sorted(p_) for p_ in paths]),
                           structure_function_agreement=float(agree), fault_tree=dict(top_event=f"series {t} {v} in G0 (NSE < {K.OK})", gate="OR", children=[dict(gate="AND", basic_events=list(c)) for c in cuts],
                           basic_event_probability=q, top_event_probability_analytical_independent=p_top, top_event_probability_MC_independent=mc_i, MC_standard_error=se, top_event_probability_real_joint_days=float(real)))
            res[dn]["fault_tree"]["minimal_cut_sets"] = [list(c) for c in cuts]
        out[f"{t}_{v}"] = res
    json.dump(out[f"{k1[0]}_{k1[1]}"], open(rv4.RES / "p2_rbd_faulttree_k1_target.json", "w"), indent=1); json.dump(out[f"{k2[0]}_{k2[1]}"], open(rv4.RES / "p2_rbd_faulttree_k2_target.json", "w"), indent=1)
    rowsft = [dict(target=f"{t} {v}", design=dn, n_components=len(r[dn]["components"]), n_path_sets=len(r[dn]["rbd"]["path_sets"]), n_cut_sets=len(r[dn]["fault_tree"]["minimal_cut_sets"]), structure_agreement=r[dn]["structure_function_agreement"],
                   P_top_analytical=r[dn]["fault_tree"]["top_event_probability_analytical_independent"], P_top_MC_independent=r[dn]["fault_tree"]["top_event_probability_MC_independent"], MC_se=r[dn]["fault_tree"]["MC_standard_error"],
                   P_top_real_joint=r[dn]["fault_tree"]["top_event_probability_real_joint_days"]) for (t, v), r in zip(targets, [out[f"{k1[0]}_{k1[1]}"], out[f"{k2[0]}_{k2[1]}"]]) for dn in r if dn in ("current", "k2_P2")]
    pd.DataFrame(rowsft).to_csv(rv4.RES / "p2_faulttree_summary.csv", index=False); print(pd.DataFrame(rowsft).round(5).to_string())
    # ---- sensitivity of network reliability to station failure rates (independent states; station = one component with unavailability U_i) ----------------------------------
    NIND = 20000; Uind = {v: rng.random((NIND, len(d.stations))) < A[v][None, :] for v in ("H", "Q")}; srows = []; conv = []
    for dn in [x for x in ("current", "k2_P2") if x in net.design]:
        att = [j for j, k in enumerate(keys) if net.full_nse(dn, k) >= th[1]]; st = net.to_state(net.nse_matrix(dn, U=Uind), th); r1 = (st >= 1)[:, att]; R0 = r1.mean()
        used = sorted({s for k in keys for s in net.design[dn][k]})
        for s in used:
            sel = [j for j, k in enumerate(keys) if s in net.design[dn][k]]
            up = net.to_state(net.nse_matrix(dn, U=Uind, series=[keys[j] for j in sel], force=(s, True)), th) >= 1; dnn = net.to_state(net.nse_matrix(dn, U=Uind, series=[keys[j] for j in sel], force=(s, False)), th) >= 1
            full = r1.copy() if False else (st >= 1); u1 = full.copy(); d1 = full.copy(); u1[:, sel] = up; d1[:, sel] = dnn
            IB = u1[:, att].mean() - d1[:, att].mean(); Ui = float(1 - np.mean([A[v][net.stn[s]] for v in ("H", "Q")]))
            for dlt in (-0.5, -0.25, 0.25, 0.5):
                srows.append(dict(design=dn, station=s, role=cm.roles()[s], lat=cm.meta().loc[s, "latitude"], lon=cm.meta().loc[s, "longitude"], unavailability_baseline=Ui, failure_rate_change=dlt,
                                  birnbaum_network=IB, network_reliability_baseline=R0, network_reliability_new=R0 - IB * Ui * dlt, delta_reliability=-IB * Ui * dlt))
        # convergence of the running estimate (independent MC and day-resampling of the real pattern)
        samp = r1.mean(1); cs = np.cumsum(samp); idx = np.unique(np.round(np.logspace(1, np.log10(NIND), 60)).astype(int))
        for n in idx:
            mean = cs[n - 1] / n; sd = samp[:n].std(ddof=1); conv.append(dict(design=dn, method="independent_MC", n_samples=int(n), estimate=mean, ci_lo=mean - 1.96 * sd / np.sqrt(n), ci_hi=mean + 1.96 * sd / np.sqrt(n)))
        days = rng.integers(0, d.T, NIND); stt = net.to_state(net.nse_matrix(dn), th)[:, att] >= 1; samp2 = stt[days].mean(1); cs2 = np.cumsum(samp2)
        for n in idx:
            mean = cs2[n - 1] / n; sd = samp2[:n].std(ddof=1); conv.append(dict(design=dn, method="real_pattern_day_resampling", n_samples=int(n), estimate=mean, ci_lo=mean - 1.96 * sd / np.sqrt(n), ci_hi=mean + 1.96 * sd / np.sqrt(n)))
    S = pd.DataFrame(srows); S["rank_abs_delta_at_plus50"] = S.assign(a=S.delta_reliability.abs()).where(S.failure_rate_change == 0.5).groupby("design").a.rank(ascending=False)
    S["top15_flag"] = S.groupby(["design", "station"]).rank_abs_delta_at_plus50.transform("min") <= 15; S.to_csv(rv4.RES / "p2_sensitivity_failure_rate.csv", index=False); pd.DataFrame(conv).to_csv(rv4.RES / "p2_mc_convergence.csv", index=False)
    print(S[(S.failure_rate_change == 0.5) & S.top15_flag].sort_values(["design", "rank_abs_delta_at_plus50"]).head(15).round(5).to_string())

def frontier_run(args):
    Rstar, seed = args
    import h3_nsga
    from de_reliability import Net
    class NetMS(Net):
        def __init__(self):
            super().__init__(); self.demand = {k: K.OK for k in self.keys}
    h3_nsga.Net = NetMS
    from fv2.optB import run_nsga2
    ev = h3_nsga.RelEval(Rstar); res = run_nsga2(ev, seed, ngen=300); X = np.asarray(res.pop.get("X"), bool); G = res.pop.get("G")[:, 0]; recs = []
    for x, g in zip(X, G):
        if g <= 0:
            r = ev.evaluate(x); recs.append(dict(Rstar=Rstar, seed=seed, n_active=r["n_active"], links=r["links"], network_reliability=r["mean_nse"], resilience_loss=-r["H"], min_series_reliability=float(r["Rs"].min()), removed=",".join(np.array(ev.rem)[~x])))
    return recs, len(ev.deficient)

def part_c():
    jobs = [(R, s) for R in (0.90, 0.95, 0.99) for s in (11, 22, 33, 44, 55)]; out = []; dfc = {}
    with mp.Pool(15) as pool:
        for recs, nd in pool.imap_unordered(frontier_run, jobs): out += recs
    A = pd.DataFrame(out).drop_duplicates(subset=["Rstar", "n_active", "links", "network_reliability", "resilience_loss"]); keep = []
    for R, g in A.groupby("Rstar"):
        F = np.column_stack([g.n_active, g.links, -g.network_reliability, g.resilience_loss]); ok = np.ones(len(F), bool)
        for i in range(len(F)):
            if (np.all(F <= F[i], 1) & np.any(F < F[i], 1)).any(): ok[i] = False
        keep.append(g[ok])
    A = pd.concat(keep); A.to_csv(rv4.RES / "p2_cost_reliability_frontier_nsga.csv", index=False); print(A.groupby("Rstar").agg(n=("n_active", "size"), min_st=("n_active", "min"), min_links=("links", "min"), max_rel=("network_reliability", "max")))

if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "a"
    {"a": part_a, "b": part_b, "c": part_c}[w]()
