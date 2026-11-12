"""1.1 multi-state reliability measures: state probabilities, importance, SPOF/min cuts, robustness, event resilience, threshold sensitivity.
usage: s1_reliability_ms.py raw|final   (designs from designs_ms_raw.pkl or the verified/repaired designs_ms_final.pkl)"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx
from scipy import stats
import rv4
from rv4 import config as K
from rv4.core import MSNet, pmf, score_of, block_ci, min_cut, perf_removed, episodes
from rv3 import common as cm

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "raw"
    net = MSNet(designs_file="designs_ms_raw.pkl" if which == "raw" else "designs_ms_final.pkl"); d = net.d; keys = net.keys; rng = np.random.default_rng(K.SEED)
    meta = cm.meta(); th = (K.GOOD, K.OK); sfx = "" if which == "raw" else "_final"
    designs = ["current"] + [n for n in ("k1_P1", "k1_P2", "k2_P1", "k2_P2", "k3_P2") if n in net.design]
    N = len(keys); rows_p, rows_n = [], []; S = {}
    for dn in designs:
        st = net.to_state(net.nse_matrix(dn), th); S[dn] = st; pm = pmf(st); r1 = (st >= 1)
        att = [j for j, k in enumerate(keys) if net.full_nse(dn, k) >= th[1]]
        for j, k in enumerate(keys): rows_p.append(dict(design=dn, target=k[0], var=k[1], P_G2=pm[j, 0], P_G1=pm[j, 1], P_G0=pm[j, 2], P_ge_G1=pm[j, 0] + pm[j, 1], expected_score=pm[j, 0] * K.SCORE[2] + pm[j, 1] * K.SCORE[1],
                                                       all_up_nse=net.full_nse(dn, k), structurally_failed=j not in att, flagged=k in cm.FLAGGED))
        lo, hi = block_ci(r1.astype(float), rng); lo2, hi2 = block_ci(r1[:, att].astype(float), rng)
        rows_n.append(dict(design=dn, n_active_stations=len({s for k in keys for s in net.design[dn][k]} | set(d.targets)), n_series=N, n_series_attainable=len(att),
                           network_reliability_all=r1.mean(), ci_lo=lo, ci_hi=hi, network_reliability_attainable=r1[:, att].mean(), ci_lo_att=lo2, ci_hi_att=hi2,
                           expected_score_all=score_of(st).mean(), P_G2_mean=(st == 2).mean(), P_G1_mean=(st == 1).mean(), P_G0_mean=(st == 0).mean(),
                           P_all_attainable_ge_G1=r1[:, att].all(1).mean(), n_series_reliability_below_0p95=int((r1[:, att].mean(0) < 0.95).sum())))
    PM = pd.DataFrame(rows_p); NR = pd.DataFrame(rows_n); PM.to_csv(rv4.RES / f"ms_series_pmf{sfx}.csv", index=False); NR.to_csv(rv4.RES / f"ms_network_reliability{sfx}.csv", index=False); print(NR.round(4).T.to_string())
    # ---------------- importance (current + k2_P2) -----------------------------------------------------------------------------------------
    A = {v: net.U[v].mean(0) for v in ("H", "Q")}; imp_rows, net_rows = [], []
    for dn in [x for x in ("current", "k2_P2") if x in net.design]:
        st = S[dn]; r1 = (st >= 1); sc = score_of(st); used = sorted({s for k in keys for s in net.design[dn][k]}); att = [j for j, k in enumerate(keys) if net.full_nse(dn, k) >= th[1]]
        Qn = 1 - r1[:, att].mean()
        for s in used:
            sel = [j for j, k in enumerate(keys) if s in net.design[dn][k]]
            up = net.to_state(net.nse_matrix(dn, series=[keys[j] for j in sel], force=(s, True)), th); dnst = net.to_state(net.nse_matrix(dn, series=[keys[j] for j in sel], force=(s, False)), th)
            for jj, j in enumerate(sel):
                k = keys[j]; Ai = A[k[1]][net.stn[s]]; Rs = r1[:, j].mean(); Qs = 1 - Rs
                ib = (up[:, jj] >= 1).mean() - (dnst[:, jj] >= 1).mean(); ibu = score_of(up[:, jj]).mean() - score_of(dnst[:, jj]).mean()
                imp_rows.append(dict(design=dn, station=s, target=k[0], var=k[1], birnbaum_ge_G1=ib, birnbaum_score=ibu, RAW=(1 - (dnst[:, jj] >= 1).mean()) / Qs if Qs > 1e-9 else np.nan,
                                     RRW=Qs / max(1 - (up[:, jj] >= 1).mean(), 1e-9) if Qs > 1e-9 else np.nan, criticality=ib * (1 - Ai) / Qs if Qs > 1e-9 else np.nan))
            # network level (series attainable only)
            fullr = r1.copy(); upr = r1.copy(); dnr = r1.copy(); upr[:, sel] = up >= 1; dnr[:, sel] = dnst >= 1
            pu, pdn = upr[:, att].mean(), dnr[:, att].mean(); Ai = np.mean([A[v][net.stn[s]] for v in ("H", "Q")]); ib = pu - pdn
            scu = sc.copy(); scd = sc.copy(); scu[:, sel] = score_of(up); scd[:, sel] = score_of(dnst)
            net_rows.append(dict(design=dn, station=s, role=cm.roles()[s], lat=meta.loc[s, "latitude"], lon=meta.loc[s, "longitude"], n_series_fed=len(sel), unavailability=1 - Ai,
                                 birnbaum_network_ge_G1=ib, birnbaum_network_score=float(scu[:, att].mean() - scd[:, att].mean()), criticality_network=ib * (1 - Ai) / Qn if Qn > 0 else np.nan,
                                 RAW_network=(1 - pdn) / Qn if Qn > 0 else np.nan, RRW_network=Qn / max(1 - pu, 1e-9)))
    pd.DataFrame(imp_rows).to_csv(rv4.RES / f"ms_importance_per_series{sfx}.csv", index=False); IN = pd.DataFrame(net_rows); IN.to_csv(rv4.RES / f"ms_importance_network{sfx}.csv", index=False)
    print(IN[IN.design == "current"].sort_values("birnbaum_network_ge_G1", ascending=False).head(10).round(4).to_string())
    # ---------------- SPOF / minimum cuts -----------------------------------------------------------------------------------------------------------
    crow = []
    for dn in designs:
        for k in keys:
            r, C_ = min_cut(net, dn, k, th); full = net.full_nse(dn, k)
            crow.append(dict(design=dn, target=k[0], var=k[1], n_inputs=len(net.design[dn][k]), all_up_nse=full, structurally_failed=full < th[1], min_cut_size=r, min_cut=",".join(C_),
                             spof=(r == 1), spof_station=C_[0] if r == 1 else "", flagged=k in cm.FLAGGED))
    CUT = pd.DataFrame(crow); CUT.to_csv(rv4.RES / f"ms_mincut_spof{sfx}.csv", index=False)
    kfrozen = pd.read_csv(rv4.V3 / "results/A_node_table.csv", dtype={"station": str}).set_index("station")
    srow = []
    for dn in designs:
        g = CUT[(CUT.design == dn) & ~CUT.structurally_failed]; tg = g[g.spof].target.nunique()
        srow.append(dict(design=dn, n_series_evaluated=len(g), n_series_structurally_failed=int(CUT[(CUT.design == dn)].structurally_failed.sum()), n_spof_series=int(g.spof.sum()), n_spof_stations=len({s for s in g.spof_station if s}),
                         n_targets_with_spof=int(tg), median_min_cut=float(g.min_cut_size.median()), n_series_cut_ge2=int((g.min_cut_size >= 2).sum()),
                         graph_targets_in_degree_1=int((kfrozen[kfrozen.role != "source"].k_direct_frozen == 1).sum()) if dn == "current" else np.nan))
    SP = pd.DataFrame(srow); SP.to_csv(rv4.RES / f"ms_spof_summary{sfx}.csv", index=False); print(SP.to_string())
    # ---------------- robustness curves ---------------------------------------------------------------------------------------------------------------
    G, _ = cm.physical_forest(); rob = []; curves = []
    for dn in [x for x in ("current", "k1_P2", "k2_P2", "k2_P1") if x in net.design]:
        used = sorted({s for k in keys for s in net.design[dn][k]}); n = len(used); base_ok, _ = perf_removed(net, dn, set(), th)
        birn = {s: base_ok - perf_removed(net, dn, {s}, th)[0] for s in used}; deg = {s: sum(1 for k in keys if s in net.design[dn][k]) for s in used}
        U = nx.DiGraph(G); [U.add_edge(s, k[0]) for k in keys for s in net.design[dn][k] if s not in net.mem[k]["direct"]]; btw = nx.betweenness_centrality(U)
        orders = {"Birnbaum-targeted": sorted(used, key=lambda s: -(birn[s] + 1e-6 * deg[s])), "degree-targeted": sorted(used, key=lambda s: -deg[s]), "betweenness-targeted": sorted(used, key=lambda s: -btw.get(s, 0))}
        res = {}
        for on, od_ in orders.items(): res[on] = np.array([perf_removed(net, dn, set(od_[:r]), th) for r in range(n + 1)])
        Rn = np.array([[perf_removed(net, dn, set(o[:r]), th) for r in range(n + 1)] for o in (list(rng.permutation(used)) for _ in range(150))])
        res["random_mean"] = Rn.mean(0); res["random_p05"] = np.percentile(Rn, 5, axis=0); res["random_p95"] = np.percentile(Rn, 95, axis=0)
        x = np.arange(n + 1) / n
        for on in ("random_mean", "Birnbaum-targeted", "degree-targeted", "betweenness-targeted"):
            c = res[on]; rob.append(dict(design=dn, order=on, n_stations_in_use=n, auc_share_ge_G1=np.trapezoid(c[:, 0], x), auc_expected_score=np.trapezoid(c[:, 1], x), share_ge_G1_after_10pct_removed=c[int(0.1 * n), 0]))
        for on, c in res.items():
            for r in range(n + 1): curves.append(dict(design=dn, order=on, n_removed=r, fraction_removed=r / n, share_ge_G1=c[r, 0], expected_score=c[r, 1]))
    pd.DataFrame(rob).to_csv(rv4.RES / f"ms_robustness_auc{sfx}.csv", index=False); pd.DataFrame(curves).to_csv(rv4.RES / f"ms_robustness_curves{sfx}.csv", index=False)
    print(pd.DataFrame(rob).round(3).to_string())
    # ---------------- event resilience -------------------------------------------------------------------------------------------------------------------
    used_all = sorted({s for dn in ("current", "k2_P2") if dn in net.design for k in keys for s in net.design[dn][k]}); ep, cnt, sm = episodes(net, used_all)
    ev_rows, daily = [], []
    for ei, c in enumerate(ep):
        lo, hi = c - 45, min(c + 91, d.T); thr = 0.5 * sm[c]; ext = np.flatnonzero(sm[lo:hi] >= thr) + lo; e0, e1 = int(ext.min()), int(ext.max())
        for dn in [x for x in ("current", "k1_P2", "k2_P1", "k2_P2") if x in net.design]:
            st = S[dn][lo:hi]; r1 = (st >= 1); sc = score_of(st)
            # no-fallback baseline: a series is failed whenever ANY of its design inputs is down
            ups = np.ones((hi - lo, len(keys)), bool)
            for j, k in enumerate(keys):
                for s in net.design[dn][k]: ups[:, j] &= net.U[k[1]][lo:hi, net.stn[s]]
            att = [j for j, k in enumerate(keys) if net.full_nse(dn, k) >= th[1]]
            full_r1 = np.array([net.full_nse(dn, k) >= th[1] for k in keys])[None, :].repeat(hi - lo, 0)
            for lab, curve in (("fallback", r1[:, att].mean(1)), ("no_fallback", (full_r1 & ups)[:, att].mean(1)), ("fallback_score", sc[:, att].mean(1))):
                base = np.full(hi - lo, (full_r1[:, att].mean(1)).mean() if lab != "fallback_score" else np.mean([K.SCORE[int(net.to_state(np.array([net.full_nse(dn, keys[j])]), th)[0])] for j in att]))
                perf = pd.Series(curve).rolling(3, center=True, min_periods=1).mean().to_numpy(); after = np.flatnonzero((np.arange(lo, hi) > e1) & (perf >= 0.99 * base))
                ev_rows.append(dict(episode=ei + 1, date=str(d.index[c].date()), event_start=str(d.index[e0].date()), event_end=str(d.index[e1].date()), design=dn, mode=lab, min_performance=perf.min(), full_service_level=base[0],
                                    robustness_min_rel=perf.min() / base[0], loss_performance_days=float(np.sum(np.maximum(base - perf, 0))), recovery_days=int(lo + after[0] - e1 - 1) if len(after) else np.nan))
                for i_, v_ in enumerate(perf): daily.append(dict(episode=ei + 1, design=dn, mode=lab, day_index=lo + i_, date=str(d.index[lo + i_].date()), performance=v_, in_event=bool(e0 <= lo + i_ <= e1)))
    pd.DataFrame(ev_rows).to_csv(rv4.RES / f"ms_event_resilience{sfx}.csv", index=False); pd.DataFrame(daily).to_csv(rv4.RES / f"ms_event_daily{sfx}.csv", index=False)
    print(pd.DataFrame(ev_rows).query("mode=='fallback' or mode=='no_fallback'").round(3).to_string())
    # ---------------- threshold sensitivity (current network) ---------------------------------------------------------------------------------------------------
    trow = []
    for (g_, o_) in [(K.GOOD, K.OK)] + K.SENSITIVITY:
        t_ = (g_, o_)
        for dn in designs:
            st = net.to_state(net.nse_matrix(dn), t_); att = [j for j, k in enumerate(keys) if net.full_nse(dn, k) >= o_]; r1 = (st >= 1)
            cuts = [min_cut(net, dn, keys[j], t_)[0] for j in att]
            trow.append(dict(thresholds=f"{g_}/{o_}", design=dn, n_series_attainable=len(att), network_reliability_attainable=r1[:, att].mean(), P_G2_mean=(st == 2).mean(), P_G0_mean=(st == 0).mean(),
                             n_spof_series=int(sum(1 for c in cuts if c == 1)), median_min_cut=float(np.median([c for c in cuts if c])) if any(cuts) else np.nan))
    pd.DataFrame(trow).to_csv(rv4.RES / f"ms_threshold_sensitivity{sfx}.csv", index=False); print(pd.DataFrame(trow).round(3).query("design in ['current','k2_P2']").to_string())

if __name__ == "__main__":
    main()
