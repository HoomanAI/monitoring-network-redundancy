"""Assemble figdata CSVs, reconciliation table, numbers registry and workbook (no final figures are made here)."""
import sys, json, pickle, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import rv4
from rv4 import config as K
from rv4.core import MSNet, pmf, score_of, block_ci
from rv3 import common as cm

R = rv4.RES; FD = rv4.FD; V3R = rv4.V3 / "results"
meta = cm.meta(); coords = meta[["latitude", "longitude"]].rename(columns={"latitude": "lat_deg", "longitude": "lon_deg"})
DESC = {}
def out(name, df, desc, source):
    df.to_csv(FD / f"{name}.csv", index=False); DESC[name] = (desc, source, list(df.columns), len(df))
REG = []
def reg(name, value, unit, source, how): REG.append(dict(name=name, value=value, unit=unit, source_file=source, how_computed=how))

def main():
    rng = np.random.default_rng(K.SEED)
    # ------------------------------------------------------------------ reconciliation (raw and repaired designs) ----------------------------------------------------------------
    VD = pickle.load(open(rv4.CACHE / "verified_designs.pkl", "rb")); VR = pd.read_csv(R / "verification_repair_summary.csv").set_index("design")
    net = MSNet(designs_file="designs_ms_final.pkl"); keys = net.keys; th = (K.GOOD, K.OK); d = net.d; dirs = {k: set(net.mem[k]["direct"]) for k in keys}
    rec = []
    for dn, v in VD.items():
        for ver in ("raw", "repaired"):
            nm = f"{dn}__{ver}"; net.design[nm] = {k: list(v[ver]["inputs"][k]) for k in keys}; st = net.to_state(net.nse_matrix(nm), th); att = [j for j, k in enumerate(keys) if net.full_nse(nm, k) >= th[1]]
            r1 = st >= 1; act = set(v[ver]["active"]); links = sum(len([s for s in net.design[nm][k] if s not in dirs[k]]) for k in keys); sv = net.served(nm).mean()
            lo, hi = block_ci(r1[:, att].astype(float), rng)
            rec.append(dict(design=dn, family=VR.loc[dn, "family"], version=ver, level_k=v["level"], pool=v["pool"], stations_removed=len(set(net.d.stations) - act), pct_of_119_removed=100 * (119 - len(act)) / 119,
                            links_added=links, network_reliability_ge_G1=r1[:, att].mean(), ci_lo=lo, ci_hi=hi, n_series_attainable=len(att), P_G2=(st[:, att] == 2).mean(), P_G0=(st[:, att] == 0).mean(),
                            expected_score=score_of(st)[:, att].mean(), v3_binary_share_served=sv, n_series_unrepaired=len(v["unrepaired"]) if ver == "repaired" else np.nan,
                            removed_stations=",".join(sorted(set(net.d.stations) - act))))
    REC = pd.DataFrame(rec); REC.to_csv(R / "reconciliation_table.csv", index=False); print(REC.drop(columns="removed_stations").round(4).to_string())
    # ------------------------------------------------------------------ figdata ---------------------------------------------------------------------------------------------------------------------
    W = pd.read_csv(R / "p2_weibull_probplot_data.csv"); out("fig01_weibull_probplot", W, "Weibull probability-plot points for 6 stations (2 low, 2 median, 2 high availability; >= 8 samples): x = ln t, y = ln(-ln(1-F)), F by median ranks; y_fitted = shape*(x - ln scale).", "p2_weibull_probplot_data.csv")
    WF = pd.read_csv(R / "p2_weibull_fits_station.csv"); out("fig02_weibull_parameters", WF, "Weibull shape/scale (95% bootstrap CIs), KS test and AIC vs exponential per station, variable (either/Q/H) and quantity (time between failures, repair time); fits need >= 5 samples.", "p2_weibull_fits_station.csv")
    AV = pd.read_csv(R / "p2_availability_mtbf_mttr.csv").merge(coords, left_on="station", right_index=True); out("fig03_availability_distribution", AV, "Availability per station and variable with year-block bootstrap 95% CIs (300 resamples), plus coordinates.", "p2_availability_mtbf_mttr.csv")
    out("fig04_mtbf_vs_mttr", AV[AV.variable == "either"][["station", "role", "n_events", "MTBF_days", "MTBF_ci_lo", "MTBF_ci_hi", "MTTR_days", "MTTR_ci_lo", "MTTR_ci_hi", "availability", "lambda_per_year", "lat_deg", "lon_deg"]], "MTBF vs MTTR (event: H or Q missing) per station with CIs.", "p2_availability_mtbf_mttr.csv")
    out("fig05_monthly_availability", pd.read_csv(R / "p2_monthly_availability_long.csv"), "Station x year x month availability (fraction of days with data present), 2005-2016, for H, Q and either.", "p2_monthly_availability_long.csv")
    J = pd.read_csv(V3R / "B_cooccurrence_jaccard.csv", index_col=0); L = J.stack().reset_index(); L.columns = ["station_a", "station_b", "jaccard_outage_days"]; out("fig06a_cooccurrence_jaccard", L[L.station_a != L.station_b], "Jaccard overlap of outage days between station pairs (H or Q missing), 2005-2016 (unchanged from ress_v3).", "ress_v3/results/B_cooccurrence_jaccard.csv")
    out("fig06b_outage_seasonality", pd.read_csv(V3R / "B_seasonality.csv"), "Monthly share of stations down (mean, 25th/75th percentile across stations) (unchanged from ress_v3).", "ress_v3/results/B_seasonality.csv")
    cc = pd.read_csv(V3R / "B_common_cause_summary.csv", index_col=0)["value"]; out("fig06c_common_cause_beta", cc.reset_index().rename(columns={"index": "quantity", "value": "value"}), "Beta factor (observed vs independent time-shift null) and co-occurrence summary (unchanged from ress_v3).", "ress_v3/results/B_common_cause_summary.csv")
    NT = pd.read_csv(V3R / "A_node_table.csv", dtype={"station": str}); T_ = NT[NT.role != "source"]
    out("fig07a_target_indegree", T_.groupby("k_direct_frozen").size().rename("n_targets").reset_index().rename(columns={"k_direct_frozen": "k_direct_inputs"}), "Number of targets by number of direct upstream inputs k (frozen links).", "ress_v3/results/A_node_table.csv")
    out("fig07b_chain_depth", NT[["station", "role", "depth_up", "n_ancestors", "component"]], "Chain depth (longest upstream path) and number of ancestors per station.", "ress_v3/results/A_node_table.csv")
    out("fig07c_component_sizes", NT.groupby("component").agg(n_stations=("station", "size"), n_targets=("role", lambda r: int((r != "source").sum()))).reset_index(), "Component sizes of the physical forest.", "ress_v3/results/A_node_table.csv")
    # performance vs number of inputs
    rows = []
    for k, arr in net.T.items():
        n = len(net.mem[k]["members"]); pc = np.array([bin(m).count("1") for m in range(2 ** n)])
        for j in range(n + 1): x = arr[pc == j]; rows.append(dict(target=k[0], var=k[1], n_available_inputs=j, n_subsets=len(x), nse_mean=float(np.nanmean(x)), nse_min=float(np.nanmin(x)), nse_max=float(np.nanmax(x)), nse_full_pool=float(arr[-1]), nse_season_only=float(arr[0])))
    out("fig08_performance_vs_inputs", pd.DataFrame(rows), "Ridge validation NSE (2013-2016) over all subsets of each series' candidate pool (<= 12 inputs): mean/min/max NSE by number of available inputs.", "rv3 fallback tables (ress_v3/cache/fallback_tables.pkl)")
    PM = pd.read_csv(R / "ms_series_pmf_final.csv", dtype={"target": str}).merge(coords, left_on="target", right_index=True)
    out("fig09_multistate_distribution", PM, f"P(G2), P(G1), P(G0) per target series and design (current, k1/k2 physical and with statistical links, k3 Ridge-only). G2: NSE >= {K.GOOD}; G1: >= {K.OK}; G0 below. Real joint outage states 2005-2016.", "ms_series_pmf_final.csv")
    tp = PM.groupby(["design", "target"]).agg(P_ge_G1_min_HQ=("P_ge_G1", "min"), P_G2_min_HQ=("P_G2", "min"), expected_score_mean_HQ=("expected_score", "mean"), lat_deg=("lat_deg", "first"), lon_deg=("lon_deg", "first")).reset_index()
    out("fig10_target_reliability", tp, "Per-target reliability = worst of H and Q probability of state >= G1, per design, with coordinates.", "ms_series_pmf_final.csv")
    IN = pd.read_csv(R / "ms_importance_network_final.csv", dtype={"station": str}).rename(columns={"lat": "lat_deg", "lon": "lon_deg"}); IN["rank_birnbaum"] = IN.groupby("design").birnbaum_network_ge_G1.rank(ascending=False, method="min")
    IN["rank_criticality"] = IN.groupby("design").criticality_network.rank(ascending=False, method="min"); out("fig11_importance_multistate", IN, "Network-level multi-state Birnbaum importance (change in expected share of series >= G1 between station up/down), criticality, RAW and RRW per station, with coordinates.", "ms_importance_network_final.csv")
    CT = pd.read_csv(R / "ms_mincut_spof_final.csv", dtype={"target": str, "spof_station": str}).merge(coords.add_prefix("target_"), left_on="target", right_index=True, how="left").merge(coords.add_prefix("spof_station_"), left_on="spof_station", right_index=True, how="left")
    out("fig12_spof_mincut", CT, "Minimum vertex cut (smallest set of simultaneous input failures taking a series from >= G1 to G0) and single points of failure per series and design, with target and SPOF-station coordinates.", "ms_mincut_spof_final.csv")
    out("fig13a_robustness_curves", pd.read_csv(R / "ms_robustness_curves_final.csv"), "Share of series >= G1 and expected score as input stations are removed (random mean/5-95%, Birnbaum-, degree-, betweenness-targeted).", "ms_robustness_curves_final.csv")
    out("fig13b_robustness_auc", pd.read_csv(R / "ms_robustness_auc_final.csv"), "Area under the robustness curves per design and order.", "ms_robustness_auc_final.csv")
    ED = pd.read_csv(R / "ms_event_daily_final.csv"); EM = pd.read_csv(R / "ms_event_resilience_final.csv"); out("fig14a_event_resilience_curves", ED, "Daily performance (share of series >= G1, or expected score) during three real multi-station outage episodes, with fallback and without fallback.", "ms_event_daily_final.csv"); out("fig14b_event_resilience_metrics", EM, "Robustness, loss (performance-days) and recovery time per episode, design and mode.", "ms_event_resilience_final.csv")
    cur = EM[(EM.design == "current") & (EM["mode"] == "fallback")]; ep = int(cur.sort_values("loss_performance_days").iloc[-1].episode); tri = ED[(ED.episode == ep) & (ED["mode"].isin(["fallback", "no_fallback"])) & (ED.design.isin(["current", "k2_P2"]))]
    out("fig15a_resilience_triangle_daily", tri, f"Resilience triangle: daily performance of the current and k=2 (with statistical links) designs for the real outage episode {ep} (largest loss for the current network).", "ms_event_daily_final.csv")
    out("fig15b_resilience_triangle_metrics", EM[(EM.episode == ep) & (EM.design.isin(["current", "k2_P2"]))], "Outage start/end, minimum performance, loss and recovery time for the triangle episode.", "ms_event_resilience_final.csv")
    NR = pd.read_csv(R / "ms_network_reliability_final.csv"); kmap = {"current": (0, "-"), "k1_P1": (1, "P1"), "k1_P2": (1, "P2"), "k2_P1": (2, "P1"), "k2_P2": (2, "P2"), "k3_P2": (3, "P2")}
    NR["k"] = NR.design.map(lambda x: kmap[x][0]); NR["pool"] = NR.design.map(lambda x: kmap[x][1]); NR["verification"] = NR.design.map(lambda x: "Ridge-only (not four-model verified)" if x == "k3_P2" else ("n/a" if x == "current" else "four-model verified and repaired"))
    out("fig16_reliability_vs_k", NR, "Network reliability P(state >= G1) (mean over attainable series, 95% 30-day-block bootstrap CI) and other measures vs redundancy level k.", "ms_network_reliability_final.csv")
    # cost-reliability frontier
    import h3_nsga
    ev = h3_nsga.RelEval(0.9); days = ev.days; rows = []
    for _, r in REC.iterrows():
        nm = f"{r.design}__{r.version}"; st = net.to_state(net.nse_matrix(nm), th); all_r = (st >= 1)
        rows.append(dict(type="design_point", name=nm, family=r.family, version=r.version, n_active=119 - r.stations_removed, links=r.links_added, network_reliability_all_series=all_r.mean(), resilience_loss=1 - all_r[days].mean(), four_model_verified=(r.version == "repaired" or r.family == "v2 NSGA-II")))
    for dn in ("current", "k1_P1", "k1_P2", "k2_P1", "k2_P2", "k3_P2"):
        st = net.to_state(net.nse_matrix(dn), th); all_r = st >= 1; kk = sum(len([s for s in net.design[dn][k] if s not in dirs[k]]) for k in keys)
        rows.append(dict(type="design_point", name=f"{dn}__final", family="v4 multi-state final", version="final", n_active=len({s for k in keys for s in net.design[dn][k]} | set(d.targets)), links=kk, network_reliability_all_series=all_r.mean(), resilience_loss=1 - all_r[days].mean(), four_model_verified=dn not in ("k3_P2",)))
    FR = pd.read_csv(R / "p2_cost_reliability_frontier_nsga.csv")
    for _, r in FR.iterrows(): rows.append(dict(type="nsga_pareto_Rstar_%.2f" % r.Rstar, name=f"seed{int(r.seed)}", family="NSGA-II multi-state (Ridge only)", version="archive", n_active=r.n_active, links=r.links, network_reliability_all_series=r.network_reliability, resilience_loss=r.resilience_loss, four_model_verified=False))
    out("fig17_cost_reliability_frontier", pd.DataFrame(rows), "Cost-reliability frontier: stations, extra links, network reliability (mean P(>=G1) over all 114 series) and resilience loss (unserved share in three outage episodes) for design points and the NSGA-II Pareto archive under R* in {0.90, 0.95, 0.99} (Ridge only).", "reconciliation_table.csv; p2_cost_reliability_frontier_nsga.csv")
    SE = pd.read_csv(R / "p2_sensitivity_failure_rate.csv", dtype={"station": str}).rename(columns={"lat": "lat_deg", "lon": "lon_deg"}); out("fig18_sensitivity", SE, "Network reliability when a station's failure rate changes by -50/-25/+25/+50% (independent states; station treated as one component; reliability change = Birnbaum importance x change in unavailability); top-15 flag at +50%.", "p2_sensitivity_failure_rate.csv")
    out("fig19_mc_convergence", pd.read_csv(R / "p2_mc_convergence.csv"), "Running estimate of network reliability and 95% CI vs number of Monte Carlo samples (independent states; and resampling of real outage days).", "p2_mc_convergence.csv")
    FA = pd.read_csv(R / "fault_ar_by_k.csv"); out("fig20a_fault_detection", FA, "Fault detection by k (1, 2, >=3), method (A = v3 all-input residual, B = separate-model residuals, C = disagreement of separate estimates; C is primary for k >= 2, B for k = 1), fault type and size: detection rate, median delay (days), isolation rates, validation false-alarm rate (threshold set for 1% on 2005-2012).", "fault_ar_by_k.csv")
    out("fig20b_fault_02GD011_EXPLORATORY", pd.read_csv(R / "fault_ar_02GD011_series_EXPLORATORY.csv"), "EXPLORATORY (2016-07 to 2018-06 test data): detection statistic / threshold for the 02GD011 H step through 02GD016 H, with current and added redundant inputs.", "fault_ar_02GD011_series_EXPLORATORY.csv")
    SB = pd.read_csv(V3R / "I_synthetic_benchmark.csv"); SB["gap_greedy_pct"] = (SB.greedy - SB.ilp) / SB.ilp * 100; SB["gap_nsga_pct"] = (SB.nsga - SB.ilp) / SB.ilp * 100; out("fig21_algorithm_comparison", SB, "Synthetic dendritic benchmark: sizes 50-1000 nodes, 0-30% non-tree edges, 5 replicates; runtimes (s), objective values and gaps (unchanged from ress_v3).", "ress_v3/results/I_synthetic_benchmark.csv")
    # final design map: k=2 with statistical links (repaired)
    f2 = VD["ms_k2_P2"]["repaired"]; act = set(f2["active"]); crit = IN[(IN.design == "current")].sort_values("criticality_network", ascending=False).head(15).station.tolist()
    nodes = pd.DataFrame([dict(station=s, role=cm.roles()[s], lat_deg=meta.loc[s, "latitude"], lon_deg=meta.loc[s, "longitude"], status="kept" if s in act else "removed_redundant", critical_top15_current=s in crit) for s in net.d.stations])
    out("fig22a_design_map_nodes", nodes, "Stations of the verified k=2 design with statistical links: kept or removed (redundant), and top-15 critical stations of the current network (multi-state criticality).", "verified_designs.pkl; ms_importance_network_final.csv")
    lrows = []
    for k in keys:
        for s in f2["inputs"][k]:
            if s not in dirs[k]:
                lrows.append(dict(target=k[0], var=k[1], added_input=s, kind="physical_ancestor" if s in net.mem[k]["connected"] else "statistical", target_lat=meta.loc[k[0], "latitude"], target_lon=meta.loc[k[0], "longitude"], input_lat=meta.loc[s, "latitude"], input_lon=meta.loc[s, "longitude"]))
    out("fig22b_design_map_added_links", pd.DataFrame(lrows), "Links added in the verified k=2 design (series level) with endpoint coordinates.", "verified_designs.pkl")
    JN = pd.read_csv(V3R / "J_new_node_candidates.csv", dtype={"target": str}); JN["still_below_k2_v4_design"] = JN.target.isin({k[0] for k in net.keys if f"{k[0]}|{k[1]}" in VD["ms_k2_P2"]["unrepaired"]})
    out("fig22c_candidate_new_node_reaches", JN.rename(columns={"lat": "lat_deg", "lon": "lon_deg"}), "Candidate HydroRIVERS reaches for new stations on targets with k < 2 (graph-level recommendation, NOT validated by data; unchanged from ress_v3) with a flag for targets still failing in the verified k=2 design.", "ress_v3/results/J_new_node_candidates.csv")
    for nm_, f in (("fig23a_rbd_faulttree_k1_target", "p2_rbd_faulttree_k1_target.json"), ("fig23b_rbd_faulttree_k2_target", "p2_rbd_faulttree_k2_target.json")): shutil.copy(R / f, FD / f"{nm_}.json"); DESC[nm_] = ("Reliability block diagram (parallel of series blocks = minimal path sets) and fault tree (OR of AND cut sets) with basic-event probabilities and top-event probabilities (analytical, Monte Carlo, real joint days).", f, ["json"], 1)
    G, _ = cm.physical_forest(); nn = NT.merge(coords, left_on="station", right_index=True)[["station", "role", "lat_deg", "lon_deg", "component", "depth_up", "n_ancestors"]]; out("fig24a_graph_nodes", nn, "Graph nodes: id, type (source/relay/sink), coordinates, component, chain depth.", "ress_v3/results/A_node_table.csv")
    PE = pd.read_csv(V3R / "A_physical_edges.csv", dtype={"upstream": str, "downstream": str}); SE_ = pd.read_csv(V3R / "A_statistical_edges.csv", dtype={"source": str, "target": str})
    ed = [dict(from_station=r.upstream, to_station=r.downstream, type="physical", river_link=r.river_link, in_forest=(r.upstream, r.downstream) in set(G.edges)) for r in PE.itertuples()] + [dict(from_station=r.source, to_station=r.target, type="statistical", var=r.var, abs_corr_train=r.abs_corr_train) for r in SE_.itertuples()]
    out("fig24b_graph_edges", pd.DataFrame(ed), "Graph edges: physical (frozen + HydroRIVERS; in_forest False = non-tree link) and statistical (top-5 correlated, 2005-2012).", "ress_v3/results/A_physical_edges.csv; A_statistical_edges.csv")
    # README_figdata
    md = ["# figdata - data behind the planned figures", "", "All coordinates are WGS84 degrees (`lat_deg`, `lon_deg`). Probabilities/availabilities are fractions (0-1); `*_days` are days; NSE is dimensionless; thresholds: G2 NSE >= %.2f, G1 >= %.2f." % (K.GOOD, K.OK), "Long format: one row per observation; the `design` column takes values `current`, `k1_P1`, `k1_P2`, `k2_P1`, `k2_P2`, `k3_P2` (P1 = physical inputs only, P2 = with statistical links; k1/k2 are four-model verified and repaired, k3 is Ridge-only).", ""]
    for n, (desc, src, cols, nrow) in sorted(DESC.items()): md += [f"## {n}", f"{desc}", f"- source: `{src}`; rows: {nrow}", f"- columns: {', '.join(cols)}", ""]
    (FD / "README_figdata.md").write_text("\n".join(md), encoding="utf-8")
    # ------------------------------------------------------------------ numbers registry ----------------------------------------------------------------------------------------------------------------------------------------
    nr = NR.set_index("design"); cur = nr.loc["current"]; sp = pd.read_csv(R / "ms_spof_summary_final.csv").set_index("design"); ts = pd.read_csv(R / "ms_threshold_sensitivity_final.csv")
    reg("G2_threshold_NSE", K.GOOD, "NSE", "rv4/config.py", "config"); reg("G1_threshold_NSE", K.OK, "NSE", "rv4/config.py", "config")
    for dn in ("current", "k1_P1", "k1_P2", "k2_P1", "k2_P2", "k3_P2"):
        x = nr.loc[dn]; reg(f"network_reliability_ge_G1_attainable_{dn}", round(x.network_reliability_attainable, 4), "probability", "ms_network_reliability_final.csv", "mean over attainable series and days of 1[state>=G1], real joint outage states 2005-2016")
        reg(f"network_reliability_ge_G1_attainable_ci_{dn}", f"{x.ci_lo_att:.4f}-{x.ci_hi_att:.4f}", "probability", "ms_network_reliability_final.csv", "30-day block bootstrap 95% CI (300 resamples)")
        reg(f"P_all_attainable_series_ge_G1_{dn}", round(x.P_all_attainable_ge_G1, 4), "probability", "ms_network_reliability_final.csv", "share of days on which every attainable series is >= G1")
        reg(f"P_G2_mean_{dn}", round(x.P_G2_mean, 4), "probability", "ms_network_reliability_final.csv", "mean P(G2) over all 114 series")
        reg(f"n_active_stations_{dn}", int(x.n_active_stations), "stations", "ms_network_reliability_final.csv", "distinct stations used as inputs plus targets")
    for dn in ("current", "k1_P1", "k1_P2", "k2_P1", "k2_P2", "k3_P2"):
        x = sp.loc[dn]; reg(f"n_spof_series_{dn}", int(x.n_spof_series), "series", "ms_spof_summary_final.csv", "series with all-up state >= G1 whose single input failure gives G0"); reg(f"n_spof_stations_{dn}", int(x.n_spof_stations), "stations", "ms_spof_summary_final.csv", "distinct stations that are a SPOF for >= 1 series"); reg(f"n_targets_with_spof_{dn}", int(x.n_targets_with_spof), "targets", "ms_spof_summary_final.csv", "targets with a SPOF in H or Q")
    reg("graph_targets_with_in_degree_1", int(sp.loc["current", "graph_targets_in_degree_1"]), "targets", "ress_v3/results/A_node_table.csv", "frozen links, targets with exactly one upstream station")
    reg("v3_binary_rule_spof_series_current", 112, "series", "ress_v3/results/E_spof_summary.csv", "v3 demand = current NSE - tolerance (replaced by the multi-state rule)")
    for _, r in ts.iterrows():
        if r.design in ("current", "k2_P2"): reg(f"sensitivity_{r.thresholds}_{r.design}_network_reliability", round(r.network_reliability_attainable, 4), "probability", "ms_threshold_sensitivity_final.csv", "same computation with thresholds G2/G1 = " + r.thresholds); reg(f"sensitivity_{r.thresholds}_{r.design}_n_spof_series", int(r.n_spof_series), "series", "ms_threshold_sensitivity_final.csv", "same computation with thresholds G2/G1 = " + r.thresholds)
    for _, r in REC.iterrows():
        reg(f"stations_removed_{r.design}_{r.version}", int(r.stations_removed), "stations", "reconciliation_table.csv", "119 - active stations"); reg(f"links_added_{r.design}_{r.version}", int(r.links_added), "links", "reconciliation_table.csv", "series-level inputs that are not frozen direct links")
        reg(f"network_reliability_{r.design}_{r.version}", round(r.network_reliability_ge_G1, 4), "probability", "reconciliation_table.csv", "as above on attainable series")
    for _, r in VR.reset_index().iterrows(): reg(f"series_failing_four_model_rules_raw_{r.design}", int(r.n_fail_raw), "series", "verification_repair_summary.csv", "R-v2 and R-ms (see s2_verify_repair.py)"); reg(f"series_failing_after_repair_{r.design}", int(r.n_fail_after_repair), "series", "verification_repair_summary.csv", "after minimum repair")
    fp = pd.read_csv(R / "fault_ar_primary.csv")
    for _, r in fp[fp.kind.isin(["step"])].iterrows(): reg(f"fault_detection_rate_{r.group}_step_{r.magnitude}", round(r.detection_rate, 3), "fraction", "fault_ar_primary.csv", "primary method (B for k=1, C for k>=2); 6 onsets x inputs x series, validation 2013-2016")
    for g in ("k=1", "k=2", "k>=3"):
        x = FA[(FA.group == g)]; m = "B" if g == "k=1" else "C"; y = x[x.method == m]; reg(f"fault_false_alarm_rate_validation_{g}", round(y.false_alarm_rate_validation.iloc[0], 4), "fraction", "fault_ar_by_k.csv", "fault-free validation days above the threshold set for 1% on 2005-2012")
    reg("fault_isolation_rate_k>=3_LOO", round(FA[(FA.group == "k>=3") & (FA.method == "C")].isolation_rate_loo.mean(), 3), "fraction", "fault_ar_by_k.csv", "leave-one-out agreement at the alarm day"); reg("fault_isolation_rate_k=2_with_target", round(FA[(FA.group == "k=2") & (FA.method == "C")].isolation_rate_k2_with_target.mean(), 3), "fraction", "fault_ar_by_k.csv", "tie-break with the target observation")
    wf = WF[(WF.variable == "either") & (WF.quantity == "time_between_failures_days") & WF["shape"].notna()]; reg("weibull_tbf_median_shape", round(wf["shape"].median(), 3), "-", "p2_weibull_fits_station.csv", "median over stations with >=5 time-between-failure samples"); reg("weibull_tbf_share_shape_lt_1", round((wf["shape"] < 1).mean(), 3), "fraction", "p2_weibull_fits_station.csv", "share of fitted stations with shape < 1"); reg("weibull_tbf_n_stations_fitted", len(wf), "stations", "p2_weibull_fits_station.csv", "n >= 5")
    reg("weibull_tbf_median_AIC_advantage_over_exponential", round(wf.weibull_better_by_AIC.median(), 2), "AIC units", "p2_weibull_fits_station.csv", "AIC_exponential - AIC_weibull, median"); reg("availability_median_either", round(AV[AV.variable == "either"].availability.median(), 4), "fraction", "p2_availability_mtbf_mttr.csv", "median over 119 stations (H or Q present)")
    mc = pd.read_csv(R / "p2_mc_convergence.csv"); last = mc[(mc.design == "current") & (mc.method == "independent_MC")].iloc[-1]; reg("mc_independent_current_N20000_estimate", round(last.estimate, 4), "probability", "p2_mc_convergence.csv", "running mean at N = 20000"); reg("mc_independent_current_N20000_ci_halfwidth", round((last.ci_hi - last.ci_lo) / 2, 5), "probability", "p2_mc_convergence.csv", "1.96 se")
    ft = pd.read_csv(R / "p2_faulttree_summary.csv")
    for _, r in ft.iterrows(): reg(f"fault_tree_top_event_{r.target.replace(' ', '_')}_{r.design}_analytical", r.P_top_analytical, "probability", "p2_faulttree_summary.csv", "exact enumeration, independent basic events"); reg(f"fault_tree_top_event_{r.target.replace(' ', '_')}_{r.design}_MC", r.P_top_MC_independent, "probability", "p2_faulttree_summary.csv", "N=200000 independent")
    dl = pd.read_csv(R / "design_levels_ms.csv"); dl = dl[dl.thresholds == f"{K.GOOD}_{K.OK}"]
    for _, r in dl.iterrows(): reg(f"ms_design_k{r.level}_{r.pool}_removed_RidgeOnly_raw", int(r.ilp_removed), "stations", "design_levels_ms.csv", "exact ILP under multi-state G1 rule, Ridge fallback tables (not four-model verified)")
    reg("tree_dp_equals_ilp_components", f"{int(pd.read_csv(R / 'tree_dp_vs_ilp_ms.csv').match.sum())}/{len(pd.read_csv(R / 'tree_dp_vs_ilp_ms.csv'))}", "components x levels", "tree_dp_vs_ilp_ms.csv", "tree DP optimum equals ILP optimum")
    RG = pd.DataFrame(REG); RG.to_csv(rv4.V4 / "numbers_registry.csv", index=False); print(len(RG), "registry rows")
    # ------------------------------------------------------------------ workbook ---------------------------------------------------------------------------------------------------------------------------------------------
    with pd.ExcelWriter(rv4.V4 / "RESS_Results_v4.xlsx", engine="openpyxl") as xw:
        RG.to_excel(xw, sheet_name="Headline_numbers", index=False); REC.to_excel(xw, sheet_name="Reconciliation", index=False)
        for f in sorted(R.glob("*.csv")):
            if f.stem in ("reconciliation_table",): continue
            df = pd.read_csv(f); nm = f.stem.replace("ms_", "ms").replace("p2_", "p2").replace("fault_ar_", "far_")[:31]
            if len(df) > 100000: df = df.head(100000)
            df.to_excel(xw, sheet_name=nm, index=False)
    print("workbook written")

if __name__ == "__main__":
    main()
