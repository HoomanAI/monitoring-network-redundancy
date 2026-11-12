"""1.2 four-model verification and repair of designs.
Rules (validation 2013-2016, models fitted on 2005-2012; constrained series = 104 multi-state constrained series):
 R-v2 : loss = NSE_current,m - NSE_design,m <= tau(t,v,m) in >= 3 of 4 models                  (v2 acceptance rule, full input set)
 R-ms : NSE_design,m >= G1 boundary in >= min(3, n_ref) models, where n_ref = max number of models reaching G1 with the current inputs or with
        every pool member (so structurally weak series are not blamed on the design); for k = 2 designs every single-failure subset must also pass.
Repair (per failing series, minimum added cost, in this order): add active pool members (ranked), then restore removed stations (+1 station each),
then revert to the current-network (direct) inputs.  Raw and repaired versions are both kept.  usage: s2_verify_repair.py [run|report]"""
import sys, pickle, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import rv4
from rv4 import config as K
from rv4.design import Problem
from rv3 import common as cm
from fv2.tasks import TaskCache
from fv2.configs import MODELS

def preload(cache):
    for f in (rv4.ROOT / "followup_v2/cache/partA_val_cache.pkl", rv4.ROOT / "followup_v2/cache/partB_cache.pkl", rv4.V3 / "cache/c_recheck_cache.pkl"):
        if f.exists():
            for k, v in pickle.load(open(f, "rb")).items():
                if k[4] == "val" and k[5] == 0: cache.res.setdefault(k, v)

def key(t, v, ins, m): return (t, v, tuple(sorted(ins)), m, "val", 0)

def n_ok(cache, t, v, ins, ok):
    return sum(cache.res[key(t, v, ins, m)]["nse"] >= ok for m in MODELS)

def tasks_for(t, v, ins, level):
    sub = [tuple(ins)] + ([tuple(x for x in ins if x != y) for y in ins] if level >= 2 else [])
    return [key(t, v, s, m) for s in sub for m in MODELS]

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    P = Problem(kmax=3, th=K.OK); mem = P.mem; keys = P.constrained; tol = cm.tolerance_val(); ok = K.OK
    cache = TaskCache(rv4.CACHE / "v4_cache.pkl"); preload(cache)
    log = lambda m: open(rv4.LOG / "s2.log", "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
    D = pickle.load(open(rv4.CACHE / "designs_ms_raw.pkl", "rb")); tag = f"{K.GOOD}_{K.OK}"; v3d = pickle.load(open(rv4.V3 / "cache/optimal_designs.pkl", "rb")); v2 = cm.v2_solutions()
    designs = {}
    for lev in (1, 2):
        for pk in ("P1", "P2"):
            v = D[(tag, lev, pk)]; designs[f"ms_k{lev}_{pk}"] = dict(inputs={k: list(v["inputs"][k]) for k in P.keys}, active=set(v["active"]), level=lev, pool=pk, family="v4 multi-state")
            w = v3d[(lev, pk)]; designs[f"v3_k{lev}_{pk}"] = dict(inputs={k: list(w["inputs"][k]) for k in P.keys}, active=set(w["active"]), level=lev, pool=pk, family="v3 binary")
    for nm, lab in (("B1_S2_knee", "v2_knee"), ("B1_S1_aggressive", "v2_aggressive")):
        designs[lab] = dict(inputs={k: list(v2[nm]["inputs"][f"{k[0]}|{k[1]}"]) for k in P.keys}, active=set(v2[nm]["active_stations"]), level=1, pool="P2", family="v2 NSGA-II")
    cur = {k: sorted(mem[k]["direct"]) for k in P.keys}
    def pool_members(k, pool):
        mm = mem[k]["members"]; return [s for s in mm if (pool == "P2" or s in mem[k]["direct"] or s in mem[k]["connected"])]
    # ---- reference counts: current inputs and all pool members
    tasks = []
    for k in keys:
        for ins in (cur[k], pool_members(k, "P2"), pool_members(k, "P1")): tasks += [key(k[0], k[1], ins, m) for m in MODELS]
    for dn, dd in designs.items():
        for k in keys: tasks += tasks_for(k[0], k[1], dd["inputs"][k], dd["level"])
    cache.run(tasks, include_test=False, procs=15, log=log, label="verify-initial")
    nref = {}
    for pool in ("P1", "P2"):
        for k in keys: nref[(k, pool)] = max(n_ok(cache, k[0], k[1], cur[k], ok), n_ok(cache, k[0], k[1], pool_members(k, pool), ok))
    def check(k, ins, level, pool):
        t, v = k; r = {}
        loss = [cache.res[key(t, v, cur[k], m)]["nse"] - cache.res[key(t, v, ins, m)]["nse"] for m in MODELS]; r["v2"] = sum(l <= tol[(t, v, m)] + 1e-12 for l, m in zip(loss, MODELS)) >= 3
        need = min(3, nref[(k, pool)]); sub = [list(ins)] + ([[x for x in ins if x != y] for y in ins] if level >= 2 else [])
        r["ms"] = all(n_ok(cache, t, v, s, ok) >= need for s in sub); return r["v2"] and r["ms"], r
    rows, final = [], {}
    for dn, dd in designs.items():
        level, pool = dd["level"], dd["pool"]; inputs = {k: list(v) for k, v in dd["inputs"].items()}; active = set(dd["active"]) | set(P.must); raw_active = set(active)
        rawfail = [k for k in keys if not check(k, inputs[k], level, pool)[0]]; failing = list(rawfail); raw_inputs = {k: list(v) for k, v in inputs.items()}
        steps = {k: 0 for k in keys}; reverted = set(); rnd = 0
        while failing and rnd < 10:
            rnd += 1; new_tasks = []
            for k in failing:
                cand_all = pool_members(k, pool); cands_active = [s for s in cand_all if s not in inputs[k] and s in active]; cands_inactive = [s for s in cand_all if s not in inputs[k] and s not in active]
                if rnd <= 4 and (cands_active or cands_inactive):
                    s = (cands_active or cands_inactive)[0]
                    if s not in active: active.add(s)
                    inputs[k].append(s)
                else:
                    for s in cur[k]: active.add(s)
                    inputs[k] = list(cur[k]); reverted.add(k)
                new_tasks += tasks_for(k[0], k[1], inputs[k], level)
            cache.run(new_tasks, include_test=False, procs=15, log=log, label=f"{dn} repair round {rnd}")
            failing = [k for k in failing if not check(k, inputs[k], level, pool)[0]]
            if all(k in reverted for k in failing): break
        removed_raw = sorted(set(P.removable) - raw_active); removed_final = sorted(set(P.removable) - active)
        dir_ = {k: set(mem[k]["direct"]) for k in P.keys}
        links = lambda inp: sum(len([s for s in inp[k] if s not in dir_[k]]) for k in P.keys)
        rows.append(dict(design=dn, family=dd["family"], level=level, pool=pool, n_series_checked=len(keys), n_fail_raw=len(rawfail), n_fail_after_repair=len(failing), n_reverted_to_current_inputs=len(reverted),
                         stations_removed_raw=len(removed_raw), stations_removed_repaired=len(removed_final), stations_restored=len(set(removed_raw) - set(removed_final)), links_raw=links(raw_inputs), links_repaired=links(inputs),
                         restored_stations=",".join(sorted(set(removed_raw) - set(removed_final))), removed_raw=",".join(removed_raw), removed_repaired=",".join(removed_final), failing_series_raw=";".join(f"{k[0]} {k[1]}" for k in rawfail),
                         failing_series_after=";".join(f"{k[0]} {k[1]}" for k in failing)))
        final[dn] = dict(raw=dict(inputs=raw_inputs, active=sorted(raw_active)), repaired=dict(inputs=inputs, active=sorted(active)), level=level, pool=pool, unrepaired=[f"{k[0]}|{k[1]}" for k in failing])
        print({k_: v for k_, v in rows[-1].items() if k_ in ("design", "n_fail_raw", "n_fail_after_repair", "stations_removed_raw", "stations_removed_repaired", "links_raw", "links_repaired")}, flush=True)
    pickle.dump(final, open(rv4.CACHE / "verified_designs.pkl", "wb")); R = pd.DataFrame(rows); R.to_csv(rv4.RES / "verification_repair_summary.csv", index=False)
    # final designs for downstream analyses: repaired multi-state designs (+ raw k3 which is Ridge-only)
    out = {}
    for lev in (1, 2):
        for pk in ("P1", "P2"):
            f = final[f"ms_k{lev}_{pk}"]["repaired"]; out[(tag, lev, pk)] = dict(inputs=f["inputs"], active=f["active"], removed=sorted(set(P.removable) - set(f["active"])))
    out[(tag, 3, "P2")] = D[(tag, 3, "P2")]
    pickle.dump(out, open(rv4.CACHE / "designs_ms_final.pkl", "wb")); print(R[["design", "n_fail_raw", "n_fail_after_repair", "stations_removed_raw", "stations_removed_repaired", "links_raw", "links_repaired"]].to_string())

if __name__ == "__main__":
    main()
