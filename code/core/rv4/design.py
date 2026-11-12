"""Redundancy-allocation core: k-redundancy feasibility from the fallback tables, minimal feasible input sets, exact tree DP, exact ILP,
greedy cover, evaluation of any station set.

Definitions (see README):
  series j = (target t, variable v); members_j = candidate inputs (capped pool); demand_j = base NSE - tolerance.
  A set X of inputs is k-redundant for series j iff NSE[X \\ F] >= demand_j for EVERY failure set F with |F| <= k-1.
  A station set A is feasible at level k iff every constrained series has some k-redundant X subset of members_j that are active in A.
  Cost: number of active REMOVABLE stations (primary), number of extra links = inputs used that are not frozen direct links (secondary)."""
import numpy as np, pickle, time, itertools
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import networkx as nx
import rv3
import rv4
from rv3 import common as cm
from ssopt import config as C

INF = 10 ** 6

def popcount(n):
    pc = np.zeros(1 << n, np.int16)
    for i in range(n): pc[1 << i: 1 << (i + 1)] = pc[: 1 << i] + 1
    return pc

def feas_levels(nse, demand, n, kmax=3):
    """feas[k][mask] for k=1..kmax (mask over n members); mask = set of active inputs."""
    f = {1: nse >= demand}
    idx = np.arange(1 << n)
    for k in range(2, kmax + 1):
        g = f[k - 1].copy(); res = f[1].copy()
        for i in range(n):
            has = (idx >> i) & 1 == 1
            sub = np.where(has, g[idx ^ (1 << i)], True)       # drop member i (if present) -> must still be (k-1)-redundant
            res &= sub
        f[k] = res
    return f

def subset_or(feas, n):
    F = feas.copy(); idx = np.arange(1 << n)
    for i in range(n):
        has = (idx >> i) & 1 == 1; F = np.where(has, F | F[idx ^ (1 << i)], F)
    return F

def subset_min(vals, n):
    V = vals.copy(); idx = np.arange(1 << n)
    for i in range(n):
        has = (idx >> i) & 1 == 1; V = np.where(has, np.minimum(V, V[idx ^ (1 << i)]), V)
    return V

class Problem:
    def __init__(self, kmax=3, th=None):
        Z = pickle.load(open(rv3.CACHE / "fallback_tables.pkl", "rb")); self.T, self.mem = Z["tables"], Z["members"]
        d = pickle.load(open(rv3.CACHE / "design_meta.pkl", "rb")) if (rv3.CACHE / "design_meta.pkl").exists() else None
        tol = cm.tolerance_val(); self.keys = sorted(self.mem)
        self.base = {}; self.demand = {}
        for k in self.keys:
            mm = self.mem[k]["members"]; m0 = sum(1 << i for i, s in enumerate(mm) if s in self.mem[k]["direct"])
            self.base[k] = float(self.T[k][m0]); self.demand[k] = self.base[k] - tol[(k[0], k[1], "Ridge")]
        S = pickle.load(open(C.CACHE / "sets.pkl", "rb")); import json
        add = json.load(open(C.ROOT / "followup_v2/results/a4_must_keep_additions.json"))["extra_must_keep_from_diagnosis"]
        if th is not None:
            self.demand = {k: float(th) for k in self.keys}          # multi-state: absolute NSE threshold (G1 boundary)
        self.must = sorted(set(S["must"]) | {"02KF013"} | set(add)); self.removable = [s for s in S["removable"] if s not in self.must]
        self.rset = set(self.removable); self.kmax = kmax
        gd = cm.physical_forest()[0]
        self.constrained = [k for k in self.keys if k not in cm.FLAGGED and not (k[1] == "H" and "02GD011" in self.mem[k]["members"])]
        if th is not None:       # series that never reach G1 even with every input up are structurally failed and cannot constrain the design
            self.structurally_failed = [k for k in self.constrained if float(np.nanmax(self.T[k])) < th]
            self.constrained = [k for k in self.constrained if k not in set(self.structurally_failed)]
        self.series = {}
        for k in self.constrained:
            mm = self.mem[k]["members"]; n = len(mm); dr = set(self.mem[k]["direct"]); fa = set(nx.ancestors(gd, k[0])) if k[0] in gd else set()
            corr = set(self.mem[k]["corr_only"]) | {x for x in mm if x not in fa and x not in dr}   # statistical or non-tree links
            pc = popcount(n); idx = np.arange(1 << n)
            phys_allowed = np.ones(1 << n, bool)
            for i, s in enumerate(mm):
                if s in corr: phys_allowed &= ((idx >> i) & 1) == 0
            extras_bits = [i for i, s in enumerate(mm) if s not in dr]
            ex = np.zeros(1 << n, np.int16)
            for i in extras_bits: ex += ((idx >> i) & 1).astype(np.int16)
            self.series[k] = dict(members=mm, n=n, feas=feas_levels(self.T[k], self.demand[k], n, kmax), phys_allowed=phys_allowed, extras=ex)

    # ---- achievable redundancy ------------------------------------------------------------------------------
    def kmax_series(self, k, physical_only):
        s = self.series[k]; best = 0
        for lvl in range(1, self.kmax + 1):
            f = s["feas"][lvl] & (s["phys_allowed"] if physical_only else True)
            if f.any(): best = lvl
        return best

    def avail_mask(self, k, active):
        mm = self.series[k]["members"]; return sum(1 << i for i, x in enumerate(mm) if (x in active))

    def prep(self, level, physical_only):
        """Per-series feasibility arrays at requested level (capped by achievable level)."""
        out = {}
        for k in self.constrained:
            s = self.series[k]; kk = min(level, self.kmax_series(k, physical_only))
            if kk == 0: out[k] = None; continue
            allowed = s["phys_allowed"] if physical_only else np.ones(1 << s["n"], bool)
            feas = s["feas"][kk] & allowed
            F = subset_or(feas, s["n"]); me = np.where(feas, s["extras"].astype(np.int32), INF).astype(np.int32); ME = subset_min(me, s["n"])
            # minimal feasible masks (no feasible proper subset)
            n = s["n"]; idx = np.arange(1 << n); proper = np.zeros(1 << n, bool)
            for i in range(n):
                has = (idx >> i) & 1 == 1; proper |= np.where(has, F[idx ^ (1 << i)], False)
            minimal = feas & ~proper
            out[k] = dict(level=kk, F=F, ME=ME, minimal=np.flatnonzero(minimal), n=n, members=s["members"], extras=s["extras"])
        return out

    # ---- evaluate any active set ---------------------------------------------------------------------------------------
    def evaluate(self, active, prep):
        active = set(active) | set(self.must); links = 0; bad = []
        for k, p in prep.items():
            if p is None: continue
            am = self.avail_mask(k, active)
            if not p["F"][am]: bad.append(k)
            else: links += int(p["ME"][am])
        return dict(n_removable_kept=len(active & self.rset), n_active=len(active | set(self.must)), feasible=not bad, n_unsatisfied=len(bad), links=links, unsatisfied=bad)

    # ---- exact ILP ---------------------------------------------------------------------------------------------------------
    def ilp(self, prep, series=None, stations=None, timelimit=600):
        ks = [k for k in (series or self.constrained) if prep[k] is not None]
        xs = sorted({s for k in ks for s in prep[k]["members"] if s in self.rset}) if stations is None else list(stations)
        xid = {s: i for i, s in enumerate(xs)}; nv = len(xs); zid = {}; wid = {}; cost = [1000.0] * nv
        for k in ks:
            p = prep[k]
            for m in p["minimal"]: zid[(k, int(m))] = nv; nv += 1; cost.append(0.0)
            for i, s in enumerate(p["members"]):
                if p["extras"][1 << i] and s not in self.mem[k]["direct"]: wid[(k, s)] = nv; nv += 1; cost.append(1.0)
        rows, cols, vals, lo, hi = [], [], [], [], []; r = 0
        for k in ks:                               # choose exactly one configuration
            p = prep[k]
            for m in p["minimal"]: rows.append(r); cols.append(zid[(k, int(m))]); vals.append(1)
            lo.append(1); hi.append(1); r += 1
            for i, s in enumerate(p["members"]):   # usage constraints: target >= sum of configurations that use member i
                need_x = s in self.rset; need_w = (k, s) in wid
                zs = [zid[(k, int(m))] for m in p["minimal"] if (int(m) >> i) & 1]
                if not zs: continue
                for tgt in ([xid[s]] if need_x else []) + ([wid[(k, s)]] if need_w else []):
                    for z in zs: rows.append(r); cols.append(z); vals.append(-1)
                    rows.append(r); cols.append(tgt); vals.append(1); lo.append(0); hi.append(np.inf); r += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
        res = milp(c=np.array(cost), constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1), options=dict(time_limit=timelimit, mip_rel_gap=0))
        if res.x is None: return dict(status=res.message, feasible=False)
        kept = [s for s in xs if res.x[xid[s]] > 0.5]
        return dict(feasible=True, n_removable_kept=len(kept), kept=kept, links=int(round(sum(res.x[v] for v in wid.values()))), status=res.message, objective=res.fun, optimal=bool(res.status == 0))

    def configure(self, active, prep):
        """Input set per series for an active station set: min-extra-links feasible subset of the available members (else all available members)."""
        active = set(active) | set(self.must); out = {}
        for k in self.keys:
            mm = self.mem[k]["members"]
            if k not in prep or prep[k] is None:
                out[k] = [s for s in self.mem[k]["direct"] if s in active] if k not in self.series else [s for s in mm if s in active and s in self.mem[k]["direct"]]; continue
            p = prep[k]; am = self.avail_mask(k, active); best = None
            for m in np.flatnonzero(p["F"] & (p["ME"] < INF)) if False else p["minimal"]:
                if int(m) & ~am == 0:
                    e_ = int(p["extras"][int(m)])
                    if best is None or e_ < best[0]: best = (e_, int(m))
            out[k] = [mm[i] for i in range(len(mm)) if (best[1] >> i) & 1] if best else [s for s in mm if s in active]
        return out

    # ---- greedy cover ---------------------------------------------------------------------------------------------------------
    def greedy(self, prep):
        active = set(self.must); cand = [s for s in self.removable]
        def unsat(act): return [k for k, p in prep.items() if p is not None and not p["F"][self.avail_mask(k, act)]]
        while True:
            u = unsat(active)
            if not u: break
            best, bs = None, -1
            for s in cand:
                if s in active: continue
                a2 = active | {s}; gain = len(u) - len(unsat(a2))
                tie = sum(1 for k in u if s in prep[k]["members"])
                sc = gain * 1000 + tie
                if sc > bs: bs, best = sc, s
            if best is None or bs <= 0: return None
            active.add(best)
        for s in sorted(active & self.rset, key=lambda s: s):          # reverse-delete pruning
            if not unsat(active - {s}): active.discard(s)
        return active

    # ---- exact tree DP (physical forest) ------------------------------------------------------------------------------------------------
    def tree_dp(self, prep, component_nodes, G):
        """Exact min number of kept removable stations for one in-tree component of the physical forest (members restricted to physical pool)."""
        sub = G.subgraph(component_nodes); root = [n for n in component_nodes if sub.out_degree(n) == 0]
        assert len(root) == 1, "component is not an in-tree"
        root = root[0]; anc = {n: nx.ancestors(sub, n) for n in component_nodes}
        ser_by_t = {}
        for k in self.constrained:
            if k[0] in component_nodes and prep[k] is not None: ser_by_t.setdefault(k[0], []).append(k)
        def members_of(t): return {s for k in ser_by_t.get(t, []) for s in prep[k]["members"]}
        down = {n: [m for m in nx.descendants(sub, n)] for n in component_nodes}
        def iface_plus(u):        # stations strictly below u needed by u's own constraint or by anything downstream
            need = set(members_of(u))
            for t in down[u]: need |= members_of(t)
            return need & anc[u]
        order = list(nx.dfs_postorder_nodes(sub.reverse(), root))
        g = {}
        for u in order:
            children = list(sub.predecessors(u)); nip = iface_plus(u)
            states = {frozenset(): 0}
            for c in children:                                  # sequential merge of child subtrees
                keep = (nip & (anc[c] | {c}))
                new = {}
                for sa, ca in states.items():
                    for sb, cb in g[c].items():
                        key = frozenset((sa | sb) & nip); cost = ca + cb
                        if key not in new or cost < new[key]: new[key] = cost
                states = new
            # decide u itself, check u's constraints
            res = {}
            u_active_opts = [True] if (u not in self.rset) else [False, True]
            down_needed = set()
            for t in down[u]: down_needed |= members_of(t)
            for sa, ca in states.items():
                for ua in u_active_opts:
                    act = set(sa) | ({u} if ua else set()) | (set(self.must) & anc[u]) | ({u} if u in self.must else set())
                    act |= {s for s in anc[u] if s in self.must}
                    ok = all(prep[k]["F"][self.avail_mask(k, act)] for k in ser_by_t.get(u, []))
                    if not ok: continue
                    key = frozenset((set(sa) | ({u} if ua else set())) & (down_needed | set())); cost = ca + (1 if (ua and u in self.rset) else 0)
                    if key not in res or cost < res[key]: res[key] = cost
            g[u] = res
            if not res: return None
        return min(g[root].values())

def timeit(fn, *a, **k):
    t = time.time(); r = fn(*a, **k); return r, time.time() - t
