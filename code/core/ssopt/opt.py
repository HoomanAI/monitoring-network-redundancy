"""Solution evaluator + pymoo problem/runner."""
import pickle
import numpy as np
import pandas as pd
from . import config as C
from . import entropy as EN
from .skill import Skill


class Evaluator:
    def __init__(self, d, pools, removable, must, tol=0.01, cv=False, obj3="entropy"):
        self.d, self.pools, self.rem, self.must = d, pools, list(removable), list(must)
        self.tol, self.cv, self.obj3 = tol, cv, obj3
        self.nrem = len(self.rem)
        self.ridx = {s: i for i, s in enumerate(self.rem)}
        self.keys = [(t, v) for t in d.targets for v in ("H", "Q")]
        self.folds = [0, 1, 2] if cv else [2]
        self.S = Skill(d)
        self.sidx = {s: i for i, s in enumerate(d.stations)}
        self.M = np.load(C.CACHE / "MI.npy"); self.Hm = np.load(C.CACHE / "Hm.npy")
        self.must_idx = np.array([self.sidx[s] for s in self.must])
        self.rem_idx = np.array([self.sidx[s] for s in self.rem])
        # per (t,v): fixed vs removable members of direct and pool input lists
        self.spec = []
        for (t, v) in self.keys:
            p = pools[(t, v)]
            dr, pl = p["direct"], p["inputs"]
            self.spec.append(dict(
                dir_fixed=[s for s in dr if s not in self.ridx], dir_rem=[self.ridx[s] for s in dr if s in self.ridx],
                pool_fixed=[s for s in pl if s not in self.ridx], pool_rem=[self.ridx[s] for s in pl if s in self.ridx],
                dr=dr, pl=pl))
        self.cache = {}
        self.tgt_pool_inputs = {t: pools[(t, "Q")]["inputs"] for t in d.targets}
        self.tidx = {t: self.sidx[t] for t in d.targets}
        # baseline: current network = frozen direct links, all stations active
        self.base = np.array([[self.S.nse(t, v, p["direct"], f) for f in self.folds] for (t, v), p in
                              ((k, pools[k]) for k in self.keys)])
        allpool = np.array([[self.S.nse(t, v, pools[(t, v)]["inputs"], f) for f in self.folds] for (t, v) in self.keys])
        self.allpool = allpool

    # skill of both input options for individual x (bool, len nrem): returns array (K, nfolds) and chosen option
    def skill_for(self, x):
        K = len(self.keys); nf = len(self.folds)
        out = np.full((K, nf), np.nan); opt = np.zeros(K, int)
        for k, ((t, v), sp) in enumerate(zip(self.keys, self.spec)):
            rd, rp = sp["dir_rem"], sp["pool_rem"]
            kd = (k, 0, x[rd].tobytes()); kp = (k, 1, x[rp].tobytes())
            a = self.cache.get(kd)
            if a is None:
                ins = sp["dir_fixed"] + [self.rem[i] for i in rd if x[i]]
                a = np.array([self.S.nse(t, v, ins, f) for f in self.folds]); self.cache[kd] = a
            b = self.cache.get(kp)
            if b is None:
                ins = sp["pool_fixed"] + [self.rem[i] for i in rp if x[i]]
                b = np.array([self.S.nse(t, v, ins, f) for f in self.folds]); self.cache[kp] = b
            if len(self.folds) == 1:
                # option chosen on validation NSE only
                pick_pool = (np.isnan(a[-1]) and not np.isnan(b[-1])) or (b[-1] > a[-1])
            else:
                # CV-constrained variant: option with the better worst-fold margin against the current network
                ma = np.nanmin(a - self.base[k]) if not np.isnan(a).any() else -9
                mb = np.nanmin(b - self.base[k]) if not np.isnan(b).any() else -9
                pick_pool = mb > ma
            out[k] = b if pick_pool else a; opt[k] = int(pick_pool)
        return out, opt

    def evaluate(self, x):
        x = np.asarray(x, bool)
        sk, opt = self.skill_for(x)
        act = np.concatenate([self.must_idx, self.rem_idx[x]])
        val = sk[:, -1]
        viol_f = np.maximum(0.0, self.base - self.tol - sk)           # (K, nf)
        viol_f = np.where(np.isnan(self.base), 0.0, np.where(np.isnan(sk), 1.0, viol_f))
        viol = viol_f.max(axis=1)
        if self.obj3 == "entropy":
            f3 = -EN.joint_entropy_cl(self.Hm, self.M, act)
        else:
            m = np.zeros(len(self.d.stations), bool); m[act] = True
            f3 = -EN.mrmr_score(self.tgt_pool_inputs, self.M, self.Hm, m, self.tidx, self.sidx)
        return dict(n_active=len(act), mean_nse=float(np.nanmean(val)), f3=f3, viol=float(viol.sum()),
                    max_viol=float(viol.max()), val=val, opt=opt, sk=sk, n_active_pool_opt=int(opt.sum()))


def run_nsga2(ev, seed, pop=C.POP, ngen=C.NGEN, log=None):
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.optimize import minimize

    n = ev.nrem

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=3, n_ieq_constr=1, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *a, **k):
            F = np.zeros((len(X), 3)); G = np.zeros((len(X), 1))
            for i, x in enumerate(X):
                r = ev.evaluate(x)
                F[i] = [r["n_active"], -r["mean_nse"], r["f3"]]; G[i, 0] = r["viol"]
            out["F"] = F; out["G"] = G

    class Samp(Sampling):
        """All-on seed solution + random individuals with keep-probability ~ U(0.35, 1)."""
        def _do(self, problem, n_samples, **kw):
            rng = np.random.default_rng(seed)
            X = np.zeros((n_samples, problem.n_var), bool); X[0] = True
            for i in range(1, n_samples):
                X[i] = rng.random(problem.n_var) < rng.uniform(0.35, 1.0)
            return X

    algo = NSGA2(pop_size=pop, sampling=Samp(), crossover=TwoPointCrossover(prob=C.CX_PROB),
                 mutation=BitflipMutation(prob=1.0, prob_var=1.0 / n), eliminate_duplicates=True)
    cb = None
    if log is not None:
        from pymoo.core.callback import Callback
        class CB(Callback):
            def notify(self, algorithm):
                g = algorithm.n_gen
                if g % 25 == 0 or g == 1:
                    F = algorithm.pop.get("F"); G = algorithm.pop.get("G")[:, 0]
                    feas = G <= 0
                    log(f"gen {g} feasible {feas.sum()}/{len(G)} min_active {F[feas,0].min() if feas.any() else 'NA'} "
                        f"cache {len(ev.cache)} fits {ev.S.misses}")
        cb = CB()
    res = minimize(Prob(), algo, ("n_gen", ngen), seed=seed, verbose=False, callback=cb)
    return res
