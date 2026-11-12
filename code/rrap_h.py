# -*- coding: utf-8 -*-
"""RRAP-H: joint redundancy allocation and component hardening.
Lever 1 (x): which stations to retain / which links to use  -> structural, k-multicover.
Lever 2 (h): which retained stations to harden (cut MTTR by RHO) -> changes availability only.
Exact tree DP on the physical forest; Birnbaum-guided greedy with exact re-evaluation elsewhere."""
import sys, itertools, numpy as np, pandas as pd, networkx as nx
from pathlib import Path
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
import rv4
from rv4.core import MSNet
from rv4 import config as K
from rv3 import common as cm

RHO = 0.5                      # hardening halves mean time to repair
TH  = K.OK

class RRAPH:
    def __init__(self, designs_file="designs_ms_final.pkl"):
        self.net = MSNet(designs_file=designs_file)
        self.keys = self.net.keys; self.d = self.net.d
        A = pd.read_csv(rv4.RES/'p2_availability_mtbf_mttr.csv', dtype={'station':str})
        self.A0, self.A1 = {}, {}
        for v in ('H','Q'):
            emp = self.net.U[v].mean(0)                       # empirical, matches the paper's independent MC
            Av = A[A.variable==v].set_index('station')
            for i,s in enumerate(self.d.stations):
                a0 = float(emp[i]); self.A0[(s,v)] = a0
                if s in Av.index and np.isfinite(Av.loc[s,'MTBF_days']) and np.isfinite(Av.loc[s,'MTTR_days']):
                    mtbf,mttr = float(Av.loc[s,'MTBF_days']), float(Av.loc[s,'MTTR_days'])
                    a1 = mtbf/(mtbf+RHO*mttr)
                    self.A1[(s,v)] = max(a0, min(a1, 1.0))    # hardening cannot lower availability
                else:
                    self.A1[(s,v)] = a0                       # no outages observed -> nothing to harden
        self.gain = {s: max(self.A1[(s,'H')]-self.A0[(s,'H')], self.A1[(s,'Q')]-self.A0[(s,'Q')])
                     for s in self.d.stations}

    # ---- exact series reliability under the independent law -------------------------------
    def _feas(self, k, inputs):
        """boolean over the 2^n masks of `inputs`: does the fallback model reach G1?"""
        mm = self.net.mem[k]["members"]; n = len(inputs)
        bit = [1 << mm.index(s) for s in inputs]
        masks = np.zeros(1<<n, np.int64)
        for i,b in enumerate(bit):
            idx = np.arange(1<<n); masks |= np.where((idx>>i)&1, b, 0)
        return self.net.T[k][masks] >= TH
    def series_P(self, k, inputs, hard):
        """P(state >= G1) for series k with these inputs, `hard` = set of hardened stations."""
        n = len(inputs); v = k[1]
        if n == 0: return 0.0
        f = self._cache_feas(k, tuple(inputs))
        a = np.array([(self.A1 if s in hard else self.A0)[(s,v)] for s in inputs])
        idx = np.arange(1<<n); p = np.ones(1<<n)
        for i in range(n):
            up = ((idx>>i)&1).astype(bool)
            p *= np.where(up, a[i], 1-a[i])
        return float(p[f].sum())
    _fc = {}
    def _cache_feas(self, k, inputs):
        key=(k,inputs)
        if key not in self._fc: self._fc[key]=self._feas(k,list(inputs))
        return self._fc[key]

    def R(self, design, hard=frozenset(), series=None):
        ks = series or [k for k in self.keys if self.net.full_nse(design,k) >= TH]
        return float(np.mean([self.series_P(k, self.net.design[design][k], hard) for k in ks]))

    # ---- lever 2, general graph: greedy with exact re-evaluation ---------------------------
    def greedy(self, design, m, cand=None):
        ks = [k for k in self.keys if self.net.full_nse(design,k) >= TH]
        used = sorted({s for k in self.keys for s in self.net.design[design][k]})
        cand = [s for s in (cand or used) if self.gain[s] > 1e-9]
        hard=set(); trace=[(0, self.R(design,frozenset(),ks), None)]
        for _ in range(m):
            best=None
            for s in cand:
                if s in hard: continue
                v=self.R(design, frozenset(hard|{s}), ks)
                if best is None or v>best[0]: best=(v,s)
            if best is None: break
            hard.add(best[1]); trace.append((len(hard),best[0],best[1]))
        return hard, trace

    def birnbaum_bound(self, design, m):
        """first-order upper estimate: rank by I_B * availability gain"""
        ks=[k for k in self.keys if self.net.full_nse(design,k) >= TH]
        used=sorted({s for k in self.keys for s in self.net.design[design][k]})
        R0=self.R(design,frozenset(),ks); est=[]
        for s in used:
            if self.gain[s]<=1e-9: continue
            est.append((self.R(design,frozenset({s}),ks)-R0, s))
        est.sort(reverse=True)
        return R0+sum(e for e,_ in est[:m]), [s for _,s in est[:m]]

    # ---- lever 2, physical forest: exact tree DP ------------------------------------------
    def tree_dp(self, design, comp, G, m):
        """Exact max of the summed series reliability over hardening sets of size <= m,
        on one in-tree component. State per node: hardened or not (it is an input to its parent)."""
        ks={k for k in self.keys if self.net.full_nse(design,k)>=TH and k[0] in comp}
        tgt_of={}
        for k in ks: tgt_of.setdefault(k[0],[]).append(k)
        root=[n_ for n_ in comp if G.out_degree(n_)==0][0]
        NEG=-1e18
        def solve(v):
            """returns dp[b][s] = best summed reliability in subtree(v) using b hardenings, v in state s"""
            kids=sorted(G.predecessors(v))
            sub=[solve(c) for c in kids]
            own=tgt_of.get(v,[])
            dp=[[NEG]*2 for _ in range(m+1)]
            for combo in itertools.product([0,1],repeat=len(kids)):
                hard=frozenset([kids[i] for i,c in enumerate(combo) if c])
                val_own=sum(self.series_P(k,self.net.design[design][k],hard) for k in own)
                acc=[NEG]*(m+1); acc[0]=0.0
                for i,c in enumerate(kids):
                    nxt=[NEG]*(m+1)
                    for b in range(m+1):
                        if acc[b]==NEG: continue
                        for bb in range(m+1-b):
                            if sub[i][bb][combo[i]]==NEG: continue
                            t=acc[b]+sub[i][bb][combo[i]]
                            if t>nxt[b+bb]: nxt[b+bb]=t
                    acc=nxt
                for b in range(m+1):
                    if acc[b]==NEG: continue
                    tot=acc[b]+val_own
                    for s in (0,1):
                        cost=b+(1 if s else 0)
                        if cost<=m and tot>dp[cost][s]: dp[cost][s]=tot
            if self.gain.get(v,0)<=1e-9:
                for b in range(m+1): dp[b][1]=NEG          # nothing to harden
            return dp
        dp=solve(root)
        return max(max(dp[b]) for b in range(m+1)), dp
