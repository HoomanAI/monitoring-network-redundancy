"""Verify the tree DP against exhaustive enumeration, then run both levers."""
import sys, time, itertools, numpy as np, pandas as pd, networkx as nx
sys.path.insert(0,'/home/claude/work'); sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
from rrap_h import RRAPH, TH, RHO
from rv3 import common as cm
from pathlib import Path
OUT=Path('/home/claude/work/out')
P=RRAPH(); net=P.net
G,_=cm.physical_forest()
comps=[sorted(c) for c in nx.weakly_connected_components(G)]
print("physical forest: %d components"%len(comps))

def exhaustive(design, comp, m):
    ks=[k for k in P.keys if net.full_nse(design,k)>=TH and k[0] in comp]
    cand=[s for s in comp if P.gain[s]>1e-9]
    best=-1
    for r in range(m+1):
        for H in itertools.combinations(cand,r):
            v=sum(P.series_P(k,net.design[design][k],frozenset(H)) for k in ks)
            if v>best: best=v
    return best

print("\n=== DP vs exhaustive, physical design (current), m = 2 ===")
rows=[]
for ci,comp in enumerate(comps):
    ks=[k for k in P.keys if net.full_nse('current',k)>=TH and k[0] in comp]
    if not ks: continue
    cand=[s for s in comp if P.gain[s]>1e-9]
    m=2
    if len(cand)>14: continue                       # keep the exhaustive check affordable
    t0=time.time(); dpv,_=P.tree_dp('current',set(comp),G,m); t_dp=time.time()-t0
    t0=time.time(); exv=exhaustive('current',comp,m); t_ex=time.time()-t0
    rows.append(dict(component=ci+1,n_nodes=len(comp),n_series=len(ks),n_candidates=len(cand),
                     dp=dpv,exhaustive=exv,match=abs(dpv-exv)<1e-9,dp_s=t_dp,exhaustive_s=t_ex))
V=pd.DataFrame(rows); V.to_csv(OUT/'x7_treedp_vs_exhaustive.csv',index=False)
print(V[['component','n_nodes','n_series','n_candidates','dp','exhaustive','match','dp_s','exhaustive_s']].round(6).to_string(index=False))
print("\nDP == exhaustive on %d/%d components; DP total %.3fs vs exhaustive %.3fs"
      %(int(V.match.sum()),len(V),V.dp_s.sum(),V.exhaustive_s.sum()))

# ===== exact global optimum on the physical forest: per-component DP + knapsack across components
def forest_optimum(design, M):
    per=[]
    ks_all=[k for k in P.keys if net.full_nse(design,k)>=TH]
    covered=set()
    for comp in comps:
        ks=[k for k in ks_all if k[0] in comp]
        if not ks: continue
        covered|=set(ks)
        _,dp=P.tree_dp(design,set(comp),G,M)
        best=[max(dp[b]) for b in range(M+1)]
        for b in range(1,M+1): best[b]=max(best[b],best[b-1])      # monotone in budget
        per.append(best)
    NEG=-1e18; acc=[NEG]*(M+1); acc[0]=0.0
    for best in per:
        nxt=[NEG]*(M+1)
        for b in range(M+1):
            if acc[b]==NEG: continue
            for bb in range(M+1-b):
                t=acc[b]+best[bb]
                if t>nxt[b+bb]: nxt[b+bb]=t
        acc=nxt
    n=len(covered)
    return [acc[b]/n for b in range(M+1)], n

M=12
print("\n=== lever 2 on the physical network: exact tree DP vs greedy ===")
t0=time.time(); dp_curve,nser=forest_optimum('current',M); t_dp=time.time()-t0
t0=time.time(); _,tr=P.greedy('current',M); t_gr=time.time()-t0
gr_curve=[v for _,v,_ in tr]
rows=[]
for b in range(M+1):
    gap=100*(dp_curve[b]-gr_curve[b])/dp_curve[b] if dp_curve[b] else 0
    rows.append(dict(budget_m=b,R_tree_dp=dp_curve[b],R_greedy=gr_curve[b],gap_pct=gap))
F=pd.DataFrame(rows); F.to_csv(OUT/'x7_lever2_forest.csv',index=False)
print(F.round(5).to_string(index=False))
print("tree DP %.2fs (exact, %d series) | greedy %.2fs | max gap %.4f%%"%(t_dp,nser,t_gr,F.gap_pct.max()))

# ===== joint two-lever frontier
print("\n=== joint frontier: redundancy level x hardening budget ===")
rows=[]
for dn in ['current','k1_P1','k2_P1','k1_P2','k2_P2']:
    ks=[k for k in P.keys if net.full_nse(dn,k)>=TH]
    used=sorted({s for k in P.keys for s in net.design[dn][k]})
    _,tr=P.greedy(dn,M)
    for b,v,s in tr:
        rows.append(dict(design=dn,n_active_stations=len(set(used)|set(P.d.targets)),
                         budget_m=b,R=v,last_hardened=s,
                         n_series_attainable=len(ks)))
J=pd.DataFrame(rows); J.to_csv(OUT/'x7_joint_frontier.csv',index=False)
piv=J.pivot(index='budget_m',columns='design',values='R')
print(piv[['current','k2_P1','k1_P2','k2_P2']].round(4).to_string())
cur0=piv.loc[0,'current']; k2=piv.loc[0,'k2_P2']
need=[b for b in piv.index if piv.loc[b,'current']>=k2]
print("\ncurrent R=%.4f ; k2_P2 R=%.4f"%(cur0,k2))
print("hardening alone reaches k2_P2 at m =", need[0] if need else ">%d (never within budget)"%M)
print("hardening 5 stations on the current network: %.4f (+%.4f)"%(piv.loc[5,'current'],piv.loc[5,'current']-cur0))
print("hardening 5 stations on the k=2 design:     %.4f (+%.4f)"%(piv.loc[5,'k2_P2'],piv.loc[5,'k2_P2']-k2))
first=J[(J.design=='current')&(J.budget_m.between(1,5))].last_hardened.tolist()
print("first 5 stations the greedy hardens (current):", first)
