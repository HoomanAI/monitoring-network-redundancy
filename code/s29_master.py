# -*- coding: utf-8 -*-
"""One evaluation pipeline for every design reported in the paper, so that
Table 1 and Table 2 share a single source and a single bootstrap."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v3/scripts')
import rv4
from rv4.core import MSNet, min_cut
from rv4.design import Problem
from rv4 import config as K
from f_resilience import episodes
from pathlib import Path
OUT=Path('out'); RHO=0.5; TH=K.OK
rng=np.random.default_rng(K.SEED)
net=MSNet(designs_file="designs_ms_final.pkl"); d=net.d
STN={s:i for i,s in enumerate(d.stations)}
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm24=[s for s in rm27 if s not in ('02CF013','02ED015','02HD021')]
rm23=[s for s in rm24 if s!='02GD004']          # 02GD001 H exempted
def build(removed):
    cfg=P.configure(set(d.stations)-set(removed),P.prep(2,False))
    return {k:(cfg.get(k) or list(net.mem[k]["direct"])) for k in net.keys}
net.design['k2_P2_rec']=build(rm24)
net.design['k2_P2_rec23']=build(rm23)
net.design['k2_P2_hard']=dict(net.design['k2_P2'])
HARD=set(pd.read_csv(OUT/'x7_joint_frontier.csv').query("design=='k2_P2' and budget_m<=8").last_hardened.dropna())
REMOVED={'current':[], 'k1_P1':sorted(net.designs_loaded['k1_P1']['removed']),
         'k2_P1':sorted(net.designs_loaded['k2_P1']['removed']),
         'k1_P2':sorted(net.designs_loaded['k1_P2']['removed']),
         'k2_P2':rm27,'k2_P2_rec':rm24,'k2_P2_rec23':rm23,'k2_P2_hard':rm27,
         'k3_P2':sorted(net.designs_loaded['k3_P2']['removed'])}
DES=['current','k1_P1','k2_P1','k1_P2','k2_P2','k2_P2_rec23','k2_P2_rec','k2_P2_hard','k3_P2']
HARDOF={dn:(HARD if dn=='k2_P2_hard' else set()) for dn in DES}

def runs(col):
    dn=~col; idx=np.flatnonzero(np.diff(np.concatenate(([0],dn.view(np.int8),[0])))!=0)
    return [(idx[i],idx[i+1]-idx[i]) for i in range(0,len(idx),2)]
def U_of(hard):
    if not hard: return net.U
    out={}
    for v in ('H','Q'):
        A=net.U[v].copy(); T=A.shape[0]
        for s in hard:
            i=STN[s]; new=np.ones(T,bool)
            for a,L in runs(net.U[v][:,i]): new[a:min(T,a+max(1,int(round(L*RHO))))]=False
            A[:,i]=new
        out[v]=A
    return out
# one fixed set of bootstrap block draws, shared by every design
T=net.U['H'].shape[0]; blocks=np.array_split(np.arange(T),T//30); B=300
BOOT=[np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]) for _ in range(B)]
def ci(X):
    m=np.array([X[idx].mean() for idx in BOOT])
    return float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
Qm=np.nanmean(np.column_stack([d.mod_arr['Q'][s] for s in d.stations]),axis=1)
HI=Qm>=np.nanquantile(Qm,0.90)
def links(dn): return sum(1 for k in net.keys for s in net.design[dn][k] if s not in net.mem[k]["direct"])
used=sorted({s for k in net.keys for s in net.design['current'][k]})
ep,_,_=episodes(net,used)
rows=[]
for dn in DES:
    hard=HARDOF[dn]; U=U_of(hard)
    ks=[k for k in net.keys if net.full_nse(dn,k)>=TH]
    X=(net.nse_matrix(dn,U=U,series=ks)>=TH).astype(np.float32)
    R=float(X.mean()); lo,hi=ci(X)
    own=np.column_stack([U[v][:,STN[t]] for (t,v) in ks])
    cuts=[c for c in (min_cut(net,dn,k,(K.GOOD,K.OK))[0] for k in ks) if c]
    per=X.mean(1)
    rows.append(dict(design=dn,removed=len(REMOVED[dn]),links=links(dn),n_attainable=len(ks),
        R=R,ci_lo=lo,ci_hi=hi,days_all_G1=float((X.mean(1)==1).mean()),
        R_highflow=float(X[HI].mean()),
        unserved_per_1e4=1e4*float(((~own)&(X<1)).mean()),
        spof=sum(1 for c in cuts if c==1),median_mincut=float(np.median(cuts)),
        resilience_loss=float(np.mean([ (1.0-per[c-45:c+91]).sum() for c in ep])),
        removed_stations=','.join(REMOVED[dn])))
    print("%-12s rm %2d links %3d  R=%.4f [%.4f,%.4f]  allG1=%.3f  Rhi=%.4f  unsv=%.2f  SPOF=%2d  cut=%.0f  loss=%.2f"%(
        dn,len(REMOVED[dn]),links(dn),R,lo,hi,rows[-1]['days_all_G1'],rows[-1]['R_highflow'],
        rows[-1]['unserved_per_1e4'],rows[-1]['spof'],rows[-1]['median_mincut'],rows[-1]['resilience_loss']))
M=pd.DataFrame(rows); M.to_csv(OUT/'x22_master_designs.csv',index=False)
print("\nwrote x22_master_designs.csv")
