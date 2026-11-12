# -*- coding: utf-8 -*-
"""Master comparison of the five designs under one consistent evaluation."""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v3/scripts')
import rv4
from rv4.core import MSNet, min_cut
from rv4.design import Problem
from rv4 import config as K
from rv3 import common as cm
from ssopt import config as C
from ssopt.skill import Skill
from ssopt.models import nse
from f_resilience import episodes
from pathlib import Path
OUT=Path('/home/claude/work/out'); RHO=0.5; TH=K.OK
rng=np.random.default_rng(K.SEED)
net=MSNet(designs_file="designs_ms_final.pkl"); d=net.d
STN={s:i for i,s in enumerate(d.stations)}
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm24=[s for s in rm27 if s not in ('02CF013','02ED015','02HD021')]
cfg=P.configure(set(d.stations)-set(rm24),P.prep(2,False))
net.design['k2_P2_rec']={k:(cfg.get(k) or list(net.mem[k]["direct"])) for k in net.keys}
net.design['k2_P2_hard']=dict(net.design['k2_P2'])
HARD=set(pd.read_csv(OUT/'x7_joint_frontier.csv').query("design=='k2_P2' and budget_m<=8").last_hardened.dropna())
DES=['current','k2_P1','k2_P2','k2_P2_rec','k2_P2_hard']
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
            for a,L in runs(net.U[v][:,i]):
                new[a:min(T,a+max(1,int(round(L*RHO))))]=False
            A[:,i]=new
        out[v]=A
    return out
def block_ci(X,B=300,bl=30):
    T=X.shape[0]; blocks=np.array_split(np.arange(T),T//bl); m=np.empty(B)
    for b in range(B):
        idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]); m[b]=X[idx].mean()
    return float(np.percentile(m,2.5)),float(np.percentile(m,97.5))

# high-flow days: top 10% of network mean discharge
Qm=np.nanmean(np.column_stack([d.mod_arr['Q'][s] for s in d.stations]),axis=1)
HI=Qm>=np.nanquantile(Qm,0.90)
removed={'current':[], 'k2_P1':sorted(set(net.designs_loaded['k2_P1']['removed'])),
         'k2_P2':rm27,'k2_P2_rec':rm24,'k2_P2_hard':rm27}
def links(dn):
    return sum(1 for k in net.keys for s in net.design[dn][k] if s not in net.mem[k]["direct"])

rows=[]
for dn in DES:
    hard=HARDOF[dn]; U=U_of(hard)
    ks=[k for k in net.keys if net.full_nse(dn,k)>=TH]
    M=net.nse_matrix(dn,U=U,series=ks); X=(M>=TH).astype(np.float32)
    R=float(X.mean()); lo,hi=block_ci(X)
    allG1=float((X.mean(1)==1).mean())
    Rhi=float(X[HI].mean())
    # unserved target-days: own record missing AND estimate below G1
    own=np.column_stack([U[v][:,STN[t]] for (t,v) in ks])
    unserved=float(((~own)&(X<1)).mean())
    # SPOF and minimum cut
    cuts=[min_cut(net,dn,k,(K.GOOD,K.OK))[0] for k in ks]
    cuts=[c for c in cuts if c]
    spof=sum(1 for c in cuts if c==1); medcut=float(np.median(cuts))
    rows.append(dict(design=dn,removed=len(removed[dn]),links=links(dn),n_series=len(ks),
                     R=R,ci_lo=lo,ci_hi=hi,days_all_G1=allG1,R_highflow=Rhi,
                     unserved_rate=unserved,spof=spof,median_mincut=medcut))
    print("%-12s removed %2d links %3d  R=%.4f  allG1=%.3f  Rhi=%.4f  unserved=%.5f  SPOF=%3d  medcut=%.1f"
          %(dn,len(removed[dn]),links(dn),R,allG1,Rhi,unserved,spof,medcut))

# ---- resilience loss over the three outage episodes ----------------------
used=sorted({s for k in net.keys for s in net.design['current'][k]})
ep,cnt,sm=episodes(net,used)
for r in rows:
    dn=r['design']; U=U_of(HARDOF[dn])
    ks=[k for k in net.keys if net.full_nse(dn,k)>=TH]
    X=(net.nse_matrix(dn,U=U,series=ks)>=TH).astype(np.float32).mean(1)
    tot=0.0
    for c in ep:
        a,b=c-45,c+91; tot+=float((1.0-X[a:b]).sum())
    r['resilience_loss']=tot/len(ep)
T=pd.DataFrame(rows)
base=T[T.design=='current'].iloc[0]
for c in ('R','days_all_G1','R_highflow','unserved_rate','spof','median_mincut','resilience_loss'):
    T['d_'+c]=T[c]-base[c]
T.to_csv(OUT/'x18_design_comparison.csv',index=False)
print("\n",T[['design','R','days_all_G1','R_highflow','unserved_rate','spof','median_mincut','resilience_loss']].round(4).to_string(index=False))
