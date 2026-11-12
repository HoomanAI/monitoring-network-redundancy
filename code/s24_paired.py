# -*- coding: utf-8 -*-
"""Per-series reliability: current network against the recommended design."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization')
import rv4
from rv4.core import MSNet
from rv4.design import Problem
from rv4 import config as K
from pathlib import Path
OUT=Path('/home/claude/work/out'); TH=K.OK
net=MSNet(designs_file="designs_ms_final.pkl"); d=net.d
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm24=[s for s in rm27 if s not in ('02CF013','02ED015','02HD021')]
cfg=P.configure(set(d.stations)-set(rm24),P.prep(2,False))
net.design['k2_P2_rec']={k:(cfg.get(k) or list(net.mem[k]["direct"])) for k in net.keys}

def per_series(dn,ks):
    M=net.nse_matrix(dn,series=ks)
    return (M>=TH).astype(np.float32).mean(0)
ks=[k for k in net.keys if net.full_nse('current',k)>=TH and net.full_nse('k2_P2_rec',k)>=TH]
r0=per_series('current',ks); r1=per_series('k2_P2_rec',ks)
rows=[]
for j,k in enumerate(ks):
    t,v=k
    cur=set(net.design['current'][k]); new=set(net.design['k2_P2_rec'][k])
    direct=set(net.mem[k]["direct"])
    added=sorted(new-cur)
    n_stat_new=sum(1 for s in new if s not in direct)
    rows.append(dict(target=t,variable=v,R_current=float(r0[j]),R_recommended=float(r1[j]),
                     n_inputs_current=len(cur),n_inputs_new=len(new),n_added=len(added),
                     n_statistical_new=n_stat_new,
                     redundancy_type=('statistical' if n_stat_new>0 else 'physical'),
                     added=','.join(added)))
D=pd.DataFrame(rows); D['delta']=D.R_recommended-D.R_current
D.to_csv(OUT/'x19_paired_series_reliability.csv',index=False)
print("series compared: %d"%len(D))
print("improved %d | unchanged %d | worsened %d"%( (D.delta>1e-6).sum(), (D.delta.abs()<=1e-6).sum(), (D.delta<-1e-6).sum()))
print("median delta %+.4f | mean %+.4f | worst %+.4f (%s)"
      %(D.delta.median(),D.delta.mean(),D.delta.min(),D.loc[D.delta.idxmin(),'target']+'|'+D.loc[D.delta.idxmin(),'variable']))
print("\nby redundancy type:")
print(D.groupby('redundancy_type').agg(n=('delta','size'),median_delta=('delta','median'),
      median_R_cur=('R_current','median'),median_R_new=('R_recommended','median')).round(4).to_string())
print("\nseries below 0.95 in the current network: %d, of which improved: %d"
      %((D.R_current<0.95).sum(), ((D.R_current<0.95)&(D.delta>1e-6)).sum()))
print("lowest-decile current reliability: median %.4f -> %.4f"
      %(D[D.R_current<=D.R_current.quantile(0.1)].R_current.median(),
        D[D.R_current<=D.R_current.quantile(0.1)].R_recommended.median()))
if (D.delta<-1e-6).any():
    print("\nworsened series:"); print(D[D.delta<-1e-6][['target','variable','R_current','R_recommended','delta']].round(4).to_string(index=False))
