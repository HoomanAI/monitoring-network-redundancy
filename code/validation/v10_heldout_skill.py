# -*- coding: utf-8 -*-
"""Results II: predictive validation on the held-out period.

Train portion  2005-2016 (4383 days, 75% of the record)
Test  portion  2017-2020 (1461 days, 25%), used once, after every design was fixed.
Nothing is re-fitted, re-selected or re-tuned on the test portion: the retained
stations, the link assignment and the state thresholds are those of Results I.
"""
import sys, pickle, json
from pathlib import Path
R='/home/claude/ws/station_subset_optimization'
for p in (R, R+'/ress_v3', R+'/ress_v4'): sys.path.insert(0,p)
import numpy as np, pandas as pd
import rv3, rv4
from rv4 import config as K
from rv4.design import Problem
from ssopt.data import Data
from ssopt.models import build_X, ridge_fast
from ssopt import config as C
OUT=Path('/home/claude/work/out')

def nse(y,p):
    m=np.isfinite(y)&np.isfinite(p)
    return float(1-np.sum((y[m]-p[m])**2)/np.sum((y[m]-np.mean(y[m]))**2))

d=Data.load(include_test=True)
tr,te=d.split_masks('test')                       # fit 2005-2016, score 2017-2020
trv,ev=d.fold_masks(2)                            # fit 2005-2012, score 2013-2016
print('train days',int(tr.sum()),'test days',int(te.sum()))

Z=pickle.load(open(rv3.CACHE/'fallback_tables.pkl','rb')); mem=Z['members']
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm23=[s for s in rm27 if s not in ('02CF013','02ED015','02HD021','02GD004')]
cfg=P.configure(set(d.stations)-set(rm23),P.prep(2,False))
DES={'current':{k:list(mem[k]['direct']) for k in sorted(mem)},
     'k2_rec23':{k:(cfg.get(k) or list(mem[k]['direct'])) for k in sorted(mem)}}
rows=[]
for dn,inp in DES.items():
    for k in sorted(mem):
        t,v=k; ins=[s for s in inp[k] if s in mem[k]['members']] or list(mem[k]['direct'])
        ym=d.mod_arr[v][t]; yr=d.raw_arr[v][t]
        out={}
        for lab,(fit,sc) in (('val',(trv,ev)),('test',(tr,te))):
            sco=sc&np.isfinite(yr)
            if sco.sum()<60: out[lab]=np.nan; continue
            p=ridge_fast(build_X(d,ins,v),ym,fit,sco)
            if v=='Q': p=np.clip(np.expm1(p),0,None)
            out[lab]=nse(yr[sco],p)
        rows.append(dict(design=dn,target=t,variable=v,n_inputs=len(ins),
                         nse_val=out['val'],nse_test=out['test']))
S=pd.DataFrame(rows); S.to_csv(OUT/'x23_heldout_skill.csv',index=False)
G1,G2=K.OK,K.GOOD
for dn,g in S.groupby('design'):
    a=g.nse_val.dropna(); b=g.nse_test.dropna()
    print('%-9s n=%d  median val %.3f -> test %.3f (delta %+.3f)  IQR test %.3f-%.3f'
          %(dn,len(b),a.median(),b.median(),(g.nse_test-g.nse_val).median(),b.quantile(.25),b.quantile(.75)))
    print('           G1+ %.1f%% -> %.1f%%   G2 %.1f%% -> %.1f%%   below G1 on test: %d'
          %(100*(a>=G1).mean(),100*(b>=G1).mean(),100*(a>=G2).mean(),100*(b>=G2).mean(),int((b<G1).sum())))
piv=S.pivot_table(index=['target','variable'],columns='design',values='nse_test')
d2=(piv['k2_rec23']-piv['current']).dropna()
print('design minus current on test: median %+.4f, better %d, worse %d, |delta|>0.02 %d'
      %(d2.median(),(d2>0).sum(),(d2<0).sum(),(d2.abs()>0.02).sum()))
for v in ('H','Q'):
    g=S[(S.design=='k2_rec23')&(S.variable==v)]
    print('  %s: median test NSE %.3f, G1+ %.1f%%'%(v,g.nse_test.median(),100*(g.nse_test>=G1).mean()))
