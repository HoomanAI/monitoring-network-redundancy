# -*- coding: utf-8 -*-
"""Supplementary data package: the graph, the pools and the recommended design as machine-readable files."""
import sys, pickle
from pathlib import Path
R='/home/claude/ws/station_subset_optimization'
for p in (R, R+'/ress_v3', R+'/ress_v4'): sys.path.insert(0,p)
import numpy as np, pandas as pd, networkx as nx
import rv3, rv4
from rv4 import config as K
from rv4.design import Problem
from rv3 import common as cm
from ssopt.data import Data
from ssopt import config as C
from ssopt.graph import corr_ranking
OUT=Path('/home/claude/work/supp'); OUT.mkdir(exist_ok=True)

d=Data.load(include_test=False); trm=d.period(C.TRAIN)
Z=pickle.load(open(rv3.CACHE/'fallback_tables.pkl','rb')); mem=Z['members']
G,_=cm.physical_forest()
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm23=set(s for s in rm27 if s not in ('02CF013','02ED015','02HD021','02GD004'))
cfg=P.configure(set(d.stations)-rm23, P.prep(2,False))
DESIGN={k:(cfg.get(k) or list(mem[k]['direct'])) for k in sorted(mem)}

# ---- S1 stations ----------------------------------------------------------
N=pd.read_csv(rv3.RES/'A_node_table.csv',dtype={'station':str})
A=pd.read_csv(rv4.RES/'p2_availability_mtbf_mttr.csv',dtype={'station':str})
av=A.pivot_table(index='station',columns='variable',values='availability')
mt=A[A.variable=='either'].set_index('station')[['MTBF_days','MTTR_days','n_events']]
S1=N[['station','role','component','lat','lon','da_km2']].copy()
S1['availability_H']=S1.station.map(av['H'] if 'H' in av else {})
S1['availability_Q']=S1.station.map(av['Q'] if 'Q' in av else {})
for c in ('MTBF_days','MTTR_days','n_events'): S1[c]=S1.station.map(mt[c])
S1['must_keep']=S1.station.isin(P.must).astype(int)
S1['status_recommended']=np.where(S1.station.isin(rm23),'retired','kept')
S1=S1.sort_values('station'); S1.to_csv(OUT/'S1_stations.csv',index=False)

# ---- S2 physical edges, station level -------------------------------------
_G,_nt=cm.physical_forest()
S2=pd.DataFrame([dict(upstream=u,downstream=v,in_forest=1) for u,v in G.edges()]
                +[dict(upstream=u,downstream=v,in_forest=0) for u,v in _nt])
S2['downstream_is_target']=S2.downstream.isin({k[0] for k in mem}).astype(int)
S2=S2.sort_values(['downstream','upstream']); S2.to_csv(OUT/'S2_physical_edges.csv',index=False)

# ---- S3 series-level candidate edges --------------------------------------
rows=[]
for (t,v),p in sorted(mem.items()):
    anc=set(nx.ancestors(G,t)) if t in G else set()
    dr=set(p['direct']); cn=set(p['connected']); _,r=corr_ranking(d,t,v,set(),trm)
    used=set(DESIGN[(t,v)])
    for u in p['members']:
        if u in dr: et='direct'
        elif u in cn: et='physical_upstream'
        else: et='statistical'
        rows.append(dict(target=t,variable=v,input=u,edge_type=et,
                         via_non_tree_link=int(et=='physical_upstream' and u not in anc),
                         abs_corr_train=round(float(r.get(u,np.nan)),4) if u in r else np.nan,
                         in_recommended_design=int(u in used)))
S3=pd.DataFrame(rows); S3.to_csv(OUT/'S3_series_edges.csv',index=False)

# ---- S4 recommended design ------------------------------------------------
rows=[]
for (t,v) in sorted(mem.items() and mem):
    ins=sorted(DESIGN[(t,v)]); dr=set(mem[(t,v)]['direct'])
    rows.append(dict(target=t,variable=v,n_inputs=len(ins),n_direct=len(set(ins)&dr),
                     n_added=len(set(ins)-dr),inputs=';'.join(ins)))
S4=pd.DataFrame(rows); S4.to_csv(OUT/'S4_recommended_design.csv',index=False)

# ---- S5 adjacency matrix of the candidate pools ---------------------------
cols=['%s_%s'%(t,v) for (t,v) in sorted(mem)]
M=pd.DataFrame(0,index=sorted(d.stations),columns=cols,dtype=int)
for (t,v),p in mem.items():
    for u in p['members']: M.loc[u,'%s_%s'%(t,v)]=1
M.index.name='station'; M.to_csv(OUT/'S5_pool_adjacency.csv')

# ---- S6, S7 results -------------------------------------------------------
pd.read_csv('/home/claude/work/out/x22_master_designs.csv').to_csv(OUT/'S6_design_comparison.csv',index=False)
pd.read_csv('/home/claude/work/out/x23_heldout_skill.csv').to_csv(OUT/'S7_heldout_skill.csv',index=False)

print('S1 stations',len(S1),'| S2 edges',len(S2),'| S3 series edges',len(S3),
      '(direct %d, ancestor %d, statistical %d)'%tuple(S3.edge_type.value_counts().reindex(['direct','physical_upstream','statistical']).fillna(0).astype(int)),
      '| S4 series',len(S4),'| S5',M.shape,'ones',int(M.values.sum()))
