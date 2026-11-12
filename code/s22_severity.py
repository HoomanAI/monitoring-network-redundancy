# -*- coding: utf-8 -*-
"""Reliability under increasing component-failure severity, for five designs,
under independent, observed-joint and common-mode failure mechanisms.
Severity is a repair-time multiplier m: every outage lasts m times longer.
Hardening is its inverse (RHO = 0.5), so both levers act on the same axis."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization')
import rv4
from rv4.core import MSNet
from rv4 import config as K
from rv3 import common as cm
from pathlib import Path
OUT=Path('/home/claude/work/out'); RHO=0.5; TH=K.OK
rng=np.random.default_rng(K.SEED)

net=MSNet(designs_file="designs_ms_final.pkl"); d=net.d
STN={s:i for i,s in enumerate(d.stations)}
lat=d.meta.set_index('station_number').latitude.to_dict()
lon=d.meta.set_index('station_number').longitude.to_dict()

# ---------- designs --------------------------------------------------------
from rv4.design import Problem
P=Problem(kmax=3,th=K.OK)
rec=pd.read_csv(rv4.RES/'reconciliation_table.csv')
rm27=sorted(rec[(rec.design=='ms_k2_P2')&(rec.version=='repaired')].removed_stations.iloc[0].split(','))
rm23=[s for s in rm27 if s not in ('02CF013','02ED015','02HD021','02GD004')]
prep=P.prep(2,False); cfg=P.configure(set(d.stations)-set(rm23),prep)
net.design['k2_P2_rec']={k:(cfg.get(k) or list(net.mem[k]["direct"])) for k in net.keys}
net.design['k2_P2_hard']=dict(net.design['k2_P2'])
HARD=set(pd.read_csv(OUT/'x7_joint_frontier.csv').query("design=='k2_P2' and budget_m<=8").last_hardened.dropna())
DESIGNS=['current','k2_P1','k2_P2','k2_P2_rec','k2_P2_hard']
HARDOF={dn:(HARD if dn=='k2_P2_hard' else set()) for dn in DESIGNS}
print("hardened stations (m=8):",sorted(HARD))
for dn in DESIGNS:
    print("  %-12s attainable series: %d"%(dn,len([k for k in net.keys if net.full_nse(dn,k)>=TH])))

# ---------- run surgery on the observed record -----------------------------
def runs(col):
    """(start, length) of each down-run in a boolean 'up' column"""
    dn=~col; idx=np.flatnonzero(np.diff(np.concatenate(([0],dn.view(np.int8),[0])))!=0)
    return [(idx[i],idx[i+1]-idx[i]) for i in range(0,len(idx),2)]

def scale_U(factor, extra_down=None):
    """rebuild U with every down-run rescaled by factor[station]; extra_down adds forced outages"""
    out={}
    for v in ('H','Q'):
        A=net.U[v].copy(); T=A.shape[0]
        for s,i in STN.items():
            f=factor.get(s,1.0)
            if f==1.0: continue
            col=net.U[v][:,i]
            new=np.ones(T,bool)
            for a,L in runs(col):
                L2=max(1,int(round(L*f))); new[a:min(T,a+L2)]=False
            A[:,i]=new
        if extra_down is not None:
            A &= ~extra_down[v]
        out[v]=A
    return out

BETA=0.3; RADIUS_KM=50.0
_nb={}
for s in STN:
    la,lo=lat[s],lon[s]
    _nb[s]=[t for t in STN if t!=s and
            np.hypot((lat[t]-la)*111.0,(lon[t]-lo)*111.0*np.cos(np.radians(la)))<=RADIUS_KM]
# fixed, design-independent choice of which observed outage runs propagate
_shared={}
_r=np.random.default_rng(K.SEED+7)
for v in ('H','Q'):
    for s,i in STN.items():
        _shared[(v,s)]=[_r.random()<BETA for _ in runs(net.U[v][:,i])]

def common_mode(factor):
    """the same observed runs always propagate; only their rescaled length depends on the design"""
    T=net.U['H'].shape[0]; extra={v:np.zeros((T,len(STN)),bool) for v in ('H','Q')}
    for v in ('H','Q'):
        for s,i in STN.items():
            f=factor.get(s,1.0); flag=_shared[(v,s)]
            for j,(a,L) in enumerate(runs(net.U[v][:,i])):
                if not flag[j]: continue
                L2=max(1,int(round(L*f)))
                for t in _nb[s]: extra[v][a:min(T,a+L2),STN[t]]=True
    return extra

def R_of(design, U, series=None):
    ks=series or [k for k in net.keys if net.full_nse(design,k)>=TH]
    M=net.nse_matrix(design,U=U,series=ks)
    return (M>=TH).astype(np.float32), ks

def block_band(X,B=300,blocklen=30):
    T=X.shape[0]; blocks=np.array_split(np.arange(T),T//blocklen); m=np.empty(B)
    for b in range(B):
        idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]); m[b]=X[idx].mean()
    return float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

# ---------- independent mechanism ------------------------------------------
Aemp={(s,v):float(net.U[v][:,STN[s]].mean()) for s in STN for v in ('H','Q')}
AV=pd.read_csv(rv4.RES/'p2_availability_mtbf_mttr.csv',dtype={'station':str})
MT={}
for v in ('H','Q'):
    g=AV[AV.variable==v].set_index('station')
    for s in STN:
        if s in g.index and np.isfinite(g.loc[s,'MTBF_days']) and np.isfinite(g.loc[s,'MTTR_days']):
            MT[(s,v)]=(float(g.loc[s,'MTBF_days']),float(g.loc[s,'MTTR_days']))

def A_indep(s,v,m,hard):
    mt=MT.get((s,v))
    if mt is None: return Aemp[(s,v)]
    mtbf,mttr=mt; f=m*(RHO if s in hard else 1.0)
    return float(np.clip(mtbf/(mtbf+f*mttr),0.0,1.0))

_UR={}
def U_indep(m,hard,reps=5):
    T=net.U['H'].shape[0]*reps; out={}
    for v in ('H','Q'):
        if v not in _UR:
            _UR[v]=np.random.default_rng(K.SEED+hash(v)%1000).random((T,len(d.stations)))
        a=np.array([A_indep(s,v,m,hard) for s in d.stations])
        out[v]=_UR[v]<a
    return out

# ---------- sweep ----------------------------------------------------------
MULT=[1,2,3,4,6,8]
rows=[]
for mech in ('independent','observed_joint','common_mode'):
    for dn in DESIGNS:
        hard=HARDOF[dn]
        for m in MULT:
            factor={s:m*(RHO if s in hard else 1.0) for s in STN}
            if mech=='independent':   U=U_indep(m,hard)
            elif mech=='observed_joint': U=scale_U(factor)
            else:                     U=scale_U(factor,extra_down=common_mode(factor))
            X,ks=R_of(dn,U)
            R=float(X.mean()); lo,hi=block_band(X)
            rows.append(dict(mechanism=mech,design=dn,multiplier=m,n_series=len(ks),
                             R=R,ci_lo=lo,ci_hi=hi,
                             share_all_G1=float((X.mean(1)==1).mean()),
                             mean_down=float((~U['H']).mean(0).sum())))
            print("  %-15s %-12s m=%d  R=%.4f [%.4f,%.4f]"%(mech,dn,m,R,lo,hi))
S=pd.DataFrame(rows); S.to_csv(OUT/'x17_severity_sweep.csv',index=False)
print("\nwrote x17_severity_sweep.csv:",len(S),"rows")
