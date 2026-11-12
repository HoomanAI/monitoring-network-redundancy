# -*- coding: utf-8 -*-
"""Remaining case-study figures, redrawn from figdata in the new style."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import pandas as pd, numpy as np
import matplotlib.lines as mlines
from matplotlib.ticker import NullFormatter, FixedLocator
FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'
R4='/home/claude/ws/station_subset_optimization/ress_v4/results/'
O='/home/claude/work/out/'
ORD=['current','k1_P1','k2_P1','k1_P2','k2_P2','k3_P2']

# ============ common cause =================================================
b=pd.read_csv(FD+'fig06c_common_cause_beta.csv').set_index('quantity').value
nul=pd.read_csv(O+'x5_fig02a_beta_null_distribution.csv')
pv=pd.read_csv(O+'x20_commoncause_pvalue.csv').iloc[0]
fig,ax=plt.subplots(figsize=(W1*1.12,60*MM))
ax.hist(nul.beta_null,bins=22,color=SEQ[1],edgecolor=SEQ[2],lw=0.6,
        label='independent time-shift null\n($n=%d$ replicates)'%pv.n_null_replicates)
ax.axvline(float(b['beta_factor_observed']),color=C['red'],lw=1.5,
           label='observed $\\beta=%.2f$,  $p=%.2f$'%(b['beta_factor_observed'],pv.empirical_p))
ax.axvline(float(b['beta_null_p95']),color=C['grey'],lw=0.9,ls=(0,(4,2)),label='null 95th percentile')
ax.set_xlabel('Common-cause factor $\\beta$'); ax.set_ylabel('Null replicates')
ax.set_ylim(0,ax.get_ylim()[1]*1.42)
ax.legend(loc='upper left',fontsize=6.4); save(fig,'fig08b_common_cause')

# ============ structure function ===========================================
P=pd.read_csv(FD+'fig08_performance_vs_inputs.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,47*MM))
ax=axes[0]
for (t,v),g in P.groupby(['target','var']):
    g=g.sort_values('n_available_inputs')
    ax.plot(g.n_available_inputs,g.nse_mean,color=C['lgrey'],lw=0.4,alpha=0.5,zorder=2)
m=P.groupby('n_available_inputs').nse_mean.agg(['median','count'])
m=m[m['count']>=5]
q=P.groupby('n_available_inputs').nse_mean.quantile([0.25,0.75]).unstack()
ax.fill_between(m.index,q.loc[m.index,0.25],q.loc[m.index,0.75],color=SEQ[1],alpha=0.55,lw=0,zorder=3)
ax.plot(m.index,m['median'],color=C['blue'],lw=1.6,marker='o',ms=3.4,zorder=4,label='median over series')
ax.plot([],[],color=C['lgrey'],lw=0.8,label='one series')
ax.axhline(0.8,color=C['orange'],lw=0.8,ls=(0,(4,2)),zorder=5); ax.axhline(0.6,color=C['grey'],lw=0.8,ls=(0,(4,2)),zorder=5)
ax.text(P.n_available_inputs.max(),0.815,'G2',fontsize=6.4,color=C['orange'],ha='right')
ax.text(P.n_available_inputs.max(),0.615,'G1',fontsize=6.4,color=C['grey'],ha='right')
ax.set_xlabel('Number of available inputs'); ax.set_ylabel('Validation NSE')
ax.set_ylim(-0.1,1.02); title(ax,'Skill against the number of inputs'); panel(ax,'a')
ax.legend(loc='best')
ax=axes[1]
first=P[P.n_available_inputs==1].groupby(['target','var']).nse_mean.mean()
full=P.groupby(['target','var']).nse_full_pool.first()
seas=P.groupby(['target','var']).nse_season_only.first()
d=pd.DataFrame({'season':seas,'one':first,'full':full}).dropna()
parts=[d.season,d.one-d.season,d.full-d.one]
labs=['seasonality only','first input','remaining inputs']
cs=[C['lgrey'],C['blue'],SEQ[1]]
bot=np.zeros(3)
for i,(p,l,c) in enumerate(zip(parts,labs,cs)):
    ax.bar(i,p.median(),0.62,color=c,edgecolor='white',lw=0.6)
    ax.text(i,p.median()+0.012,'%+.2f'%p.median() if i else '%.2f'%p.median(),ha='center',fontsize=6.6)
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=7)
ax.set_ylabel('Median contribution to NSE'); ax.set_ylim(0,0.8)
title(ax,'Where the skill comes from'); panel(ax,'b')
fig.tight_layout(pad=0.4); save(fig,'fig09_performance_vs_inputs')

# ============ multi-state performance ======================================
M=pd.read_csv(FD+'fig09_multistate_distribution.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,46*MM))
ax=axes[0]
ds=[d for d in ORD if d in set(M.design)]
x=np.arange(len(ds)); bot=np.zeros(len(ds))
for col,c,l in (('P_G2',G2C,'G2'),('P_G1',G1C,'G1'),('P_G0',G0C,'G0')):
    v=[M[M.design==d][col].mean() for d in ds]
    ax.bar(x,v,0.64,bottom=bot,color=c,edgecolor='white',lw=0.6,label=l); bot+=np.array(v)
ax.set_xticks(x); ax.set_xticklabels([LABC[d] for d in ds],fontsize=6.4)
ax.set_ylabel('Mean state probability'); ax.set_ylim(0,1.17)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
title(ax,'Multi-state performance by design'); panel(ax,'a')
ax.legend(loc='best',ncol=3)
ax=axes[1]
for d in ('current','k1_P2','k2_P2'):
    g=M[M.design==d].P_ge_G1.dropna().sort_values().values
    ax.plot(g,np.linspace(0,1,len(g)),color=COL[d],lw=1.4,label=LAB[d])
ax.set_xlabel('Series reliability $R_t$'); ax.set_ylabel('Cumulative share of series')
ax.set_xlim(0.5,1.002); title(ax,'Distribution over series'); panel(ax,'b')
ax.legend(loc='best')
fig.tight_layout(pad=0.4); save(fig,'fig11_multistate')

# ============ importance ===================================================
B2=pd.read_csv(O+'x21_importance_bootstrap.csv',dtype={'station':str})
fig,axes=plt.subplots(1,2,figsize=(W2,47*MM))
ax=axes[0]
top=B2.nlargest(15,'criticality').sort_values('criticality')
y=np.arange(len(top))
ax.barh(y,top.criticality,0.66,color=C['blue'],edgecolor='white',lw=0.5,zorder=3)
ax.hlines(y,top.ic_lo,top.ic_hi,color=C['ink'],lw=0.8,zorder=4)
ax.set_yticks(y); ax.set_yticklabels(top.station,fontsize=6.3)
ax.set_xlabel('Criticality importance $I^C_u$'); ax.set_xlim(0,top.ic_hi.max()*1.06)
panel(ax,'a')
ax=axes[1]
r=B2.sort_values('criticality',ascending=False).reset_index(drop=True)
x=np.arange(len(r))
ax.plot(x,r.birnbaum,ls='',marker='o',ms=2.8,color=C['orange'],label='Birnbaum $I^B_u$',zorder=3)
ax.plot(x,r.criticality,ls='',marker='s',ms=2.8,color=C['blue'],label='criticality $I^C_u$',zorder=3)
ax.vlines(x,r.ic_lo,r.ic_hi,color=C['blue'],lw=0.5,alpha=0.5,zorder=2)
ax.axvline(14.5,color=C['grey'],lw=0.8,ls=(0,(3,2)),zorder=1)
ax.set_yscale('log'); ax.set_ylim(1e-4,0.4)
ax.set_xlabel('Station, ordered by criticality importance'); ax.set_ylabel('Importance')
ax.legend(loc='upper right',fontsize=6.6)
ax.text(15.6,0.25,'the 15 most critical stations stay\nin the top 15 in all 300\nblock-bootstrap resamples',fontsize=6.0,
        va='top',ha='left',color=C['grey'])
panel(ax,'b')
fig.tight_layout(pad=0.4); save(fig,'fig13_importance')

# ============ SPOF and minimum cut =========================================
S=pd.read_csv(FD+'fig12_spof_mincut.csv',dtype={'target':str,'spof_station':str})
fig,axes=plt.subplots(1,2,figsize=(W2,44*MM))
ax=axes[0]
ds=[d for d in ORD if d in set(S.design)]
x=np.arange(len(ds)); bot=np.zeros(len(ds))
cuts=[1,2,3]
for i,k in enumerate(cuts+['>3']):
    if k=='>3': v=[((S[S.design==d].min_cut_size>3)).sum() for d in ds]
    else:       v=[((S[S.design==d].min_cut_size==k)).sum() for d in ds]
    ax.bar(x,v,0.64,bottom=bot,color=SEQ[min(i,3)],edgecolor='white',lw=0.6,
           label='$\\kappa_t$ = %s'%k); bot+=np.array(v)
ax.set_xticks(x); ax.set_xticklabels([LABC[d] for d in ds],fontsize=6.4)
ax.set_ylabel('Attainable series'); ax.set_ylim(0,132)
title(ax,'Minimum cut size by design'); panel(ax,'a')
ax.legend(loc='best',ncol=4,fontsize=6.4,columnspacing=0.8)
ax=axes[1]
nser=[int(S[(S.design==d)&S.spof].shape[0]) for d in ds]
nstn=[int(S[(S.design==d)&S.spof].spof_station.nunique()) for d in ds]
w=0.38
ax.bar(x-w/2,nser,w,color=C['blue'],edgecolor='white',lw=0.5,label='series with a single point of failure')
ax.bar(x+w/2,nstn,w,color=SEQ[1],edgecolor=SEQ[2],lw=0.5,label='distinct stations responsible')
for xi,(a_,b_) in enumerate(zip(nser,nstn)):
    ax.text(xi-w/2,a_+1.2,str(a_),ha='center',fontsize=6.3)
    ax.text(xi+w/2,b_+1.2,str(b_),ha='center',fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels([LABC[d] for d in ds],fontsize=6.4)
ax.set_ylabel('Count'); ax.set_ylim(0,max(nser)*1.22)
title(ax,'Single points of failure by design'); panel(ax,'b'); ax.legend(loc='best')
fig.tight_layout(pad=0.4); save(fig,'fig14_spof_mincut')

# ============ reliability against k ========================================
N=pd.read_csv(FD+'fig16_reliability_vs_k.csv').set_index('design')
fig,ax=plt.subplots(figsize=(W1,60*MM))
for pool,lab,c in (('P1','physical links only',C['orange']),('P2','with statistical links',C['blue'])):
    ks=[1,2,3]; X=[];Y=[];LO=[];HI=[]
    for k in ks:
        dn='k%d_%s'%(k,pool)
        if dn in N.index:
            X.append(k); Y.append(N.loc[dn,'network_reliability_attainable'])
            LO.append(N.loc[dn,'ci_lo_att']); HI.append(N.loc[dn,'ci_hi_att'])
    ax.errorbar(X,Y,yerr=[np.array(Y)-np.array(LO),np.array(HI)-np.array(Y)],marker='o',ms=3.6,
                color=c,capsize=2.4,elinewidth=0.8,lw=1.2,label=lab)
ax.axhline(N.loc['current','network_reliability_attainable'],color=C['grey'],lw=1.0,ls=(0,(4,2)),label='current network')
ax.set_xlabel('Redundancy level $k$'); ax.set_ylabel('Network reliability $R$'); ax.set_xticks([1,2,3])
title(ax,'Reliability against redundancy level'); ax.legend(loc='best')
save(fig,'fig12a_reliability_k')

# ============ resilience: episodes and robustness ==========================
E=pd.read_csv(FD+'fig14b_event_resilience_metrics.csv')
Mm=E[E['mode'].isin(['fallback','no_fallback'])].groupby(['design','mode']).loss_performance_days.mean().unstack()
AU=pd.read_csv(FD+'fig13b_robustness_auc.csv')
ds=[d for d in ORD if d in Mm.index]
fig,axes=plt.subplots(1,2,figsize=(W2,46*MM))
ax=axes[0]
x=np.arange(len(ds)); w=0.38
ax.bar(x-w/2,Mm.loc[ds,'fallback'],w,color=C['blue'],edgecolor='white',lw=0.5,label='with fallback models')
ax.bar(x+w/2,Mm.loc[ds,'no_fallback'],w,color=SEQ[1],edgecolor=SEQ[2],lw=0.5,label='without')
ax.set_xticks(x); ax.set_xticklabels([LABC[d] for d in ds],fontsize=6.4)
ax.set_ylabel('Mean resilience loss (series-days)')
ax.set_ylim(0,max(Mm.loc[ds,'no_fallback'])*1.3)
ax.legend(loc='upper right',fontsize=6.6); panel(ax,'a')
ax=axes[1]
ords=[o for o in ('random_mean','Birnbaum-targeted','degree-targeted','betweenness-targeted') if o in set(AU.order)]
oc={'random_mean':C['grey'],'Birnbaum-targeted':C['red'],'degree-targeted':C['orange'],'betweenness-targeted':C['purple']}
dsa=[d for d in ORD if d in set(AU.design)]
x=np.arange(len(dsa)); w=0.8/len(ords)
for i_,o in enumerate(ords):
    v=[float(AU[(AU.design==d)&(AU.order==o)].auc_share_ge_G1.mean()) for d in dsa]
    ax.bar(x+(i_-(len(ords)-1)/2)*w,v,w,color=oc[o],edgecolor='white',lw=0.5,
           label=o.replace('_mean','').replace('-targeted',''))
ax.set_xticks(x); ax.set_xticklabels([LABC[d] for d in dsa],fontsize=6.4)
ax.set_ylabel('Area under the robustness curve'); ax.set_ylim(0,0.95)
ax.legend(loc='upper left',fontsize=6.4,ncol=2,title='removal order',title_fontsize=6.4)
panel(ax,'b')
fig.tight_layout(pad=0.4); save(fig,'fig16b_resilience_loss')
print("batch B ok")
