# -*- coding: utf-8 -*-
"""Single-panel replacements for figures whose second panel was cut,
plus the two new figures requested in review."""
import sys; sys.path.insert(0,'/home/claude/ws/station_subset_optimization/paper_v4')
from style import *
import pandas as pd, numpy as np
from matplotlib.lines import Line2D
O='/home/claude/work/out/'; R4='/home/claude/ws/station_subset_optimization/ress_v4/results/'

# ---------- common cause: beta factor against the independent null ----------
obs=pd.read_csv(O+'x5_fig02a_beta_observed.csv'); nul=pd.read_csv(O+'x5_fig02a_beta_null_distribution.csv')
fig,ax=plt.subplots(figsize=(W1,58*MM))
ax.hist(nul.beta_null,bins=22,color='#c6dbef',edgecolor='#6baed6',lw=0.5)
b=float(obs.beta_observed.iloc[0]); p95=float(obs.beta_null_p95.iloc[0])
ax.axvline(b,color='#D55E00',lw=1.4); ax.axvline(p95,color='#777777',lw=0.9,ls=(0,(4,2)))
ax.annotate('observed $\\beta$ = %.2f'%b,xy=(b,ax.get_ylim()[1]*0.92),xytext=(-4,0),
            textcoords='offset points',ha='right',fontsize=7,color='#D55E00')
ax.annotate('null 95th pct',xy=(p95,ax.get_ylim()[1]*0.60),xytext=(4,0),
            textcoords='offset points',ha='left',fontsize=7,color='#777777')
ax.set_xlabel('Common-cause factor $\\beta$'); ax.set_ylabel('Replicates')
save(fig,'fig08b_common_cause')

# ---------- reliability against redundancy level ----------------------------
N=pd.read_csv(R4+'ms_network_reliability_final.csv').set_index('design')
fig,ax=plt.subplots(figsize=(W1,58*MM))
for pool,lab,c in (('P1','Physical links only','#D55E00'),('P2','With statistical links','#0072B2')):
    ks=[1,2,3] if pool=='P2' else [1,2]
    x=[]; y=[]; lo=[]; hi=[]
    for k in ks:
        dn='k%d_%s'%(k,pool)
        if dn not in N.index: continue
        x.append(k); y.append(N.loc[dn,'network_reliability_attainable'])
        lo.append(N.loc[dn,'ci_lo_att']); hi.append(N.loc[dn,'ci_hi_att'])
    ax.errorbar(x,y,yerr=[np.array(y)-np.array(lo),np.array(hi)-np.array(y)],
                marker='o',ms=3.4,color=c,label=lab,capsize=2.2,elinewidth=0.7,lw=1.0)
cur=N.loc['current','network_reliability_attainable']
ax.axhline(cur,color='#3d3d3d',lw=0.9,ls=(0,(4,2)),label='Current network')
ax.set_xlabel('Redundancy level $k$'); ax.set_ylabel('Network reliability $R$')
ax.set_xticks([1,2,3]); ax.legend(loc='lower right',handlelength=1.6)
save(fig,'fig12a_reliability_k')

# ---------- resilience loss with and without fallback models ----------------
E=pd.read_csv(R4+'ms_event_resilience_final.csv')
M=E[E['mode'].isin(['fallback','no_fallback'])].groupby(['design','mode']).loss_performance_days.mean().unstack()
order=[dn for dn in ('current','k1_P1','k2_P1','k1_P2','k2_P2') if dn in M.index]
fig,ax=plt.subplots(figsize=(W1,58*MM))
x=np.arange(len(order)); w=0.37
ax.bar(x-w/2,M.loc[order,'fallback'],w,color='#0072B2',label='With fallback models')
ax.bar(x+w/2,M.loc[order,'no_fallback'],w,color='#c6dbef',edgecolor='#6baed6',lw=0.5,label='Without')
ax.set_xticks(x); ax.set_xticklabels([LAB[o].replace(', ','\n') for o in order],fontsize=6.6)
ax.set_ylabel('Mean resilience loss (series-days)')
ax.legend(loc='upper right',handlelength=1.1)
save(fig,'fig16b_resilience_loss')
print("panel rebuilds done")

# ================= NEW FIGURE: severity sweep ===============================
S=pd.read_csv(O+'x17_severity_sweep.csv')
LB={'current':'Current','k2_P1':'$k$=2, physical','k2_P2':'$k$=2, +statistical',
    'k2_P2_rec':'$k$=2, +recoverability','k2_P2_hard':'$k$=2 + hardening'}
CO={'current':'#3d3d3d','k2_P1':'#E69F00','k2_P2':'#0072B2','k2_P2_rec':'#009E73','k2_P2_hard':'#CC79A7'}
TT={'independent':'Independent failures','observed_joint':'Observed joint outages','common_mode':'Common-mode failures'}
YL={'independent':(0.83,1.004),'observed_joint':(0.76,1.004),'common_mode':(0.38,0.96)}
fig,axes=plt.subplots(1,3,figsize=(W2,49*MM))
for ax,mech in zip(axes,('independent','observed_joint','common_mode')):
    g=S[S.mechanism==mech]
    for dn in LB:
        h=g[g.design==dn].sort_values('multiplier')
        ax.plot(h.multiplier,h.R,marker='o',ms=2.8,color=CO[dn],label=LB[dn],
                ls={'k2_P1':(0,(4,1.6)),'k2_P2_hard':(0,(3,1,1,1))}.get(dn,'-'))
        ax.fill_between(h.multiplier,h.ci_lo,h.ci_hi,color=CO[dn],alpha=0.15,lw=0)
    ax.set_xlabel('Repair-time multiplier $m$'); ax.set_title(TT[mech],fontsize=7.6)
    ax.set_xticks([1,2,3,4,6,8]); ax.set_ylim(*YL[mech]); ax.set_ylabel('Network reliability $R$')
axes[0].legend(loc='lower left',handlelength=1.7,labelspacing=0.3,fontsize=6.3)
for ax,l in zip(axes,'abc'): panel(ax,l,-0.01,1.07)
fig.tight_layout(pad=0.35)
save(fig,'fig24_severity')

# ================= NEW FIGURE: paired series reliability ====================
D=pd.read_csv(O+'x19_paired_series_reliability.csv')
fig,ax=plt.subplots(figsize=(W1*1.22,74*MM))
ax.plot([0.74,1.004],[0.74,1.004],color='#999999',lw=0.8,ls=(0,(4,2)),zorder=1)
for typ,c in (('physical','#E69F00'),('statistical','#0072B2')):
    g=D[D.redundancy_type==typ]
    ax.scatter(g.R_current,g.R_recommended,s=9+13*g.n_added,c=c,alpha=0.78,
               linewidths=0.35,edgecolors='white',zorder=3,label=typ.capitalize()+' redundancy')
w=D[D.delta<-1e-6].nsmallest(1,'delta').iloc[0]
ax.annotate('%s %s'%(w.target,w.variable),xy=(w.R_current,w.R_recommended),xytext=(-8,-12),
            textcoords='offset points',fontsize=6.4,ha='right',color='#c0392b',
            arrowprops=dict(arrowstyle='-',color='#c0392b',lw=0.5))
ax.set_xlabel('Series reliability, current network')
ax.set_ylabel('Series reliability, recommended design')
ax.set_xlim(0.745,1.006); ax.set_ylim(0.745,1.006)
ax.set_xticks([0.8,0.85,0.9,0.95,1.0]); ax.set_yticks([0.8,0.85,0.9,0.95,1.0])
h=[Line2D([],[],ls='',marker='o',color='#E69F00',ms=4,label='Physical redundancy'),
   Line2D([],[],ls='',marker='o',color='#0072B2',ms=4,label='Statistical redundancy'),
   Line2D([],[],ls='',marker='o',mfc='none',mec='#777777',ms=np.sqrt(9)*1.0,label='no input added'),
   Line2D([],[],ls='',marker='o',mfc='none',mec='#777777',ms=np.sqrt(35)*1.0,label='2 inputs added')]
ax.legend(handles=h,loc='lower right',handlelength=1.0,labelspacing=0.45,fontsize=6.6)
save(fig,'fig25_paired_reliability')
print("new figures done")
