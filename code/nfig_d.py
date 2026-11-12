# -*- coding: utf-8 -*-
"""Figures that come from this study's own result files, in the new style."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import pandas as pd, numpy as np
import matplotlib.lines as mlines
O='/home/claude/work/out/'; FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'

# ============ allocation levers: redundancy and hardening ==================
FD2='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'
N=pd.read_csv(FD2+'fig16_reliability_vs_k.csv').set_index('design')
jf=pd.read_csv(O+'x7_joint_frontier.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,46*MM))
ax=axes[0]
for pool,lab,c in (('P1','physical links only',C['orange']),('P2','with statistical links',C['blue'])):
    X=[];Y=[];LO=[];HI=[]
    for k in (1,2,3):
        dn='k%d_%s'%(k,pool)
        if dn in N.index:
            X.append(k); Y.append(N.loc[dn,'network_reliability_attainable'])
            LO.append(N.loc[dn,'ci_lo_att']); HI.append(N.loc[dn,'ci_hi_att'])
    ax.errorbar(X,Y,yerr=[np.array(Y)-np.array(LO),np.array(HI)-np.array(Y)],marker='o',ms=3.6,
                color=c,capsize=2.4,elinewidth=0.8,lw=1.2,label=lab)
ax.axhline(N.loc['current','network_reliability_attainable'],color=C['grey'],lw=1.0,
           ls=(0,(4,2)),label='current network')
ax.set_xlabel('Redundancy level $k$'); ax.set_ylabel('Network reliability $R$')
ax.set_xticks([1,2,3]); ax.set_xlim(0.7,3.3); ax.set_ylim(0.9555,1.0005)
ax.legend(loc='upper left',fontsize=6.6); panel(ax,'a')
ax=axes[1]
for d in ('current','k1_P2','k2_P2'):
    g=jf[jf.design==d].sort_values('budget_m')
    ax.plot(g.budget_m,g.R,marker='o',ms=3.0,color=COL[d],label=LAB[d])
k3=0.99506564619501
ax.axhline(k3,color=C['purple'],ls=(0,(4,2)),lw=1.0,label='$k$=3, +statistical, no hardening')
ax.set_xlabel('Hardened stations $m$'); ax.set_ylabel('Network reliability $R$')
ax.set_xlim(-0.3,12.3); ax.set_ylim(0.9555,1.0005)
ax.legend(loc='lower right',fontsize=6.6); panel(ax,'b')
fig.tight_layout(pad=0.4); save(fig,'fig10_allocation_levers')

# ============ graph structure against statistical redundancy ===============
P=pd.read_csv(O+'x15_corr_vs_graph.csv'); A=pd.read_csv(O+'x16_pool_class_ablation.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,47*MM))
ax=axes[0]
for v,c,mk in (('H',C['red'],'o'),('Q',C['blue'],'s')):
    g=P[(P.variable==v)&P.same_component&P.graph_distance.notna()]
    m=g.groupby('graph_distance').abs_corr.agg(['median','count']); m=m[m['count']>=10]
    ax.plot(m.index,m['median'],marker=mk,ms=3.4,color=c,label='%s, same component'%v)
    ax.axhline(P[(P.variable==v)&~P.same_component].abs_corr.median(),color=c,ls=(0,(4,2)),lw=0.9)
ax.text(4.0,0.622,'different component (median)',fontsize=6.6,color=C['grey'],ha='center')
ax.set_xlabel('Graph distance (edges)'); ax.set_ylabel('Median $|\\rho|$ over station pairs')
ax.set_ylim(0.55,1.0); title(ax,'Correlation against graph distance'); panel(ax,'a')
ax.legend(loc='best')
ax=axes[1]
lab=['All pool\ninputs','Graph-connected\ninputs only','Correlation-only\ninputs']
x=np.arange(3); w=0.36
for i,(v,c) in enumerate((('H',C['red']),('Q',C['blue']))):
    a=A[A.variable==v]
    y=[a.nse_full.median(),a.nse_conn_only.median(),a.nse_corr_only.median()]
    ax.bar(x+(i-0.5)*w,y,w,color=c,edgecolor='white',lw=0.5,label=v)
    for xi,yi in zip(x+(i-0.5)*w,y): ax.text(xi,yi+0.008,'%.3f'%yi,ha='center',fontsize=6.4,color=c)
ax.axhline(0.8,color=C['grey'],ls=(0,(2,1.5)),lw=0.8)
ax.text(2.46,0.806,'G2',fontsize=6.6,color=C['grey'],va='bottom',ha='right')
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=7); ax.set_ylabel('Median validation NSE')
ax.set_ylim(0.7,1.0); title(ax,'Contribution of each class of input'); panel(ax,'b')
ax.legend(loc='best',ncol=2)
fig.tight_layout(pad=0.4); save(fig,'fig23_graph_vs_corr')

# ============ severity sweep ===============================================
S=pd.read_csv(O+'x17_severity_sweep.csv')
LB={'current':'Current','k2_P1':'$k$=2, physical','k2_P2':'$k$=2, +statistical',
    'k2_P2_rec':'$k$=2, +recoverability','k2_P2_hard':'$k$=2, +hardening'}
TT={'independent':'Independent failures','observed_joint':'Observed joint outages','common_mode':'Common-mode failures'}
YL={'independent':(0.83,1.004),'observed_joint':(0.76,1.004),'common_mode':(0.38,0.96)}
LS={'k2_P1':(0,(4,1.6)),'k2_P2_hard':(0,(3,1,1,1))}
fig,axes=plt.subplots(1,3,figsize=(W2,55*MM))
for ax,mech in zip(axes,('independent','observed_joint','common_mode')):
    g=S[S.mechanism==mech]
    for dn in LB:
        h=g[g.design==dn].sort_values('multiplier')
        ax.plot(h.multiplier,h.R,marker='o',ms=2.9,color=COL[dn],label=LB[dn],ls=LS.get(dn,'-'))
        ax.fill_between(h.multiplier,h.ci_lo,h.ci_hi,color=COL[dn],alpha=0.15,lw=0)
    ax.set_xlabel('Repair-time multiplier $\gamma$'); title(ax,TT[mech])
    ax.set_xticks([1,2,3,4,6,8]); ax.set_ylim(*YL[mech]); ax.set_ylabel('Network reliability $R$')
for ax,l in zip(axes,'abc'): panel(ax,l,-0.012,1.055)
h,lb=axes[0].get_legend_handles_labels()
fig.legend(h,lb,loc='lower center',ncol=5,fontsize=6.4,frameon=False,
           handlelength=1.6,handletextpad=0.4,columnspacing=1.3,bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(pad=0.4,rect=(0,0.085,1,1)); save(fig,'fig24_severity')

# ============ paired series reliability ====================================
D=pd.read_csv(O+'x19_paired_series_reliability.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,52*MM))
def draw(ax,lo,hi,ticks):
    ax.plot([lo,hi],[lo,hi],color=C['lgrey'],lw=0.9,ls=(0,(4,2)),zorder=1)
    for typ,c in (('physical',C['orange']),('statistical',C['blue'])):
        g=D[D.redundancy_type==typ]
        ax.scatter(g.R_current,g.R_recommended,s=9+13*g.n_added,c=c,alpha=0.8,
                   linewidths=0.35,edgecolors='white',zorder=3)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel('Series reliability, current network')
draw(axes[0],0.745,1.006,[0.8,0.85,0.9,0.95,1.0])
axes[0].add_patch(plt.Rectangle((0.985,0.985),0.021,0.021,fill=False,ec=C['red'],lw=0.9,zorder=6))
axes[0].set_ylabel('Series reliability, recommended design')
w=D[D.delta<-1e-6].nsmallest(1,'delta').iloc[0]
axes[0].annotate('%s %s'%(w.target,w.variable),xy=(w.R_current,w.R_recommended),xytext=(-9,-13),
            textcoords='offset points',fontsize=6.6,ha='right',color=C['red'],
            arrowprops=dict(arrowstyle='-',color=C['red'],lw=0.6))
panel(axes[0],'a')
draw(axes[1],0.985,1.0012,[0.985,0.99,0.995,1.0])
axes[1].set_ylabel('Series reliability, recommended design')
for sp in axes[1].spines.values(): sp.set_color(C['red']); sp.set_linewidth(0.9)
panel(axes[1],'b')
h=[mlines.Line2D([],[],ls='',marker='o',color=C['orange'],ms=4,label='physical redundancy'),
   mlines.Line2D([],[],ls='',marker='o',color=C['blue'],ms=4,label='statistical redundancy'),
   mlines.Line2D([],[],ls='',marker='o',mfc='none',mec=C['grey'],ms=3.0,label='no input added'),
   mlines.Line2D([],[],ls='',marker='o',mfc='none',mec=C['grey'],ms=5.9,label='2 inputs added')]
axes[0].legend(handles=h,loc='lower right',labelspacing=0.45,fontsize=6.5)
fig.tight_layout(pad=0.4); save(fig,'fig25_paired_reliability')
print("batch D ok")
