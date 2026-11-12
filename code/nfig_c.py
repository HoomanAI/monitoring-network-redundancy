# -*- coding: utf-8 -*-
"""Appendix figures, redrawn from figdata in the new style."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import pandas as pd, numpy as np
FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'
ORD=['current','k1_P1','k2_P1','k1_P2','k2_P2','k3_P2']

# ============ graph properties =============================================
a=pd.read_csv(FD+'fig07a_target_indegree.csv'); b=pd.read_csv(FD+'fig07b_chain_depth.csv')
c=pd.read_csv(FD+'fig07c_component_sizes.csv')
fig,axes=plt.subplots(1,3,figsize=(W2,41*MM))
ax=axes[0]
ax.bar(a.k_direct_inputs,a.n_targets,0.68,color=C['blue'],edgecolor='white',lw=0.5)
for _,r in a.iterrows(): ax.text(r.k_direct_inputs,r.n_targets+0.8,int(r.n_targets),ha='center',fontsize=6.4)
ax.set_xlabel('Direct inputs per target'); ax.set_ylabel('Targets')
ax.set_xticks(sorted(a.k_direct_inputs)); ax.set_ylim(0,a.n_targets.max()*1.16)
title(ax,'In-degree'); panel(ax,'a')
ax=axes[1]
h=b.depth_up.value_counts().sort_index()
ax.bar(h.index,h.values,0.68,color=C['teal'],edgecolor='white',lw=0.5)
ax.set_xlabel('Upstream chain depth'); ax.set_ylabel('Stations')
title(ax,'Chain depth'); panel(ax,'b')
ax=axes[2]
h=c.n_stations.value_counts().sort_index()
ax.bar(h.index,h.values,0.68,color=C['purple'],edgecolor='white',lw=0.5)
ax.set_xlabel('Stations per component'); ax.set_ylabel('Components')
title(ax,'Component size'); panel(ax,'c')
fig.tight_layout(pad=0.4); save(fig,'fig05_graph_properties')

# ============ seasonality of outages (single panel) ========================
fig,ax=plt.subplots(figsize=(W2,44*MM))
for v,col,mk,lab in (('H',C['red'],'o','$H$'),('Q',C['blue'],'s','$Q$')):
    g=pd.read_csv(FD+'fig05_monthly_availability.csv',dtype={'station':str})
    g=g[g.variable==v].groupby('month').availability.mean()
    ax.plot(g.index,1-g.values,marker=mk,ms=3.4,lw=1.1,color=col,label=lab)
ax.set_xticks(range(1,13)); ax.set_xticklabels(list('JFMAMJJASOND'))
ax.set_xlim(0.6,12.4)
ax.set_xlabel('Month'); ax.set_ylabel('Mean share of stations down')
ax.legend(loc='upper right',fontsize=7,handletextpad=0.4,borderpad=0.35)
fig.tight_layout(pad=0.4); save(fig,'fig07_availability_time')

# ============ failure-rate sensitivity =====================================
S=pd.read_csv(FD+'fig18_sensitivity.csv',dtype={'station':str})
S=S[(S.design=='current')&(S.top15_flag==True)]
fig,ax=plt.subplots(figsize=(W15,70*MM))
order=S[S.failure_rate_change==0.5].sort_values('delta_reliability').station.tolist()
MK={-0.5:('o',C['blue']),-0.25:('s',SEQ[2]),0.25:('^',C['orange']),0.5:('D',C['red'])}
for ch,(m,col) in MK.items():
    g=S[S.failure_rate_change==ch].set_index('station').reindex(order)
    ax.plot(g.delta_reliability,np.arange(len(order)),ls='',marker=m,ms=3.4,color=col,
            label='%+d%%'%(100*ch))
ax.axvline(0,color=C['grey'],lw=0.6)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order,fontsize=6.3)
ax.set_xlabel('Change in network reliability $\\Delta R$')
title(ax,'Sensitivity to the failure rate of the fifteen most important stations')
ax.legend(loc='best',title='failure rate',title_fontsize=6.6,ncol=2)
save(fig,'fig17_sensitivity')

# ============ cost-reliability frontier ====================================
F=pd.read_csv(FD+'fig17_cost_reliability_frontier.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,44*MM))
fr=F[F.type.str.contains('frontier|pareto|nsga',case=False,na=False)] if 'type' in F else F
if len(fr)==0: fr=F
ax=axes[0]
sc=ax.scatter(fr.n_active,fr.network_reliability_all_series,c=fr.links,cmap='viridis',s=14,
              linewidths=0.25,edgecolors='white',zorder=3)
cb=fig.colorbar(sc,ax=ax,pad=0.02,fraction=0.046); cb.set_label('links added',fontsize=7)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
ax.set_xlabel('Active stations'); ax.set_ylabel('Network reliability')
title(ax,'Cost against reliability'); panel(ax,'a')
ax=axes[1]
sc=ax.scatter(fr.n_active,fr.resilience_loss,c=fr.links,cmap='viridis',s=14,
              linewidths=0.25,edgecolors='white',zorder=3)
cb=fig.colorbar(sc,ax=ax,pad=0.02,fraction=0.046); cb.set_label('links added',fontsize=7)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
ax.set_xlabel('Active stations'); ax.set_ylabel('Resilience loss')
title(ax,'Cost against resilience loss'); panel(ax,'b')
fig.tight_layout(pad=0.4); save(fig,'fig18_cost_reliability')

# ============ robustness ===================================================
Rc=pd.read_csv(FD+'fig13a_robustness_curves.csv'); Ra=pd.read_csv(FD+'fig13b_robustness_auc.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,44*MM),sharey=True)
OC={'Birnbaum-targeted':C['red'],'degree-targeted':C['orange'],'betweenness-targeted':C['purple']}
for ax,d,t in zip(axes,('current','k2_P2'),('Current network','$k$=2, +statistical')):
    g=Rc[Rc.design==d]
    lo=g[g.order=='random_p05'].sort_values('fraction_removed')
    hi=g[g.order=='random_p95'].sort_values('fraction_removed')
    me=g[g.order=='random_mean'].sort_values('fraction_removed')
    if len(lo) and len(hi):
        ax.fill_between(lo.fraction_removed,lo.share_ge_G1,hi.share_ge_G1,color=C['lgrey'],alpha=0.45,lw=0,zorder=1)
    ax.plot(me.fraction_removed,me.share_ge_G1,color=C['grey'],lw=1.3,zorder=3,label='random')
    for o,col in OC.items():
        h=g[g.order==o].sort_values('fraction_removed')
        ax.plot(h.fraction_removed,h.share_ge_G1,color=col,lw=1.2,zorder=3,label=o.replace('-targeted',''))
    ax.set_xlabel('Share of input stations removed'); title(ax,t); ax.set_xlim(0,1); ax.set_ylim(0,1.0)
axes[0].set_ylabel('Share of series at G1 or better')
panel(axes[0],'a'); panel(axes[1],'b')
axes[0].legend(loc='best',title='removal order',title_fontsize=6.6)
fig.tight_layout(pad=0.4); save(fig,'fig15_robustness')

# ============ fault detection ==============================================
D=pd.read_csv(FD+'fig20a_fault_detection.csv')
st=D[D.kind.str.contains('step',case=False,na=False)]
fig,axes=plt.subplots(1,2,figsize=(W2,43*MM))
ax=axes[0]
for g,col in zip(['k=1','k=2','k>=3'],[C['orange'],C['teal'],C['blue']]):
    h=st[st.group==g].groupby('magnitude').detection_rate.mean()
    ax.plot(h.index,h.values,marker='o',ms=3.4,color=col,label=str(g))
ax.set_xlabel('Step size (m, or $Q$ equivalent)'); ax.set_ylabel('Detection rate')
ax.set_ylim(0,1); title(ax,'Detection of step faults'); panel(ax,'a')
ax.legend(loc='best',title='inputs',title_fontsize=6.6)
ax=axes[1]
kd=D.groupby(['group','kind']).detection_rate.mean().unstack()
ks=[g for g in ['k=1','k=2','k>=3'] if g in kd.index]; kinds=[c for c in kd.columns][:4]
x=np.arange(len(kinds)); w=0.8/len(ks)
for i,g in enumerate(ks):
    ax.bar(x+(i-(len(ks)-1)/2)*w,kd.loc[g,kinds],w,color=[C['orange'],C['teal'],C['blue']][i%3],
           edgecolor='white',lw=0.5,label=str(g))
ax.set_xticks(x); ax.set_xticklabels(kinds,fontsize=6.6); ax.set_ylabel('Detection rate')
ax.set_ylim(0,1); title(ax,'Detection by fault type'); panel(ax,'b')
ax.legend(loc='best',title='inputs',title_fontsize=6.6)
fig.tight_layout(pad=0.4); save(fig,'fig19_fault_detection')

# ============ algorithm comparison =========================================
A=pd.read_csv(FD+'fig21_algorithm_comparison.csv')
fig,axes=plt.subplots(1,2,figsize=(W2,43*MM))
ax=axes[0]
pure=A[A.f_nontree==0]
for col,lab,c in (('dp_s','tree DP',C['blue']),('ilp_s','integer programming',C['teal']),
                  ('greedy_s','greedy',C['orange']),('nsga_s','NSGA-II',C['purple'])):
    g=pure.groupby('N')[col].median()
    ax.plot(g.index,g.values,marker='o',ms=3.0,color=c,label=lab)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Network size (nodes)'); ax.set_ylabel('Runtime (s)')
title(ax,'Runtime on pure forests'); panel(ax,'a'); ax.legend(loc='best')
ax=axes[1]
for col,lab,c in (('gap_greedy_pct','greedy',C['orange']),('gap_nsga_pct','NSGA-II',C['purple'])):
    g=A.groupby('f_nontree')[col].agg(['mean','max'])
    ax.plot(100*g.index,g['mean'],marker='o',ms=3.0,color=c,label=lab)
    ax.fill_between(100*g.index,0,g['max'],color=c,alpha=0.14,lw=0)
ax.set_xlabel('Non-tree edges (%)'); ax.set_ylabel('Optimality gap (%)')
title(ax,'Optimality gap'); panel(ax,'b'); ax.legend(loc='best')
fig.tight_layout(pad=0.4); save(fig,'fig20_algorithms')
print("batch C ok")
