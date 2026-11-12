# -*- coding: utf-8 -*-
"""Component reliability: main-text panels (a,b) and the appendix Weibull figure."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import pandas as pd, numpy as np
from scipy.stats import spearmanr
FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'

# ---------------- main text: availability and its two components ----------
A=pd.read_csv(FD+'fig03_availability_distribution.csv',dtype={'station':str})
B=pd.read_csv(FD+'fig04_mtbf_vs_mttr.csv',dtype={'station':str}).dropna(subset=['MTBF_days','MTTR_days','availability'])
B=B[(B.MTBF_days>0)&(B.MTTR_days>0)&(B.availability<1)]
fig,axes=plt.subplots(1,3,figsize=(W2,49*MM))

ax=axes[0]
bins=np.linspace(0.70,1.0,31)
for v,c,lab in (('H',C['red'],'$H$'),('Q',C['blue'],'$Q$'),('either',C['grey'],'either')):
    g=A[A.variable==v].availability.dropna()
    ax.hist(g,bins=bins,histtype='step',lw=1.3,color=c,label='%s  (median %.3f)'%(lab,g.median()))
ax.set_xlabel('Station availability $A_u$'); ax.set_ylabel('Stations')
ax.set_xlim(0.70,1.005); title(ax,'Observed availability'); panel(ax,'a')
ax.legend(loc='best')

ax=axes[1]
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('seq',SEQ[::-1])
sc=ax.scatter(B.MTBF_days,B.MTTR_days,c=B.availability,cmap=cmap,s=16,
              linewidths=0.3,edgecolors='#555555',vmin=0.76,vmax=1.0,zorder=3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('MTBF (days)'); ax.set_ylabel('MTTR (days)')
cb=fig.colorbar(sc,ax=ax,pad=0.02,fraction=0.046); cb.set_label('$A_u$',fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
cb.outline.set_linewidth(0.5)
for a_ in (0.90,0.95,0.99):
    xx=np.logspace(np.log10(B.MTBF_days.min()),np.log10(B.MTBF_days.max()),50)
    ax.plot(xx,xx*(1-a_)/a_,color=C['lgrey'],lw=0.7,ls=(0,(4,2)),zorder=1)
    ax.text(xx[0],xx[0]*(1-a_)/a_,' $A_u$=%.2f'%a_,fontsize=6.0,color=C['grey'],va='bottom',ha='left')
rs_t=spearmanr(B.availability,B.MTTR_days).correlation
rs_b=spearmanr(B.availability,B.MTBF_days).correlation
y=np.log(1-B.availability.values); X=np.column_stack([np.log(B.MTBF_days),np.log(B.MTTR_days)])
Xc=(X-X.mean(0))/X.std(0); yc=(y-y.mean())/y.std()
bb,*_=np.linalg.lstsq(np.column_stack([np.ones(len(Xc)),Xc]),yc,rcond=None)
ax.text(0.035,0.975,'$\\rho_s$(MTTR) $=%.2f$\n$\\rho_s$(MTBF) $=%+.2f$'%(rs_t,rs_b),
        transform=ax.transAxes,fontsize=6.0,va='top',ha='left',
        bbox=dict(fc='white',ec='none',alpha=0.88,pad=1.4))
title(ax,'Availability against its two components'); panel(ax,'b')

# ---- (c) common-cause factor against the independent null ----------------
O='/home/claude/work/out/'
bdat=pd.read_csv(FD+'fig06c_common_cause_beta.csv').set_index('quantity').value
nul=pd.read_csv(O+'x5_fig02a_beta_null_distribution.csv')
pv=pd.read_csv(O+'x20_commoncause_pvalue.csv').iloc[0]
ax=axes[2]
ax.hist(nul.beta_null,bins=20,color=SEQ[1],edgecolor=SEQ[2],lw=0.6,
        label='null, $n=%d$'%pv.n_null_replicates)
ax.axvline(float(bdat['beta_factor_observed']),color=C['red'],lw=1.4,
           label='observed $\\beta=%.2f$,  $p=%.2f$'%(bdat['beta_factor_observed'],pv.empirical_p))
ax.axvline(float(bdat['beta_null_p95']),color=C['grey'],lw=0.9,ls=(0,(4,2)),label='null 95th pct.')
ax.set_xlabel('Common-cause factor $\\beta$'); ax.set_ylabel('Null replicates')
ax.set_ylim(0,ax.get_ylim()[1]*1.38)
ax.legend(loc='upper left',fontsize=6.0,handlelength=1.4,handletextpad=0.4,borderpad=0.3)
panel(ax,'c')
fig.tight_layout(pad=0.4,w_pad=1.4); save(fig,'fig06_component_reliability')

# ---------------- appendix: Weibull behaviour -----------------------------
P=pd.read_csv(FD+'fig01_weibull_probplot.csv',dtype={'station':str})
W=pd.read_csv(FD+'fig02_weibull_parameters.csv',dtype={'station':str})
W=W[(W.quantity=='time_between_failures_days')&W['shape'].notna()].sort_values('shape').reset_index(drop=True)
fig,axes=plt.subplots(1,2,figsize=(W2,50*MM))

ax=axes[0]
GC={'low':C['red'],'median':C['orange'],'high':C['blue']}
for grp,g in P[P.quantity=='time_between_failures_days'].groupby('availability_group'):
    for st,h in g.groupby('station'):
        ax.plot(h.x_ln_t,h.y_ln_neg_ln_1mF,ls='',marker='o',ms=2.4,mfc=GC.get(grp,C['grey']),
                mec='none',alpha=0.75,zorder=3)
        ax.plot(h.x_ln_t,h.y_fitted,color=GC.get(grp,C['grey']),lw=0.9,alpha=0.85,zorder=2)
import matplotlib.lines as mlines
ax.legend(handles=[mlines.Line2D([],[],ls='',marker='o',ms=3.2,color=GC[k],
          label='%s availability'%k) for k in ('low','median','high')],loc='lower right')
ax.set_xlabel('$\\ln t$  (time between failures, days)'); ax.set_ylabel('$\\ln[-\\ln(1-F)]$')
title(ax,'Weibull probability plot, six example stations'); panel(ax,'a')

ax=axes[1]
x=np.arange(len(W))
ax.vlines(x,W.shape_ci_lo,W.shape_ci_hi,color=C['lgrey'],lw=0.7,zorder=2)
below=(W.shape_ci_hi<1).values
above=(W.shape_ci_lo>1).values
mid=~below&~above
ax.plot(x[below],W['shape'][below],ls='',marker='o',ms=2.6,color=C['blue'],zorder=3,
        label='interval entirely below 1  (%d)'%below.sum())
ax.plot(x[mid],W['shape'][mid],ls='',marker='o',ms=2.6,mfc='white',mec=C['grey'],
        mew=0.7,zorder=3,label='interval includes 1  (%d)'%mid.sum())
ax.plot(x[above],W['shape'][above],ls='',marker='o',ms=2.6,color=C['red'],zorder=3,
        label='interval entirely above 1  (%d)'%above.sum())
ax.axhline(1.0,color=C['red'],lw=0.9,zorder=4)
ax.text(len(W)*0.42,1.06,'exponential  ($\\beta=1$)',fontsize=6.3,color=C['red'],ha='left',va='bottom')
from matplotlib.ticker import NullFormatter, FixedLocator
ax.set_yscale('log'); ax.set_ylim(0.18,6.0)
ax.yaxis.set_major_locator(FixedLocator([0.2,0.5,1,2,5]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticklabels(['0.2','0.5','1','2','5'])
ax.set_xlabel('Station, ordered by shape parameter'); ax.set_ylabel('Weibull shape')
title(ax,'Shape of the time between failures, all stations'); panel(ax,'b'); ax.legend(loc='best')
fig.tight_layout(pad=0.4); save(fig,'figA_weibull')
print("ok")
