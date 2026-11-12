# -*- coding: utf-8 -*-
"""Figure for Results II: predictive validation on the held-out period 2017-2020."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import numpy as np, pandas as pd
from pathlib import Path
OUTD=Path('/home/claude/work/out')
S=pd.read_csv(OUTD/'x23_heldout_skill.csv')
G1,G2=0.6,0.8
MM=1/25.4
fig,ax=plt.subplots(1,2,figsize=(W2,47*MM))

# ---- (a) paired NSE, validation vs held-out -------------------------------
a=ax[0]
for dn,lab,c,mk in (('current','Current network',C['grey'],'o'),
                    ('k2_rec23','$k=2$ design',C['blue'],'s')):
    g=S[S.design==dn].dropna(subset=['nse_val','nse_test'])
    a.scatter(g.nse_val,g.nse_test,s=11,facecolor=c,edgecolor='none',alpha=0.75,marker=mk,label=lab,zorder=3)
lim=(0.0,1.0)
a.plot(lim,lim,color=C['ink'],lw=0.7,ls='-',zorder=2)
for t in (G1,G2):
    a.axhline(t,color=C['lgrey'],lw=0.6,ls=':',zorder=1); a.axvline(t,color=C['lgrey'],lw=0.6,ls=':',zorder=1)
a.text(0.02,G1+0.012,'G1',fontsize=6.5,color=C['grey']); a.text(0.02,G2+0.012,'G2',fontsize=6.5,color=C['grey'])
a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal')
a.set_xlabel('series NSE, 2013$-$2016'); a.set_ylabel('series NSE, 2017$-$2020')
a.legend(loc='upper left',fontsize=6.8,handletextpad=0.4,borderpad=0.35)
panel(a,'(a)')

# ---- (b) state shares ------------------------------------------------------
b=ax[1]
grp=[('current','val'),('current','test'),('k2_rec23','val'),('k2_rec23','test')]
lab=['Current\n2013$-$2016','Current\n2017$-$2020','$k=2$\n2013$-$2016','$k=2$\n2017$-$2020']
g2=[];g1=[];g0=[]
for dn,per in grp:
    v=S[S.design==dn]['nse_'+per].dropna()
    g2.append(100*(v>=G2).mean()); g1.append(100*((v>=G1)&(v<G2)).mean()); g0.append(100*(v<G1).mean())
x=np.arange(4); w=0.62
b.bar(x,g2,w,color=C['blue'],label='G2',zorder=3)
b.bar(x,g1,w,bottom=g2,color=C['teal'],label='G1',zorder=3)
b.bar(x,g0,w,bottom=np.array(g2)+np.array(g1),color=C['red'],label='below G1',zorder=3)
for i in range(4):
    b.text(i,g2[i]/2,'%.0f'%g2[i],ha='center',va='center',fontsize=6.5,color='white')
    b.text(i,g2[i]+g1[i]/2,'%.0f'%g1[i],ha='center',va='center',fontsize=6.5,color='white')
b.set_xticks(x); b.set_xticklabels(lab,fontsize=6.8)
b.set_ylabel('share of series (\\%)'); b.set_ylim(0,112)
b.set_yticks([0,20,40,60,80,100])
b.legend(loc='upper center',ncol=3,fontsize=6.8,handletextpad=0.4,columnspacing=0.9,borderpad=0.3,frameon=True)
panel(b,'(b)')
fig.tight_layout(pad=0.4,w_pad=1.6)
save(fig,'fig27_heldout')
print('saved')
