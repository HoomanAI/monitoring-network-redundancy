# -*- coding: utf-8 -*-
"""Figure 2: two branches that converge, one chain, two loops.
The arrows carry the quantities that pass between the steps."""
import sys; sys.path.insert(0,'/home/claude/work')
from nstyle import *
import matplotlib.patches as mp

MM=1/25.4
FW,FH=137.0,96.0
fig=plt.figure(figsize=(FW*MM,FH*MM))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,FW); ax.set_ylim(0,FH); ax.axis('off')
K='#000000'; G='#5A5A5A'; LG='#AAAAAA'
DASH=(0,(2.6,1.9))
TOP,MID,BOT=86.0,62.0,28.0
BH=15.0

def step(x,w,yc,num,title,sub):
    ax.add_patch(mp.Rectangle((x,yc-BH/2),w,BH,lw=0.6,edgecolor=K,facecolor='white',zorder=3))
    ax.text(x+2.3,yc+BH/2-3.2,num,ha='left',va='center',fontsize=7.0,color=K,fontweight='bold',zorder=5)
    ax.text(x+6.6,yc+BH/2-3.2,title,ha='left',va='center',fontsize=7.0,color=K,zorder=5)
    ax.plot([x,x+w],[yc+BH/2-6.0,yc+BH/2-6.0],lw=0.45,color=LG,zorder=4)
    ax.text(x+w/2,yc-2.4,sub,ha='center',va='center',fontsize=6.1,color=G,zorder=5,linespacing=1.35)
    return (x,x+w,yc)

def token(x,w,yc,txt,h=11.0):
    ax.add_patch(mp.Rectangle((x,yc-h/2),w,h,lw=0.6,edgecolor=K,facecolor='white',
                 linestyle=DASH,zorder=3))
    ax.text(x+w/2,yc,txt,ha='center',va='center',fontsize=6.1,color=G,zorder=5,linespacing=1.4)
    return (x,x+w,yc)

def ar(p,q,ls='-'):
    ax.annotate('',xy=q,xytext=p,zorder=6,arrowprops=dict(arrowstyle='-|>',lw=0.7,color=K,
                ls=ls,mutation_scale=7,shrinkA=0,shrinkB=0))
def seg(x0,y0,x1,y1,ls='-'):
    ax.plot([x0,x1],[y0,y1],lw=0.7,color=K,ls=ls,zorder=6)

i1=token(1.5,25.0,TOP,'river network and\nfrozen station links')
i2=token(1.5,25.0,MID,'station outage\nrecord, 2005$-$2016')
b1=step(33.0,36.0,TOP,'1','Graph construction','physical and statistical edges\nover a forest of in-trees')
b3=step(78.0,36.0,TOP,'3','Structure function','ridge models over every\nsubset of each pool')
b2=step(33.0,36.0,MID,'2','Component reliability','outage runs to MTBF,\nMTTR and availability')
b4=step(1.5,36.0,BOT,'4','System evaluation','reliability, cut sets,\nimportance and resilience')
b5=step(48.0,38.0,BOT,'5','Allocation','tree DP, integer programming,\ngreedy search and NSGA-II')
b6=step(94.5,41.0,BOT,'6','Verification','four model families, repair\nwhere three disagree')

ar((i1[1],TOP),(b1[0],TOP))
ar((i2[1],MID),(b2[0],MID))
ar((b1[1],TOP),(b3[0],TOP))
ax.text((b1[1]+b3[0])/2,TOP+2.6,'$N(t)$',ha='center',va='center',fontsize=6.3,color=K,zorder=7)

yA,yP,yR=48.0,44.0,40.0
# component reliability down into system evaluation
seg(41.0,MID-BH/2,41.0,yA); seg(41.0,yA,12.0,yA); ar((12.0,yA),(12.0,BOT+BH/2))
ax.text(13.6,yA-4.0,'$A_u,\\ z(\\tau)$',ha='left',va='center',fontsize=6.3,color=K,zorder=7)
# structure function down into system evaluation
seg(96.0,TOP-BH/2,96.0,yP); seg(96.0,yP,30.0,yP); ar((30.0,yP),(30.0,BOT+BH/2))
ax.text(31.6,yP-4.0,'$\\bar\\varphi_t$',ha='left',va='center',fontsize=6.3,color=K,zorder=7)

ar((b4[1],BOT),(b5[0],BOT))
ax.text((b4[1]+b5[0])/2,BOT+3.4,'$R,\\kappa_t,$\n$I^{B}$',ha='center',va='center',fontsize=6.0,color=K,zorder=7,linespacing=1.25)
ar((b5[1],BOT),(b6[0],BOT))
ax.text((b5[1]+b6[0])/2,BOT+3.0,'$(X,L)$',ha='center',va='center',fontsize=6.0,color=K,zorder=7)

# repair loop, verification back to allocation
seg(115.0,BOT+BH/2,115.0,yR,DASH); seg(115.0,yR,68.0,yR,DASH); ar((68.0,yR),(68.0,BOT+BH/2),DASH)
ax.text(91.5,yR+2.2,'repair a failing series',ha='center',va='center',fontsize=6.1,color=G,zorder=7)
# recoverability cutting loop inside allocation
yc_=BOT-BH/2-5.5
seg(53.0,BOT-BH/2,53.0,yc_,DASH); seg(53.0,yc_,79.0,yc_,DASH); ar((79.0,yc_),(79.0,BOT-BH/2),DASH)
ax.text(66.0,yc_-3.2,'force back an unrecoverable station',ha='center',va='center',
        fontsize=6.1,color=G,zorder=7)
# output
oy=10.5
ax.add_patch(mp.Rectangle((94.5,oy-5.0),41.0,10.0,lw=0.6,edgecolor=K,facecolor='white',
             linestyle=DASH,zorder=3))
ax.text(115.0,oy,'stations to retire, links to add,\nmaintenance priority, new sites',
        ha='center',va='center',fontsize=6.1,color=G,zorder=5,linespacing=1.4)
ar((115.0,BOT-BH/2),(115.0,oy+5.0))
save(fig,'fig26_framework')
print('saved')
