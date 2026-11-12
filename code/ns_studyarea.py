# -*- coding: utf-8 -*-
"""Study area: southern Ontario and the in-tree of one river system. Two equal panels."""
import sys, os, numpy as np, pandas as pd, networkx as nx
from pathlib import Path
P4='/home/claude/ws/station_subset_optimization/paper_v4'
sys.path.insert(0,'/home/claude/work'); sys.path.insert(0,P4)
from nstyle import *
from style3 import n3
RED=C['red']
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import basemap

FD=Path('/home/claude/ws/station_subset_optimization/ress_v4/figdata')
OUT=Path('/home/claude/work/figures_x')
N=pd.read_csv(FD/'fig24a_graph_nodes.csv',dtype={'station':str})
E=pd.read_csv(FD/'fig24b_graph_edges.csv',dtype={'from_station':str,'to_station':str})
ph=E[E.type=='physical']
co={r.station:(r.lon_deg,r.lat_deg) for r in N.itertuples()}
role=dict(zip(N.station,N.role))
G=nx.DiGraph(); G.add_nodes_from(N.station); G.add_edges_from(zip(ph.from_station,ph.to_station))
COMP=sorted(max(nx.weakly_connected_components(G),key=len)); sub=G.subgraph(COMP).copy()

AR=float(os.environ.get('AR','1.33'))          # display aspect every panel is fitted to
def rfac(box): return 1/np.cos(np.radians((box[1]+box[3])/2))
def fit(box, ar=AR):
    """Grow a lon/lat box (never shrink) until its drawn aspect equals ar."""
    r=rfac(box); w=box[2]-box[0]; h=(box[3]-box[1])*r
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
    if w/h < ar: w=ar*h
    else:        h=w/ar
    return (cx-w/2, cy-h/(2*r), cx+w/2, cy+h/(2*r))

A=fit((-141.0,41.5,-52.0,70.0))
B=fit((-82.6,42.2,-74.3,46.4))
la=[co[s][1] for s in COMP]; lo=[co[s][0] for s in COMP]
C=fit((min(lo)-0.20,min(la)-0.14,max(lo)+0.20,max(la)+0.14))

MK={'source':dict(s=7,c='black',lw=0,ec='none'),
    'relay' :dict(s=20,c='white',lw=0.7,ec='black'),
    'sink'  :dict(s=26,c='black',lw=0.5,ec='black')}
def stations(ax,box,scale=1.0,lw=0.5):
    ids=[s for s in N.station if box[0]<=co[s][0]<=box[2] and box[1]<=co[s][1]<=box[3]]
    S=set(ids)
    for _,r in ph.iterrows():
        if r.from_station in S and r.to_station in S:
            p_,q_=co[r.from_station],co[r.to_station]
            ax.plot([p_[0],q_[0]],[p_[1],q_[1]],color='#5f5f5f',lw=lw,zorder=3)
    for k,m in MK.items():
        pts=[co[s] for s in ids if role[s]==k]
        if pts: ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=m['s']*scale,c=m['c'],
                           marker='o',linewidths=m['lw'],edgecolors=m['ec'],zorder=5)
    return ids
def frame(ax,box):
    ax.set_xlim(box[0],box[2]); ax.set_ylim(box[1],box[3]); ax.set_aspect(rfac(box))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_color('#8f8f8f'); s_.set_linewidth(0.6)
def box_on(ax,box):
    ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                           fill=False,ec=RED,lw=1.0,zorder=9))

fig=plt.figure(figsize=(W2,W2*float(os.environ.get('HF','0.42'))))
gs=fig.add_gridspec(1,2,width_ratios=[1,1],
                    wspace=0.045,left=0.006,right=0.996,top=0.955,bottom=0.012)
b=fig.add_subplot(gs[0,0]); d=fig.add_subplot(gs[0,1])

basemap.draw(b,B,rivers='wide',min_order=3,river_scale=1.25); stations(b,B,scale=1.05,lw=0.55); frame(b,B); box_on(b,C)
b.legend(handles=[Line2D([],[],ls='',marker='o',color='black',ms=2.6,label='source'),
                  Line2D([],[],ls='',marker='o',mfc='white',mec='black',ms=4.2,label='relay'),
                  Line2D([],[],ls='',marker='o',color='black',ms=5.0,label='sink (target)'),
                  Line2D([],[],color='#5f5f5f',lw=0.9,label='physical link')],
         loc='lower right',fontsize=6.4,handletextpad=0.5,labelspacing=0.32,
         frameon=True,facecolor='white',framealpha=0.9,edgecolor='none')
panel(b,'a',-0.004,1.012)

def arrow(ax,p_,q_,rq=0.17,rp=0.09,lw=0.6):
    p_=np.array(p_,float); q_=np.array(q_,float); dv=q_-p_; u=dv/np.hypot(*dv)
    ax.annotate('',xy=q_-u*(rq+0.02),xytext=p_+u*rp,zorder=3,arrowprops=dict(
        arrowstyle='-|>,head_length=0.26,head_width=0.10',color='black',lw=lw,shrinkA=0,shrinkB=0))
root=[n_ for n_ in sub.nodes() if sub.out_degree(n_)==0][0]
pos={}; _x=[0.0]
def place(n_,lev):
    kids=sorted(sub.predecessors(n_),key=lambda s_: co[s_][0])
    if not kids:
        pos[n_]=(_x[0],lev); _x[0]+=1.0; return pos[n_][0]
    xs_=[place(k_,lev+1) for k_ in kids]
    pos[n_]=(float(np.mean(xs_)),lev); return pos[n_][0]
place(root,0)
pos={n_:(v[0]*1.15,v[1]*1.05) for n_,v in pos.items()}
for u,v in sub.edges(): arrow(d,pos[u],pos[v])
for s_ in sub.nodes(): n3(d,pos[s_][0],pos[s_][1],role[s_],s=1.15)
for s_ in sub.nodes():
    x0,y0=pos[s_]
    if sub.in_degree(s_)==0:
        d.annotate(s_[-4:],(x0,y0),textcoords='offset points',xytext=(0,5.6),ha='center',fontsize=5.3,color='#555555')
    else:
        near=[pos[o][0]-x0 for o in sub.nodes() if o!=s_ and abs(pos[o][1]-y0)<0.5]
        rgap=min([g for g in near if g>0], default=9.9)
        lgap=min([-g for g in near if g<0], default=9.9)
        right = rgap>=lgap
        d.annotate(s_[-4:],(x0,y0),textcoords='offset points',
                   xytext=(6.5 if right else -6.5, -1.6), ha='left' if right else 'right',
                   fontsize=5.3,color='#555555')
xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
x0,x1=min(xs)-1.6,max(xs)+1.4; y0,y1=min(ys)-0.9,max(ys)+1.0
w_=x1-x0; h_=y1-y0
if w_/h_<AR: m=(AR*h_-w_)/2; x0-=m; x1+=m
else:        m=(w_/AR-h_)/2; y0-=m; y1+=m
d.set_xlim(x0,x1); d.set_ylim(y0,y1); d.set_aspect('equal'); d.axis('off'); panel(d,'b',-0.004,1.012)
d.annotate('',xy=(min(xs)-1.05,min(ys)-0.3),xytext=(min(xs)-1.05,max(ys)+0.5),
           arrowprops=dict(arrowstyle='-|>,head_length=0.4,head_width=0.16',color='#999999',lw=0.7))
d.text(min(xs)-1.28,(min(ys)+max(ys))/2,'flow',rotation=90,va='center',ha='right',
       fontsize=6.4,style='italic',color='#999999')

save(fig,'fig02_study_area_2panel')
print("panels fitted to aspect %.2f; comp %d stations"%(AR,len(COMP)))
