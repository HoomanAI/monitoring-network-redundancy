# -*- coding: utf-8 -*-
"""Maps: study area (two panels) and the recommended design."""
import sys, os; sys.path.insert(0,'/home/claude/work')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/paper_v4')
from nstyle import *
import pandas as pd, numpy as np, networkx as nx
import basemap
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'

N=pd.read_csv(FD+'fig22a_design_map_nodes.csv',dtype={'station':str})
L=pd.read_csv(FD+'fig22b_design_map_added_links.csv',dtype={'target':str,'added_input':str})
Cn=pd.read_csv(FD+'fig22c_candidate_new_node_reaches.csv',dtype={'target':str})
Cn=Cn.sort_values('rank').groupby('target',as_index=False).first()
def rfac(b): return 1/np.cos(np.radians((b[1]+b[3])/2))
def fit(b,ar=1.52):
    r=rfac(b); w=b[2]-b[0]; h=(b[3]-b[1])*r; cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
    if w/h<ar: w=ar*h
    else: h=w/ar
    return (cx-w/2,cy-h/(2*r),cx+w/2,cy+h/(2*r))
B=fit((-82.6,42.2,-74.3,46.4))

fig,ax=plt.subplots(figsize=(W2,W2/1.52))
basemap.draw(ax,B,rivers='wide',min_order=3,river_scale=1.2)
for _,r in L.iterrows():
    ax.plot([r.input_lon,r.target_lon],[r.input_lat,r.target_lat],color=C['blue'],
            lw=0.45,alpha=0.42,zorder=3)
kept=N[N.status.str.contains('kept|keep|active',case=False,na=False)]
rem =N[N.status.str.contains('remov|retire',case=False,na=False)]
crit=N[N.critical_top15_current==True] if 'critical_top15_current' in N else N.iloc[:0]
ax.scatter(rem.lon_deg,rem.lat_deg,s=16,facecolor='white',edgecolor=C['grey'],lw=0.7,zorder=5)
ax.scatter(kept.lon_deg,kept.lat_deg,s=14,c=C['ink'],lw=0,zorder=6)
ax.scatter(crit.lon_deg,crit.lat_deg,s=44,facecolor='none',edgecolor=C['red'],lw=1.0,zorder=7)
if len(Cn): ax.scatter(Cn.lon_deg,Cn.lat_deg,s=22,marker='^',c=C['green'],lw=0,zorder=6)
ax.set_xlim(B[0],B[2]); ax.set_ylim(B[1],B[3]); ax.set_aspect(rfac(B))
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_linewidth(0.6); s_.set_color(C['ink'])
ax.legend(handles=[
    Line2D([],[],ls='',marker='o',color=C['ink'],ms=3.6,label='station kept'),
    Line2D([],[],ls='',marker='o',mfc='white',mec=C['grey'],ms=4.0,label='station retired'),
    Line2D([],[],ls='',marker='o',mfc='none',mec=C['red'],ms=6.4,label='15 most critical'),
    Line2D([],[],ls='',marker='^',color=C['green'],ms=4.2,label='candidate new station'),
    Line2D([],[],color=C['blue'],lw=1.0,alpha=0.6,label='statistical link added')],
    loc='lower right',frameon=True,facecolor='white',framealpha=0.92,edgecolor='none',fontsize=6.8)
save(fig,'fig21_final_design')
print("map ok | kept %d removed %d critical %d candidates %d links %d"%(len(kept),len(rem),len(crit),len(Cn),len(L)))
