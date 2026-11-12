# -*- coding: utf-8 -*-
"""Recommended design: regional map and a graph view of one river system."""
import sys; sys.path.insert(0,'/home/claude/work')
sys.path.insert(0,'/home/claude/ws/station_subset_optimization/paper_v4')
from nstyle import *
import pandas as pd, numpy as np, networkx as nx
import basemap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
FD='/home/claude/ws/station_subset_optimization/ress_v4/figdata/'
N=pd.read_csv(FD+'fig22a_design_map_nodes.csv',dtype={'station':str})
L=pd.read_csv(FD+'fig22b_design_map_added_links.csv',dtype={'target':str,'added_input':str})
Cn=pd.read_csv(FD+'fig22c_candidate_new_node_reaches.csv',dtype={'target':str})
Cn=Cn.sort_values('rank').groupby('target',as_index=False).first()
GN=pd.read_csv(FD+'fig24a_graph_nodes.csv',dtype={'station':str})
GE=pd.read_csv(FD+'fig24b_graph_edges.csv',dtype={'from_station':str,'to_station':str})
ph=GE[GE.type=='physical']
co={r.station:(r.lon_deg,r.lat_deg) for r in GN.itertuples()}
role=dict(zip(GN.station,GN.role))
G=nx.DiGraph(); G.add_nodes_from(GN.station); G.add_edges_from(zip(ph.from_station,ph.to_station))
COMP=sorted(max(nx.weakly_connected_components(G),key=len)); sub=G.subgraph(COMP).copy()
status=dict(zip(N.station,N.status))
import sys as _s; _s.path.insert(0,'/home/claude/ws/station_subset_optimization/ress_v4')
_M=pd.read_csv('/home/claude/work/out/x22_master_designs.csv')
REM23=set(_M[_M.design=='k2_P2_rec23'].removed_stations.iloc[0].split(','))
status={st:('removed' if st in REM23 else 'kept') for st in N.station}
crit=set(N[N.critical_top15_current==True].station) if 'critical_top15_current' in N else set()
def is_rem(s): return bool(pd.notna(status.get(s)) and ('remov' in str(status[s]).lower() or 'retire' in str(status[s]).lower()))
def rfac(b): return 1/np.cos(np.radians((b[1]+b[3])/2))
def fit(b,ar):
    r=rfac(b); w=b[2]-b[0]; h=(b[3]-b[1])*r; cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
    if w/h<ar: w=ar*h
    else: h=w/ar
    return (cx-w/2,cy-h/(2*r),cx+w/2,cy+h/(2*r))

fig=plt.figure(figsize=(W2,58*MM))
gs=fig.add_gridspec(1,2,width_ratios=[1,1],wspace=0.045,left=0.004,right=0.997,top=0.965,bottom=0.01)
A=fig.add_subplot(gs[0,0]); B_=fig.add_subplot(gs[0,1])

# ---------------- (a) regional map, stations only -------------------------
BB=fit((-82.6,42.2,-74.3,46.4),1.30)
basemap.draw(A,BB,rivers='wide',min_order=3,river_scale=1.15)
kept=N[~N.station.map(is_rem)]; rem=N[N.station.map(is_rem)]
A.scatter(rem.lon_deg,rem.lat_deg,s=19,facecolor='white',edgecolor=C['grey'],lw=0.8,zorder=5)
A.scatter(kept.lon_deg,kept.lat_deg,s=15,c=C['ink'],lw=0,zorder=6)
A.scatter(N[N.station.isin(crit)].lon_deg,N[N.station.isin(crit)].lat_deg,s=52,
          facecolor='none',edgecolor=C['red'],lw=1.0,zorder=7)
A.scatter(Cn.lon_deg,Cn.lat_deg,s=20,marker='^',c=C['green'],lw=0,zorder=6)
la=[co[s][1] for s in COMP]; lo=[co[s][0] for s in COMP]
CB=fit((min(lo)-0.22,min(la)-0.16,max(lo)+0.22,max(la)+0.16),1.30)
A.add_patch(Rectangle((CB[0],CB[1]),CB[2]-CB[0],CB[3]-CB[1],fill=False,ec=C['red'],lw=1.0,zorder=9))
A.set_xlim(BB[0],BB[2]); A.set_ylim(BB[1],BB[3]); A.set_aspect(rfac(BB))
A.set_xticks([]); A.set_yticks([])
for s_ in A.spines.values(): s_.set_linewidth(0.6); s_.set_color(C['ink'])
A.legend(handles=[
  Line2D([],[],ls='',marker='o',color=C['ink'],ms=3.6,label='station kept'),
  Line2D([],[],ls='',marker='o',mfc='white',mec=C['grey'],ms=4.2,label='station retired'),
  Line2D([],[],ls='',marker='o',mfc='none',mec=C['red'],ms=6.6,label='15 most critical'),
  Line2D([],[],ls='',marker='^',color=C['green'],ms=4.2,label='candidate new station')],
  loc='lower right',fontsize=6.4,labelspacing=0.34,framealpha=0.92)
panel(A,'a',-0.004,1.012)

# ---------------- (b) graph of one river system ---------------------------
root=[n for n in sub.nodes() if sub.out_degree(n)==0][0]
pos={}; _x=[0.0]
def place(n,lev):
    kids=sorted(sub.predecessors(n),key=lambda s: co[s][0])
    if not kids: pos[n]=(_x[0],lev); _x[0]+=1.0; return pos[n][0]
    xs=[place(k,lev+1) for k in kids]; pos[n]=(float(np.mean(xs)),lev); return pos[n][0]
place(root,0)
pos={n:(v[0]*1.18,v[1]*1.06) for n,v in pos.items()}
CS=set(COMP)
for _,r in L.iterrows():
    if r.target in CS and r.added_input in CS:
        p,q=pos[r.added_input],pos[r.target]
        B_.annotate('',xy=q,xytext=p,zorder=2,arrowprops=dict(
            arrowstyle='-|>,head_length=0.22,head_width=0.10',color=C['blue'],lw=0.9,
            linestyle=(0,(2.6,1.6)),shrinkA=5,shrinkB=6,connectionstyle='arc3,rad=0.22'))
for u,v in sub.edges():
    p,q=np.array(pos[u],float),np.array(pos[v],float); d=q-p; un=d/np.hypot(*d)
    B_.annotate('',xy=q-un*0.20,xytext=p+un*0.12,zorder=3,arrowprops=dict(
        arrowstyle='-|>,head_length=0.26,head_width=0.11',color=C['ink'],lw=0.8,shrinkA=0,shrinkB=0))
_order=sorted(sub.nodes(),key=lambda n:(round(pos[n][1],2),pos[n][0]))
_OFF={n:(7.0 if i%2==0 else -9.4) for i,n in enumerate(_order)}
for s in sub.nodes():
    x,y=pos[s]
    if is_rem(s): B_.add_patch(Circle((x,y),0.15,fc='white',ec=C['grey'],lw=0.9,ls=(0,(1.8,1.2)),zorder=5))
    else:
        r=0.19 if role.get(s)=='sink' else 0.14
        B_.add_patch(Circle((x,y),r,fc=C['ink'],ec='none',zorder=5))
        if role.get(s)=='relay':
            B_.add_patch(Circle((x,y),0.21,fc='none',ec=C['ink'],lw=0.7,zorder=5))
    if s in crit: B_.add_patch(Circle((x,y),0.30,fc='none',ec=C['red'],lw=1.0,zorder=6))
    _dy=_OFF[s]
    B_.annotate(s[-4:],(x,y),textcoords='offset points',xytext=(0,_dy),ha='center',
                fontsize=4.9,color='#555555',zorder=7,
                bbox=dict(boxstyle='square,pad=0.08',fc='white',ec='none',alpha=0.85))
xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
x0,x1=min(xs)-1.0,max(xs)+0.9; y0,y1=min(ys)-0.75,max(ys)+0.95
w_,h_=x1-x0,y1-y0; ar=1.30
if w_/h_<ar: m=(ar*h_-w_)/2; x0-=m; x1+=m
else: m=(w_/ar-h_)/2; y0-=m; y1+=m
B_.set_xlim(x0,x1); B_.set_ylim(y0,y1); B_.set_aspect('equal'); B_.axis('off')
B_.annotate('',xy=(min(xs)-0.70,min(ys)-0.25),xytext=(min(xs)-0.70,max(ys)+0.45),
            arrowprops=dict(arrowstyle='-|>,head_length=0.4,head_width=0.16',color='#999999',lw=0.7))
B_.text(min(xs)-0.86,(min(ys)+max(ys))/2,'flow',rotation=90,va='center',ha='right',
        fontsize=6.2,style='italic',color='#999999')
B_.legend(handles=[
  Line2D([],[],ls='',marker='o',color=C['ink'],ms=4.4,label='sink (target)'),
  Line2D([],[],ls='',marker='o',mfc=C['ink'],mec=C['ink'],ms=3.2,label='source'),
  Line2D([],[],ls='',marker='o',mfc='white',mec=C['grey'],ms=4.4,label='retired'),
  Line2D([],[],color=C['ink'],lw=1.0,label='physical link'),
  Line2D([],[],color=C['blue'],lw=1.0,ls=(0,(2.6,1.6)),label='statistical link added')],
  loc='lower right',fontsize=6.3,labelspacing=0.34,framealpha=0.92)
panel(B_,'b',-0.004,1.012)
save(fig,'fig21_final_design')
print("design map ok | component %d stations | retired in it %d"%(len(COMP),sum(is_rem(s) for s in COMP)))
