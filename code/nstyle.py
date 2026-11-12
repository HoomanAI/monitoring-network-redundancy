# -*- coding: utf-8 -*-
"""Figure style matching the ENSO/North America paper: Arial-metric sans, full axes
box, lowercase bold panel letters, panel titles, saturated semantic palette."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path
MM=1/25.4; W1=67*MM; W2=137*MM; W15=104*MM
OUT=Path('/home/claude/figs2'); OUT.mkdir(exist_ok=True)
DPI=900
plt.rcParams.update({
 'font.family':'sans-serif',
 'font.sans-serif':['Arial','Helvetica','Liberation Sans','Nimbus Sans','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'axes.linewidth':0.6,
 'xtick.direction':'out','ytick.direction':'out',
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.6,'ytick.major.size':2.6,
 'xtick.minor.width':0.45,'ytick.minor.width':0.45,'xtick.minor.size':1.4,'ytick.minor.size':1.4,
 'lines.linewidth':1.1,'lines.markersize':3.2,
 'axes.spines.top':True,'axes.spines.right':True,          # full box, as in the reference
 'legend.frameon':True,'legend.framealpha':0.88,'legend.edgecolor':'none',
 'legend.facecolor':'white','legend.handlelength':1.5,'legend.labelspacing':0.32,
 'legend.borderpad':0.2,'legend.handletextpad':0.5,'legend.columnspacing':1.0,
 'axes.grid':False,'savefig.dpi':DPI,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','figure.dpi':150,'axes.axisbelow':True,
})
# --- semantic palette, saturated, print-safe ------------------------------
C={'red':'#C0392B','blue':'#1F5FA9','green':'#2E8B57','purple':'#7B4EA3','orange':'#E08214',
   'teal':'#179E8E','grey':'#6E6E6E','ink':'#111111','lgrey':'#B8B8B8'}
BAND={'a':'#E3F0E6','b':'#EFE6F3'}                     # light highlight bands
COL={'current':'#4D4D4D','k1_P1':'#E08214','k1_P2':'#C0392B','k2_P1':'#E08214',
     'k2_P2':'#1F5FA9','k2_P2_rec':'#2E8B57','k2_P2_hard':'#7B4EA3','k3_P2':'#7B4EA3'}
LAB={'current':'Current','k1_P1':'$k$=1, physical','k1_P2':'$k$=1, +statistical',
     'k2_P1':'$k$=2, physical','k2_P2':'$k$=2, +statistical','k3_P2':'$k$=3, +statistical',
     'k2_P2_rec':'$k$=2, +recoverability','k2_P2_hard':'$k$=2, +hardening'}
# compact two-line forms for crowded category axes
LABC={'current':'Current','k1_P1':'$k$=1\nphys','k1_P2':'$k$=1\nstat',
      'k2_P1':'$k$=2\nphys','k2_P2':'$k$=2\nstat','k3_P2':'$k$=3\nstat',
      'k2_P2_rec':'$k$=2\nrecov','k2_P2_hard':'$k$=2\nhard'}
G2C,G1C,G0C='#1F5FA9','#8FB8DE','#E08214'
SEQ=['#DEEBF7','#9ECAE1','#4A90C4','#1F5FA9']

def panel(ax,l,x=-0.015,y=1.015,fs=9.0):
    """lowercase bold letter at the top-left, outside the axes"""
    ax.text(x,y,l,transform=ax.transAxes,fontweight='bold',fontsize=fs,ha='right',va='bottom')
def title(ax,s=None,fs=8.5):
    """titles are carried by the caption, not drawn above the axes"""
    return None
def zeroline(ax,y=0.0): ax.axhline(y,color=C['grey'],lw=0.5,zorder=1)
def tidy(ax):
    for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color(C['ink'])
def save(fig,name,pad=0.015):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=pad)
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',pad_inches=pad,dpi=600)
    plt.close(fig)
