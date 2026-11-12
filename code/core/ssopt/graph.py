"""Directed station graph (edge = upstream -> downstream) and candidate-input pools."""
import numpy as np
import pandas as pd
import networkx as nx
from . import config as C


def frozen_graph(links):
    G = nx.DiGraph()
    G.add_edges_from(zip(links.upstream_station, links.target_station))
    return G


def augmented_graph(links, added):
    G = frozen_graph(links)
    for a, b in added:
        G.add_edge(a, b)
    return G


def graph_table(G, stations, role, targets):
    """Per-station graph descriptors: sub-network id, chain depth (longest upstream path), ancestors, descendants."""
    H = G.copy()
    H.add_nodes_from(stations)
    comp = {}
    for i, c in enumerate(sorted(nx.weakly_connected_components(H), key=lambda c: (-len(c), sorted(c)[0]))):
        for s in c:
            comp[s] = i + 1
    rows = []
    for s in stations:
        anc = nx.ancestors(H, s); des = nx.descendants(H, s)
        up_depth = max((nx.shortest_path_length(H, a, s) for a in anc), default=0)  # DAG; longest below
        # longest path length from any source to s
        lp = 0
        for a in anc:
            for p in nx.all_simple_paths(H, a, s):
                lp = max(lp, len(p) - 1)
        rows.append(dict(station=s, role=role[s], subnetwork=comp[s], n_ancestors=len(anc),
                         n_descendants=len(des), chain_depth_up=lp,
                         ancestors=",".join(sorted(anc)), descendants=",".join(sorted(des))))
    return pd.DataFrame(rows)


def chains(G):
    """Maximal source->sink chains (paths) of the graph."""
    out = []
    srcs = [n for n in G if G.in_degree(n) == 0]
    sinks = [n for n in G if G.out_degree(n) == 0]
    for a in srcs:
        for b in sinks:
            for p in nx.all_simple_paths(G, a, b):
                out.append(p)
    return out


def corr_ranking(d, t, v, exclude, train_mask):
    """|Pearson r| (train period only) between target series and every other station, same variable."""
    y = d.mod_arr[v][t]
    res = {}
    for s in d.stations:
        if s in exclude:
            continue
        x = d.mod_arr[v][s]
        ok = train_mask & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < C.MIN_OVERLAP:
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(r):
            res[s] = abs(r)
    return sorted(res, key=res.get, reverse=True), res


def build_pools(d, G_conn, variant, train_mask):
    """pools[(t,v)] = dict(direct=[...], inputs=[...], connected=set, corr_only=set).
    variant 'pool'      : direct + graph ancestors + top-5 correlated (train only)
    variant 'connected' : direct + graph ancestors only (hydrologically connected)"""
    direct = d.links.groupby("target_station").upstream_station.apply(list).to_dict()
    pools = {}
    for t in d.targets:
        anc = sorted(nx.ancestors(G_conn, t)) if t in G_conn else []
        conn = list(dict.fromkeys(direct.get(t, []) + anc))
        for v in ("H", "Q"):
            corr_only = []
            if variant == "pool":
                rank, _ = corr_ranking(d, t, v, set(conn) | {t}, train_mask)
                corr_only = rank[: C.TOP_CORR]
            pools[(t, v)] = dict(direct=list(direct.get(t, [])), connected=conn, corr_only=corr_only,
                                 inputs=conn + corr_only)
    return pools
