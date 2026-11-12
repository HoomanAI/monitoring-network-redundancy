"""1.1 (designs): exact multi-state k-redundant designs. k-redundant = service state >= G1 (NSE >= OK threshold) survives any k-1 input failures.
Exact ILP + tree DP check + greedy, for the main thresholds and the two sensitivity pairs (G1 boundary = second element of each pair)."""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, networkx as nx
import rv4
from rv4 import config as K
from rv4.design import Problem
from rv3 import common as cm
from rv3.design import timeit

def main():
    G, nontree = cm.physical_forest(); comps = [sorted(c) for c in nx.weakly_connected_components(G)]
    settings = [(K.GOOD, K.OK)] + list(K.SENSITIVITY); designs = {}; rows = []; crow = []
    for (good, ok) in settings:
        tag = f"{good}_{ok}"
        P = Problem(kmax=3, th=ok)
        for level in (1, 2, 3):
            for pk, phys in (("P1", True), ("P2", False)):
                prep = P.prep(level, phys)
                ilp, t_ilp = timeit(P.ilp, prep); gr, t_gr = timeit(P.greedy, prep); ev = P.evaluate(gr, prep) if gr is not None else None
                row = dict(thresholds=tag, G1_boundary=ok, level=level, pool=pk, n_constrained=len(P.constrained), n_structurally_failed=len(P.structurally_failed),
                           n_series_below_level=sum(1 for k in P.constrained if prep[k] is None or prep[k]["level"] < level), n_removable=len(P.removable),
                           ilp_removed=len(P.removable) - ilp["n_removable_kept"], ilp_links=ilp["links"], ilp_optimal=ilp.get("optimal"), ilp_s=t_ilp,
                           greedy_removed=len(P.removable) - ev["n_removable_kept"] if ev else None, greedy_links=ev["links"] if ev else None, greedy_s=t_gr)
                kept = set(ilp["kept"]); removed = sorted(set(P.removable) - kept); row["removed_stations"] = ",".join(removed)
                designs[(tag, level, pk)] = dict(active=sorted(kept | set(P.must)), removed=removed, inputs=P.configure(kept | set(P.must), prep),
                                                 below=[k for k in P.constrained if prep[k] is None or prep[k]["level"] < level])
                rows.append(row)
                if pk == "P1" and tag == f"{K.GOOD}_{K.OK}":
                    for comp in comps:
                        ser = [k for k in P.constrained if k[0] in comp and prep[k] is not None]
                        if not ser: continue
                        dp, t_dp = timeit(P.tree_dp, prep, set(comp), G); r2 = P.ilp(prep, ser)
                        crow.append(dict(level=level, root=[n for n in comp if G.out_degree(n) == 0][0], n_nodes=len(comp), dp_kept=dp, ilp_kept=r2.get("n_removable_kept"), match=dp == r2.get("n_removable_kept")))
                print({k_: v for k_, v in row.items() if k_ != "removed_stations"}, flush=True)
    pickle.dump(designs, open(rv4.CACHE / "designs_ms_raw.pkl", "wb"))
    pd.DataFrame(rows).to_csv(rv4.RES / "design_levels_ms.csv", index=False); pd.DataFrame(crow).to_csv(rv4.RES / "tree_dp_vs_ilp_ms.csv", index=False)
    C_ = pd.DataFrame(crow); print("tree DP == ILP:", int(C_.match.sum()), "/", len(C_))

if __name__ == "__main__":
    main()
