import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]      # station_subset_optimization/
V4 = Path(__file__).resolve().parents[1]
V3 = ROOT / "ress_v3"
for p in (ROOT, ROOT / "followup_v2", V3, V3 / "scripts", V4):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
RES = V4 / "results"; FD = V4 / "figdata"; CACHE = V4 / "cache"; DIAG = V4 / "diagnostics"; LOG = V4 / "logs"
