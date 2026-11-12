"""Central configuration. Nothing here depends on test-period data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # station_subset_optimization/
PKG = ROOT.parent / "hooman_independent_ai_data_package"   # read-only original data
CACHE = ROOT / "cache"
RES = ROOT / "results"
FIG = ROOT / "figures"
LOG = ROOT / "logs"
EXT = ROOT / "external_data"

F_META = PKG / "01_station_metadata/Hooman_119_Required_Network_Stations_Master.csv"
F_LINKS = PKG / "02_network/Hooman_57_Target_Upstream_Relationships.csv"
F_WIDE = PKG / "03_timeseries/Hooman_HQ_Wide_119_Stations_2005_2020.csv"
F_MANIFEST = PKG / "05_quality_checks/Hooman_File_Manifest_SHA256.csv"

# ---- strict chronological split -------------------------------------------------
TRAIN = ("2005-01-01", "2012-12-31")
VAL = ("2013-01-01", "2016-12-31")
TEST = ("2017-01-01", "2020-12-31")
DEV = ("2005-01-01", "2016-12-31")          # train + validation (refit period for test)
# blocked year-wise CV inside 2005-2016 (fold 3 == the standard train/validation split)
CV_FOLDS = [(2005, 2008), (2009, 2012), (2013, 2016)]

LAGS = (0, 1, 2, 3)
RIDGE_ALPHA = 1.0           # fixed a priori (not tuned)
N_BINS = 10
TOP_CORR = 5
MIN_OVERLAP = 365

# ---- fixed stations -------------------------------------------------------------
RHBN_N = ["02AB008", "02CF008", "02EC002", "02GA010", "02HL004"]
FLOOD_CRITICAL = ["02GD008", "02HL003", "02HL005"]
# user-supplied data flags ("requires review"); H series only
H_FLAGGED = {
    "02GD011": "H steps up about +0.25 m from 2017 (supplied by data owner/user; requires review)",
    "02HB001": "H level shifts between 2005-2016 and 2017-2020 (supplied by user; requires review)",
}

# ---- optimisation ---------------------------------------------------------------
SEEDS = [11, 22, 33, 44, 55]
POP = 120
NGEN = 300
CX_PROB = 0.9
