"""Single place for the multi-state thresholds and seeds (NSE of the fallback model). G2 'good' >= GOOD; G1 'acceptable' >= OK; else G0 'failed'."""
GOOD = 0.8
OK = 0.6
SENSITIVITY = [(0.85, 0.7), (0.75, 0.5)]
SEED = 20261110
FAULT_FAR = 0.01
SCORE = {2: 1.0, 1: 0.5, 0: 0.0}      # utility of a state, used for 'expected performance'
