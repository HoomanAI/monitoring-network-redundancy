Supplementary data for
"Reliability-Based Redundancy Allocation for Data-Driven Monitoring Sensor Networks
 Under Component Failures: Application to a Hydrometric Network in Ontario"

The files below reconstruct the monitoring graph of Section 3, the candidate pools of
Eq. (7) and the recommended design of Section 5.15 exactly as they were used. Station
records themselves are the property of the Water Survey of Canada and are not
redistributed here; every station is identified by its WSC gauge number.

S1_stations.csv          119 rows, one per station.
  station, role (source / relay / sink), component (index of its in-tree), lat, lon,
  da_km2 (drainage area), availability_H, availability_Q, MTBF_days, MTTR_days,
  n_events (observed outages 2005-2016), must_keep (1 if the station cannot be removed),
  status_recommended (kept / retired in the recommended k = 2 design).

S2_physical_edges.csv    88 rows, station-level physical links.
  upstream, downstream, in_forest (0 for the single link excluded to keep out-degree
  at most one, 02KF001 -> 02LA004), downstream_is_target.
  The 87 rows with in_forest = 1 are the arc set A_p of Proposition 1.

S3_series_edges.csv      850 rows, the candidate pools at series level.
  target, variable (H or Q), input, edge_type:
    direct            166 rows, the input is the immediate downstream neighbour, N_p(t)
    physical_upstream 157 rows, the input lies further upstream, A(t) \ N_p(t)
    statistical       527 rows, selected by Eq. (6)
  via_non_tree_link  1 for the 12 physical_upstream rows reached only through the
    excluded link of S2,
  abs_corr_train  absolute Pearson correlation with the target series on 2005-2012,
  in_recommended_design  1 if the edge is used by the recommended design.
  Pools are capped at 12 members (Eq. 7); the cap binds on 12 of the 114 series.

S4_recommended_design.csv  114 rows, the input set of each target series in the
  recommended 23-removal design: n_inputs, n_direct, n_added, inputs (semicolon list).

S5_pool_adjacency.csv    119 x 114 incidence matrix of S3, stations by series
  (column names are station_variable). Row sums give the number of series a station
  serves; 850 non-zero cells, 6.3 per cent of the matrix.

S6_design_comparison.csv  the designs of Table 2 with their reliability, service,
  structural and resilience quantities under one evaluation pipeline and one set of
  bootstrap resamples.

S7_heldout_skill.csv     per-series NSE on the selection period (2013-2016) and on the
  held-out period (2017-2020) for the current network and the recommended design,
  the data behind Section 6.

Periods. Models are fitted on 2005-2012 and scored on 2013-2016; 2017-2020 is the
held-out test portion, read once; 2021-2025 is withheld by the data owner and was not
used at any stage.
