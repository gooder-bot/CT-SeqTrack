-- Portable materialization of the reviewed M0-3/M0-4 report snapshot.
-- The statistical derivations and raw-source identities are preserved in
-- m0_m03_m04_snapshot_20260721.json and m0_m03_m04_analysis_20260721.md.

DROP VIEW IF EXISTS headline_metrics;
CREATE TEMP VIEW headline_metrics AS
SELECT
  'm03' AS metric_group,
  1.117648 AS oracle_gain_mean_m,
  1.039871 AS dynamics_gain_mean_m,
  0.81312 AS dynamics_better_rate,
  0.87324 AS dynamics_positive_tracklet_rate,
  NULL AS velocity_threshold_multiple,
  NULL AS acceleration_threshold_multiple,
  NULL AS matched_error_delta_mean_m,
  NULL AS positive_tracklet_rate,
  1311 AS primary_endpoints,
  213 AS primary_tracklets,
  NULL AS matched_comparisons,
  NULL AS matched_tracklets,
  'GO_M2_PROPOSAL_INNOVATION' AS decision
UNION ALL
SELECT
  'm04', NULL, NULL, NULL, NULL, 12.22, 21.28, 0.010436, 0.817,
  NULL, NULL, 8515, 235, 'FREEZE_M1_SHARED_SE2';

DROP VIEW IF EXISTS m03_proposal_errors;
CREATE TEMP VIEW m03_proposal_errors AS
SELECT 'Observation' AS proposal, 1.3492 AS mean_error_m, 0.363567 AS median_error_m,
       6.02805 AS p95_error_m, 1311 AS endpoints, 213 AS tracklets,
       'primary' AS cohort, 'GO_M2_PROPOSAL_INNOVATION' AS decision
UNION ALL
SELECT 'Dynamics', 0.309334, 0.117888, 1.27717, 1311, 213,
       'primary', 'GO_M2_PROPOSAL_INNOVATION'
UNION ALL
SELECT 'Segment oracle', 0.231557, 0.09248, 0.986982, 1311, 213,
       'primary', 'GO_M2_PROPOSAL_INNOVATION';

DROP VIEW IF EXISTS m04_candidate_jitter;
CREATE TEMP VIEW m04_candidate_jitter AS
SELECT 'Velocity' AS derivative, 0.62683 AS mean, 0.611018 AS p50,
       1.02177 AS p95, 0.05 AS threshold, 12.22 AS threshold_multiple,
       'm/s' AS unit, 10450 AS rows, 'candidate1/2/3' AS candidate_group,
       1 AS candidate0_exact_zero
UNION ALL
SELECT 'Acceleration', 2.20189, 2.128118, 3.9929, 0.1, 21.28,
       'm/s²', 10450, 'candidate1/2/3', 1;

DROP VIEW IF EXISTS stage_decisions;
CREATE TEMP VIEW stage_decisions AS
SELECT 1 AS "order", 'M0-3' AS stage, 'GO' AS decision,
       'Implement bounded innovation' AS unlocked,
       'Offline evidence; no tracking promotion' AS boundary
UNION ALL
SELECT 2, 'M0-4', 'FREEZE',
       'Build shared SE(2) data path',
       'No smooth drift; no guaranteed gain'
UNION ALL
SELECT 3, 'M0 overall', 'IN PROGRESS', 'Prepare M1/M2 implementation',
       'M0-2 outputs and provenance remain';

DROP VIEW IF EXISTS robustness_checks;
CREATE TEMP VIEW robustness_checks AS
SELECT 1 AS "order", 'M0-3 dynamics-only vs observation' AS "check",
       '+0.803 m tracklet mean gain' AS estimate, '[+0.633, +0.988] m' AS ci95,
       'Non-oracle complementarity is positive' AS interpretation
UNION ALL
SELECT 2, 'M0-3 long-gap oracle', '+0.717 m tracklet mean gain',
       '[+0.493, +0.967] m', 'Long-gap challenge bin is supported'
UNION ALL
SELECT 3, 'M0-3 top-5%-trimmed oracle', '+0.816 m endpoint mean gain',
       'Not applicable', 'Signal is not owned by extreme tail'
UNION ALL
SELECT 4, 'M0-4 matched candidate penalty', '+0.0104 m endpoint mean error',
       '[+0.0093, +0.0155] m', 'Stable harm, small absolute mean';
