# Latest 5 Runs Comparison

## Summary

| model | success final | success best | precision final | precision best | note |
| --- | ---: | ---: | ---: | ---: | --- |
| A3-conf-res best-e14 retest | 28.06 | 28.06 | 37.70 | 37.70 | single checkpoint test; does not reproduce old 62/76 best signal |
| A2-order-dyn seed43 | 23.64 | 45.92 | 23.77 | 54.88 | large seed regression |
| A2-order-dyn seed44 | 46.90 | 50.23 | 52.62 | 58.19 | better than seed43 but still below old seed42 60ep report |
| A2-order-dyn+TWC w0.01 seed42 | 22.88 | 30.27 | 24.27 | 32.16 | lower TWC weight still collapses |
| A3-conf-res rerun seed42 | 32.11 | 34.55 | 31.87 | 36.50 | rerun remains low and unstable |

## Diagnostics

| model | diagnostic | final | mean | tail1000 mean | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A2-order-dyn+TWC w0.01 seed42 | loss_twc | 0.0086 | 0.0153 | 0.0083 | 0.0010 | 1.1429 |
| A2-order-dyn+TWC w0.01 seed42 | twc_valid_ratio | 0.8125 | 0.7501 | 0.7541 | 0.2500 | 1.0000 |
| A2-order-dyn+TWC w0.01 seed42 | twc_center_gap | 0.1987 | 0.2030 | 0.1748 | 0.0609 | 2.4381 |
| A2-order-dyn+TWC w0.01 seed42 | twc_angle_gap | 0.0182 | 0.0223 | 0.0103 | 0.0023 | 1.5926 |
| A3-conf-res rerun seed42 | obs_alpha_dyn_mean | 0.5691 | 0.4906 | 0.4988 | 0.0372 | 0.7635 |
| A3-conf-res rerun seed42 | obs_alpha_dyn_clamped_mean | 0.1875 | 0.1807 | 0.1810 | 0.0372 | 0.2000 |
| A3-conf-res rerun seed42 | obs_dyn_residual_norm | 0.0136 | 0.0315 | 0.0314 | 0.0001 | 0.0902 |

## Readout

1. `A3-conf-res best-e14 retest` only reaches 28.06 / 37.70 on the tested checkpoint, so the earlier 62.04 / 76.30 best point should no longer be treated as confirmed until its exact evaluation path is reconciled.
2. `A2-order-dyn` now shows high seed sensitivity: seed43 collapses to 23.64 / 23.77 while seed44 is 46.90 / 52.62.
3. Reducing TWC from 0.05 to 0.01 does not rescue the A2+dynamics combination: the final result is 22.88 / 24.27, despite valid TWC diagnostics.
4. The A3 conf-res rerun remains low at 32.11 / 31.87; gate/conf-res should move to diagnostic analysis before more structure changes.

## Generated files

- `../data/latest_5runs_metrics_points.csv`
- `../data/latest_5runs_metrics_summary.csv`
- `../data/latest_5runs_diagnostics_summary.csv`
- `../data/latest_5runs_hparams_summary.csv`
- `../figures/line_charts/latest_5runs_success_curve.svg`
- `../figures/line_charts/latest_5runs_precision_curve.svg`
- `../figures/bar_charts/latest_5runs_best_final_summary.svg`
- `../figures/diagnostics/latest_5runs_diagnostics_tail_mean.svg`
