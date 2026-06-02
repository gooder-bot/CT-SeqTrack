# Cand1 / Displacement Dynamics Ablation

Compared runs: SeqTrack baseline, A2-order-dyn, A2-order-dyn-cand1, A2-order-dyn-disp.

Important caveat: cand1 has `num_candidates=1`, so 60 epochs contain about one quarter of the optimizer steps of the `num_candidates=4` runs. The final cand1 point is therefore not strictly step-aligned with A2-order-dyn or disp.

| model              | metric         |   final |    best |   best_final_gap |    mean |   late_mean_epoch40_60 |   final_delta_vs_baseline |   final_delta_vs_a2_order_dyn |
|:-------------------|:---------------|--------:|--------:|-----------------:|--------:|-----------------------:|--------------------------:|------------------------------:|
| SeqTrack baseline  | success/test   | 50.9858 | 52.2834 |           1.2976 | 47.9679 |                49.4527 |                    0.0000 |                        0.0241 |
| SeqTrack baseline  | precision/test | 59.9617 | 65.2144 |           5.2527 | 55.9259 |                57.4746 |                    0.0000 |                       -3.3523 |
| A2-order-dyn       | success/test   | 50.9617 | 51.5416 |           0.5799 | 46.8277 |                50.6011 |                   -0.0241 |                        0.0000 |
| A2-order-dyn       | precision/test | 63.3140 | 63.5733 |           0.2593 | 57.7629 |                62.5208 |                    3.3523 |                        0.0000 |
| A2-order-dyn-cand1 | success/test   | 26.6772 | 41.9858 |          15.3085 | 30.4462 |                26.7442 |                  -24.3086 |                      -24.2845 |
| A2-order-dyn-cand1 | precision/test | 24.4989 | 54.6247 |          30.1258 | 31.5898 |                24.9133 |                  -35.4628 |                      -38.8151 |
| A2-order-dyn-disp  | success/test   | 50.5416 | 52.4409 |           1.8993 | 44.8018 |                50.4059 |                   -0.4442 |                       -0.4201 |
| A2-order-dyn-disp  | precision/test | 63.8479 | 64.8085 |           0.9606 | 54.7297 |                63.3735 |                    3.8862 |                        0.5339 |

## Reading

1. `A2-order-dyn-disp` is essentially tied with `A2-order-dyn` on success and is slightly better on final/best precision. The displacement auxiliary loss is not a clear breakthrough, but it does not hurt at weight 0.01.
2. `A2-order-dyn-cand1` collapses after the first 10 epochs. This argues against simply removing non-zero candidates under the current 60-epoch protocol. Because cand1 also has 4x fewer optimizer steps, a 240-epoch step-aligned run would be needed before making a final statement about candidate noise.
3. The current evidence favors keeping multi-candidate training and treating displacement supervision as a small optional stabilizer, not as the main explanation for A2-order-dyn gains.

## Generated files

- `compare_results/cand1_disp_dynamics_curves.png`
- `compare_results/cand1_disp_dynamics_success_curve.png`
- `compare_results/cand1_disp_dynamics_precision_curve.png`
- `compare_results/cand1_disp_dynamics_best_final_summary.png`
- `compare_results/cand1_disp_dynamics_metrics_points.csv`
- `compare_results/cand1_disp_dynamics_metrics_summary.csv`
