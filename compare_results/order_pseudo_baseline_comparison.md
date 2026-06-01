# Baseline, A1-pseudo, A1-order, and A2-order-dyn

Validation points are plotted at epochs 5, 10, ..., 60. Baseline and A1-pseudo are reused from `a1_time_encoding_metrics_points.csv`; A1-order and A2-order-dyn are extracted from TensorBoard event files under `output/`.

| Model | Metric | Final | Best | Mean | Std | Final Delta vs baseline | Best Delta vs baseline | Final Delta vs A1-pseudo | Best Delta vs A1-pseudo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SeqTrack baseline | success/test | 50.9858 | 52.2834 | 47.9679 | 3.4748 | 0.0000 | 0.0000 | 2.6477 | 2.3917 |
| SeqTrack baseline | precision/test | 59.9617 | 65.2144 | 55.9259 | 6.3958 | 0.0000 | 0.0000 | 7.7112 | -0.0241 |
| A1-pseudo | success/test | 48.3381 | 49.8917 | 43.8953 | 4.4349 | -2.6477 | -2.3917 | 0.0000 | 0.0000 |
| A1-pseudo | precision/test | 52.2505 | 65.2385 | 50.8270 | 7.7315 | -7.7112 | 0.0241 | 0.0000 | 0.0000 |
| A1-order | success/test | 51.2287 | 53.2987 | 49.0018 | 3.5593 | 0.2429 | 1.0153 | 2.8906 | 3.4070 |
| A1-order | precision/test | 57.8632 | 62.0153 | 56.9589 | 4.7958 | -2.0985 | -3.1991 | 5.6127 | -3.2232 |
| A2-order-dyn | success/test | 50.9617 | 51.5416 | 46.8277 | 6.5465 | -0.0241 | -0.7418 | 2.6236 | 1.6499 |
| A2-order-dyn | precision/test | 63.3140 | 63.5733 | 57.7629 | 9.9031 | 3.3523 | -1.6411 | 11.0635 | -1.6652 |

## Readout
- A1-order clearly improves over A1-pseudo on final success: 51.2287 vs 48.3381 (+2.8906), and is slightly above the baseline final success.
- A1-order improves final precision over A1-pseudo: 57.8632 vs 52.2505 (+5.6127), while its best precision is 3.2232 below A1-pseudo.
- A2-order-dyn is the best final precision model in this group: success 50.9617, precision 63.3140; compared with baseline, final success is essentially tied (-0.0241) and final precision is higher by +3.3523.

## Interpretation
- Restoring baseline-style order-time in the main branch fixes most of the collapse seen in A1-scaled, supporting the hypothesis that the main branch is sensitive to order-token semantics.
- A2-order-dyn adds real-time dynamics on top of order-time main features and gives the strongest final scores here, suggesting real time is more useful as a dynamics prior than as a direct replacement for the main branch time channel.
- The remaining gap to SeqTrack baseline means order-time restoration is not the whole story; dynamics training noise, recursive test drift, and CT code-path differences still need separate ablation.

## Figures
- `compare_results/order_pseudo_baseline_curves.png`
- `compare_results/order_pseudo_baseline_success_curve.png`
- `compare_results/order_pseudo_baseline_precision_curve.png`
- `compare_results/order_pseudo_baseline_best_final_summary.png`

## Data files
- `compare_results/order_pseudo_baseline_metrics_points.csv`
- `compare_results/order_pseudo_baseline_metrics_summary.csv`
