# Baseline vs A1-scaled vs A2-scaled-dyn

Validation points are plotted at epochs 5, 10, ..., 60. Baseline values are reused from `compare_results/metrics_points.csv`; A1/A2 scaled values are extracted from TensorBoard event files under `output/`.

| Model | Metric | Final | Best | Mean | Std | Final Delta vs baseline | Best Delta vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| SeqTrack baseline | success/test | 50.9858 | 52.2834 | 47.9679 | 3.4748 | 0.0000 | 0.0000 |
| SeqTrack baseline | precision/test | 59.9617 | 65.2144 | 55.9259 | 6.3958 | 0.0000 | 0.0000 |
| A1-scaled | success/test | 31.3337 | 33.2024 | 29.8461 | 2.6705 | -19.6521 | -19.0810 |
| A1-scaled | precision/test | 31.2243 | 37.6980 | 29.9821 | 3.3708 | -28.7374 | -27.5164 |
| A2-scaled-dyn | success/test | 29.4103 | 39.9967 | 31.0080 | 3.1791 | -21.5755 | -12.2867 |
| A2-scaled-dyn | precision/test | 31.5142 | 43.6762 | 33.2384 | 4.1683 | -28.4475 | -21.5382 |

## Key readout
- success/test: A1-scaled final 31.3337 (-19.6521 vs baseline), best 33.2024 (-19.0810); A2-scaled-dyn final 29.4103 (-21.5755), best 39.9967 (-12.2867).
- precision/test: A1-scaled final 31.2243 (-28.7374 vs baseline), best 37.6980 (-27.5164); A2-scaled-dyn final 31.5142 (-28.4475), best 43.6762 (-21.5382).

## Figures
- `compare_results/scaled_a1_a2_baseline_curves.png`
- `compare_results/scaled_a1_a2_baseline_success_curve.png`
- `compare_results/scaled_a1_a2_baseline_precision_curve.png`
- `compare_results/scaled_a1_a2_baseline_best_final_summary.png`

## Data files
- `compare_results/scaled_a1_a2_baseline_metrics_points.csv`
- `compare_results/scaled_a1_a2_baseline_metrics_summary.csv`
