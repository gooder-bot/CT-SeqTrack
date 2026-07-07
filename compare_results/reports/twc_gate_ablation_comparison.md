# TWC / Gate Ablation After Validity Fix

Compared runs: SeqTrack baseline, A1-order, A2-order-dyn, cand1, disp, active TWC, gate-safe, and conf-res gate.

Data notes:
- `A1-order+TWC` and `A2-order-dyn+TWC` are the cand4 validity-fixed runs: `ct_a1_order_twc_cand4_validfix_car_60ep_bs16_gpu2` and `ct_a2_order_dyn_twc_cand4_validfix_car_60ep_bs16_gpu3`.
- `A3-order-conf-res-gate` is reconstructed from `version_1` early points plus `version_2` continuation; duplicate step `18930` keeps the later `version_2` value.
- Validation epochs are inferred from `train_dataloader_length=1262` and event steps.

| Model | Succ final | Succ best | Succ delta vs parent | Prec final | Prec best | Prec delta vs parent | Parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SeqTrack baseline | 50.99 | 52.28 | - | 59.96 | 65.21 | - |  |
| A1-order | 51.23 | 53.30 | - | 57.86 | 62.02 | - |  |
| A2-order-dyn | 50.96 | 51.54 | - | 63.31 | 63.57 | - |  |
| A2-order-dyn-cand1 | 26.68 | 41.99 | -24.28 | 24.50 | 54.62 | -38.82 | A2-order-dyn |
| A2-order-dyn-disp | 50.54 | 52.44 | -0.42 | 63.85 | 64.81 | 0.53 | A2-order-dyn |
| A1-order+TWC | 51.16 | 53.16 | -0.07 | 61.10 | 63.35 | 3.24 | A1-order |
| A2-order-dyn+TWC | 28.23 | 45.24 | -22.73 | 32.04 | 57.43 | -31.28 | A2-order-dyn |
| A3-order-gate-safe | 48.32 | 50.99 | -2.64 | 54.87 | 60.17 | -8.45 | A2-order-dyn |
| A3-order-conf-res-gate | 31.17 | 62.04 | -19.79 | 30.92 | 76.30 | -32.40 | A2-order-dyn |

## Readout

1. Active TWC is now genuinely active: `A1-order+TWC` has mean `twc_valid_ratio=0.750` and `A2-order-dyn+TWC` has mean `twc_valid_ratio=0.750`. The previous `twc_valid_ratio=0` diagnosis no longer applies to these runs.
2. `A1-order+TWC` is essentially tied with `A1-order` on final success (51.16 vs 51.23) and improves final precision by +3.24. This is a useful precision-positive TWC signal, but not a clean success gain.
3. `A2-order-dyn+TWC` collapses late despite valid TWC: final success/precision are 28.23/32.04, down -22.73/-31.28 from `A2-order-dyn`. Under the current `twc_weight=0.05` setting, TWC should not yet be combined with dynamics as the main configuration.
4. `A3-order-gate-safe` is much safer than the old P5 full result (31.19/31.89), but it is still below `A2-order-dyn` on final success/precision by -2.64/-8.45.
5. `A3-order-conf-res-gate` has a very high best point (62.04 success, 76.30 precision) but falls to 31.17/30.92 at final. Treat this as an unstable checkpoint-selection signal, not as a stable final model.

## Diagnostics

- TWC tail means: `A1-order+TWC loss_twc=0.0081`, valid ratio `0.753`; `A2-order-dyn+TWC loss_twc=0.0077`, valid ratio `0.750`.
- Gate-safe is conservative: alpha dyn mean `0.127`, tail `0.116`.
- Conf-res still wants large raw dynamics weight: raw alpha mean `0.493`, clamped alpha mean `0.181`, residual norm mean `0.0315`.

## Generated files

- `../figures/twc_gate_ablation_curves.png`
- `../figures/twc_gate_ablation_success_curve.png`
- `../figures/twc_gate_ablation_precision_curve.png`
- `../figures/twc_gate_ablation_best_final_summary.png`
- `../figures/diagnostics/twc_gate_ablation_twc_diagnostics.png`
- `../figures/diagnostics/twc_gate_ablation_gate_diagnostics.png`
- `../data/twc_gate_ablation_metrics_points.csv`
- `../data/twc_gate_ablation_metrics_summary.csv`
- `../data/twc_gate_ablation_diagnostics_summary.csv`
