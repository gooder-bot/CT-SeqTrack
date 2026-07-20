# P0-A / P0-B 诊断运行手册

这一步不训练正式模型。目标是先回答两个会改变研究路线的问题：

1. 当前 bounded residual 的 2 cm 上限和近零 gate 是否真的能修正 observation proposal；
2. 强 gap 下的目标是否在进入网络前已经离开 search crop。

所有校准统计只使用 `mini_train`。`mini_val` 和测试集不能用于调整 residual bound、crop scale 或阈值。

## 2026-07-17 回传状态

- standard/gap1124/burst-drop crop full-history 均已完成；强协议确认固定 2x expanded 也不足，而 GT-history CV oracle 仍接近 99% recall。完整分析见 `compare_results/reports/p0_ab_diagnostics_20260717.md`。
- standard residual warmup 与 active 64-batch 已完成；默认实际 correction 约 `1e-7 m`，未通过非平凡幅度验收。
- P0-A 仍缺完整 split、gap1124/burst-drop 和真正的 2-step optimizer smoke。

每次服务器运行后必须同时确认以下两个文件非空；只有日志文件不算完成：

```text
output/diagnostics/crop_reachability/<tag>/crop_reachability_summary.json
output/diagnostics/crop_reachability/<tag>/crop_reachability_endpoints.csv
```

## 0. 服务器准备

以下命令在仓库根目录运行。将 `<A1_CKPT>` 替换为同提交、seed42 的 A1-order Lightning checkpoint，将数据路径按服务器实际位置修改。

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python tools/diagnose_crop_reachability.py --self-test
python tools/check_residual_dynamics.py
```

第二条通过只说明 residual 的纯张量逻辑正确，不代替真实 batch 检查。

## 1. P0-B：先检查 crop 可达性

先用 10 条 tracklet 做 smoke：

```bash
python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --require-full-history \
  --max-tracklets 10 \
  --tag standard_smoke
```

smoke 正常后，去掉 `--max-tracklets`，依次运行三个配置：

```bash
python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history --tag standard_train

python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history --tag gap1124_train

python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history --tag burst_drop_train
```

若循环命令没有生成 gap/burst 文件，使用下面两条独立重跑，并在服务器立即检查输出：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p logs/diagnostics
set -euo pipefail

python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --tag gap1124_train \
  2>&1 | tee logs/diagnostics/p0b_gap1124_train.log

test -s output/diagnostics/crop_reachability/gap1124_train/crop_reachability_summary.json
test -s output/diagnostics/crop_reachability/gap1124_train/crop_reachability_endpoints.csv

python tools/diagnose_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --tag burst_drop_train \
  2>&1 | tee logs/diagnostics/p0b_burst_drop_train.log

test -s output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_summary.json
test -s output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_endpoints.csv
```

每次运行会生成 endpoint CSV 和汇总 JSON，默认位于：

```text
output/diagnostics/crop_reachability/<tag>/
```

三个 crop mode 的含义：

- `base`：以前一时刻 GT 框为中心，使用当前配置的 `bb_scale=1.25, bb_offset=2`；
- `expanded`：同一中心，scale 和 offset 均放大 2 倍；
- `cv_recenter`：只用更早的 GT 历史做常速度中心外推，再使用 base crop 大小。

它们都是 oracle 可达性诊断，不是在线跟踪结果。`target_point_count` 可作为可见性/稀疏程度代理；当前工具不把它等同于完整遮挡标注。

重点比较 `center_outside_rate`、`no_target_point_rate` 和 `target_point_recall`，并查看 `delta_t_buckets`、`displacement_buckets` 与 `target_point_buckets`：

- 若强 gap 的 `base` 明显下降，而 `expanded` 或 `cv_recenter` 大幅恢复，瓶颈发生在 crop 前；这只触发 GT-free predicted-history 诊断，不能从 GT oracle 直接跳到正式 recenter。
- 若三者接近且 base recall 已高，crop 不是首要瓶颈，可以继续验证 proposal 后 residual。
- 若 expanded 只增加大量背景点、target recall 改善很小，不应直接把 crop 放大作为正式方案。

## 2. P0-A：检查 residual 幅度与梯度

### 2.1 warmup 内的严格零检查

先保留配置中的 `dynamics_warmup_epoch: 5`。加载已有 A1 权重，是为了让 observation error 统计有实际意义；新增 dynamics/gate 参数会显示为 missing keys，这是预期现象。

```bash
python -u tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train \
  --batch-size 16 --workers 0 --max-steps 2 \
  --require-full-history --no-shuffle \
  --weights <A1_CKPT> \
  --residual-diagnostics --no-optimizer-step \
  --checkpoint-every 0 \
  --log-file output/diagnostics/p0a_standard_warmup.jsonl \
  --diagnostic-summary-file output/diagnostics/p0a_standard_warmup_summary.json \
  --checkpoint-dir output/diagnostics/p0a_standard_warmup_ckpt
```

这一阶段应满足：

- `applied_ratio == 0`；
- `applied_residual_norm.max == 0`；
- residual gate 的梯度为 0；
- observation、dynamics 和总 loss 均 finite。

### 2.2 warmup 后的冻结权重统计

用 `--residual-warmup-epoch 0` 只临时打开 residual，用 `--no-optimizer-step` 保证跨 batch 统计时模型权重不漂移。这里检查的是幅度、覆盖范围和梯度通路，不是 tracking 性能。

```bash
python -u tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train \
  --batch-size 16 --workers 0 --max-steps 64 \
  --require-full-history --no-shuffle \
  --weights <A1_CKPT> \
  --residual-diagnostics --residual-warmup-epoch 0 \
  --no-optimizer-step --checkpoint-every 0 \
  --log-file output/diagnostics/p0a_standard_active.jsonl \
  --diagnostic-summary-file output/diagnostics/p0a_standard_active_summary.json \
  --checkpoint-dir output/diagnostics/p0a_standard_active_ckpt
```

64 个 batch 正常后，正式训练 split 校准将 `--max-steps 64` 改为 `--max-steps 0 --keep-partial-batch`，遍历完整 loader。再把 `--cfg` 和输出文件前缀分别换为 `gap1124`、`burst_drop` 对应配置运行。正式比较至少记录：

- `observation_error_norm` 的 P50/P75/P95；
- `configured_max_residual_norm`（默认应为 `0.02`）；
- `alpha`、`applied_residual_norm`、`applied_ratio`、`clamp_ratio`；
- `gate_grad_norm` 与 `encoder_grad_norm`；
- `current_delta_t`、`num_points_in_search` 和 `dynamics_valid_ratio`。

汇总中的 `by_candidate_id` 用来检查 candidate 扰动是否制造了异常 proposal error，`by_current_delta_t` 用来区分正常帧间隔和强 gap；`invalid_applied_residual_norm.max` 必须为 0，`motion_head_input_dim` 必须为 256。

最后再做真正的 2-step optimizer smoke：复用上述命令，将 `--max-steps` 改为 2，并删除 `--no-optimizer-step` 和 `--checkpoint-every 0`。连续两步 loss/gradient 必须 finite。

## 3. 决策规则

按下面的顺序做决定：

1. 先看 crop：若 base crop 在强 gap 下不可达，先做 GT-free recursive reachability；P0-B2 已据此否定 raw CV 恒开启，当前改为可靠性控制的 dual-anchor 验证。bounded residual 暂时只保留为 refinement 候选。
2. crop 可达时，再看 `observation_error_norm`：若其 P50/P75 显著高于 2 cm，当前上限覆盖不了主要误差。
3. 只有 gate gradient finite 且非零、alpha/applied ratio 非平凡，才允许根据 `mini_train` 分位数一次性预注册新 bound。
4. 不根据 `mini_val` 指标来回调整 bound，也不在这一步宣称 dynamics 提升了跟踪性能。
5. P0-A/P0-B 定位清楚后，立即完成 P0-C 的 frozen manifest 与 `true/fixed/shuffled`，然后才启动 seed42 最小矩阵。

## 4. P0-B2：A1 predicted-history 递归可达性（已完成）

P0-B oracle 已完成后，不直接改 `models/base_model.py`。当前已新增独立诊断脚本 `tools/diagnose_recursive_crop_reachability.py`，避免把实验逻辑混入正式训练/评测路径。

第一阶段只运行一条 baseline A1 递归轨迹，并在同一个 endpoint 上被动比较四个 anchor：

```text
previous_gt             previous GT box，oracle 下界参考
previous_a1_pred        baseline A1 实际使用的上一预测框
gt_history_cv           GT history + real delta_t，oracle 上界
a1_pred_history_cv      baseline A1 最近两次预测 + real delta_t，GT-free counterfactual
```

`a1_pred_history_cv` 只改变离线 crop reachability 统计，不反向改变 baseline A1 的预测历史。这样第一阶段回答的是“已有预测历史是否包含足够运动信息”，而不是同时混入新闭环策略。

固定使用同一个 standard A1-order checkpoint 测试三种协议：

```text
output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/
lightning_logs/version_0/checkpoints/last.ckpt
```

脚本必须：

- 接受 `--reference-endpoints-csv`，逐项核对 `(tracklet_id, frame_index, frame_token)`，保证与本轮 oracle 完全相同；
- 记录 checkpoint SHA256、cfg SHA256、reference CSV SHA256、seed、device 和 virtual-rate metadata；
- 在 full-history endpoint 记录 `previous_prediction_error`、`current_target_from_previous_prediction`、`gt_cv_anchor_error`、`pred_cv_anchor_error`；
- 对四种 anchor 统一记录 center outside、target-point recall、crop 点数；
- 记录 baseline A1 当前预测误差、empty-crop fallback 和每种 anchor 的连续失败长度；
- 按 protocol、`delta_t`、真实位移、previous-prediction error 和 tracklet 分桶。

预注册判断：

- **Go**：在 gap1124 和 burst-drop 的 `>4 m` 可见目标上，`a1_pred_history_cv` 相对 `previous_a1_pred` recall 均至少提高 10 个百分点；总体均至少提高 5 个百分点；平均点数不超过 1.25 倍；standard 总体 recall 下降不超过 1 个百分点。
- **Conditional Go**：强协议达到上述收益但 standard 明显下降。下一步只测试由 gap/trajectory agreement 控制的 gated recenter，不直接全局启用。
- **No-Go**：GT-history CV 很强，但 predicted-history CV 在任一强协议总体提升不足 5 个百分点或连续失败没有缩短。此时停止直接 CV 接入，先检查预测历史漂移、速度 clipping 或 Kalman uncertainty，不增加学习式大模型。

2026-07-17 回传结果满足 **No-Go**：standard/gap1124/burst-drop 的总体 recall 仅提高 2.91/2.65/3.03 pp；gap/burst 的 `>4 m` 桶提高 8.45/9.96 pp，均未完整达到预注册门槛。点数和 standard 安全条件通过，但不能把 raw predicted-history CV 接成唯一 search anchor。完整汇总见 `compare_results/reports/p0b2_recursive_crop_reachability_20260717.md`。

如需复现 P0-B2，服务器同步 `tools/diagnose_recursive_crop_reachability.py` 后运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p logs/diagnostics
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

A1_CKPT=output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0/checkpoints/last.ckpt

python tools/diagnose_recursive_crop_reachability.py --self-test
python tools/diagnose_recursive_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" --device cuda:0 --model-load-smoke

python tools/diagnose_recursive_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --device cuda:0 --max-tracklets 10 --tag standard_a1_recursive_smoke \
  2>&1 | tee logs/diagnostics/p0b2_standard_a1_recursive_smoke.log
```

smoke 成功后运行三协议 full diagnostic：

```bash
python tools/diagnose_recursive_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/standard_train/crop_reachability_endpoints.csv \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --device cuda:0 --tag standard_a1_recursive \
  2>&1 | tee logs/diagnostics/p0b2_standard_a1_recursive.log

python tools/diagnose_recursive_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/gap1124_train/crop_reachability_endpoints.csv \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --device cuda:0 --tag gap1124_a1_recursive \
  2>&1 | tee logs/diagnostics/p0b2_gap1124_a1_recursive.log

python tools/diagnose_recursive_crop_reachability.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_endpoints.csv \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini --split mini_train --require-full-history \
  --device cuda:0 --tag burst_drop_a1_recursive \
  2>&1 | tee logs/diagnostics/p0b2_burst_drop_a1_recursive.log
```

每个 full run 必须同时满足：summary/CSV 非空，且 summary 中 `reference_match.exact_match == true`。输出位于：

```text
output/diagnostics/recursive_crop_reachability/<tag>/recursive_crop_reachability_summary.json
output/diagnostics/recursive_crop_reachability/<tag>/recursive_crop_reachability_endpoints.csv
```

上述命令保留为复现实验记录，不再是当前待运行任务。

## 5. P0-B3：测试时可靠性与 active dual-anchor

P0-B2 还发现：当 `previous_prediction_error <= 4 m` 时，pred-history CV recall 为 97.34%–98.64%；当误差超过 4 m 时只有 0.80%–1.61%。这个 GT 分桶只能做离线标签，不能直接成为在线 gate。下一阶段先扩展诊断，不先改训练网络：

1. 记录 best-box confidence、proposal score、foreground count/score、crop point count、empty fallback、`||c_cv-c_prev||`、速度、CV shift 和 local/trajectory agreement；
2. 用离线 GT 评估这些测试时信号预测 `previous_prediction_error > 4 m` 与 next-crop failure 的 AUROC、AUPRC、calibration，并检查三协议稳定性；
3. 只有代理信号有效时，固定同一 A1 checkpoint 做无训练 active dual-anchor：previous-pred 与 clipped-CV/Kalman 两个 crop 各 forward 一次，再按 confidence/agreement 选择或保守融合；
4. 验收重点是首次失控时间、连续失败长度、empty fallback 和标准跟踪指标，而不是被动 oracle recall；
5. 若代理信号无预测力或 active 版本无收益，停止 gate/dual-anchor 路线，不增加 Mamba、ODE、occupancy memory 或更大 trajectory encoder。
