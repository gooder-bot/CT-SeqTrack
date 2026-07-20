# P0-A / P0-B 诊断运行手册

这一步不训练正式模型。目标是先回答两个会改变研究路线的问题：

1. 当前 bounded residual 的 2 cm 上限和近零 gate 是否真的能修正 observation proposal；
2. 强 gap 下的目标是否在进入网络前已经离开 search crop。

所有校准统计只使用 `mini_train`。`mini_val` 和测试集不能用于调整 residual bound、crop scale 或阈值。

## 2026-07-20 回传状态

- standard/gap1124/burst-drop crop full-history 均已完成；强协议确认固定 2x expanded 也不足，而 GT-history CV oracle 仍接近 99% recall。完整分析见 `compare_results/reports/p0_ab_diagnostics_20260717.md`。
- standard residual warmup 与 active 64-batch 已完成；默认实际 correction 约 `1e-7 m`，未通过非平凡幅度验收。
- P0-A 仍缺完整 split、gap1124/burst-drop 和真正的 2-step optimizer smoke。
- P0-B2 recursive predicted-history、P0-B3 passive reliability 与 P0-B4 independent mini_val 均已完成。P0-B4 判定为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`：当前 observation calibrator、raw-CV candidate、frozen-state/active anchor 与 selector 全部停止。

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

P0-B2 还发现：当 `previous_prediction_error <= 4 m` 时，pred-history CV recall 为 97.34%–98.64%；当误差超过 4 m 时只有 0.80%–1.61%。这个 GT 分桶只能做离线标签，不能直接成为在线 gate。A1 当前也没有独立的 best-box confidence/proposal score；P0-B3 只使用现有 forward 可取得的 foreground、motion-state、crop points、empty fallback 与几何一致性信号，不新增或训练 confidence head。

2026-07-20 已新增两个独立工具：

- `tools/diagnose_reliability_signals.py`：运行正常 A1 observation 递归轨迹，同时被动 forward raw real-dt predicted-history CV crop；只有 observation candidate 更新后续历史。
- `tools/summarize_reliability_signals.py`：不依赖 scikit-learn，以稳定 `tracklet_key` 分组，只在 standard 上拟合 NumPy logistic calibrator 和运行阈值，再原样评估 gap1124/burst-drop。

### 5.1 特征与标签的时间边界

汇总器分开三个任务，不能互相替代：

1. `trigger`：用上一 endpoint 的 `prev_obs_*` 质量信号，加当前帧已知的真实 `delta_t`、CV speed/shift，预测**当前 observation crop miss**。这等价于上一时刻预测 next-crop failure，全部特征在当前 crop 前可得。
2. `current_evidence`：用当前 observation crop 的 foreground/point 信号判断当前 crop 是否已经 miss；它只能用于决定是否需要第二分支或判断当前候选质量，不能冒充 pre-crop trigger。
3. `selector`：两个分支都 forward 后，只在当前目标可见且至少一个 crop 可达的 endpoint 上，用 foreground/agreement 预测 trajectory candidate 是否比 observation candidate 至少好 `0.25 m`；只用于 post-crop 候选选择诊断。

CSV 中 `previous_prediction_error`、`current_gt_*`、target/crop GT metrics、candidate error、drift 与 selector label 全部只是离线标签。任何 active gate 都不得读取这些字段。

### 5.2 服务器自测与 smoke

以下命令在仓库根目录运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p logs/diagnostics
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

A1_CKPT=output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0/checkpoints/last.ckpt
DATA_ROOT=/home/lishengjie/data/nuscenes-mini

python tools/diagnose_reliability_signals.py --self-test
python tools/summarize_reliability_signals.py --self-test

python tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" --device cuda:0 --model-load-smoke

python tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/standard_train/crop_reachability_endpoints.csv \
  --path "$DATA_ROOT" --version v1.0-mini --split mini_train \
  --require-full-history --device cuda:0 --max-tracklets 10 \
  --tag standard_p0b3_smoke \
  2>&1 | tee logs/diagnostics/p0b3_standard_smoke.log

test -s output/diagnostics/reliability_signals/standard_p0b3_smoke/reliability_endpoints.csv
test -s output/diagnostics/reliability_signals/standard_p0b3_smoke/reliability_summary.json
```

smoke summary 必须满足：

- `reference_match.exact_match == true`；
- `weights_sha256` 与 P0-B2 相同；
- `obs_candidate_error`、`traj_candidate_error` 和 foreground stats 均无非有限异常；
- summary note 明确 trajectory branch 为 passive，递归历史只由 observation candidate 更新。

### 5.3 三协议 full passive diagnostic

```bash
python tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/standard_train/crop_reachability_endpoints.csv \
  --path "$DATA_ROOT" --version v1.0-mini --split mini_train \
  --require-full-history --device cuda:0 --tag standard_p0b3 \
  2>&1 | tee logs/diagnostics/p0b3_standard.log

python tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/gap1124_train/crop_reachability_endpoints.csv \
  --path "$DATA_ROOT" --version v1.0-mini --split mini_train \
  --require-full-history --device cuda:0 --tag gap1124_p0b3 \
  2>&1 | tee logs/diagnostics/p0b3_gap1124.log

python tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml \
  --weights "$A1_CKPT" \
  --reference-endpoints-csv output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_endpoints.csv \
  --path "$DATA_ROOT" --version v1.0-mini --split mini_train \
  --require-full-history --device cuda:0 --tag burst_drop_p0b3 \
  2>&1 | tee logs/diagnostics/p0b3_burst_drop.log

for tag in standard_p0b3 gap1124_p0b3 burst_drop_p0b3; do
  test -s "output/diagnostics/reliability_signals/$tag/reliability_endpoints.csv"
  test -s "output/diagnostics/reliability_signals/$tag/reliability_summary.json"
done
```

三份 summary 都必须满足 `reference_match.exact_match == true`，且 endpoint 数分别与 P0-B2 对应协议一致。

### 5.4 grouped reliability 汇总

```bash
python tools/summarize_reliability_signals.py \
  --input standard=output/diagnostics/reliability_signals/standard_p0b3/reliability_endpoints.csv \
  --input gap1124=output/diagnostics/reliability_signals/gap1124_p0b3/reliability_endpoints.csv \
  --input burst_drop=output/diagnostics/reliability_signals/burst_drop_p0b3/reliability_endpoints.csv \
  --standard-protocol standard \
  --strong-protocols gap1124,burst_drop \
  --folds 5 --seed 42 --target-recall 0.80 \
  --go-auroc 0.75 --go-auprc-margin 0.15 \
  --raw-cv-union-gain 0.05 \
  --output-dir output/diagnostics/reliability_signals/analysis \
  --tag p0b3_reliability_20260720 \
  2>&1 | tee logs/diagnostics/p0b3_reliability_summary.log

test -s output/diagnostics/reliability_signals/analysis/p0b3_reliability_20260720_summary.json
test -s output/diagnostics/reliability_signals/analysis/p0b3_reliability_20260720_report.md
```

### 5.5 预注册决策

- `NO_GO_RELIABILITY_PROXY`：gap1124 或 burst-drop 任一 trigger AUROC `< 0.75`，或 AUPRC 未高于本协议 prevalence `0.15`。停止 learned gate/dual-anchor，不增加 Mamba、ODE、occupancy memory 或大 trajectory encoder。
- `RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`：可靠性通过，但任一强协议 passive raw-CV dual oracle target-point recall 相对 observation 增益 `< 5 pp`。不实现 raw-CV active anchor；只允许测试一次拒绝低可靠观测更新的 timestamp-aware Kalman/frozen-state anchor。
- `GO_ACTIVE_RAW_CV_DUAL_ANCHOR`：可靠性与 raw-CV crop complementarity 均通过。固定 standard calibrator、同一 A1 checkpoint 和阈值，进入无训练 active inference。

active 阶段仍必须报告首次失控时间、连续失败长度、empty fallback、Success/Precision、FPS，并与 2x expanded、random second anchor 和相同计算成本控制对照。P0-B3 passive diagnostic 本身不是 tracking 性能结果。

### 5.6 2026-07-20 回传、独立复算与路线决定

第 5.2–5.4 节命令已经执行完毕，现保留为复现记录。下载到本地的三协议结果通过以下完整性检查：

- endpoints：standard/gap1124/burst-drop 为 `4246 / 2127 / 2098`；visible + full-history eligibility 为 `2996 / 1503 / 1478`。
- tracklet：`260 / 243 / 243`；`(tracklet_key, frame_index, frame_token)` 无重复。
- reference endpoints exact match，且三协议 checkpoint SHA256 均为 `a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82`。
- CSV hash 与 summary 完全一致；重新运行 `summarize_reliability_signals.py` 后，报告指标最大绝对差约 `5.3e-16`。
- 无关键字段缺失、非有限值、非正 `delta_t`、范围异常或 label/crop-union/selector 矛盾。
- `prev_obs_*` 前景特征缺失与 `prev_obs_empty_fallback=True` 一一对应，缺失率为 `14.79% / 15.37% / 16.24%`，属于预期的结构性缺失。

预注册 13 特征 trigger 的 grouped 5-fold 结果：

| protocol | prevalence | AUROC | AUPRC | Brier | ECE | activation | recall | precision | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 0.283 | 0.857 | 0.742 | 0.120 | 0.024 | 0.366 | 0.764 | 0.590 | 0.209 |
| gap1124 | 0.349 | 0.787 | 0.660 | 0.198 | 0.134 | 0.603 | 0.847 | 0.490 | 0.472 |
| burst_drop | 0.357 | 0.785 | 0.671 | 0.212 | 0.155 | 0.571 | 0.833 | 0.521 | 0.425 |

trigger 的 AUROC 和 `AUPRC - prevalence` 通过预注册门槛，但强协议校准与 FPR 明显恶化。passive crop complementarity 为：

| protocol | observation recall | raw-CV recall | oracle union | union gain |
| --- | ---: | ---: | ---: | ---: |
| standard | 0.698 | 0.727 | 0.728 | +3.04 pp |
| gap1124 | 0.637 | 0.663 | 0.666 | +2.88 pp |
| burst_drop | 0.632 | 0.661 | 0.663 | +3.15 pp |

两个强协议都低于 `+5 pp`，所以预注册决定是 **`RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`**。post-crop selector 在 standard/gap/burst 的 AUROC 为 `0.729/0.605/0.433`，FPR 为 `0.313/0.750/0.826`，也判定 No-Go。

为解释跨协议失准，额外做了 post-hoc feature ablation；这些结果是诊断，不是新的预注册 confirmatory test：

| features | standard AUROC | gap AUROC | burst AUROC | gap ECE/FPR | burst ECE/FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| all 13 | 0.857 | 0.787 | 0.785 | 0.134 / 0.472 | 0.155 / 0.425 |
| `prev_obs_only` | 0.853 | 0.867 | 0.873 | 0.061 / 0.202 | 0.057 / 0.200 |
| all without raw `current_delta_t` | 0.857 | 0.865 | 0.872 | 0.061 / 0.171 | 0.060 / 0.159 |
| time/CV geometry only | 0.529 | 0.553 | 0.557 | 0.273 / >0.81 | 0.292 / >0.81 |

解释边界：当前模型主要在识别“上一 observation 是否已经不可靠”；raw `current_delta_t` 是 standard-to-HTV 分布外失准的主要来源，CV geometry 单独接近随机。因此不能写“timestamp-aware reliability 已验证”。此外，标签以当前 target visible 为条件，只覆盖可见目标的 crop miss，不是全遮挡或总体 uncertainty。

P0-B3 当时固定的后续入口为：

1. 代码以 GitHub commit 为版本源；服务器允许手动同步，但工具必须保存 git/dirty、脚本/config/data/checkpoint hashes。当前 P0-B3 summary 记录 commit `f28f495...` 且 dirty，复现声明继续保留该 caveat。
2. 在独立 split/held-out tracklet 上确认精简 `prev_obs_*` trigger；只有该入口通过，才允许实现 state anchor。

P0-B4 已执行这个入口并失败，所以上述 state anchor 与 active 对照不再执行。

完整验证报告见 `compare_results/reports/p0b3_reliability_validation_20260720.md`；消融数据见 `compare_results/data/p0b3_reliability_feature_ablation_20260720.csv`。

### 5.7 P0-B4：独立 observation-only reliability 验证

本地已新增：

```text
tools/validate_observation_reliability.py
tools/run_p0b4_observation_validation.sh
```

`observation_v1` 固定使用以下5个上一 observation 特征，不再包含 raw `current_delta_t` 或 CV geometry：

```text
log1p(prev_obs_search_point_count)
prev_obs_empty_fallback
prev_obs_mean_fg_score
prev_obs_fg_margin_mean
prev_obs_motion_dynamic_probability
```

删除字段的结构理由是：`prev_obs_forward_ran` 与 empty fallback 完全互补；`pred_cv_available` 在 full-history 行中恒真；soft foreground count 严格等于 `1024 * mean_fg_score`；estimated foreground points 基本等于 `search_points * mean_fg_score`；entropy 与 margin 高度冗余。这个特征集已经冻结，不能根据 mini_val 结果再挑选。

只需把上面两个新文件同步到服务器。原有三个 P0-B3 Python 工具和 A1/HTV YAML 不变。先运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
chmod +x tools/run_p0b4_observation_validation.sh

MAX_TRACKLETS=10 bash tools/run_p0b4_observation_validation.sh
```

smoke 通过后运行完整 mini_val：

```bash
bash tools/run_p0b4_observation_validation.sh \
  2>&1 | tee logs/diagnostics/p0b4_observation_validation_full.log
```

若服务器路径不同，可显式覆盖，不需要修改脚本：

```bash
REPO_ROOT=/path/to/CT-SeqTrack \
DATA_ROOT=/path/to/nuscenes-mini \
A1_CKPT=/path/to/a1-order-last.ckpt \
DEVICE=cuda:0 \
bash tools/run_p0b4_observation_validation.sh
```

full run 自动生成并检查：

```text
output/diagnostics/crop_reachability/{standard,gap1124,burst_drop}_val_reference/
output/diagnostics/reliability_signals/{standard,gap1124,burst_drop}_p0b4_val/
output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_summary.json
output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_calibrator.json
output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_report.md
```

验证器强制 mini_train fitting tracklet 与所有 evaluation tracklet 不重叠，并预注册 strong-protocol Go 门槛：AUROC `>=0.75`、`AUPRC-prevalence >=0.15`、ECE `<=0.10`、FPR `<=0.30`、operating recall `>=0.70`。任一强协议失败即 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`，不允许在 mini_val 上调参后重跑。只有输出 `GO_PASSIVE_INDEPENDENT_STATE_ANCHOR`，才进入独立 frozen-state passive anchor。

### 5.8 P0-B4 回传结果与最终决定

10-tracklet smoke 与完整 mini_val 均已完成。正式结果：

| protocol | N | prevalence | AUROC | AUPRC-prev | ECE | recall | FPR | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1623 | 0.111 | 0.794 | 0.414 | 0.073 | 0.711 | 0.220 | report-only |
| gap1124 | 829 | 0.089 | 0.680 | 0.282 | 0.093 | 0.568 | 0.258 | False |
| burst_drop | 815 | 0.085 | 0.712 | 0.328 | 0.089 | 0.609 | 0.248 | False |

完整性与复算：

- evaluation CSV `1979/984/978` 行，visible+labeled `1623/829/815` 行；无重复 endpoint，fit/eval tracklet 无交集。
- 三协议 reference exact match，checkpoint SHA256 均为 `a2fbffb...24a82`。
- 本地验证器复算得到相同 model、metrics 和 verdict，差异仅约 `1e-15`。
- mini_train fit prevalence `0.283`，mini_val strong prevalence `0.089/0.085`；强协议 Brier 略差于 prevalence 常数基线。
- 同批 raw-CV 第二 crop 的 trajectory-only endpoint 在 gap/burst 均为 0，union gain 均为 `0.00 pp`。

最终决定：

```text
NO_GO_OBSERVATION_RELIABILITY_VALIDATION
```

禁止动作：

- 不根据 mini_val 重挑 `prev_obs_*`、L2、threshold 或 crop scale；
- 不实现当前 calibrator 控制的 Kalman/frozen-state 或 active dual-anchor；
- 不增加 learned gate、trajectory encoder、Mamba、ODE/CDE 或复杂 memory；
- 不把 P0-B3 post-hoc 高 AUROC 写成独立验证通过。

下一工作入口改为 P0-C frozen variable-rate/held-out-cadence protocol；P0-A residual 只做 reachable-subset 的一次性机制收尾。完整报告见 `compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。
