# CT-SeqTrack 已完成记录

更新时间：2026-06-02

这份文件统一记录已经完成的工程验收、历史实验和可供回查的关键输出。当前和未来任务只维护在 `need_to_do.md`；研究定位和论文边界见 `refined_plan.md`；简洁实验结论见 `sum_results.md`。

注意：本文件是历史归档。下方旧日志里的“下一步”“待后续确认”只代表当时上下文，不代表当前任务；当前任务一律以 `need_to_do.md` 为准。

---

## 0. 完成总览

### 工程链路

- [x] P0：真实时间字段主链路闭合。
- [x] P1：真实时间 batch 字段、CPU forward、GPU loss、2-step train smoke test 通过。
- [x] P2：scalar-preserving `TimeEncoding` 已实现，`raw / mlp / fourier` smoke test 通过。
- [x] P3：`DynamicsEncoder` / Velocity Branch 已实现，forward / loss / 2-step train smoke test 通过。
- [x] P4：Time-resampling Consistency 已实现，paired batch / forward / loss / 2-step train smoke test 通过。
- [x] P5：Observability Gate 已实现，forward / loss / 2-step train smoke test 通过。
- [x] 当前六组新消融 YAML 已创建：

```text
cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml
cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml
cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml
cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml
```

### 已完成实验

完整结果文件位于 `compare_results/`，简洁叙事见 `sum_results.md`。

- [x] `SeqTrack baseline vs CT-SeqTrack P5 full`：P5 full final 明显退化，说明不能把 real time、dynamics、gate 混在一起下结论。
- [x] `A1-raw vs A2 raw-dyn`：A1-raw 崩坏，A2 raw-dyn 明显恢复，说明 dynamics 分支有价值，但 raw real-time 主干路径有问题。
- [x] `A1-pseudo / A1-MLP / A1-Fourier`：A1-pseudo 接近 baseline，MLP/Fourier 没有救回 real-time A1，说明问题不是简单时间编码函数。
- [x] `A1-scaled / A2-scaled-dyn`：缩放 real time 仍未修复，说明主干对时间 token 语义敏感，不只是数值尺度问题。
- [x] `A1-order / A2-order-dyn`：恢复 order-time 主干后，A1-order 基本修复 A1 崩坏；A2-order-dyn final precision 高于 baseline，是当前最强正向信号。
- [x] `A2-order-dyn-cand1 / A2-order-dyn-disp`：cand1 在 60 epoch 协议下明显退化但 step 未对齐；disp 与 A2-order-dyn 基本持平，precision 小幅更高。
- [x] `A1-order+TWC / A2-order-dyn+TWC`：两组已跑完并整理，但 `twc_valid_ratio=0`，说明 TWC 项未激活；这些结果不能作为 TWC 有效或无效的最终结论。

### 当前结论

```text
真实时间方向没有被否定；
当前最稳路线是保留 SeqTrack3D 主干的 order-time 语义，
把真实 delta_t 放进 DynamicsEncoder 作为运动先验。
```

后续要做的事情不要写在本文件，统一放到 `need_to_do.md`。

---

## 2026-06-02：cand1 / disp / order+TWC 正式结果整理

### cand1 / displacement 诊断

关键结果：

```text
SeqTrack baseline final:    success 50.99, precision 59.96
A2-order-dyn final:         success 50.96, precision 63.31
A2-order-dyn-cand1 final:   success 26.68, precision 24.50
A2-order-dyn-disp final:    success 50.54, precision 63.85
```

判断：

- `A2-order-dyn-cand1` 明显退化，但 `num_candidates=1` 让 60 epoch 只有约 18899 step，而 cand4 的 A2-order-dyn 是 75719 step；当前只能说明 60 epoch cand1 协议不稳，不能彻底判死 candidate noise 假设。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本持平，final precision 高 0.53；小权重 displacement loss 不伤主线，但不是决定性收益来源。

归档文件：

```text
compare_results/cand1_disp_dynamics_comparison.md
compare_results/cand1_disp_dynamics_metrics_summary.csv
compare_results/cand1_disp_dynamics_metrics_points.csv
compare_results/cand1_disp_dynamics_curves.png
compare_results/cand1_disp_dynamics_best_final_summary.png
```

### order+TWC 诊断

关键结果：

```text
A1-order final:          success 51.23, precision 57.86
A1-order+TWC final:      success 45.61, precision 50.77
A2-order-dyn final:      success 50.96, precision 63.31
A2-order-dyn+TWC final:  success 38.27, precision 38.85
```

关键诊断：

```text
两组 order+TWC 的 loss_twc / twc_valid_ratio / twc_center_gap / twc_angle_gap
全程为 0。
```

判断：

- 当前两组 order+TWC 不是 active-TWC 训练结果，而是 paired-view / cand1 / reduced-step 训练结果。
- 下降幅度不能解释成 TWC 与 order-time 主干或 dynamics prior 不兼容。
- 下一步应先修复 TWC validity / logging，使 `twc_valid_ratio` 非 0 后，先重跑 `A1-order+TWC`，再重跑 `A2-order-dyn+TWC`。

归档文件：

```text
compare_results/twc_order_ablation_comparison.md
compare_results/twc_order_ablation_metrics_summary.csv
compare_results/twc_order_ablation_metrics_points.csv
compare_results/twc_order_ablation_twc_diagnostics_summary.csv
compare_results/twc_order_ablation_curves.png
compare_results/twc_order_ablation_step_aligned_curves.png
compare_results/twc_order_ablation_delta_summary.png
compare_results/twc_order_ablation_twc_diagnostics.png
```

---

## 2026-05-26：P0-P2 验收状态

### 总结

- `P0` 真实时间主链路的代码修改已完成。
- `P1-1` 真实 batch 时间字段验收已完成。
- `P1-2` CPU forward smoke test 已完成，输出 tensor 全部 finite。
- `P1-2b` GPU loss smoke test 已完成，forward 输出和所有 loss 均 finite。
- `P1-3` 真实时间小训练步已通过，`loss_total` 和梯度均 finite，没有 NaN。
- `P2-1` scalar-preserving `TimeEncoding` 已实现。
- `P2-2` `raw / mlp / fourier` forward smoke test 已通过。
- `P2-3` `raw / mlp / fourier` GPU loss smoke test 已通过。

### 已完成文件

```text
CT-SeqTrack/datasets/misc_utils.py
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/models/base_model.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/models/time_encoding.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

### 时间字段链路

训练侧 `motion_processing_mf()` 已输出：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
velocity_label
```

测试侧 `MotionBaseModelMF.build_input_dict()` 已输出：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
```

模型侧已改为：

```python
delta_T = input_dict["delta_T"]
corner_stamps = create_corner_timestamps_from_deltas(delta_T, 8)
corner_stamps = self.time_encoder(corner_stamps)
box_seq_corners = torch.cat((box_seq_corners, corner_stamps), dim=-1)
```

点云时间也走同一个 `TimeEncoding`：

```python
points = self.encode_point_time(input_dict["points"])
```

### P1-1：真实 batch 时间字段

真实时间关键输出：

```text
points shape: (2, 4096, 5)
valid_mask: [1 1 1]
timestamps: [-0.499305 -1.049506 -1.549402  0.      ]
delta_T:    [-0.499305 -1.049506 -1.549402]
delta_t:    [0.499305  0.55020094 0.49989605]
current_delta_t: 0.499305
```

判断：

- 历史帧 `points[..., 3]` 为负数，当前帧为 `0`。
- `delta_T` 为负数。
- `delta_t` 为正数。
- nuScenes keyframe 约 2Hz，`delta_t` 在 `0.5s` 附近正确。
- 出现 `0.5502s` 说明确实读取了真实 timestamp，不是硬编码 `0.5`。

固定伪时间对照：

```text
valid_mask: [1 1 1]
timestamps: [-0.1 -0.2 -0.3  0. ]
delta_T:    [-0.1 -0.2 -0.3]
delta_t:    [0.1 0.1 0.1]
current_delta_t: 0.1
```

判断：

```text
use_real_time=True/False 已经能在 batch 级正确切换。
```

### P1-2：CPU forward smoke test

关键输出：

```text
device: cpu
valid_mask: [1 1 1]
delta_T: [-0.499305 -1.049506 -1.549402]
delta_t: [0.499305 0.55020094 0.49989605]

pred_bc: shape=(1, 4096, 9), finite=True
motion_cls: shape=(1, 2), finite=True
estimation_boxes: shape=(1, 4), finite=True
seg_logits: shape=(1, 2, 4096), finite=True
motion_pred: shape=(1, 4), finite=True
aux_estimation_boxes: shape=(1, 4), finite=True
ref_boxs: shape=(1, 3, 4), finite=True
valid_mask: shape=(1, 3), finite=True
updated_ref_boxs: shape=(1, 3, 4), finite=True
```

判断：

```text
delta_T -> create_corner_timestamps_from_deltas() -> box corner timestamp -> Transformer forward
主链路已通过 shape 和 NaN/Inf 检查。
```

### P1-2b：GPU loss smoke test

关键输出：

```text
device: cuda
valid_mask: [1 1 1]
delta_T: [-0.499305 -1.049506 -1.549402]
delta_t: [0.499305 0.55020094 0.49989605]

loss_motion_cls: 0.686676, finite=True
loss_center: 0.000131, finite=True
loss_angle: 0.000000, finite=True
loss_total: 3.694282, finite=True
loss_seg: 0.665731, finite=True
loss_center_aux: 0.523946, finite=True
loss_center_motion: 0.000000, finite=True
loss_angle_aux: 0.035597, finite=True
loss_angle_motion: 0.000000, finite=True
loss_center_ref: 0.447077, finite=True
loss_angle_ref: 0.031530, finite=True
loss_bc: 2.033965, finite=True
```

判断：

```text
GPU forward + compute_loss() 已通过 finite 检查。
真实时间字段已经能进入模型并参与完整 loss 计算。
```

### P1-3：真实时间小训练步

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=9.924500 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=2 loss_total=12.464478 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p1_3_real_time_gpu0_loss.jsonl
last checkpoint: output/p1_3_real_time_gpu0_ckpt/last.pt
```

判断：

```text
真实时间字段不仅能进入 forward 和 loss，而且已经能参与最小 optimizer 更新。
```

注意：

- 当前只跑了 `max_steps=2`，足够作为 smoke test，不等价于正式小 epoch。
- `check_train_steps.py` 保存普通 PyTorch `state_dict` checkpoint，不是 Lightning `.ckpt`。

### P2：TimeEncoding

当前实现：

- 新增 `CT-SeqTrack/models/time_encoding.py`。
- `seqtrack3d.py` 中 point time 和 box corner time 共用同一个 `TimeEncoding`。
- `seqtrack3d_nuscenes.yaml` 和 `seqtrack3d_waymo.yaml` 已加入 `time_encoding / time_scale / time_clip / time_fourier_bands / time_hidden_dim`。
- 默认 `time_encoding: raw`，保持真实时间 scalar 行为。
- `mlp` 和 `fourier` 仍保持一个时间通道，不改变 PointNet/Transformer 输入维度。
- `raw / mlp / fourier` 的 forward smoke test 和 GPU loss smoke test 均 finite。
- 当前帧 `t=0` 输出保持为 `0`。

---

## 常用验收命令

服务器路径：

```text
CT-SeqTrack: /home/lishengjie/study/lcyu/CT-SeqTrack
nuScenes-mini: /home/lishengjie/data/nuscenes-mini
config: /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
```

真实 batch 时间字段：

```bash
python tools/check_time_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

固定伪时间对照：

```bash
python tools/check_time_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history \
  --pseudo-time
```

CPU forward：

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/check_forward_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --no-loss
```

小训练步：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0
```

共享服务器上的低占用长版命令见：

```text
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

注意：`compute_loss()` 内部仍有 `.cuda()`，CPU 下不能直接算 loss。CPU forward 只验证模型 forward 主链路。

---

## 2026-05-26：P3 Dynamics / Velocity Branch 实现

### 代码改动

新增：

```text
CT-SeqTrack/models/dynamics.py
```

修改：

```text
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
```

### 实现内容

- 新增 `DynamicsEncoder`，输入 `ref_boxs / delta_t / valid_mask`。
- 按历史框序列构造真实时间差分动力学特征：

```text
d_i     = c_i - c_{i-1}
v_i     = d_i / max(delta_t_i, eps)
omega_i = wrap(theta_i - theta_{i-1}) / max(delta_t_i, eps)
speed_i = ||v_i||
gap_i   = delta_t_i
valid_i = valid_mask_i * valid_mask_{i-1}
```

- 使用 per-step MLP + masked mean/max pooling 得到 `z_dyn`。
- 输出 `velocity_pred` 作为轻量速度监督分支。
- 在 `seqtrack3d.py` 中通过 `use_dynamics_encoder` 开关控制是否启用。
- 默认配置保持关闭，不影响 P0-P2 baseline。
- 启用后 coarse motion branch 使用 `torch.cat([point_feature, z_dyn], dim=1)`。
- `compute_loss()` 在存在 `velocity_label` 时加入：

```text
loss_velocity = SmoothL1(velocity_pred, velocity_label)
loss_total += velocity_weight * loss_velocity
```

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/models/dynamics.py CT-SeqTrack/models/seqtrack3d.py
```

已通过直接加载 `dynamics.py` 的纯张量 smoke test：

```text
z shape: (1, 128), finite=True
velocity_pred shape: (1, 3), finite=True
dynamics_valid shape: (1, 1), value=[[1.0]]
invalid-history case: z sum=0.0, velocity sum=0.0, dynamics_valid=[[0.0]]
```

未在本地跑完整 `check_forward_batch.py`，因为本机缺少部分训练依赖和数据集。完整 forward/loss 与训练步检查已在服务器完成。

```text
check_forward_batch.py
check_train_steps.py --max-steps 2
```

### 服务器 P3 forward + loss smoke test

已在服务器使用 P3 配置运行：

```bash
python tools/check_forward_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p3_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history
```

关键输出：

```text
using batch_idx=12
timestamps shape=(1, 4): [-0.499305 -1.049506 -1.549402  0.      ]
delta_T shape=(1, 3): [-0.499305 -1.049506 -1.549402]
delta_t shape=(1, 3): [0.499305   0.55020094 0.49989605]
current_delta_t shape=(1,): 0.49930500984191895
valid_mask shape=(1, 3): [1 1 1]
device: cuda

pred_bc: shape=(1, 4096, 9), finite=True
velocity_pred: shape=(1, 3), finite=True
dynamics_valid: shape=(1, 1), finite=True
motion_cls: shape=(1, 2), finite=True
estimation_boxes: shape=(1, 4), finite=True
seg_logits: shape=(1, 2, 4096), finite=True
motion_pred: shape=(1, 4), finite=True
aux_estimation_boxes: shape=(1, 4), finite=True
ref_boxs: shape=(1, 3, 4), finite=True
valid_mask: shape=(1, 3), finite=True
updated_ref_boxs: shape=(1, 3, 4), finite=True

loss_motion_cls: 0.772253, finite=True
loss_center: 0.001527, finite=True
loss_angle: 0.006683, finite=True
loss_total: 10.814757, finite=True
loss_seg: 0.737634, finite=True
loss_center_aux: 1.532263, finite=True
loss_center_motion: 0.000000, finite=True
loss_angle_aux: 0.481590, finite=True
loss_angle_motion: 0.000000, finite=True
loss_center_ref: 1.518901, finite=True
loss_angle_ref: 0.385201, finite=True
loss_velocity: 0.002232, finite=True
loss_bc: 2.024360, finite=True
```

判断：

```text
P3 dynamics branch 已能在真实 batch 上进入 forward 和 compute_loss。
velocity_pred / dynamics_valid / loss_velocity / loss_total 均 finite。
```

注意：

- PointNet2 的 `SyntaxWarning: "is" with a literal` 仍是旧代码警告，不影响本次结果。
- 当前已完成 forward + loss smoke test，输出 `velocity_pred / dynamics_valid / loss_velocity` 均正常。

### 服务器 P3 train-step smoke test

已在服务器继续使用同一份 P3 配置运行 2-step 训练检查：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
timeout 20m nice -n 19 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p3_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --log-file output/p3_dyn_real_time_gpu0_loss.jsonl \
  --checkpoint-dir output/p3_dyn_real_time_gpu0_ckpt \
  --tag p3_dyn_real_time_gpu0
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=22.146883 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=4.546315 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p3_dyn_real_time_gpu0_loss.jsonl
last checkpoint: output/p3_dyn_real_time_gpu0_ckpt/last.pt
```

判断：

```text
P3 dynamics branch 不仅能完成 forward 和 compute_loss，也已经能完成 backward、梯度裁剪、optimizer step、loss log 写出和 checkpoint 保存。
loss_total 与 grad_norm 均 finite，P3 工程验收完成，可以进入 P4。
```

### 当前 GitHub 同步范围

本轮同步建议包含以下文件，构成 P0-P3 的可复现工程快照：

```text
.gitignore
need_to_do.md
refined_plan.md
done.md
CT-SeqTrack/models/time_encoding.py
CT-SeqTrack/models/dynamics.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_train_steps.py
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

同步后建议从 P4 开始新一轮改动，不再继续在同一个提交里混入 TWC 代码。

---

## 2026-05-27：P4 Time-resampling Consistency 代码实现

### 代码改动

新增：

```text
CT-SeqTrack/tools/check_twc_batch.py
```

修改：

```text
CT-SeqTrack/datasets/misc_utils.py
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
```

### 实现内容

- `get_history_frame_ids_and_masks()` 增加 `offsets` 参数，默认保持 `[1, 2, ..., hist_num]` 行为。
- `MotionTrackingSamplerMF` 在 `use_twc=True` 时返回 `view_a / view_b` paired batch。
- 第一版 paired view 使用同当前帧、同最近历史 anchor、不同旧历史路径：

```text
view_a: [t-1, t-2, t-3] -> t
view_b: [t-1, t-3, t-5] -> t
```

- TWC 默认关闭，不影响 P0-P3 baseline。
- `SEQTRACK3D.forward()` 和 `compute_loss()` 支持 nested paired batch。
- 新增 `compute_twc_loss()`，只约束最终 `aux_estimation_boxes`。
- paired supervised loss 使用 `0.5 * (L_a + L_b)`，避免监督项权重翻倍。
- `check_train_steps.py` 和 `check_forward_batch.py` 已支持递归移动 nested dict。
- `check_time_batch.py`、`check_forward_batch.py`、`check_train_steps.py` 增加 `--twc` 临时开关，可在不修改 YAML 的情况下启用 paired-view 检查。

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/datasets/misc_utils.py CT-SeqTrack/datasets/sampler.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_train_steps.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_time_batch.py CT-SeqTrack/tools/check_twc_batch.py
python -m py_compile CT-SeqTrack/datasets/sampler.py
```

已通过 offsets 纯函数检查：

```text
(7, 3, None)      -> ([6, 5, 4], [1, 1, 1])
(7, 3, [1,3,5])  -> ([6, 4, 2], [1, 1, 1])
(2, 3, [1,3,5])  -> ([1, 0, 0], [1, 0, 0])
bad offsets       -> correctly raises ValueError
```

### 服务器 P4 paired batch 检查

已在服务器运行：

```bash
python tools/check_twc_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

关键输出：

```text
using batch_idx=3
view_a prev_frame_ids: [5 4 3]
view_a history_offsets: [1 2 3]
view_a timestamps: [-0.49572396 -0.95052814 -1.4509721   0.        ]
view_a delta_T:    [-0.49572396 -0.95052814 -1.4509721]
view_a delta_t:    [0.49572396 0.45480418 0.50044394]
view_a current_delta_t: 0.4957239627838135
view_a current_timestamp: 1532402930.648325
view_a valid_mask: [1 1 1]

view_b prev_frame_ids: [5 3 1]
view_b history_offsets: [1 3 5]
view_b timestamps: [-0.49572396 -1.4509721  -2.500478    0.        ]
view_b delta_T:    [-0.49572396 -1.4509721  -2.500478]
view_b delta_t:    [0.49572396 0.9552481  1.049506]
view_b current_delta_t: 0.4957239627838135
view_b current_timestamp: 1532402930.648325
view_b valid_mask: [1 1 1]

shape_mismatches: none
same_current_timestamp: [True, True]
same_anchor_ref_box: [True, True]
different_delta_T: [True, True]
full_history_a: [True, True]
full_history_b: [True, True]
twc_valid: [True, True]
```

判断：

```text
paired view 数据构造正确。view_a / view_b 共享同一当前帧和最近历史 anchor，
只改变更早历史路径；nested batch 能被 DataLoader 正常 collate。
```

### 服务器 P4 forward + loss smoke test

已在服务器运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --twc
```

关键输出：

```text
using batch_idx=5
device: cuda

output view_a:
pred_bc / motion_cls / estimation_boxes / seg_logits / motion_pred /
aux_estimation_boxes / ref_boxs / valid_mask / updated_ref_boxs: finite=True

output view_b:
pred_bc / motion_cls / estimation_boxes / seg_logits / motion_pred /
aux_estimation_boxes / ref_boxs / valid_mask / updated_ref_boxs: finite=True

loss_total: 4.892723, finite=True
loss_total_sup: 4.892723, finite=True
loss_total_a: 4.889785, finite=True

loss_twc: 0.000004, finite=True
twc_valid_ratio: 1.000000, finite=True
twc_center_gap: 0.002937, finite=True
twc_angle_gap: 0.002928, finite=True
```

判断：

```text
paired forward 与 paired loss 均通过 finite 检查。
twc_valid_ratio=1.0 说明当前 batch 中所有样本均满足同当前时刻、同 anchor、
不同历史路径和完整历史条件。loss_twc 约 4e-6，远小于 loss_total，
当前权重下不会主导训练；center / angle gap 均很小，说明两条历史采样路径
在未训练初始状态下已能产生接近的最终框，TWC 项的量级是安全的。
```

### 服务器 P4 train-step smoke test

已在服务器运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --twc
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
use_twc: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=10.901030 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=9.719684 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/check_train_steps_loss.jsonl
last checkpoint: output/check_train_steps_ckpt/last.pt
```

判断：

```text
P4 TWC 不仅能完成 paired batch、forward 和 compute_loss，也已经能完成
backward、梯度裁剪、optimizer step、loss log 写出和 checkpoint 保存。
loss_total 与 grad_norm 均 finite，P4 工程 smoke test 已通过。
```

### 待后续确认

```text
1. 再跑一次 use_twc=False 的默认 forward/train-step，对照确认默认路径仍与 P0-P3 兼容。
2. 检查 output/check_train_steps_loss.jsonl 中是否写入 loss_twc / twc_valid_ratio /
   twc_center_gap / twc_angle_gap。
3. 进入小规模训练和消融前，建议为 P4 单独保存一份配置，例如
   cfgs/seqtrack3d_nuscenes_p4_twc.yaml。
```

### 2026-05-27：P4 剩余收口项取消为当前阻塞项

决定：

```text
P4 已有 paired batch / forward / loss / 2-step train smoke test 记录，足够支撑进入 P5。
默认路径回归、JSONL 字段复核和 P4 专用配置暂不作为当前阻塞任务。
这些项目后续如进入正式消融，再和实验配置一起补齐。
```

下一步：

```text
进入 P5 Observability Gate。
P5 第一版保持轻量：只在 coarse motion branch 中融合 observation feature 与 P3 dynamics prior。
不引入复杂 memory、不引入多模态、不改 TWC 和 Transformer refine。
```

---

## 2026-05-27：P5 观测可靠性统计量实现

### 代码改动

修改：

```text
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/models/base_model.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_twc_batch.py
```

### 实现内容

- 训练侧 `motion_processing_mf()` 在当前搜索区域裁剪后、`regularize_pc()` 前记录真实点数：

```text
num_points_in_search = this_frame_pc.nbr_points()
```

- 测试侧 `MotionBaseModelMF.build_input_dict()` 使用同一口径写入 `num_points_in_search`。
- `SEQTRACK3D.build_observability_stats()` 已构造第一版 P5 观测可靠性统计量：

```text
obs_stats = [
  log1p(num_points_in_search),
  log1p(soft_fg_count_current),
  mean_fg_score_current,
  valid_history_ratio,
  current_delta_t / time_scale
]
```

- `soft_fg_count_current / mean_fg_score_current` 只来自当前帧 chunk 的 `seg_logits`，不混入历史点云。
- 默认 `obs_stats_detach_seg=True`，避免后续 gate 通过统计量反向操纵 segmentation confidence。
- 检查脚本已增加 `num_points_in_search` 打印。

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/datasets/sampler.py CT-SeqTrack/models/base_model.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_time_batch.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_twc_batch.py
```

未在本地运行完整模型 import 单元测试，因为本地环境缺少 `easydict` 等训练依赖；后续服务器 smoke test 可通过 `check_forward_batch.py` 查看 `obs_stats / obs_*` 输出是否 finite。

---

## 2026-05-27：P5 Observability Gate 主体实现

### 代码改动

新增：

```text
CT-SeqTrack/models/observability.py
CT-SeqTrack/tools/check_observability_gate.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml
```

修改：

```text
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
need_to_do.md
```

### 实现内容

- 新增 `ObservabilityGate`：

```text
point_feature: B,256
z_dyn: B,dynamics_hidden_dim
obs_stats: B,5
dynamics_valid: B,1

z_dyn_proj = Linear(z_dyn) -> B,256
alpha = softmax(MLP(obs_stats)) -> [alpha_obs, alpha_dyn]
fused_feature = alpha_obs * point_feature + alpha_dyn * z_dyn_proj
```

- `gate_mlp` 最后一层权重初始化为 0，bias 初始化为 `[obs_gate_init_obs_bias, 0]`，训练初期偏向 observation。
- `dynamics_valid < obs_gate_min_dyn_valid` 时强制 `alpha_dyn=0, alpha_obs=1`。
- `SEQTRACK3D` 中新增 `use_observability_gate` 开关；打开 P5 时必须同时打开 `use_dynamics_encoder`。
- P5 打开时 motion feature 保持 256 维并复用原始 `motion_mlp`；P3 dynamics-only 路径仍使用 `torch.cat([point_feature, z_dyn])`。
- `compute_loss()` 已记录：

```text
obs_alpha_obs_mean
obs_alpha_dyn_mean
obs_alpha_dyn_min
obs_alpha_dyn_max
obs_gate_entropy
obs_num_points_search_mean
obs_soft_fg_count_mean
obs_mean_fg_score
obs_valid_history_ratio
obs_current_delta_t_ratio
```

- `obs_gate_entropy_weight` 已接入，默认 `0.0`，当前不改变 loss。
- `check_forward_batch.py` 和 `check_train_steps.py` 新增 `--obs-gate`，可临时打开 `use_dynamics_encoder=True` 和 `use_observability_gate=True`。
- 新增 P5 专用 nuScenes 配置：

```text
cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml
```

其中 `use_dynamics_encoder=True`、`use_observability_gate=True`、`use_twc=False`。

### 本地检查

已通过：

```text
python CT-SeqTrack/tools/check_observability_gate.py
python -m compileall CT-SeqTrack/models/observability.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_observability_gate.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_train_steps.py
```

纯张量 smoke test 关键输出：

```text
fused shape: (2, 256), finite=True
alpha: [[0.7310586  0.26894143]
        [1.         0.        ]]
alpha_sum_ok: True
invalid_dyn_ok: True
```

服务器 P5 forward + loss smoke test 已通过。运行命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --obs-gate
```

关键输出：

```text
using batch_idx=12
timestamps shape=(1, 4): [-0.499305 -1.049506 -1.549402  0.      ]
delta_T shape=(1, 3): [-0.499305 -1.049506 -1.549402]
delta_t shape=(1, 3): [0.499305   0.55020094 0.49989605]
current_delta_t shape=(1,): 0.49930500984191895
num_points_in_search shape=(1,): 3.0
valid_mask shape=(1, 3): [1 1 1]
device: cuda

velocity_pred: shape=(1, 3), finite=True
dynamics_valid: shape=(1, 1), finite=True
obs_alpha: shape=(1, 2), finite=True
obs_alpha_obs: shape=(1,), finite=True
obs_alpha_dyn: shape=(1,), finite=True
obs_gate_entropy: shape=(1,), finite=True
obs_stats: shape=(1, 5), finite=True
obs_num_points_search / obs_soft_fg_count / obs_mean_fg_score /
obs_valid_history_ratio / obs_current_delta_t_ratio: finite=True

loss_total: 4.333134, finite=True
loss_velocity: 0.001596, finite=True
obs_num_points_search_mean: 3.000000, finite=True
obs_soft_fg_count_mean: 497.636841, finite=True
obs_mean_fg_score: 0.485973, finite=True
obs_valid_history_ratio: 1.000000, finite=True
obs_current_delta_t_ratio: 0.998610, finite=True
obs_alpha_obs_mean: 0.731059, finite=True
obs_alpha_dyn_mean: 0.268941, finite=True
obs_alpha_dyn_min: 0.268941, finite=True
obs_alpha_dyn_max: 0.268941, finite=True
obs_gate_entropy: 0.582203, finite=True
```

判断：

```text
P5 forward + loss 主链路已通过。当前样本 num_points_in_search=3，属于极稀疏搜索区域；
obs_stats、obs_alpha、obs_gate_entropy 和所有 tracking loss 均 finite。
alpha_obs≈0.731 / alpha_dyn≈0.269 与 obs_gate_init_obs_bias=1.0 的初始化一致，
说明 gate 初始化和 dynamics_valid 有效路径正常；这不是训练后 gate 已学到策略的结论。
```

### 服务器 P5 train-step smoke test

已通过。运行命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --obs-gate \
  --log-file output/p5_obs_gate_loss.jsonl \
  --checkpoint-dir output/p5_obs_gate_ckpt \
  --tag p5_obs_gate
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
use_twc: False
use_observability_gate: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=14.486424 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=5.297028 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p5_obs_gate_loss.jsonl
last checkpoint: output/p5_obs_gate_ckpt/last.pt
```

判断：

```text
P5 Observability Gate 已能完成 forward、compute_loss、backward、梯度裁剪、
optimizer step、loss log 写出和 checkpoint 保存。P5 工程 smoke test 已通过。
默认关闭路径回归检查已取消，不再作为后续计划任务；下一步重点转向困难子集评估、
正式消融，以及观察训练后 alpha 是否随 sparse / gap / fg score 分桶发生合理变化。
```
