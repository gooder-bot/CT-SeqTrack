# CT-SeqTrack 当前执行清单

更新时间：2026-05-27

文件分工：

- `need_to_do.md`：只放当前执行清单。
- `refined_plan.md`：放研究计划、论文定位、贡献和 related work 边界。
- `log.md`：放工程日志、验收记录和常用命令。

---

## 0. 当前状态

当前只改：

- 本地：`D:\desktop\research\CT-SeqTrack`
- 服务器：`/home/lishengjie/study/lcyu/CT-SeqTrack`

不要同时改原始 `seqtrack`，避免 baseline、改进版和实验结果混在一起。

已完成：

- [x] P0：真实时间主链路闭合。
- [x] P1：真实时间 baseline smoke test 通过。
- [x] P2：scalar-preserving `TimeEncoding` 已实现，`raw / mlp / fourier` smoke test 通过。
- [x] P3：Dynamics / Velocity Branch 代码已实现，默认关闭。
- [x] P3：服务器 `check_forward_batch.py` + loss smoke test 已通过。
- [x] P3：服务器 `check_train_steps.py --max-steps 2` 已通过，backward / optimizer step 正常。
- [x] P4：Time-resampling Consistency 代码已实现，paired batch / forward / loss / 2-step train smoke test 已通过。
- [x] P5：Observability Gate 代码已实现，forward / loss / 2-step train smoke test 已通过。

下一步：

- [ ] 实验：补困难子集评估与正式消融。

P0-P3 的详细验收记录见 `log.md`。论文定位和边界见 `refined_plan.md`。

当前可提交快照：

- 默认配置仍保持 baseline 兼容：`use_dynamics_encoder: False`。
- P3 可通过复制一份实验配置并设置 `use_dynamics_encoder=True` 打开。
- 已有验收只证明工程链路可跑通，不等价于正式性能结论。
- P4 剩余收口项不再作为当前阻塞任务；先保持已有 smoke-test 记录，后续若进入正式消融再补专用配置和日志字段复核。

---

## 1. P3：Dynamics / Velocity Branch

代码状态：已实现，默认 `use_dynamics_encoder: False`，不影响现有 baseline。打开后模型会输出 `velocity_pred / dynamics_valid`，训练 batch 中有 `velocity_label` 时会额外加入 `loss_velocity`。服务器上 P3 forward + loss smoke test 和 2-step train smoke test 均已通过，P3 工程验收完成。

目标：让模型显式看到历史框在真实 `delta_t` 下对应的速度、角速度和运动趋势。

第一版只做真实时间差分，不做 ODE/SDE/CDE：

```text
d_i     = c_i - c_{i-1}
v_i     = d_i / max(delta_t_i, eps)
omega_i = wrap(theta_i - theta_{i-1}) / max(delta_t_i, eps)
speed_i = ||v_i||
gap_i   = delta_t_i
valid_i = valid_mask_i * valid_mask_{i-1}
```

建议新增：

```text
CT-SeqTrack/models/dynamics.py
```

建议配置：

```yaml
use_dynamics_encoder: False
dynamics_hidden_dim: 128
velocity_weight: 0.05
dynamics_use_acceleration: False
```

推荐结构：

```text
DynamicsEncoder
  input: ref_boxs, delta_t, valid_mask
  per_step_mlp: dyn_dim -> 64 -> 64
  masked_pool: mean/max over valid transitions
  global_mlp: 64 -> 128
  output: z_dyn [B, 128], velocity_pred [B, 3]
```

推荐接入点：

```python
point_feature = self.mini_pointnet(mask_points)
z_dyn, velocity_pred = self.dynamics_encoder(ref_boxs, delta_t, valid_mask)
motion_feature = torch.cat([point_feature, z_dyn], dim=1)
motion_pred = self.motion_mlp_dyn(motion_feature)
```

速度监督：

```text
L_vel = SmoothL1(velocity_pred, velocity_label)
L = L_original + lambda_vel * L_vel
```

注意：

- 当前 `velocity_label` 只针对最近历史帧到当前帧：

```text
velocity_label = motion_label_list[0][:3] / current_delta_t
```

- `Dynamics Encoder` 自身应该从完整 `ref_boxs / delta_t / valid_mask` 提取历史趋势。
- 不要第一版加加速度，二阶差分会放大历史框噪声。
- 不要把最终框硬约束为速度积分结果，否则会伤害急转弯和遮挡后重捕获。

验收：

- `raw / mlp / fourier` 下 forward 和 loss 仍 finite。
- `L_vel` 能下降。
- `velocity_pred` 量级合理。
- `valid_mask=0` 的历史帧不参与动力学池化。
- 主表不明显退化，大 gap 子集最好不退化或提升。

服务器 smoke test 建议：

```text
1. 已完成：复制 P3 配置并设置 use_dynamics_encoder=True。
2. 已完成：check_forward_batch.py，velocity_pred / dynamics_valid / loss_velocity / loss_total 均 finite。
3. 已完成：check_train_steps.py --max-steps 2，loss_total 和 grad_norm 均 finite，并成功写出 loss log / checkpoint。
```

已通过的服务器训练步命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

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

下一步：进入 P4，实现 Time-resampling Consistency。

---

## 2. P4：Time-resampling Consistency

目标：同一条 tracklet、同一个当前绝对时刻 `t`，在不同历史采样路径下得到的最终当前状态估计应一致。

这一节必须写窄、做窄：TWC 不是普通 temporal smoothness，也不是长时记忆一致性；它只约束 **different sampling paths to the same absolute time**。

### P4-0：已确认的采样语义

- [x] 当前 `prev_frame_ids` 顺序是从最近历史到更旧历史：默认等价于 `offsets=[1, 2, 3]`。
- [x] `valid_mask=1` 表示该历史帧真实存在；`valid_mask=0` 时使用第 0 帧 padding，不应参与 TWC。
- [x] `motion_processing_mf()` 使用 `ref_boxs[0]` 作为当前搜索区域和局部坐标系的 anchor。

关键结论：第一版 TWC 的两个 view 必须共享最近历史帧 `t-1` 作为 anchor，即 `twc_view_a_offsets[0] == twc_view_b_offsets[0] == 1`。否则两个输出框处在不同局部坐标系里，直接比较 `aux_estimation_boxes` 会把坐标系差异混进一致性 loss。

### P4-1：历史 offset 参数化

- [x] 修改 `get_history_frame_ids_and_masks(this_frame_id, hist_num, offsets=None)`。
- [x] 默认 `offsets=None` 时保持旧行为：`[1, 2, ..., hist_num]`。
- [x] `offsets` 语义：正整数 frame gap，按最近到更旧排序，例如 `[1, 3, 5]`。
- [x] 校验 `len(offsets) == hist_num`，并建议要求严格递增，避免历史顺序歧义。

推荐实现语义：

```text
offsets=[1, 2, 3] -> prev_frame_ids=[t-1, t-2, t-3]
offsets=[1, 3, 5] -> prev_frame_ids=[t-1, t-3, t-5]
frame_id < 0      -> frame_id=0, valid_mask=0
```

### P4-2：paired view 数据返回

- [x] 在 `MotionTrackingSamplerMF` 中增加可关闭 paired view 路径，默认关闭。
- [x] `use_twc: False` 时仍返回原来的 flat `data_dict`，保证 baseline 完全不变。
- [x] `use_twc: True` 时返回嵌套 batch：

```text
view_a: offsets=[1, 2, 3] 的 data_dict
view_b: offsets=[1, 3, 5] 的 data_dict
```

第一版为了让 loss 干净，paired view 必须满足：

- 两个 view 的 `this_frame_id / current_timestamp / box_label / bbox_size` 相同。
- 两个 view 的最近历史 anchor 相同：`prev_frame_ids[0]` 相同，推荐都是 `t-1`。
- 两个 view 的旧历史不同：`delta_T` 至少有一个历史位置不同。
- 两个 view 都有完整有效历史时才计算 TWC；早期 padding 样本只算各自 supervised loss。

注意随机 offset：当前 `candidate_id != 0` 时 `motion_processing_mf()` 会给历史框随机扰动。如果两个 view 独立调用，会破坏共享 anchor 的局部坐标系。第一版有两个稳妥选择，优先选 A：

```text
A. TWC paired view 固定使用 candidate_id == 0；smoke config 可临时设 num_candidates: 1。
B. 为两个 view 共享同一组 anchor sample_offsets，至少保证最近历史框 ref_boxs[0] 完全一致。
```

如果后续打开 `use_augmentation`，paired view 也必须共享同一个 transform；P4 第一版建议保持 `use_augmentation: False`，先把 TWC 主链路跑稳。

### P4-3：检查脚本

- [x] 新增 `CT-SeqTrack/tools/check_twc_batch.py`。
- [x] 检查 nested batch 能被 DataLoader 正常 collate。
- [x] 检查 `view_a/view_b` 的必要字段、shape、dtype 与普通 batch 一致。
- [x] 检查 `current_timestamp` 相同、`delta_T` 不同、`valid_mask` 合法。
- [x] 检查两个 view 的 `ref_boxs[:, 0]` 完全一致或足够接近。

建议输出：

```text
view_a prev_frame_ids: [t-1, t-2, t-3]
view_b prev_frame_ids: [t-1, t-3, t-5]
same_current_timestamp: True
same_anchor_ref_box: True
different_delta_T: True
twc_valid: True/False
```

### P4-4：TWC loss 接入

- [x] 在 `SEQTRACK3D.training_step()` 中识别 paired batch。
- [x] 对 `view_a/view_b` 分别 forward 和 `compute_loss()`。
- [x] 新增 `compute_twc_loss(out_a, out_b, data_a, data_b)`，只使用最终当前框 `aux_estimation_boxes`。
- [x] 默认配置保持 `use_twc: False`，不影响 P0-P3。
- [x] 修改 `tools/check_train_steps.py` 的 `move_to_device()`，支持递归移动 nested dict。

第一版 loss：

```text
c_a, theta_a = out_a["aux_estimation_boxes"][:, :3], out_a["aux_estimation_boxes"][:, 3]
c_b, theta_b = out_b["aux_estimation_boxes"][:, :3], out_b["aux_estimation_boxes"][:, 3]

L_center = SmoothL1(c_a, c_b)
L_theta  = SmoothL1(sin(theta_a), sin(theta_b))
         + SmoothL1(cos(theta_a), cos(theta_b))
L_twc    = L_center + twc_theta_weight * L_theta
```

paired training 的总 loss 不建议直接 `L_a + L_b`，否则打开 TWC 后 supervised loss 梯度规模翻倍。第一版用平均更稳：

```text
L_sup = 0.5 * (L_a + L_b)
L     = L_sup + twc_weight * twc_valid * L_twc
```

`twc_valid` 第一版建议为 batch 级或 sample 级 mask：

```text
twc_valid = same_current_timestamp
          & same_anchor
          & full_history_a
          & full_history_b
          & different_delta_T
```

### 建议配置

```yaml
use_twc: False
twc_weight: 0.05
twc_theta_weight: 0.5
twc_view_a_offsets: [1, 2, 3]
twc_view_b_offsets: [1, 3, 5]
twc_full_history_only: True
twc_candidate_zero_only: True
twc_warmup_epoch: 0
```

### 验收标准

- 已取消为当前阻塞项：`use_twc: False` 默认路径回归检查，后续正式消融前再补。
- [x] `check_twc_batch.py` 能证明两个 view 同当前时刻、同 anchor、不同历史路径。
- [x] paired forward 中 `out_a/out_b` 所有 tensor finite。
- [x] `loss_twc / loss_total` finite，且 `loss_twc` 量级不能主导训练。
- [x] 2-step train smoke test 通过，能写出 loss log 和 checkpoint。
- [x] 训练日志里记录 `loss_twc`、`twc_valid_ratio`、`twc_center_gap`、`twc_angle_gap`。
- 已取消为当前阻塞项：后续正式实验中再验证 TWC 是否降低不同采样路径下的预测方差；不阻塞 P5。

### 服务器 P4 smoke test 记录

已通过：

```text
check_twc_batch.py:
same_current_timestamp=[True, True]
same_anchor_ref_box=[True, True]
different_delta_T=[True, True]
twc_valid=[True, True]

check_forward_batch.py --twc:
loss_total=4.892723, finite=True
loss_twc=0.000004, finite=True
twc_valid_ratio=1.000000, finite=True
twc_center_gap=0.002937, finite=True
twc_angle_gap=0.002928, finite=True

check_train_steps.py --twc --max-steps 2:
step=1/2 loss_total=10.901030 grad_norm=1.000000
step=2/2 loss_total=9.719684 grad_norm=1.000000
finished train-step check
```

判断：P4 工程链路已通过；当前结果只证明代码可训练和 loss 量级安全，不等价于正式性能结论。

---

## 3. P5：Observability Gate

当前主任务：P5。目标是根据当前观测可靠性，在 observation feature 和 timestamp-conditioned dynamics prior 之间自适应融合。P5 不引入复杂 memory、不引入多模态、不做 ODE/SDE；它只回答一个窄问题：**当前点云观测不可靠时，是否应该更信 P3 的真实时间动力学 prior。**

### P5-0：设计边界

- [x] P5 默认关闭：`use_observability_gate: False`。
- [x] P5 第一版依赖 P3：打开 `use_observability_gate=True` 时必须同时打开 `use_dynamics_encoder=True`。
- [x] P5 只接入 coarse motion branch，不改 Transformer refine、segmentation head 和 TWC。
- [x] P5 不使用 GT-only 统计量作为 gate 输入，保证 train/test 语义一致。
- [x] P5 第一版只融合 `point_feature` 与 `z_dyn`，不融合历史 memory、context token 或 virtual cue。

### P5-1：观测可靠性统计量

第一版 gate 输入只用稳定、可训练/测试一致获得的统计量：

```text
o_t = [
  log1p(num_points_in_search),
  log1p(soft_fg_count_current),
  mean_fg_score_current,
  valid_history_ratio,
  current_delta_t / time_scale
]
```

字段来源：

- `num_points_in_search`：当前搜索区域裁剪后、regularize 之前的真实点数；需要在训练侧 `motion_processing_mf()` 和测试侧 `MotionBaseModelMF.build_input_dict()` 都写入。
- `soft_fg_count_current`：只统计当前帧 chunk 的前景概率和，不统计历史帧。
- `mean_fg_score_current`：当前帧 chunk 的平均前景概率。
- `valid_history_ratio`：`valid_mask.float().mean(dim=1)`。
- `current_delta_t / time_scale`：最近历史帧到当前帧的真实时间间隔归一化。

来自 `seg_logits` 的统计方式：

```python
current_logits = seg_logits[:, :, -chunk_size:]
fg_prob = torch.softmax(current_logits, dim=1)[:, 1, :]
soft_fg_count_current = fg_prob.sum(dim=1)
mean_fg_score_current = fg_prob.mean(dim=1)
```

注意：`point_sample_size` 是 regularize 后的固定点数，不能直接当作 `num_points_in_search`，否则 sparse gate 会失去输入信号。

当前代码状态：

- [x] 训练侧 `motion_processing_mf()` 已在 regularize 前写入 `num_points_in_search`。
- [x] 测试侧 `MotionBaseModelMF.build_input_dict()` 已在 regularize 前写入 `num_points_in_search`。
- [x] `SEQTRACK3D.build_observability_stats()` 已根据 `num_points_in_search`、当前帧 `seg_logits`、`valid_mask` 和 `current_delta_t` 构造 `obs_stats`。
- [x] `check_time_batch.py / check_forward_batch.py / check_twc_batch.py` 已打印 `num_points_in_search`。

### P5-2：新增模块

新增文件：

```text
CT-SeqTrack/models/observability.py
```

推荐结构：

```text
ObservabilityGate
  input:
    point_feature: B,256
    z_dyn: B,dynamics_hidden_dim
    obs_stats: B,5
    dynamics_valid: B,1

  modules:
    dyn_proj: dynamics_hidden_dim -> 256
    stats_norm: log / clamp handled before MLP
    gate_mlp: 5 -> hidden_dim -> hidden_dim -> 2
    softmax -> alpha_obs, alpha_dyn

  output:
    fused_feature = alpha_obs * point_feature + alpha_dyn * dyn_proj(z_dyn)
    alpha: B,2
```

安全策略：

- `gate_mlp` 最后一层 bias 初始化为 `[1.0, 0.0]`，训练初期更偏向 observation，避免一开始过度依赖未训练的 dynamics prior。
- 当 `dynamics_valid == 0` 时强制 `alpha_dyn=0, alpha_obs=1`，避免 padding 历史污染 gate。
- `alpha` 必须记录到 output：`obs_alpha / obs_stats / obs_gate_entropy`。

当前代码状态：

- [x] 新增 `CT-SeqTrack/models/observability.py`。
- [x] `ObservabilityGate` 已实现 `z_dyn -> 256` 投影、二路 softmax gate、`dynamics_valid=0` 保护和 entropy 输出。

### P5-3：模型接入点

接入位置在 `SEQTRACK3D.forward()` 的 coarse motion prediction：

```python
point_feature = self.mini_pointnet(mask_points)
z_dyn, velocity_pred, dynamics_valid = self.dynamics_encoder(...)
obs_stats = build_observability_stats(...)
motion_feature, obs_aux = self.observability_gate(
    point_feature, z_dyn, obs_stats, dynamics_valid)
motion_pred = self.motion_mlp(motion_feature)
```

第一版不要新增 `motion_mlp_gate`，而是让 gate 输出保持 `256` 维，复用当前 `motion_mlp`。这样 P5 的行为更像“选择 observation/dynamics 信息源”，不是换一个新的 motion head。

当前代码状态：

- [x] `SEQTRACK3D` 已接入 `use_observability_gate`。
- [x] `use_observability_gate=True` 且 `use_dynamics_encoder=False` 时会直接报错。
- [x] P5 打开时复用原始 `motion_mlp`；P3 dynamics-only 仍保持原来的 concat 路径。

### P5-4：配置

建议加入 nuScenes / Waymo 默认配置，但保持关闭：

```yaml
use_observability_gate: False
obs_gate_hidden_dim: 64
obs_gate_num_stats: 5
obs_gate_entropy_weight: 0.0
obs_gate_init_obs_bias: 1.0
obs_gate_min_dyn_valid: 0.5
obs_gate_log_stats: True
```

建议另存实验配置：

```text
cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml
```

其中：

```yaml
use_dynamics_encoder: True
use_observability_gate: True
use_twc: False
```

P5 第一轮不要和 TWC 同时打开；先确认 gate 独立有效，再做 `P3 + P4 + P5` 组合实验。

当前代码状态：

- [x] nuScenes / Waymo 默认配置已加入 P5 开关并保持关闭。
- [x] 新增 `cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml`，默认 `use_dynamics_encoder=True`、`use_observability_gate=True`、`use_twc=False`。

### P5-5：loss 与日志

第一版不加额外监督，只用主 tracking loss 反向学习 gate。

可选正则先保留关闭：

```text
gate_entropy = -sum(alpha * log(alpha))
obs_gate_entropy_weight = 0.0
```

必须写入日志：

```text
obs_alpha_obs_mean
obs_alpha_dyn_mean
obs_alpha_dyn_min / max
obs_gate_entropy
obs_num_points_search_mean
obs_soft_fg_count_mean
obs_mean_fg_score
obs_valid_history_ratio
```

判断 gate 是否学到东西时，不只看整体均值，还要看：

- sparse bin 中 `alpha_dyn` 是否更高。
- 大 `current_delta_t` bin 中 `alpha_dyn` 是否更高。
- 当前前景置信度高时 `alpha_obs` 是否更高。

当前代码状态：

- [x] `compute_loss()` 已记录 `obs_alpha_* / obs_gate_entropy / obs_*_mean`。
- [x] `obs_gate_entropy_weight` 已接入，默认 `0.0`，不影响当前 loss。
- [x] `check_forward_batch.py` 和 `check_train_steps.py` 已支持 `--obs-gate` 临时打开 P5。

### P5-6：工程验收

- [x] `python -m compileall` 通过。
- [x] 纯张量 smoke test：`ObservabilityGate` 输出 finite，`alpha.sum(dim=1)==1`。
- [x] `dynamics_valid=0` 时 `alpha_dyn=0`。
- [x] 服务器普通 forward smoke：输出 `obs_alpha / obs_gate_entropy`，所有 tensor finite。
- [x] 服务器 GPU loss smoke：`loss_total` finite。
- [x] 服务器 2-step train smoke：backward / optimizer step / JSONL 写出正常。

已取消任务：不再单独执行“P5 默认关闭时，P0-P4 默认路径不受影响”的回归检查；当前以后续困难子集评估和正式消融为主。

服务器 P5 smoke test 记录：

```text
check_forward_batch.py --obs-gate:
using batch_idx=12
num_points_in_search=3.0
valid_mask=[1, 1, 1]
velocity_pred / dynamics_valid / obs_alpha / obs_gate_entropy / obs_stats: finite=True
loss_total=4.333134, finite=True
loss_velocity=0.001596, finite=True
obs_num_points_search_mean=3.000000
obs_soft_fg_count_mean=497.636841
obs_mean_fg_score=0.485973
obs_valid_history_ratio=1.000000
obs_current_delta_t_ratio=0.998610
obs_alpha_obs_mean=0.731059
obs_alpha_dyn_mean=0.268941
obs_gate_entropy=0.582203

check_train_steps.py --obs-gate --max-steps 2:
use_observability_gate=True
step=1/2 batch_idx=0 loss_total=14.486424 grad_norm=1.000000
step=2/2 batch_idx=1 loss_total=5.297028 grad_norm=1.000000
finished train-step check
loss log: output/p5_obs_gate_loss.jsonl
last checkpoint: output/p5_obs_gate_ckpt/last.pt
```

判断：P5 工程链路已通过。当前 `alpha_obs≈0.731 / alpha_dyn≈0.269` 来自 `obs_gate_init_obs_bias=1.0` 的初始偏置，只说明 gate 初始化和数值链路正常，不代表已经学到可靠的稀疏/遮挡自适应策略。

服务器命令，注意不要输入 `...`：

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

### P5-7：实验顺序

先做小规模判断，不要直接跑完整大表：

```text
1. P3 dynamics only
2. P3 + P5 Observability Gate
3. P3 + P4 TWC
4. P3 + P4 + P5
```

重点子集：

- sparse bins：按当前目标点数或当前搜索区域真实点数分桶。
- large-gap bins：`current_delta_t` 分桶。
- re-appearance：连续低点数后恢复的片段。
- variable-gap：`skip=1/2/3/5`。

预期现象：

- 主表不能明显退化。
- sparse / re-appearance 子集优先提升。
- `alpha_dyn` 在 sparse、大 gap、低前景置信度样本中上升。
- `alpha_obs` 在当前观测清晰、前景概率高的样本中上升。

### P5-8：风险与止损

- 如果 `alpha` 长期塌缩到 observation：先检查 `z_dyn` 是否有梯度和 `velocity_pred` 是否合理，再考虑增大 `velocity_weight` 或打开轻量 entropy 正则。
- 如果 `alpha` 长期塌缩到 dynamics：调高 observation 初始 bias，或在前几个 epoch 冻结 gate / 降低 dynamics 分支学习率。
- 如果主表退化但 sparse 提升：保留为困难场景增强模块，降低默认 gate 强度或只在低观测可靠性样本启用。
- 如果 P5 没有收益：不要继续堆复杂 gate，优先转向困难子集评估和 TWC 方差分析。

---

## 4. 下一轮修改建议与优先级

下面这些是当前 P0-P5 工程链路跑通之后，进入正式消融和论文化前最值得补的修改。优先顺序按“论文收益 / 工程风险 / 消融清晰度”排序。

### M1：收窄论文 claim

- [ ] 暂缓：先跑出可支撑的实验现象，再正式修改方法命名、标题和主 claim。
- [ ] 将主 claim 从泛泛的 `continuous-time 3D tracking` 收窄为 `timestamp-native / variable-rate 3D SOT`。
- [ ] 在论文和文档中明确当前方法不是完整 ODE/SDE/CDE tracker，也不支持任意时刻 `state(t*)` 查询。
- [ ] 推荐表述：

```text
CT-SeqTrack converts SeqTrack3D from fixed-step frame sequence learning
to timestamp-native variable-rate state estimation.
```

- [ ] Related work 中主动区分：
  - StreamTrack：continuous stream / memory bank，不是显式真实 `delta_t` 建模。
  - HVTrack / MambaTrack3D：high temporal variation / SSM-Mamba，不是当前主线。
  - TrajTrack：trajectory prior，不是多帧点云 + 真实时间联合建模。
  - ChronoTrack：long-term memory + temporal consistency；CT-SeqTrack 的 TWC 必须限定为不同采样路径到同一绝对时刻的一致性。

### M2：移除 Transformer 中固定 4 帧假设

- [ ] 暂不修改代码，先理解固定 4 帧设计的优劣。

固定 4 帧的优点：

- 与当前 `hist_num=3` 完全匹配，风险最低，和原始 SeqTrack3D 实现最一致。
- Transformer 内部 shape 固定，训练和调试简单，不容易引入动态 reshape bug。
- 对 mini / smoke / 第一轮 baseline 对比更稳定，避免把时间建模收益和历史长度变化混在一起。

固定 4 帧的缺点：

- 无法直接做 `hist_num=2/4/6` 历史长度消融，限制论文实验完整性。
- TWC 的采样路径虽然能改成 `[1,3,5]`，但历史 token 数仍固定为 3 个，无法评估更长历史路径。
- 如果后续要做更长 gap 或更长 memory，`models/attn/Models.py` 里的 `4 * 128`、`view(..., 4, ...)` 会成为硬限制。

- [ ] 检查 `models/attn/Models.py` 中 `4 * 128`、`view(..., 4, ...)`、`reshape(-1, 4 * 128, ...)` 等硬编码。
- [ ] 将 `L = hist_num + 1` 从 `valid_mask.shape[1] + 1` 动态推断，并传入 `Seq2SeqFormer.forward()`。
- [ ] 验收：
  - `hist_num=3` 与当前行为一致。
  - `hist_num=2 / 4 / 6` forward finite。
  - 后续历史长度消融不再受硬编码限制。

### M3：让 dynamics prior 显式条件于当前查询间隔

- [x] 当前 `DynamicsEncoder` 已从只编码历史帧间差分，升级为同时接收 `current_delta_t` 作为当前查询间隔。
- [x] 已新增 `dynamics_displacement_pred = velocity_pred * current_delta_t`，用于记录 dynamics prior 对当前帧位移的显式估计。
- [x] 已新增可选配置：

```yaml
dynamics_use_query_gap: True
dynamics_motion_mode: feature # feature | residual
dynamics_displacement_weight: 0.0
```

- [x] 当前默认仍保持 `feature` 模式，即只让 query-conditioned `z_dyn` 参与 motion feature，不强行把速度积分结果加到最终框，降低退化风险。
- [ ] 后续正式消融可打开 `dynamics_motion_mode: residual`，比较显式位移 prior 是否优于纯 feature 融合。
- [x] 当前实现语义：

```text
z_dyn = DynamicsEncoder(ref_boxs, delta_t, valid_mask, current_delta_t)
velocity_pred = velocity_head(z_dyn)
displacement_pred = velocity_pred * current_delta_t
```

- [ ] 对比两种 coarse motion 接法：
  - `motion_pred = motion_mlp([point_feature, z_dyn])`
  - `motion_pred = residual_mlp([point_feature, z_dyn]) + displacement_pred`
- [ ] 验收重点：
  - 大 `current_delta_t` / `skip=3,5` 子集不能退化。
  - `velocity_pred` 量级合理。
  - 急转弯场景不被速度积分硬约束伤害。

### M4：修正 P5 sparse 统计口径

- [x] 已修正：`soft_fg_count_current = fg_prob.sum()` 仍保留为日志参考，但不再作为 gate 第二个统计输入。
- [x] 已新增更稳的统计量：

```text
estimated_fg_points = mean_fg_score_current * num_points_in_search
```

- [x] P5 gate 输入已改为：

```text
o_t = [
  log1p(num_points_in_search),
  log1p(estimated_fg_points),
  mean_fg_score_current,
  valid_history_ratio,
  current_delta_t / time_scale
]
```

- [ ] 验收重点：
  - sparse bin 中 `alpha_dyn` 是否更高。
  - dense / high-confidence bin 中 `alpha_obs` 是否更高。
  - `num_points_in_search` 必须保持 regularize 前真实搜索区域点数口径。

### M5：优先补评估协议，而不是继续堆模块

- [ ] 暂缓：当前先完成 M3 / M4 代码修正，再跑 baseline 与 CT-SeqTrack 的第一轮 mini 对比。
- [ ] 先做小规模正式消融，不直接跑完整大表：

```text
SeqTrack3D
+ real timestamp field
+ time encoding
+ dynamics / velocity branch
+ TWC
+ observability gate
```

- [ ] 时间编码消融：

```text
fixed pseudo time
raw real time
MLP time encoding
Fourier time encoding
```

- [ ] 困难子集必须补：
  - `variable-gap`: `skip=1/2/3/5`
  - `delta_t bins`: `[0,0.2), [0.2,0.5), [0.5,1.0), [1.0,+inf)`
  - `sparse bins`: `[0,5), [5,10), [10,20), [20,50), [50,+inf)`
  - `re-appearance`: 连续低点数后恢复，统计恢复后 K 帧内是否重新跟上。
- [ ] TWC 需要额外报告不同采样路径下的预测方差：

```text
variance_center = ||c(view_a) - c(view_b)||
variance_angle = angle_gap(view_a, view_b)
```

- [ ] P5 需要额外报告 gate 分桶行为：

```text
alpha_dyn by sparse bin
alpha_dyn by delta_t bin
alpha_obs by fg confidence bin
```

---

## 5. 当前执行原则

- 不再补 P1/P2 的额外 smoke test，除非改动影响时间主链路。
- 不切换到 MambaTrack3D / TrackM3D / TrajTrack 作为主 baseline。
- 不上 Neural ODE / SDE / CDE。
- 不主打任意时间查询或多传感器异步融合。
- 每个模块都必须能独立消融，避免把真实时间收益和模型容量变大混在一起。

---

## 6. 第一轮 nuScenes-mini 差距诊断

详细诊断已统一整合到 `gap_analysis.md`。后续关于第一轮 **SeqTrack baseline vs CT-SeqTrack P5 full** 的结果差距、代码原因、消融矩阵和论文表述，以 `gap_analysis.md` 为唯一来源，避免 `need_to_do.md` 与分析文档重复维护。

当前执行摘要：

- [ ] 不再直接把当前 P5 full 作为论文主结果；先定位不稳定来源。
- [ ] 先用 best checkpoint 复测 CT，不只看 last/final。
- [ ] 拉取 TensorBoard 标量：`obs_alpha_dyn_mean`、`obs_alpha_obs_mean`、`loss_velocity`、`loss_dynamics_displacement`、`obs_estimated_fg_points_mean`。
- [ ] 优先跑最小消融：

```text
A0: SeqTrack baseline
A1: CT-base = real timestamp / delta_t / delta_T / TimeEncoding, dynamics=False, gate=False
A2: CT + Dynamics, gate=False
A2-lite: CT + Dynamics, num_candidates=1
A3-safe: CT + Dynamics + Gate, obs_gate_init_obs_bias=3.0 or 4.0
A3-res: CT + Dynamics + residual Gate
```

- [ ] 两个最高优先级代码怀疑点：
  - Dynamics 分支训练/测试分布不匹配：训练中 candidate offset 污染 `ref_boxs` 差分，测试中变成递归预测分布。
  - P5 Gate 融合过强：当前是 feature replacement，建议优先试 residual fusion 或更高 observation bias。

论文层面暂定表述：

```text
The timestamp-native base path is evaluated first to verify no regression.
The current full P5 configuration is unstable on nuScenes-mini, likely because
dynamics features are weakly constrained and fused too aggressively.
```
