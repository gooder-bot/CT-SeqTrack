# CT-SeqTrack 差距诊断与消融计划

更新时间：2026-05-30

本文整合了 `need_to_do.md` 末尾的第一轮 nuScenes-mini 诊断和原 `gap_analysis.md`。后续关于 **SeqTrack baseline vs CT-SeqTrack P5 full** 的性能差距、代码原因、止损策略和最小消融计划，统一以本文为准；`need_to_do.md` 只保留执行摘要。

---

## 1. 当前实验现象

第一轮对比设置：

```text
Dataset: nuScenes-mini
Category: Car
Training: 60 epochs
Comparison:
  A0: SeqTrack baseline
  A3-old: CT-SeqTrack P5 full
```

结果：

| model | metric | final | best |
| --- | --- | ---: | ---: |
| SeqTrack baseline | success/test | 50.9858 | 52.2834 |
| SeqTrack baseline | precision/test | 59.9617 | 65.2144 |
| CT-SeqTrack P5 full | success/test | 31.1937 | 44.9836 |
| CT-SeqTrack P5 full | precision/test | 31.8851 | 62.5120 |

按 12 个验证点统计：

```text
success/test:
  SeqTrack mean=47.9679, std=3.4748
  CT       mean=33.3975, std=7.9882

precision/test:
  SeqTrack mean=55.9259, std=6.3958
  CT       mean=36.0892, std=12.5834
```

结论：

- CT-SeqTrack P5 full 的 final 明显低于 baseline，当前不能作为主结果。
- CT 在早期验证点的 `precision/test=62.5120` 接近 baseline best `65.2144`，说明模型不是完全学不到定位。
- 后续多次断崖式下降，且方差显著高于 baseline，更像是新增 P3/P5 模块造成的训练后期不稳定或递归跟踪 drift，而不是单纯欠拟合。

---

## 2. 总体判断

nuScenes-mini 的小数据量、稀疏点云和 empty point cloud 会放大波动，但不是主因。理由是：

- 两个模型使用同一数据集、同一 split、同一 60 epoch 设置。
- baseline 虽然也波动，但整体保持在较稳定区间。
- CT 的 precision/std 约为 baseline 的两倍，且退化集中出现在新增 dynamics/gate 之后。

因此当前优先假设是：

```text
P0-P2 timestamp-native 主链路可能是安全的；
P3 Dynamics 和 P5 Observability Gate 中至少一个模块引入了不稳定。
```

下一步必须先做最小消融，不能继续直接用 P5 full 和 baseline 对比。

---

## 3. 两个核心代码差距

### 3.1 差距一：Dynamics 分支存在训练-测试分布不匹配

相关代码：

- `datasets/sampler.py`：训练时 `candidate_id != 0` 会给历史 `ref_boxs` 加随机 offset。
- `models/dynamics.py`：`DynamicsEncoder` 直接从历史 `ref_boxs` 差分构造 velocity / angular velocity。
- `cfgs/seqtrack3d_nuscenes.yaml`：默认 `num_candidates: 4`。

关键代码语义：

```python
newer = ref_boxs[:, :-1, :]
older = ref_boxs[:, 1:, :]
displacement = newer[:, :, :3] - older[:, :, :3]
velocity = displacement / gap.unsqueeze(-1)
```

训练时：

- `num_candidates=4`。
- 非 0 candidate 会对历史 `ref_boxs` 添加随机 offset。
- `z_dyn` 学到的是带噪声的历史框差分分布。
- `velocity_label` 却来自 GT motion / `current_delta_t`，和 noisy ref boxes 的统计口径不完全一致。

测试时：

- `ref_boxs` 来自模型递归预测。
- 没有训练时那种随机 offset。
- 历史框误差来自 tracker 自身 drift，而不是均匀扰动。

后果：

- `z_dyn` 在测试时容易变成 OOD feature。
- 这些 OOD dynamics feature 进入 P5 gate 后，会在递归跟踪中逐帧放大 drift。
- 这能解释 CT early checkpoint 还不错，但后期/final 严重不稳定。

优先验证：

- A2 dynamics-only 是否已经退化。
- `num_candidates=1` 的 dynamics-only 是否明显更稳。
- candidate 0 与 candidate 非 0 的 `loss_velocity` 是否存在明显差异。
- `dynamics_displacement_pred` 是否能在打开小权重监督后贴近 `motion_label[:, 0, :3]`。

### 3.2 差距二：Observability Gate 使用 feature replacement，融合过强

相关代码：

- `models/observability.py`
- `models/seqtrack3d.py`
- `cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml`

当前融合语义：

```python
fused_feature = alpha[:, 0:1] * point_feature + alpha[:, 1:2] * z_dyn_proj
```

问题：

- 这是 feature replacement，不是 residual correction。
- `obs_gate_init_obs_bias=1.0` 时，valid dynamics 样本初始约为：

```text
alpha_obs = 0.731
alpha_dyn = 0.269
```

- `z_dyn_proj` 是随机初始化的 Linear。训练初期等价于用约 27% 的随机 dynamics projection 替换已经能工作的 point feature。
- gate 没有显式监督，无法保证 dense / reliable observation 时一定偏向 observation。
- 如果差距一中的 `z_dyn` 已经 OOD，feature replacement 会把 OOD feature 直接注入 coarse motion prediction。

更保守的融合方式：

```python
fused_feature = point_feature + alpha_dyn * dyn_residual_proj(z_dyn)
```

这样 dynamics 分支只提供残差修正，而不是替换主观测特征。

优先验证：

- A2 正常但 A3 退化时，基本可确认 P5 融合策略是主因。
- 提高 `obs_gate_init_obs_bias` 到 3.0 或 4.0 后是否缓解。
- 限制 `max_dyn_alpha` 到 0.1-0.2 后是否缓解。
- 前 5-10 epoch freeze gate 或降低 dynamics/gate 学习率是否缓解。

---

## 4. 次要但需要记录的因素

### 4.1 Dynamics 约束偏弱

当前配置：

```yaml
velocity_weight: 0.05
dynamics_displacement_weight: 0.0
```

含义：

- `velocity_pred` 有很弱的辅助监督。
- `dynamics_displacement_pred` 只记录，不参与优化。
- `z_dyn` 主要靠最终 tracking loss 间接学习。

风险：

- mini 数据量小，128 维 dynamics feature 容易学到不稳定模式。
- dynamics feature 被 P5 gate 使用时，缺少足够强的可解释约束。

建议：

- 在 dynamics sanity ablation 中打开 `dynamics_displacement_weight=0.01`。
- 记录 `loss_velocity` 和 `loss_dynamics_displacement` 的训练曲线。

### 4.2 P5 full 在 mini 上自由度偏大

新增模块约包含：

- `gate_mlp`：5 -> 64 -> 64 -> 2，约 4.5K 参数。
- `dyn_proj`：128 -> 256，约 33K 参数。
- `DynamicsEncoder`：per-step MLP + global MLP + velocity head，约 30K 参数。

这些参数主要依赖最终 tracking loss 学习。nuScenes-mini 的有效监督量偏小，所以容易出现 early checkpoint 尚可、后期过拟合或 drift 的现象。

### 4.3 缺少 CT-base 对照

当前对比是：

```text
SeqTrack baseline vs CT-SeqTrack P5 full
```

中间缺少关键消融点：

- P0-P2：真实时间字段和 TimeEncoding 是否安全。
- P3：Dynamics 分支是否单独退化。
- P5：Observability Gate 是否单独引入不稳定。

这也是目前最需要补的实验。

---

## 5. 最小消融矩阵

| 编号 | 配置 | dynamics | gate | 目的 |
| --- | --- | --- | --- | --- |
| A0 | SeqTrack baseline | OFF | OFF | 已有基线，确认复现实验区间 |
| A1 | CT-base：real timestamp / delta_t / delta_T / TimeEncoding | OFF | OFF | 验证 P0-P2 不退化 |
| A2 | CT + Dynamics | ON | OFF | 隔离 P3 Dynamics 的影响 |
| A2-lite | CT + Dynamics, `num_candidates=1` | ON | OFF | 排除 candidate offset 对 dynamics 的干扰 |
| A2-disp | CT + Dynamics, `dynamics_displacement_weight=0.01` | ON | OFF | 检查显式 displacement 监督是否稳定 dynamics |
| A3-safe | CT + Dynamics + Gate, obs bias=3/4 | ON | ON | 测试更保守 gate 初始化 |
| A3-res | CT + Dynamics + residual Gate | ON | ON | 测试 residual fusion 是否优于 replacement |
| A4 | Gate warmup / freeze 5-10 epoch | ON | ON | 测试训练日程是否缓解后期崩坏 |

优先顺序：

```text
A1 -> A2 -> A2-lite -> A3-safe -> A3-res
```

P4 TWC 暂时不要和 P5 混在一起。建议在 P3/P5 稳定后，再单独跑：

```text
CT-base + TWC
CT + Dynamics + TWC
```

---

## 6. 近期执行清单

1. 不再使用当前 P5 full 作为论文主结果。
2. 用 best checkpoint 复测 CT，避免只用 last/final 判断。
3. 拉取并检查 TensorBoard 标量：

```text
obs_alpha_dyn_mean
obs_alpha_obs_mean
obs_alpha_dyn_min / max
loss_velocity
loss_dynamics_displacement
obs_estimated_fg_points_mean
obs_num_points_search_mean
```

4. 先跑 A1，判断 P0-P2 是否安全。
5. 再跑 A2，判断 dynamics-only 是否退化。
6. 如果 A2 退化，优先检查 `num_candidates=1` 和 displacement 监督。
7. 如果 A2 正常但 A3 退化，优先改 P5 为 residual / higher obs bias / max dyn alpha。
8. 后续正式报告必须补困难子集：

```text
variable-gap: skip=1/2/3/5
delta_t bins: [0,0.2), [0.2,0.5), [0.5,1.0), [1.0,+inf)
sparse bins: [0,5), [5,10), [10,20), [20,50), [50,+inf)
re-appearance: 连续低点数后恢复
```

---

## 7. 服务器命令参考

### A1：CT-base

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --batch_size 2 \
  --epoch 60 \
  --seed 42 \
  --tag "ct_base_mini_car_60ep"
```

注意：`seqtrack3d_nuscenes.yaml` 当前默认 `use_dynamics_encoder=False`、`use_observability_gate=False`、`use_twc=False`，适合作为 CT-base。

### A2：CT + Dynamics only

建议复制一份配置：

```text
cfgs/seqtrack3d_nuscenes_a2_dyn.yaml
```

关键配置：

```yaml
use_dynamics_encoder: True
use_observability_gate: False
use_twc: False
```

训练命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --batch_size 2 \
  --epoch 60 \
  --seed 42 \
  --tag "ct_dyn_mini_car_60ep"
```

### A3-safe：更保守的 Gate

建议复制一份配置：

```text
cfgs/seqtrack3d_nuscenes_a3_gate_safe.yaml
```

关键配置：

```yaml
use_dynamics_encoder: True
use_observability_gate: True
use_twc: False
obs_gate_init_obs_bias: 3.0
```

如果仍不稳定，再尝试 residual fusion 或 `max_dyn_alpha`，不要继续堆更复杂的 gate。

---

## 8. 论文层面的处理

在 A1/A2/A3 消融完成前，不要把当前 P5 full 写成正向结果。推荐表述：

```text
The timestamp-native base path is evaluated first to verify no regression.
The dynamics branch and the observability-aware gate are then evaluated
incrementally to isolate their individual effects. The current full P5
configuration is unstable on nuScenes-mini, likely because dynamics features
are weakly constrained and fused too aggressively at the feature level.
```

当前论文叙事应保持克制：

- 可以继续主打 timestamp-native / variable-rate 3D SOT。
- 可以保留 P3/P5 作为方法模块，但必须通过消融证明它们不是退化来源。
- 如果 P5 最终只在 sparse / large-gap 子集提升，而主表不提升，可将其定位为困难场景增强模块，而不是默认全量模块。

---

## 9. 当前结论

当前项目不是 idea 不成立，而是 full configuration 过早合并了多个新增模块：

```text
real timestamp + dynamics prior + observability gate
```

其中最可疑的两个代码差距是：

1. Dynamics 分支训练/测试分布不匹配。
2. Gate 用 feature replacement 过强地融合未充分约束的 dynamics feature。

下一步的胜负手不是继续加模块，而是把 A1/A2/A3 消融跑干净，先证明 timestamp-native 主链路不退化，再逐步恢复 dynamics、TWC 和 gate。
