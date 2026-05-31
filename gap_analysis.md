# CT-SeqTrack 差距诊断与下一步修改计划

更新时间：2026-05-30

本文统一记录当前 **SeqTrack baseline vs CT-SeqTrack P5 full** 的退步原因、代码层诊断、下一步实验和需要修改的文件。后续关于这轮 nuScenes-mini 对比的解释，以本文为准；`need_to_do.md` 只保留执行摘要。

---

## 0. 先给结论

当前第一轮结果不能说明 CT-SeqTrack 的 timestamp-native 思路失败。它更准确地说明：

```text
当前 P5 full 配置过早合并了 real timestamp + dynamics prior + observability gate，
其中 P3 dynamics 和 P5 gate 至少有一个引入了不稳定。
```

这次退步不应直接归因给真实时间主链路，因为当前已跑的 CT 配置不是最小 CT-base，而是：

```yaml
use_real_time: True
time_encoding: raw
use_dynamics_encoder: True
use_observability_gate: True
use_twc: False
num_candidates: 4
dynamics_motion_mode: feature
velocity_weight: 0.05
dynamics_displacement_weight: 0.0
obs_gate_init_obs_bias: 1.0
obs_gate_entropy_weight: 0.0
```

因此当前最重要的工作不是继续加模块，而是先把 A1/A2/A3 消融跑干净：

```text
A1: CT-base, dynamics=False, gate=False
A2: CT + Dynamics, gate=False
A2-lite: CT + Dynamics, num_candidates=1
A2-disp: CT + Dynamics, dynamics_displacement_weight=0.01
A3-safe: CT + Dynamics + Gate, higher observation bias
A3-res: CT + Dynamics + residual gate
```

---

## 1. 当前实验现象

第一轮对比设置：

```text
Dataset: nuScenes-mini
Category: Car
Training: 60 epochs
Validation interval: every 5 epochs

A0:
  SeqTrack baseline
  path: /home/lishengjie/study/lcyu/seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16

A3-old:
  CT-SeqTrack P5 full
  path: /home/lishengjie/study/lcyu/CT-SeqTrack/output/20260528-1718-seqtrack3d_nuscenes_mini_p5_obs_gate-ct_seqtrack_mini_p5_car_60ep_bs16_gpu3
```

主结果：

| model | metric | final | best |
| --- | --- | ---: | ---: |
| SeqTrack baseline | success/test | 50.9858 | 52.2834 |
| SeqTrack baseline | precision/test | 59.9617 | 65.2144 |
| CT-SeqTrack P5 full | success/test | 31.1937 | 44.9836 |
| CT-SeqTrack P5 full | precision/test | 31.8851 | 62.5120 |

退步点数：

| metric | best delta | final delta |
| --- | ---: | ---: |
| success/test | -7.2998 | -19.7921 |
| precision/test | -2.7024 | -28.0766 |

按 12 个验证点统计：

```text
success/test:
  SeqTrack mean=47.9679, std=3.4748
  CT       mean=33.3975, std=7.9882

precision/test:
  SeqTrack mean=55.9259, std=6.3958
  CT       mean=36.0892, std=12.5834
```

关键现象：

1. CT 的 best precision 为 `62.5120`，接近 baseline best `65.2144`，说明模型不是完全学不到定位。
2. CT 的 final precision 掉到 `31.8851`，相对 baseline final 低 `28.0766` 点，说明后期递归跟踪稳定性很差。
3. CT 的方差明显大于 baseline，precision/std 约为 baseline 的两倍。这更像新增模块导致的 drift 或训练后期不稳定，而不是单纯数据集波动。
4. 本轮 CT 配置 `use_twc=False`，所以这次退步不能归因给 P4 TWC。

---

## 2. 如何理解 success 和 precision 的退步

`precision/test` 主要反映中心距离精度。CT early checkpoint precision 很高，但 final 大幅下降，说明 coarse motion 或 refine 后的中心定位在递归测试中容易漂移。

`success/test` 主要反映 3D IoU。它受中心、朝向和尺寸局部化共同影响。CT success best 已经比 baseline best 低 7.3 点，说明即使在较好 checkpoint，box 重叠质量也没有完全追上 baseline。

更重要的是 best 与 final 的差距：

```text
SeqTrack baseline:
  success best -> final: 52.2834 -> 50.9858, 下降 1.2976
  precision best -> final: 65.2144 -> 59.9617, 下降 5.2527

CT P5 full:
  success best -> final: 44.9836 -> 31.1937, 下降 13.7899
  precision best -> final: 62.5120 -> 31.8851, 下降 30.6269
```

这说明 CT 的主要问题不是一开始没有能力，而是训练过程和测试递归分布不稳定。3D SOT 的测试不是独立样本分类，而是逐帧递归：前一帧预测框会成为后一帧搜索区域和历史输入。一旦 dynamics 或 gate 给出错误先验，错误会被下一帧继续使用，最终体现为 precision 和 success 同时掉点。

---

## 3. 主要原因一：当前比较缺少 CT-base，无法证明真实时间主链路有问题

当前对比是：

```text
SeqTrack baseline vs CT-SeqTrack P5 full
```

但 P5 full 同时打开了：

```text
real timestamp
TimeEncoding(raw)
DynamicsEncoder
ObservabilityGate
```

中间缺少三个关键层级：

```text
P0-P2: real timestamp + delta_t + delta_T + TimeEncoding
P3: Dynamics only
P5: Dynamics + Observability Gate
```

因此当前不能写成：

```text
real timestamp 让模型退步
```

只能写成：

```text
full P5 configuration 退步，退步来源需要通过 A1/A2/A3 消融定位。
```

这是论文层面最重要的止损：主叙事继续保留 timestamp-native / variable-rate 3D SOT，但必须先用 A1 证明真实时间主链路不退化。

---

## 4. 主要原因二：A1-raw 可能改变了原 SeqTrack3D 的时间输入分布

当前 A1 的定义是：

```yaml
use_real_time: True
time_encoding: raw
use_dynamics_encoder: False
use_observability_gate: False
use_twc: False
```

也就是说，A1 只打开真实时间主链路，不打开 P3/P5/P4。它的退步如果成立，优先怀疑的是 **raw real time 的输入分布**，而不是 dynamics 或 gate。

### 4.1 点云时间通道尺度改变

模型输入点云不是只有 `xyz`，还有时间和前景先验通道，形式类似：

```text
[x, y, z, time, mask]
```

伪时间下，历史点云时间大致为：

```text
[-0.1, -0.2, -0.3, 0.0]
```

真实 nuScenes keyframe 约 2Hz，真实时间下变成：

```text
[-0.5, -1.0, -1.5, 0.0]
```

这会直接进入 PointNet 第一层。第一层本质上会做：

```text
feature = w_x*x + w_y*y + w_z*z + w_t*time + w_m*mask + bias
```

因此当 `time` 从 `-0.1` 变成 `-0.5` 时，`w_t*time` 这部分贡献会被放大约 5 倍。即使真实时间语义更合理，raw 数值尺度也可能破坏 PointNet 第一层、BatchNorm 和后续特征分布。

结论：

```text
A1-raw 不是“无损地加入真实时间”，而是显著改变了点云输入分布。
```

### 4.2 Box corner timestamp 分布改变

SeqTrack3D 的 Transformer 不只看点云，也看历史 box corner token。每个历史框会被转成 corner token：

```text
[x_corner, y_corner, z_corner, time]
```

伪时间下，box corner timestamp 大致为：

```text
[-0.1, -0.2, -0.3, 0.0]
```

真实时间下变成：

```text
[-0.5, -1.0, -1.5, 0.0]
```

这会改变 Transformer 中 query / key / value 的输入分布。原始 SeqTrack3D 中，`-0.1/-0.2/-0.3` 更像离散历史顺序编码；A1-raw 中，`-0.5/-1.0/-1.5` 会被当成更大幅度的连续标量输入。Transformer 可能不再按原来的方式理解历史框顺序和远近。

结论：

```text
A1-raw 同时改变了 point feature 和 box token 两条时间路径。
```

### 4.3 raw TimeEncoding 不做归一化

当前 `models/time_encoding.py` 中 raw 模式等价于：

```python
return time_values
```

也就是说：

```text
伪时间: -0.1, -0.2, -0.3
真实时间: -0.5, -1.0, -1.5
```

会原封不动进入网络。`raw` 不做 log normalize、不做 clamp、不做可学习映射。它适合用来验证“真实秒数直接输入”的最小路径，但不一定是最稳的时间表达。

相比之下：

```text
time_encoding=mlp:
  先做 signed-log normalize，再用小 MLP 输出 1 维时间通道。

time_encoding=fourier:
  先做 signed-log normalize，再用 sin/cos 多频率特征编码，最后投影回 1 维时间通道。
```

注意：当前 `mlp` 和 `fourier` 都是 scalar-preserving，最后仍输出 1 维时间通道，不直接增加 PointNet/Transformer 输入维度。因此如果它们比 raw 好，更容易归因于“时间表达更合适”，而不是简单模型容量变大。

### 4.4 A1-pseudo / A1-MLP / A1-Fourier 的诊断意义

下一步应补三个 A1 变体：

```text
A1-raw:
  use_real_time=True
  time_encoding=raw

A1-pseudo:
  use_real_time=False
  time_encoding=raw

A1-MLP:
  use_real_time=True
  time_encoding=mlp

A1-Fourier:
  use_real_time=True
  time_encoding=fourier
```

判断逻辑：

| 现象 | 解释 | 下一步 |
| --- | --- | --- |
| A1-pseudo 接近 SeqTrack baseline，A1-raw 很低 | CT 代码主链路没坏，raw real time 尺度是主要问题 | 不再用 raw 作为主配置，优先看 MLP/Fourier |
| A1-pseudo 也很低 | CT 当前实现和原 SeqTrack baseline 还有其他差异 | 查 point time、corner time、配置和 baseline 代码差异 |
| A1-MLP/Fourier 明显高于 A1-raw | 真实时间方向有价值，但需要编码/归一化 | 将时间编码作为正式消融 |
| A1-MLP/Fourier 仍很低 | 真实时间接入位置或训练策略仍不适配 | 回查时间通道注入方式 |
| A2 Dynamics 仍显著高于所有 A1 变体 | 仅靠时间编码不够，显式 dynamics prior 是当前有效模块 | 保留 P3，继续修 P5 gate |

一句话：

```text
A1-raw 退步可能不是 timestamp-native 思路错，而是 raw 秒数输入破坏了原模型习惯的时间尺度。
```

### 4.5 最新 A1 时间编码结果

已完成 A1-pseudo / A1-MLP / A1-Fourier 三组实验，并和 SeqTrack baseline 对齐到同一张表。完整文件见：

```text
compare_results/a1_time_encoding_comparison.md
compare_results/a1_time_encoding_metrics_summary.csv
compare_results/a1_time_encoding_metrics_points.csv
compare_results/a1_time_encoding_curves.png
```

主结果如下：

| model | final success | best success | final precision | best precision |
| --- | ---: | ---: | ---: | ---: |
| SeqTrack baseline | 50.9858 | 52.2834 | 59.9617 | 65.2144 |
| A1-pseudo | 48.3381 | 49.8917 | 52.2505 | 65.2385 |
| A1-MLP | 27.4387 | 31.5700 | 26.2779 | 32.7757 |
| A1-Fourier | 30.7232 | 31.0646 | 29.8151 | 30.3228 |

补充参考：之前的 A1-raw 结果为：

| model | final success | best success | final precision | best precision |
| --- | ---: | ---: | ---: | ---: |
| A1-raw | 28.2768 | 32.3556 | 27.4289 | 40.3611 |

这轮结果带来的判断变化：

1. **A1-pseudo 基本修复 success**。A1-pseudo final success 为 `48.3381`，只比 baseline final success 低 `2.6477` 点；best success 低 `2.3917` 点。这说明 CT 当前 A1 主链路不是完全坏掉，之前 A1-raw 的大幅退步确实和真实时间数值尺度强相关。
2. **A1-pseudo 仍然存在 precision gap**。A1-pseudo final precision 为 `52.2505`，比 baseline final precision 低 `7.7112` 点；但 best precision 为 `65.2385`，略高于 baseline best `65.2144`。这说明模型有能力在某些 checkpoint 达到 baseline 级别中心定位，但后期稳定性或递归测试鲁棒性仍弱。
3. **A1-MLP 和 A1-Fourier 没有救回 real-time A1**。二者 final success / precision 都接近 A1-raw，远低于 A1-pseudo 和 baseline。当前 scalar-preserving 的 MLP/Fourier 时间编码还不能直接作为有效创新点，需要继续改注入方式、初始化或尺度设计。
4. **当前不能简单写成“加时间编码有效”**。更准确的表述是：伪时间尺度可以显著恢复 A1 表现；真实秒数输入即使经过当前 MLP/Fourier 编码仍然不稳定。问题已经从“raw 是否太大”推进到“真实时间该如何以不破坏原特征分布的方式接入”。

下一步不建议继续堆更多频域变体，而应优先做一个更保守的时间尺度桥接：

```text
A1-scaled:
  use_real_time=True
  time_encoding=scaled_raw 或 normalized_raw
  目标是把真实秒数映射回接近原伪时间的数值范围
  例如 -0.5, -1.0, -1.5 -> -0.1, -0.2, -0.3
```

如果 A1-scaled 接近 A1-pseudo，说明主问题就是输入尺度；如果 A1-scaled 仍低于 A1-pseudo，则需要进一步检查 point time 和 box corner timestamp 的注入位置，而不是继续只换编码函数。

---

## 5. 主要原因三：Dynamics 分支存在训练/测试分布不匹配

### 5.1 相关代码

训练侧在 `datasets/sampler.py` 的 `motion_processing_mf()` 中构造历史 `ref_boxs`。当 `candidate_id != 0` 时，每一个历史框都会被加独立随机 offset：

```python
if candidate_id == 0:
    sample_offsets = np.zeros(3)
else:
    sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
    sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, ...)
```

当前训练配置：

```yaml
num_candidates: 4
```

所以一个 GT 样本会对应 4 个 candidate，其中 3 个 candidate 的历史 `ref_boxs` 带随机扰动。

而 `models/dynamics.py` 的 `DynamicsEncoder` 直接从 `ref_boxs` 差分构造 dynamics feature：

```python
newer = ref_boxs[:, :-1, :]
older = ref_boxs[:, 1:, :]
displacement = newer[:, :, :3] - older[:, :, :3]
velocity = displacement / gap.unsqueeze(-1)
angular_velocity = angle_delta / gap
```

### 5.2 为什么会导致掉点

训练时：

1. `ref_boxs` 是 candidate reference boxes，其中非 0 candidate 含随机 offset。
2. 多个历史框的 offset 是逐帧独立采样的，会污染历史框差分。
3. `z_dyn` 学到的是带随机扰动的历史框差分。
4. 但 `velocity_label` 来自 `motion_label_list[0][:3] / current_delta_t`，本质是干净 GT motion 的监督口径。
5. 所以 dynamics 输入分布和 dynamics 监督目标不是完全一致的。

测试时：

1. `ref_boxs` 来自 tracker 递归预测结果。
2. 没有训练时那种均匀随机 offset。
3. 历史误差来自模型自己的 drift、遮挡、搜索区域偏移。
4. 这和训练时的随机 candidate 噪声不是同一种分布。

后果：

```text
DynamicsEncoder 在训练中看到的是 random-offset history，
测试中看到的是 recursive-drift history。
```

如果 `z_dyn` 只是被轻量监督，测试时很容易变成 OOD feature。这个 OOD feature 一旦进入 coarse motion prediction，就会改变当前框；当前框又会进入下一帧搜索区域，形成逐帧放大的 tracking drift。

这能解释：

- early checkpoint 还有较高 precision，因为模型尚未过度依赖不稳定 dynamics。
- 后期 final 大幅下降，因为 dynamics/gate 可能学到了 mini 数据上的脆弱模式。
- precision 下降比 success 更严重，因为中心漂移会首先拉低中心距离精度。

### 5.3 需要怎么验证

必须先跑：

```text
A2: CT + Dynamics, gate=False
A2-lite: CT + Dynamics, num_candidates=1
A2-disp: CT + Dynamics, dynamics_displacement_weight=0.01
```

判断规则：

| 现象 | 解释 | 下一步 |
| --- | --- | --- |
| A2 已明显退化 | Dynamics 本身有问题 | 先修 P3，不要继续调 P5 |
| A2 退化但 A2-lite 恢复 | candidate offset 污染 dynamics | 优先改 dynamics 训练输入或训练策略 |
| A2 不退化但 A3 退化 | Gate 是主因 | 转向 residual gate / higher obs bias |
| A2-disp 比 A2 稳 | dynamics 需要更强显式位移监督 | 保留小权重 displacement loss |

### 5.4 建议修改

短期先不改代码，先通过配置验证：

```yaml
use_dynamics_encoder: True
use_observability_gate: False
use_twc: False
num_candidates: 1
dynamics_displacement_weight: 0.01
```

如果 A2-lite 明显更稳，再做代码修改：

1. 在 `datasets/sampler.py` 的 `data_dict` 中加入 `candidate_id`，用于日志和按 candidate 分桶分析。
2. 增加 dynamics 诊断脚本或日志，分别统计 candidate 0 和 candidate 非 0 的：

```text
loss_velocity
loss_dynamics_displacement
||ref_boxs[i] - ref_boxs[i+1]||
velocity_pred norm
dynamics_displacement_pred norm
```

3. 给 dynamics 加一个保守训练选项，例如：

```yaml
dynamics_candidate_zero_only: True
```

含义：只在 clean candidate 上启用 dynamics loss 或 dynamics feature，用于判断随机 candidate 是否污染 P3。

4. 更稳的长期方案是让历史框扰动在时间维度上更一致，而不是每个历史帧独立随机 offset。也就是让 candidate noise 模拟整体搜索框偏移，而不是制造不真实的历史速度噪声。

---

## 6. 主要原因四：Observability Gate 当前是 feature replacement，融合过强

### 6.1 相关代码

`models/observability.py` 当前融合方式：

```python
z_dyn_proj = self.dyn_proj(z_dyn)
fused_feature = alpha[:, 0:1] * point_feature + alpha[:, 1:2] * z_dyn_proj
```

这不是 residual correction，而是 feature replacement。

当前配置：

```yaml
obs_gate_init_obs_bias: 1.0
obs_gate_entropy_weight: 0.0
```

初始化时 valid dynamics 样本大约是：

```text
alpha_obs = 0.731
alpha_dyn = 0.269
```

### 6.2 为什么会导致掉点

P5 的意图是：

```text
当前观测可靠时信 point_feature；
当前点云稀疏或 gap 大时信 dynamics prior。
```

但当前实现一开始就会把约 27% 的 `point_feature` 替换成 `z_dyn_proj`。问题在于：

1. `z_dyn_proj` 是随机初始化 Linear。
2. `z_dyn` 来自约束偏弱的 DynamicsEncoder。
3. gate 没有显式监督，不能保证 dense / high-confidence observation 时一定偏向 observation。
4. 如果 P3 dynamics 已经存在 OOD 问题，P5 会把 OOD dynamics feature 直接注入 coarse motion branch。

这会破坏原本已经能工作的 observation feature。尤其在递归测试时，错误先验不是只影响当前帧，而是会改变下一帧搜索区域。

### 6.3 需要怎么验证

如果 A2 不退化，但 A3 退化，基本可以确认 P5 融合方式是主因。此时按顺序跑：

```text
A3-safe: obs_gate_init_obs_bias=3.0 or 4.0
A3-cap:  obs_gate_max_dyn_alpha=0.1 or 0.2
A3-res:  residual gate
```

判断规则：

| 现象 | 解释 | 下一步 |
| --- | --- | --- |
| 提高 obs bias 后恢复 | 早期 dynamics 注入过强 | 保留更高 obs bias 或 warmup |
| 限制 max dyn alpha 后恢复 | dynamics 只能做弱补偿 | 增加 alpha cap |
| residual gate 恢复 | feature replacement 是主因 | 改成 residual fusion |
| 都不恢复 | P3 dynamics 或 obs_stats 本身无效 | 回到 A2 和 gate 分桶分析 |

### 6.4 建议修改

在 `models/observability.py` 或 `models/seqtrack3d.py` 中加入 gate 融合模式：

```yaml
obs_gate_fusion_mode: replacement  # replacement | residual
obs_gate_max_dyn_alpha: null       # e.g. 0.2
obs_gate_residual_scale: 0.1
```

推荐 residual 形式：

```python
dyn_residual = self.dyn_proj(z_dyn)
fused_feature = point_feature + residual_scale * alpha_dyn * dyn_residual
```

并建议：

1. residual 分支最后一层或 `dyn_proj` 可用小初始化或零初始化，避免训练初期破坏 point feature。
2. `obs_gate_init_obs_bias` 先从 `3.0` 或 `4.0` 开始。
3. 加 `obs_gate_max_dyn_alpha=0.2` 做安全上限。
4. 后续如果要更稳，可加：

```yaml
obs_gate_warmup_epoch: 5
```

前 5 epoch 只用 observation 或极小 dynamics residual。

---

## 7. 次要但需要同步处理的问题

### 7.1 Dynamics 监督偏弱

当前：

```yaml
velocity_weight: 0.05
dynamics_displacement_weight: 0.0
```

`velocity_pred` 有轻量监督，但 `dynamics_displacement_pred` 只记录不优化。这样 `z_dyn` 主要依赖最终 tracking loss 间接学习，mini 数据量下容易不稳。

建议在 A2-disp 中尝试：

```yaml
dynamics_displacement_weight: 0.01
```

如果 loss 量级安全，再试：

```yaml
velocity_weight: 0.1
dynamics_displacement_weight: 0.01
```

### 7.2 缺少 gate 行为分桶分析

仅看 `obs_alpha_dyn_mean` 不够。必须按困难条件分桶：

```text
alpha_dyn by sparse bin
alpha_dyn by delta_t bin
alpha_obs by fg confidence bin
```

如果 gate 学对了，应该看到：

```text
sparse / large delta_t / low fg score -> alpha_dyn 更高
dense / high fg score -> alpha_obs 更高
```

如果看不到这种趋势，P5 即使主表偶然涨点，也缺少论文解释力。

### 7.3 Transformer 中仍有固定 4 帧假设

`models/attn/Models.py` 中仍有：

```python
enc_output.reshape(-1, 4 * 128, self.d_model)
dec_output.view(dec_output.shape[0], 4, self.d_model * 8)
```

这暂时不影响当前 `hist_num=3`，因为 `hist_num + 1 = 4`。但它会限制后续历史长度消融：

```text
hist_num=2 / 4 / 6
```

建议暂缓修改，等 A1/A2/A3 跑清楚后再处理。否则容易把 shape 改动和模块收益混在一起。

### 7.4 优化稳定性

当前 P5 full 的正式训练 hparams 中：

```yaml
gradient_clip_val: 0.0
```

smoke test 中常用 `grad_clip=1.0`。如果后续发现 loss 或 gate alpha 曲线有尖峰，可以在正式消融中统一加入：

```yaml
gradient_clip_val: 1.0
```

但这必须对 baseline 和各 ablation 保持一致，否则会引入新的变量。

---

## 8. 下一步实验顺序

### Step 1：复测已保存 CT checkpoint

不要只看 last。当前 CT 已保存：

```text
epoch=4-step=6310.ckpt      # precision best 附近
epoch=24-step=31550.ckpt    # success best 附近
epoch=29-step=37860.ckpt
epoch=34-step=44170.ckpt
epoch=49-step=63100.ckpt
last.ckpt
```

先复测：

```text
epoch=4-step=6310.ckpt
epoch=24-step=31550.ckpt
last.ckpt
```

目的：

1. 确认 best checkpoint 是否稳定复现。
2. 区分“训练后期退化”和“评估随机波动”。
3. 为论文保留一个诚实的 early checkpoint 观察，而不是只用 final 下结论。

### Step 2：跑 A1 时间编码诊断

先补 A1-pseudo、A1-MLP、A1-Fourier，用来判断 A1-raw 退步是否来自 raw 时间尺度。

基础配置：

```yaml
use_dynamics_encoder: False
use_observability_gate: False
use_twc: False
```

三组变体：

```text
A1-pseudo:  use_real_time=False, time_encoding=raw
A1-MLP:     use_real_time=True,  time_encoding=mlp
A1-Fourier: use_real_time=True,  time_encoding=fourier
```

目的：

```text
区分 CT 主链路问题、raw real time 尺度问题和时间编码形式问题。
```

### Step 3：跑 A2 Dynamics-only

配置：

```yaml
use_dynamics_encoder: True
use_observability_gate: False
use_twc: False
num_candidates: 4
dynamics_displacement_weight: 0.0
```

目的：

```text
隔离 P3 Dynamics 是否单独造成退步。
```

### Step 4：跑 A2-lite 和 A2-disp

A2-lite：

```yaml
use_dynamics_encoder: True
use_observability_gate: False
use_twc: False
num_candidates: 1
```

A2-disp：

```yaml
use_dynamics_encoder: True
use_observability_gate: False
use_twc: False
dynamics_displacement_weight: 0.01
```

目的：

```text
验证 candidate offset 污染和 dynamics 监督偏弱是否是 P3 的主要问题。
```

### Step 5：跑 A3-safe

配置：

```yaml
use_dynamics_encoder: True
use_observability_gate: True
use_twc: False
obs_gate_init_obs_bias: 3.0
```

如果仍不稳：

```yaml
obs_gate_init_obs_bias: 4.0
obs_gate_max_dyn_alpha: 0.2
```

目的：

```text
测试更保守 gate 是否缓解 P5 full 后期崩坏。
```

### Step 6：实现并跑 A3-res

代码修改：

```text
models/observability.py
models/seqtrack3d.py
cfgs/*.yaml
```

新增配置：

```yaml
obs_gate_fusion_mode: residual
obs_gate_residual_scale: 0.1
obs_gate_max_dyn_alpha: 0.2
```

目标语义：

```python
fused_feature = point_feature + obs_gate_residual_scale * alpha_dyn * dyn_residual
```

目的：

```text
让 dynamics prior 只做残差修正，不替换 observation feature。
```

### Step 7：P4 TWC 暂后

当前 P4 工程链路已经通过 smoke test，但本轮 P5 full 退步与 TWC 无关，因为 `use_twc=False`。建议等 A1/A2/A3 稳定后再跑：

```text
CT-base + TWC
CT + Dynamics + TWC
```

不要现在把 TWC 和 P5 混在一起。

---

## 9. 需要新增或修改的配置文件

建议新增：

```text
cfgs/seqtrack3d_nuscenes_mini_a1_ct_base.yaml
cfgs/seqtrack3d_nuscenes_mini_a1_pseudo.yaml
cfgs/seqtrack3d_nuscenes_mini_a1_mlp.yaml
cfgs/seqtrack3d_nuscenes_mini_a1_fourier.yaml
cfgs/seqtrack3d_nuscenes_mini_a2_dyn.yaml
cfgs/seqtrack3d_nuscenes_mini_a2_dyn_cand1.yaml
cfgs/seqtrack3d_nuscenes_mini_a2_dyn_disp.yaml
cfgs/seqtrack3d_nuscenes_mini_a3_gate_safe.yaml
cfgs/seqtrack3d_nuscenes_mini_a3_gate_residual.yaml
```

如果不想维护太多文件，也可以先复制服务器上的 mini 配置，每次只改关键开关。但正式论文化前建议固定成独立 YAML，避免实验记录混乱。

---

## 10. 需要新增或修改的代码

### 10.1 `datasets/sampler.py`

短期建议：

1. 在 `data_dict` 中加入：

```python
'candidate_id': np.int64(candidate_id)
```

2. 保留 `history_offsets`、`prev_frame_ids`、`num_points_in_search` 当前逻辑。

目的：

```text
支持 candidate 0 / 非 0 的 dynamics loss 和 ref_boxs 差分诊断。
```

### 10.2 `models/seqtrack3d.py`

建议增加日志：

```text
dynamics_velocity_norm
dynamics_displacement_norm
velocity_label_norm
dynamics_valid_ratio
candidate0_loss_velocity
candidate_nonzero_loss_velocity
```

如果加入 `candidate_id`，可以在 `compute_loss()` 中按 candidate 分桶记录，不一定参与 loss。

后续如果 A2-lite 证明 candidate offset 是主因，再增加：

```yaml
dynamics_candidate_zero_only: True
```

让非 0 candidate 不参与 dynamics loss，或者把其 `z_dyn` 置零做对照。

### 10.3 `models/observability.py`

建议增加：

```yaml
obs_gate_fusion_mode: replacement  # replacement | residual
obs_gate_max_dyn_alpha: null
obs_gate_residual_scale: 0.1
```

实现 residual gate：

```python
if fusion_mode == "replacement":
    fused_feature = alpha_obs * point_feature + alpha_dyn * z_dyn_proj
else:
    fused_feature = point_feature + residual_scale * alpha_dyn * z_dyn_proj
```

并支持：

```python
if max_dyn_alpha is not None:
    alpha_dyn = alpha_dyn.clamp(max=max_dyn_alpha)
```

注意：如果 clamp 以后要重新归一化，只用于 replacement；如果 residual 模式，`alpha_dyn` 可以直接作为残差强度，不必强制 `alpha_obs + alpha_dyn = 1`。

### 10.4 `tools/`

建议新增一个轻量分析脚本：

```text
tools/check_dynamics_stats.py
```

功能：

```text
读取一个 train batch
打印 candidate_id
打印 ref_boxs 差分速度分布
打印 velocity_label 分布
打印 dynamics_displacement_pred 与 motion_label 的量级
```

这比直接跑完整训练更快定位 P3 问题。

---

## 11. 需要拉取的 TensorBoard 标量

在服务器上优先检查这些标量：

```text
loss_total
loss_velocity
loss_dynamics_displacement
obs_alpha_obs_mean
obs_alpha_dyn_mean
obs_alpha_dyn_min
obs_alpha_dyn_max
obs_gate_entropy
obs_num_points_search_mean
obs_estimated_fg_points_mean
obs_mean_fg_score
obs_current_delta_t_ratio
```

重点看：

1. `obs_alpha_dyn_mean` 是否后期升高。
2. `obs_alpha_dyn_max` 是否长期接近 1。
3. `loss_velocity` 是否真的下降。
4. `loss_dynamics_displacement` 虽然权重为 0，量级是否合理。
5. `obs_estimated_fg_points_mean` 和 `obs_num_points_search_mean` 是否在 mini 上过于稀疏，导致 gate 学到极端策略。

如果 alpha 后期偏向 dynamics，同时 precision/test 下降，说明 P5 在放大不可靠 dynamics prior。

---

## 12. 论文表述

当前不要写：

```text
CT-SeqTrack full model outperforms SeqTrack3D.
```

应该写成：

```text
The timestamp-native base path is evaluated first to verify no regression.
The dynamics branch and the observability-aware gate are then evaluated
incrementally to isolate their individual effects. The current full P5
configuration is unstable on nuScenes-mini, likely because dynamics features
are weakly constrained and fused too aggressively at the feature level.
```

中文解释：

```text
当前 full P5 不是最终正向结果。它暴露的是：真实时间主链路、动力学先验和可观测性门控不能一次性混合评价，必须逐级消融。只要 A1 能接近 baseline，timestamp-native 主线仍然成立；P3/P5 则作为后续增强模块继续修正。
```

---

## 13. 当前最终判断

本轮退步最可能来自两个叠加因素：

1. **P3 Dynamics 分支训练/测试分布不匹配。**  
   训练中 `num_candidates=4` 和独立随机 historical offsets 会污染 `ref_boxs` 差分；测试中 history 来自递归预测 drift。DynamicsEncoder 学到的 `z_dyn` 在测试时容易 OOD。

2. **P5 Observability Gate 融合过强。**  
   当前 gate 是 feature replacement，初始化就约有 27% dynamics 注入。若 dynamics feature 不稳，gate 会直接破坏 observation feature，并在递归跟踪中放大误差。

下一步优先级：

```text
1. 复测 saved best checkpoints，不只看 last。
2. 跑 A1，证明 timestamp-native 主链路是否安全。
3. 跑 A2/A2-lite/A2-disp，定位 dynamics 是否退化及原因。
4. 若 A2 正常，再跑 A3-safe/A3-res，修正 gate。
5. P4 TWC 暂后，等 P3/P5 稳定后再接回。
```

一句话：

```text
先证明时间主链路不退化，再修 dynamics 的分布口径，最后把 gate 从强替换改成保守残差。
```
