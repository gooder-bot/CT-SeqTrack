# CT-SeqTrack 当前执行清单

更新时间：2026-06-01

## 文件分工

- `need_to_do.md`：只放下一步和未来要做的事情。
- `done.md`：统一放已经完成的工程验收、实验记录和历史参考。
- `sum_results.md`：简洁总结已有实验说明了什么，以及下一步为什么这么做。
- `refined_plan.md`：放研究定位、论文边界、贡献设计和 related work 边界。
- `compare_results/`：放完整实验表格、曲线和结果文件。

当前只改：

- 本地：`D:\desktop\research\CT-SeqTrack`
- 服务器：`/home/lishengjie/study/lcyu/CT-SeqTrack`

不要同时改原始 `seqtrack`，避免 baseline、改进版和实验结果混在一起。

## 0. 当前原则

- 当前主线优先基于 `A2-order-dyn`：主干保留 SeqTrack3D 的 order-time token，DynamicsEncoder 使用真实 `delta_t/current_delta_t`。
- 暂时不要继续投入 raw / MLP / Fourier real-time 主干，因为已有实验说明它们没有修复 A1 崩坏。
- 下一批实验只改机制变量，学习率、batch size、epoch、seed、验证间隔等训练条件保持统一。
- `main.py` 会用命令行覆盖 YAML 中的 `batch_size / epoch / workers / check_val_every_n_epoch / tag`，正式命令必须显式写这些参数。
- 已有工程 smoke test 和已完成实验统一归档到 `done.md`，这里不再重复维护。

### 0.1 术语速查

| 名称 | 含义 |
| --- | --- |
| `cand1` | `num_candidates=1`，不是 `candidate_id=1` |
| `cand4` | 默认多 candidate，包含 `candidate_id=0/1/2/3` |
| `disp` | 增加小权重 `dynamics_displacement_weight`，检查位移监督是否必要 |
| `A1-order` | 主干 order-time，无 dynamics / TWC / gate |
| `A2-order-dyn` | 主干 order-time，真实时间只进入 `DynamicsEncoder` |
| `TWC` | 不同历史采样路径到同一当前时刻的一致性 |
| `gate-safe` | 保守 observation-biased gate，避免旧 P5 full 的强融合问题 |

## 1. 当前五组消融实验

当前要跑：

```text
1. A2-order-dyn-cand1
2. A2-order-dyn-disp
3. A1-order+TWC
4. A2-order-dyn+TWC
5. A3-order-gate-safe
```

对应配置：

```text
cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml
cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml
cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml
```

统一训练命令参数：

```text
--batch_size 16
--epoch 60
--workers 12
--seed 42
--preloading
--check_val_every_n_epoch 5
```

### 1.0 实验状态总表

| 实验 | cfg | tag | 状态 | final success | final precision | 结论 |
| --- | --- | --- | --- | ---: | ---: | --- |
| A2-order-dyn-cand1 | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml` | `ct_a2_order_dyn_cand1_car_60ep_bs16` | 未跑 | - | - | - |
| A2-order-dyn-disp | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml` | `ct_a2_order_dyn_disp_car_60ep_bs16` | 未跑 | - | - | - |
| A1-order+TWC | `cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml` | `ct_a1_order_twc_car_60ep_bs16` | 未跑 | - | - | - |
| A2-order-dyn+TWC | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml` | `ct_a2_order_dyn_twc_car_60ep_bs16` | 未跑 | - | - | - |
| A3-order-gate-safe | `cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml` | `ct_a3_order_gate_safe_car_60ep_bs16` | 未跑 | - | - | - |

跑完每组后，先填这个表，再更新 `sum_results.md` 和 `done.md`。如果要多卡并行，保持命令参数不变，只替换 `CUDA_VISIBLE_DEVICES=<GPU>` 和 tag 中必要的实验名。

### 1.1 A2-order-dyn-cand1

目的：

```text
检查非 0 candidate 的随机历史框扰动是否污染 DynamicsEncoder。
```

关键设置：

```yaml
num_candidates: 1
use_dynamics_encoder: true
use_observability_gate: false
use_twc: false
main_time_source: order
```

结果解读：

- 如果 `cand1` 比 `A2-order-dyn` 更稳或更好，说明 noisy candidate history 可能污染 dynamics。
- 如果 `cand1` 明显变差，说明多 candidate 仍提供鲁棒性，不能简单移除。
- 当前先按 60 epoch 统一跑；若后续要严格按 optimizer step 对齐，再考虑 240 epoch 版本。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a2_order_dyn_cand1_car_60ep_bs16
```

### 1.2 A2-order-dyn-disp

目的：

```text
检查 dynamics 是否需要额外 displacement 监督。
```

关键设置：

```yaml
dynamics_displacement_weight: 0.01
use_dynamics_encoder: true
use_observability_gate: false
use_twc: false
main_time_source: order
```

结果解读：

- 如果 `disp` 更稳或更好，说明仅有 `loss_velocity` 不够，位移监督有必要。
- 如果没变化或退化，说明 dynamics 的主要问题不在 displacement 监督强度。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a2_order_dyn_disp_car_60ep_bs16
```

### 1.3 A1-order+TWC

目的：

```text
先在没有 dynamics 的 order-time 主干上检查 TWC 是否有效。
```

关键设置：

```yaml
use_dynamics_encoder: false
use_observability_gate: false
use_twc: true
main_time_source: order
num_candidates: 1
```

结果解读：

- 如果提升，说明 TWC 自身对 variable-rate / historical path consistency 有价值。
- 如果不提升，先检查 `twc_valid_ratio`、TWC 权重、paired view 构造和 mini 数据规模。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a1_order_twc_car_60ep_bs16
```

### 1.4 A2-order-dyn+TWC

目的：

```text
检查真实时间 dynamics prior 和 TWC 是否互补。
```

关键设置：

```yaml
use_dynamics_encoder: true
use_observability_gate: false
use_twc: true
main_time_source: order
num_candidates: 1
```

结果解读：

- 如果比 `A2-order-dyn` 更好，TWC 可以作为第二个核心贡献接入主线。
- 如果只比 `A1-order+TWC` 好，说明 dynamics 仍是主要收益来源。
- 如果退化，暂时不要继续叠 gate，先检查 `twc_weight / twc_valid_ratio / loss_twc`。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a2_order_dyn_twc_car_60ep_bs16
```

### 1.5 A3-order-gate-safe

目的：

```text
在干净的 order-time 主干上重新测试保守 gate。
```

关键设置：

```yaml
use_dynamics_encoder: true
use_observability_gate: true
use_twc: false
main_time_source: order
obs_gate_init_obs_bias: 3.0
```

结果解读：

- 如果比旧 P5 full 稳，说明旧 gate 失败部分来自 raw real-time 主干和过强 dynamics 注入。
- 如果仍退化，优先实现 residual gate 或限制 `alpha_dyn`，不要直接否定 observability-aware fusion。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a3_order_gate_safe_car_60ep_bs16
```

## 2. 五组实验完成后的整理任务

- [ ] 把每组 TensorBoard 的 `success/test` 和 `precision/test` 拉成统一 CSV。
- [ ] 生成一份新的比较表，至少包含：

```text
SeqTrack baseline
A1-order
A2-order-dyn
A2-order-dyn-cand1
A2-order-dyn-disp
A1-order+TWC
A2-order-dyn+TWC
A3-order-gate-safe
```

- [ ] 更新 `sum_results.md`，写清楚每组结果支持或反驳了什么。
- [ ] 如果某组 final 和 best 差距很大，复测 best checkpoint，不只看 last。
- [ ] 如果 TWC 退化，优先查看 `loss_twc / twc_valid_ratio / twc_center_gap / twc_angle_gap`。
- [ ] 如果 gate 退化，优先查看 `obs_alpha_dyn_mean / obs_alpha_dyn_max / obs_gate_entropy`。

## 3. Dynamics 诊断日志

当前 `cand1` 只是间接诊断。为了直接判断非 0 candidate 是否污染 dynamics，建议后续补日志。

### 3.1 数据字段

- [ ] 在 `datasets/sampler.py` 的 `data_dict` 中加入：

```python
'candidate_id': np.int64(candidate_id)
```

### 3.2 模型日志

- [ ] 在 `models/seqtrack3d.py` 中记录：

```text
dynamics_velocity_norm
dynamics_displacement_norm
velocity_label_norm
dynamics_valid_ratio
candidate0_loss_velocity
candidate_nonzero_loss_velocity
candidate0_velocity_label_norm
candidate_nonzero_velocity_label_norm
candidate0_velocity_pred_norm
candidate_nonzero_velocity_pred_norm
```

### 3.3 轻量检查脚本

- [ ] 新增：

```text
tools/check_dynamics_stats.py
```

功能：

```text
读取一个 train batch
打印 candidate_id 分布
打印 ref_boxs 差分速度分布
打印 velocity_label 分布
打印 dynamics_displacement_pred 和 motion_label 的量级
按 candidate 0 / 非 0 分桶
```

## 4. 根据结果的决策树

### 4.1 cand1

- [ ] 如果 `cand1` 优于 `A2-order-dyn`：考虑让 dynamics loss 只监督 `candidate_id=0`，或让非 0 candidate 的 dynamics loss 降权。
- [ ] 如果 `cand1` 劣于 `A2-order-dyn`：保留 multi-candidate 训练，后续用日志判断非 0 candidate 是否虽然 noisy 但提升鲁棒性。

候选改法：

```yaml
dynamics_candidate_zero_only: true
```

### 4.2 disp

- [ ] 如果 `disp` 有收益：保留小权重 displacement loss，后续在大 gap / sparse 子集上确认。
- [ ] 如果 `disp` 退化：保持 `dynamics_displacement_weight: 0.0`，只把 displacement 作为日志参考。

### 4.3 TWC

- [ ] 如果 `A1-order+TWC` 有收益：TWC 可以独立作为贡献。
- [ ] 如果 `A2-order-dyn+TWC` 进一步提升：把 `A2-order-dyn+TWC` 作为下一版主配置。
- [ ] 如果 TWC 退化：先调小 `twc_weight` 或增加 warmup，不要直接接 gate。

候选设置：

```yaml
twc_weight: 0.01
twc_warmup_epoch: 5
```

### 4.4 gate

- [ ] 如果 `A3-order-gate-safe` 稳定：继续做 gate 分桶分析。
- [ ] 如果仍退化：实现 residual gate，而不是 feature replacement。

候选配置：

```yaml
obs_gate_fusion_mode: residual
obs_gate_residual_scale: 0.1
obs_gate_max_dyn_alpha: 0.2
```

候选语义：

```python
fused_feature = point_feature + obs_gate_residual_scale * alpha_dyn * dyn_residual
```

## 5. 后续评估协议

- [ ] 困难子集评估：

```text
delta_t bins: [0,0.2), [0.2,0.5), [0.5,1.0), [1.0,+inf)
sparse bins: [0,5), [5,10), [10,20), [20,50), [50,+inf)
re-appearance: 连续低点数后恢复的片段
variable-gap: skip=1/2/3/5
```

- [ ] TWC 额外报告：

```text
twc_center_gap
twc_angle_gap
prediction variance under different sampling paths
```

- [ ] Gate 额外报告：

```text
alpha_dyn by sparse bin
alpha_dyn by delta_t bin
alpha_obs by foreground confidence bin
```

## 6. 后续代码债

- [ ] 暂缓移除 Transformer 中固定 4 帧假设，等当前五组消融完成后再做。
- [ ] 检查 `models/attn/Models.py` 中 `4 * 128`、`view(..., 4, ...)`、`reshape(-1, 4 * 128, ...)` 等硬编码。
- [ ] 后续如果要做 `hist_num=2/4/6` 历史长度消融，再把 `L = hist_num + 1` 动态传入 `Seq2SeqFormer.forward()`。
- [ ] 如果 gate 继续退化，再实现 residual gate 和 `obs_gate_max_dyn_alpha`。
- [ ] 如果 cand1 证明 candidate noise 是主因，再实现 dynamics clean-history 或 candidate 分桶 loss。

## 7. 论文与文档后续

- [ ] 根据五组新消融结果更新 `sum_results.md`。
- [ ] 根据最终正向模块更新 `README.md` 的当前实验诊断。
- [ ] 根据最终主配置更新 `refined_plan.md` 的贡献顺序。
- [ ] 不要写“CT-SeqTrack full model outperforms SeqTrack3D”，除非完整消融支持。
- [ ] 更稳的当前表述：

```text
Preserving SeqTrack3D's order-time semantics while injecting real delta_t
through a timestamp-conditioned dynamics prior is currently more stable than
directly replacing the main branch time tokens with raw timestamps.
```

## 8. 暂缓方向

- [ ] 暂不切换到 MambaTrack3D / TrackM3D / TrajTrack 作为主 baseline。
- [ ] 暂不上 Neural ODE / SDE / CDE。
- [ ] 暂不主打任意时间查询或多传感器异步融合。
- [ ] 频域 / 谱域方向只保留为后续诊断候选；在当前五组消融完成前，不改模型主干。
