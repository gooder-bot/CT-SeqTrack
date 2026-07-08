# CT-SeqTrack 当前执行清单

更新时间：2026-07-08

当前总路线：

```text
先做 variable-rate / HTV 协议，把真实时间的应用场景立住；
再把 A2 从 feature-concat dynamics 改成更保守的 residual dynamics；
最后用多 seed、delta_t/sparse 分桶和 candidate 日志解释稳定性。
```

真实时间处理原则：

- 用 `dt_norm = dt / dt_ref`，例如 nuScenes keyframe `dt_ref=0.5`，Waymo/KITTI 可按数据频率设定。
- 对 `dt` 做 clip，例如 `[0.5x, 3x]` 或按训练集分位数裁剪。
- velocity 分支里用 `max(dt, eps)`，避免小 `dt` 放大噪声。
- 加 `valid_mask` 和 frame-gap 信息，区分“真的 0.2s”与“缺了一帧”。
- 做 ablation：fixed step、true timestamp、jittered timestamp、shuffled timestamp、按 `delta_t` 分桶评估。
- 如果跨数据集训练，最好随机 temporal dropout / jitter augmentation，让网络学会 variable-rate，而不是记住某个数据集的频率。
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

- 当前主线优先围绕 `variable-rate 评测协议 + A2 residual dynamics` 做稳定性和机制诊断：主干保留 SeqTrack3D 的 order-time token，真实 `delta_t/current_delta_t` 只进入保守 dynamics prior。
- 暂时不要继续投入 raw / MLP / Fourier real-time 主干，因为已有实验说明它们没有修复 A1 崩坏。
- 下一批实验优先统一 seed / checkpoint / optimizer step 口径；不要在 variable-rate 协议和 residual dynamics 之前继续叠加 TWC 或 gate。
- 普通 fixed-step final 不是唯一成功标准；如果 long-delta_t / sparse / re-appearance 子集稳定提升，也可以支撑论文叙事。
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
| `conf-res` | confidence residual gate，只用 dynamics 做小幅 motion residual 修正 |

## 1. 当前消融实验

当前批次：

```text
已跑并整理：
1. A2-order-dyn-cand1
2. A2-order-dyn-disp
3. A1-order+TWC（validity-fixed active TWC）
4. A2-order-dyn+TWC（validity-fixed active TWC）
5. A3-order-gate-safe
6. A3-order-conf-res-gate
7. A3-order-conf-res best-e14 checkpoint retest
8. A2-order-dyn seed43 / seed44
9. A2-order-dyn+TWC w0.01
10. A3-order-conf-res rerun seed42

当前下一步：
1. 新增 variable-rate / HTV 评测协议：`skip=1/2/3/5`、temporal dropout、delta_t bins、sparse bins。
2. 新增 A2 residual dynamics 配置，与当前 A2 feature-concat dynamics 做同 seed 对照。
3. 汇总 A2-order-dyn 多 seed 稳定性，确认 seed43 collapse 是否可复现。
4. 暂停 A2-order-dyn+TWC 主线接入；0.05 和 0.01 都已崩。
5. 暂停 conf-res 正向表述；先核对旧 best 评测路径和 alpha/residual 行为。
6. 补 dynamics candidate / velocity / displacement 分桶日志，解释 cand1 / disp / seed collapse 的机制来源。
```

对应配置：

```text
cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml
cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc_w001.yaml
cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml
cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml
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
| A2-order-dyn-cand1 | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml` | `ct_a2_order_dyn_cand1_car_60ep_bs16` | 已跑已整理 | 26.68 | 24.50 | 明显退化；但 60 epoch 只有约 1/4 step，需等 step 复核才可最终判断 candidate noise。 |
| A2-order-dyn-disp | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml` | `ct_a2_order_dyn_disp_car_60ep_bs16` | 已跑已整理 | 50.54 | 63.85 | 与 A2-order-dyn 基本持平，precision 小幅更高；可作为温和稳定项继续观察。 |
| A1-order+TWC | `cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml` | `ct_a1_order_twc_cand4_validfix_car_60ep_bs16_gpu2` | 已跑已整理 | 51.16 | 61.10 | `twc_valid_ratio` 均值约 0.75；相对 A1-order success 基本持平，precision +3.24，是一个 precision-positive 但非 success-positive 的 TWC 信号。 |
| A2-order-dyn+TWC | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml` | `ct_a2_order_dyn_twc_cand4_validfix_car_60ep_bs16_gpu3` | 已跑已整理 | 28.23 | 32.04 | TWC 已激活但后期崩坏；相对 A2-order-dyn final success/precision 分别 -22.73/-31.28，当前 `twc_weight=0.05` 不适合作为主配置。 |
| A3-order-gate-safe | `cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml` | `ct_a3_order_gate_safe_car_60ep_bs16` | 已跑已整理 | 48.32 | 54.87 | 比旧 P5 full 安全很多，但仍低于 A2-order-dyn；保守 feature gate 不是当前最终收益来源。 |
| A3-order-conf-res-gate | `cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml` | `ct_a3_order_conf_res_gate_car_60ep_bs16_gpu3` | 已跑已整理 | 31.17 | 30.92 | 旧 best 很高（62.04 / 76.30）但最新 best-e14 复测未复现；不能作为确认收益。 |
| A3-conf-res best-e14 retest | `cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml` | `retest_a3_conf_res_precision_best_e14` | 已跑已整理 | 28.06 | 37.70 | 单 checkpoint 测试，未复现旧 best；需核对旧汇总评测路径。 |
| A2-order-dyn seed43 | `cfgs/seqtrack3d_nuscenes_a2_order_dyn.yaml` | `ct_a2_order_dyn_car_60ep_bs16_seed43` | 已跑已整理 | 23.64 | 23.77 | seed43 崩坏，A2 稳定性风险很大。 |
| A2-order-dyn seed44 | `cfgs/seqtrack3d_nuscenes_a2_order_dyn.yaml` | `ct_a2_order_dyn_car_60ep_bs16_seed44` | 已跑已整理 | 46.90 | 52.62 | 好于 seed43，但仍低于旧 seed42 60ep 汇总。 |
| A2-order-dyn+TWC w0.01 | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc_w001.yaml` | `ct_a2_order_dyn_twc_w001_seed42_car_60ep_gpu0_nowpreload` | 已跑已整理 | 22.88 | 24.27 | 降低 TWC 权重仍崩，A2+TWC 暂停接主线。 |
| A3-conf-res rerun seed42 | `cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml` | `ct_a3_order_conf_res_gate_rerun_seed42_car_60ep_gpu0_nowpreload` | 已跑已整理 | 32.11 | 31.87 | rerun 仍低，gate/conf-res 先转诊断。 |

本轮完整整理文件见 `compare_results/twc_gate_ablation_*` 和 `compare_results/reports/latest_5runs_comparison.md`。如果后续要多卡并行复跑，保持命令参数不变，只替换 `CUDA_VISIBLE_DEVICES=<GPU>` 和 tag 中必要的实验名。

### 1.0a 下一批优先实验

| 优先级 | 实验 | 目的 | 成功信号 |
| --- | --- | --- | --- |
| P0 | `A1-order` variable-rate eval | 建立 fixed-step baseline 在长 gap / sparse 下的退化曲线 | 指标随 `skip/delta_t` 退化，证明问题设置有必要 |
| P1 | `A2-residual-dyn` seed42/43 | 检查保守 residual 是否比 feature concat 更稳 | final 不崩，或 long-gap/sparse bin 提升 |
| P2 | `A2-feature-dyn` 同协议复评 | 与当前 A2 做公平对照 | 解释 seed collapse 是否来自接入方式 |
| P3 | dynamics candidate 日志 | 判断非 0 candidate 是否污染 velocity 监督 | candidate0 与 nonzero candidate 的 velocity/residual 误差可分 |
| P4 | TWC on A1 variable-rate | 检查 TWC 是否在 variable-rate 下更有价值 | prediction variance 降低，precision 或 long-gap bin 提升 |

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
num_candidates: 4
twc_candidate_zero_only: false
```

结果解读：

- 如果提升，说明 TWC 自身对 variable-rate / historical path consistency 有价值。
- 如果不提升，先检查 `twc_valid_ratio`、TWC 权重、paired view 构造和 mini 数据规模。

当前实测结果：

```text
validity-fixed cand4 版本：
final success 51.16，final precision 61.10；
best success 53.16，best precision 63.35。

twc_valid_ratio 均值约 0.750，tail1000 约 0.753；
loss_twc tail1000 约 0.0081。
所以这已经是 active-TWC 结果，不再是旧的 twc_valid_ratio=0 诊断。

相对 A1-order：
final success   -0.07
final precision +3.24
```

当前解读：TWC 在无 dynamics 的 A1-order 主干上没有带来明确 success 增益，但给 final precision 带来有意义的正向信号。论文里可以把它作为“路径一致性有助于定位精度”的候选证据，但不要写成完整指标全面提升。

命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a1_order_twc_cand4_validfix_car_60ep_bs16
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
num_candidates: 4
twc_candidate_zero_only: false
```

结果解读：

- 如果比 `A2-order-dyn` 更好，TWC 可以作为第二个核心贡献接入主线。
- 如果只比 `A1-order+TWC` 好，说明 dynamics 仍是主要收益来源。
- 如果退化，暂时不要继续叠 gate，先检查 `twc_weight / twc_valid_ratio / loss_twc`。

当前实测结果：

```text
validity-fixed cand4 版本：
final success 28.23，final precision 32.04；
best success 45.24，best precision 57.43。

twc_valid_ratio 均值约 0.750，tail1000 约 0.750；
loss_twc tail1000 约 0.0077。
所以 TWC 确实激活，但和 dynamics 组合后明显后期崩坏。

相对 A2-order-dyn：
final success   -22.73
final precision -31.28
```

当前解读：这次下降可以解释为“当前 `twc_weight=0.05` / paired-view 协议与 dynamics prior 组合不稳定”，不再是 validity 没生效的问题。后续 `twc_weight=0.01` 复核仍崩坏，因此如果继续 TWC，应优先考虑 warmup / 延后启用 / 仅保留 A1 诊断，不要把当前 A2+TWC 直接接 gate。

命令：

```bash
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a2_order_dyn_twc_cand4_validfix_car_60ep_bs16
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

当前实测结果：

```text
final success 48.32，final precision 54.87；
best success 50.99，best precision 60.17。

obs_alpha_dyn_mean 均值约 0.127，tail1000 约 0.116；
obs_alpha_dyn_max 均值约 0.152；
obs_gate_entropy 均值约 0.365。

相对 A2-order-dyn：
final success   -2.64
final precision -8.45
```

当前解读：gate-safe 比旧 P5 full 的 `31.19 / 31.89` 安全很多，说明保守 observation-biased gate 确实缓解了强融合灾难；但它仍低于 A2-order-dyn，暂时不能作为最终收益模块。

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

### 1.6 A3-order-conf-res-gate

目的：

```text
测试比 gate-safe 更保守的 confidence residual gate：
不在 feature 空间替换 observation，而是在 motion 空间用 dynamics 做小幅残差修正。
```

关键设置：

```yaml
use_dynamics_encoder: true
use_observability_gate: true
use_twc: false
main_time_source: order
obs_gate_fusion_mode: confidence_residual
obs_gate_init_obs_bias: 3.0
obs_gate_residual_scale: 0.1
obs_gate_max_dyn_alpha: 0.2
dynamics_motion_mode: feature
```

候选语义：

```text
obs_motion_pred = motion_mlp(point_feature)
alpha_dyn = clamp(alpha_dyn, 0, obs_gate_max_dyn_alpha)
motion_pred_xyz = obs_motion_pred_xyz
                + obs_gate_residual_scale * alpha_dyn * dynamics_displacement_pred
```

结果解读：

- 如果优于 `A3-order-gate-safe`，说明 feature replacement 太激进，residual correction 更安全。
- 如果优于 `A2-order-dyn`，说明 observability-aware dynamics 修正确实能带来额外收益。
- 如果总体持平但 sparse / long-delta_t 子集提升，也可以作为困难场景增强模块。
- 如果仍退化，优先检查 `obs_alpha_dyn_raw_mean / obs_alpha_dyn_clamped_mean / obs_dyn_residual_norm`。

当前实测结果：

```text
version_1 早期点 + version_2 续跑点合并，重复 step=18930 保留 version_2。

final success 31.17，final precision 30.92；
best success 62.04，best precision 76.30。

obs_alpha_dyn_raw_mean 均值约 0.493；
obs_alpha_dyn_clamped_mean 均值约 0.181；
obs_dyn_residual_norm 均值约 0.0315。

相对 A2-order-dyn：
final success   -19.79
final precision -32.40
best success    +10.50
best precision  +12.72
```

当前解读：conf-res 的旧 best 信号很强，但 final 崩得很重，说明它更像 checkpoint-selection / 后期稳定性问题，而不是稳定最终模型。后续 best-e14 复测只有 28.06 / 37.70，暂时不能按旧 best 肯定 conf-res；下一步应核对旧 best 的评测路径和 alpha/residual 行为。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a3_order_conf_res_gate_car_60ep_bs16
```

## 2. 当前消融实验完成后的整理任务

已完成局部整理：

```text
compare_results/cand1_disp_dynamics_*
compare_results/twc_gate_ablation_*
```

注意：旧 inactive-TWC 诊断结论是 TWC validity 出错，当前 active-TWC / gate 正式整理以 `twc_gate_ablation_*` 为准。

- [x] 把每组 TensorBoard 的 `success/test` 和 `precision/test` 拉成统一 CSV：`compare_results/twc_gate_ablation_metrics_points.csv`。
- [x] 生成新的比较表和 summary：`compare_results/twc_gate_ablation_metrics_summary.csv`、`compare_results/twc_gate_ablation_comparison.md`，包含：

```text
SeqTrack baseline
A1-order
A2-order-dyn
A2-order-dyn-cand1
A2-order-dyn-disp
A1-order+TWC
A2-order-dyn+TWC
A3-order-gate-safe
A3-order-conf-res-gate
```

- [x] 生成曲线和诊断图：

```text
compare_results/twc_gate_ablation_curves.png
compare_results/twc_gate_ablation_success_curve.png
compare_results/twc_gate_ablation_precision_curve.png
compare_results/twc_gate_ablation_best_final_summary.png
compare_results/twc_gate_ablation_twc_diagnostics.png
compare_results/twc_gate_ablation_gate_diagnostics.png
compare_results/twc_gate_ablation_diagnostics_summary.csv
```

- [x] 更新 `sum_results.md`，写清楚 active TWC、gate-safe、conf-res 分别支持或反驳了什么。
- [x] `A3-order-conf-res-gate` best checkpoint 已复测：best-e14 retest 为 28.06 / 37.70，未复现旧 62.04 / 76.30。
- [x] `A2-order-dyn+TWC` 已确认 active 但后期崩坏；`twc_weight=0.01` 复核仍为 22.88 / 24.27，低权重未救回。
- [ ] 汇总 A2-order-dyn seed42 / seed43 / seed44 稳定性，必要时补更多 seed 或重跑 seed42。
- [ ] 如果继续 gate，优先做 sparse / delta_t / foreground confidence 分桶，不只看整体 final。

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

- [x] validity 已修复并验证：两组 active TWC 的 `twc_valid_ratio` 均值约 0.75，`loss_twc / twc_center_gap / twc_angle_gap` 都有实际量级。
- [x] `A1-order+TWC`：final success 基本持平（-0.07），final precision 提升（+3.24）。TWC 可以作为 precision-positive 候选信号，但还不能说全面提升。
- [x] `A2-order-dyn+TWC`：final success / precision 明显退化（-22.73 / -31.28）。`twc_weight=0.01` 复核仍崩，当前 TWC 不适合作为 dynamics 主配置。
- [ ] 如果继续 active TWC，优先调小 `twc_weight` 或增加 warmup；不要直接接 gate。

候选设置：

```yaml
twc_weight: 0.01
twc_warmup_epoch: 5
```

### 4.4 gate

- [x] `A3-order-gate-safe` 已跑：比旧 P5 full 安全，但相对 A2-order-dyn final success / precision 仍为 -2.64 / -8.45。
- [ ] 如果继续 gate，先做分桶分析确认 sparse / long-delta_t / low-confidence 样本是否受益。
- [ ] feature replacement 版本暂时不要作为主线；后续优先 residual / confidence 约束，并严格限制 `alpha_dyn`。

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

### 4.5 置信空间 / residual confidence gate

这个想法已经落地为 `A3-order-conf-res-gate`，作为 `A3-order-gate-safe` 的并行对照。它不替代当前主线，只用于判断 feature replacement 是否过于激进，以及 residual correction 是否更安全。

当前结果：

- [x] `A3-order-conf-res-gate` 已跑，final success / precision 为 31.17 / 30.92。
- [x] best checkpoint 已复测：best-e14 retest 为 28.06 / 37.70，没有复现旧 62.04 / 76.30。
- [ ] 核对旧 best 汇总的 checkpoint / version 拼接 / test split / metric 读取路径，再决定是否继续 conf-res。
- [ ] 如果 best 可复现，再检查为什么后期崩坏：raw alpha 均值约 0.493，clamped alpha 均值约 0.181，说明 gate 仍然很想依赖 dynamics。

核心原则：

```text
不要再用 dynamics 强替换 observation feature；
改成用观测置信度和 dynamics 置信度控制一个小幅 residual 修正。
```

候选形式：

```text
conf_obs = f(num_points, foreground_score, seg_entropy, estimated_fg_points)
conf_dyn = f(dynamics_valid, valid_history_ratio, delta_t, velocity_consistency)

alpha_dyn = conf_dyn / (conf_obs + conf_dyn)
alpha_dyn = clamp(alpha_dyn, 0, obs_gate_max_dyn_alpha)

final_pred = obs_pred + obs_gate_residual_scale * alpha_dyn * dyn_residual
```

也可以考虑 precision-weighted fusion：

```text
precision = 1 / variance
pred = (precision_obs * obs_pred + precision_dyn * dyn_pred)
       / (precision_obs + precision_dyn)
```

优先实现方式：

- [ ] 先放在 box / motion residual 空间，不要直接融合 256 维 feature。
- [ ] `alpha_dyn` 必须有上限，例如 `0.1` 或 `0.2`。
- [ ] `dynamics_valid` 不足时强制 `alpha_dyn=0`。
- [ ] 记录 `conf_obs / conf_dyn / alpha_dyn / dyn_residual_norm`。
- [ ] 按 sparse bin、delta_t bin、foreground confidence bin 看是否只在困难样本上启用 dynamics。

它能体现什么：

- 如果 sparse / long delta_t / low foreground confidence 时 dynamics 权重上升且指标提升，说明 observability-aware fusion 有价值。
- 如果 confidence residual gate 仍退化，说明问题可能不在 gate 数学形式，而在 dynamics prior 质量或监督信号。
- 如果总体持平但困难子集提升，也可以作为“真实时间 dynamics prior 对困难场景有帮助”的分析点。

## 5. 后续评估协议

- [ ] 困难子集评估：

```text
delta_t bins: [0,0.2), [0.2,0.5), [0.5,1.0), [1.0,+inf)
sparse bins: [0,5), [5,10), [10,20), [20,50), [50,+inf)
re-appearance: 连续低点数后恢复的片段
variable-gap: skip=1/2/3/5
```

- [ ] variable-rate / HTV 协议：

```text
fixed-step: offsets=[1,2,3]
skip-2: offsets=[1,3,5]
skip-3: offsets=[1,4,7]
mixed-gap: per-sample random offsets with shared current frame
temporal-dropout: randomly remove earlier history while keeping nearest anchor
jittered-dt: perturb timestamps only, to test whether the model uses real time robustly
shuffled-dt: shuffle delta_t within batch, as a negative control
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
- [ ] 实现 `dynamics_motion_mode=residual_limited` 或等价配置：`scale`、`max_residual_norm`、`warmup_epoch`、long-gap-only 开关。
- [ ] 如果 gate 继续退化，再实现 residual gate 和 `obs_gate_max_dyn_alpha`。
- [ ] 如果 cand1 证明 candidate noise 是主因，再实现 dynamics clean-history 或 candidate 分桶 loss。

## 7. 论文与文档后续

- [x] 根据五组新消融结果更新 `sum_results.md`。
- [x] 根据 2026-07-08 五次复核更新 `README.md` 的当前实验诊断。
- [x] 根据当前判断更新 `refined_plan.md` 的贡献顺序：variable-rate 协议、residual dynamics、TWC 候选。
- [ ] 不要写“CT-SeqTrack full model outperforms SeqTrack3D”，除非完整消融支持。
- [ ] 更稳的当前表述：

```text
Preserving SeqTrack3D's order-time semantics while injecting real delta_t
through a conservative timestamp-conditioned residual dynamics prior is the
next most plausible route, especially under variable-rate / long-gap tracking.
```

## 8. 暂缓方向

- [ ] 暂不切换到 MambaTrack3D / TrackM3D / TrajTrack 作为主 baseline。
- [ ] 暂不上 Neural ODE / SDE / CDE。
- [ ] 暂不主打任意时间查询或多传感器异步融合。
- [ ] 频域 / 谱域方向只保留为后续诊断候选；在当前五组消融完成前，不改模型主干。
