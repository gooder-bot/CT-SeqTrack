# CT-SeqTrack 当前执行清单

更新时间：2026-07-08

本文只放还没完成、正在进行、或者后续要做的事情。已经完成的工程验收、实验记录和当时结果统一归档到 `done.md`；简洁实验结论放 `sum_results.md`；研究定位、论文边界和贡献顺序放 `refined_plan.md`。

## 0. 当前主线

```text
先用 nuScenes-mini-HTV / variable-rate 协议把真实时间的应用场景立住；
再把 A2 从 feature-concat dynamics 改成更保守的 residual dynamics；
最后用多 seed、delta_t/sparse/displacement 分桶和 candidate 日志解释稳定性。
```

当前原则：

- 不再把已完成实验结果堆在本文件；旧结果见 `done.md` 和 `compare_results/`。
- 第一批 HTV 主表只比较 `A1-order` 与 `A2-order-dyn`，暂不混入 TWC / gate。
- 同一个 protocol 内比较 `A2 - A1`，不要跨 protocol 直接比较 absolute metric。
- 如果 `burst_drop` / `gap1124` 上 A2 有提升，但 `random20` 不明显，也仍然是合理信号，因为 `random20` 更温和。
- 如果 6 组里 A2 仍整体不稳，下一步优先实现 `A2-residual-dyn`，不要先叠加 TWC / gate。
- 参考 TrajTrack 的经验时，只借鉴“低维历史 box trajectory 作为保守 proposal / residual prior”的用法；不要把 CT-SeqTrack 改成 trajectory-prior 论文，也不要直接复制完整 TrajFormer。
- TrajTrack 对当前 dyn 不稳定问题的核心启发：长期运动先验应从 point feature 主干中解耦，先生成低维 motion proposal，再按可靠性小幅修正 observation prediction；避免继续用 `concat(point_feature, z_dyn)` 让 dynamics 过早接管 motion head。

术语：

| 名称 | 含义 |
| --- | --- |
| `A1-order` | 主干 order-time，无 dynamics / TWC / gate |
| `A2-order-dyn` | 主干 order-time，真实时间只进入 `DynamicsEncoder` |
| `A2-residual-dyn` | 待实现的保守 residual dynamics 版本 |
| `A2-dyn-proposal` | TrajTrack-style 候选方向：dyn 只输出低维 motion proposal / residual，不拼接到主 motion feature |
| `dyn_reliability_gate` | 根据 gap、点数、前景置信度、obs-dyn 一致性决定 residual 权重的轻量门控 |
| `obs_uncertainty_head` | 从 observation feature 预测中心定位不确定度，第一版只做辅助监督和 gate 统计，不替代主回归 |
| `unc_conf_residual` | uncertainty / confidence-aware residual dynamics：只有观测不可靠且 dyn 相对可靠时，才允许小幅 dynamics residual |
| `nuScenes-mini-HTV` | 在 nuScenes-mini 上按固定协议构造的虚拟高时间变化评测集 |
| `gap1124` | gap pattern `[1,1,2,4]`，制造强不等间隔 |
| `burst_drop` | 短连续片段后跳过若干帧，制造 burst missing / long gap |
| `random20` | 固定 seed 随机丢 20%，较温和的 variable-rate 协议 |
| `stride-k` | 等间隔 long-gap 辅助对照，不作为核心 variable-dt 证据 |

## 1. 正在运行：HTV 第一批 6 组 60ep

服务器统一设置：

```text
server path: /home/lishengjie/study/lcyu/CT-SeqTrack
dataset: /home/lishengjie/data/nuscenes-mini
category: Car
split: mini_train / mini_val
seed: 42
batch_size: 16
epoch: 60
workers: 4
preloading: true
check_val_every_n_epoch: 5
logs: logs/vr_htv/
```

GPU 分配：

```text
GPU0:
  gap1124 A1-order
  gap1124 A2-order-dyn

GPU1:
  burst_drop A1-order
  burst_drop A2-order-dyn
  random20 A1-order
  random20 A2-order-dyn
```

运行状态表：

| 协议 | 模型 | cfg | GPU | tag | log | 当前状态 | final success | final precision |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| gap1124 | A1-order | `cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml` | 0 | `htv_gap1124_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_gap1124_a1_order_seed42_w4_60ep_bs16.log` | 后台运行中，待完成 | 待填 | 待填 |
| gap1124 | A2-order-dyn | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_gap1124.yaml` | 0 | `htv_gap1124_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_gap1124_a2_order_dyn_seed42_w4_60ep_bs16.log` | 后台运行中，待完成 | 待填 | 待填 |
| burst_drop | A1-order | `cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml` | 1 | `htv_burst_drop_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_burst_drop_a1_order_seed42_w4_60ep_bs16.log` | 后台运行中，待完成 | 待填 | 待填 |
| burst_drop | A2-order-dyn | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_burst_drop.yaml` | 1 | `htv_burst_drop_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_burst_drop_a2_order_dyn_seed42_w4_60ep_bs16.log` | 后台运行中，待完成；已见 PID `1714325` | 待填 | 待填 |
| random20 | A1-order | `cfgs/seqtrack3d_nuscenes_a1_order_vr_random20.yaml` | 1 | `htv_random20_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_random20_a1_order_seed42_w4_60ep_bs16.log` | 后台运行中，待完成 | 待填 | 待填 |
| random20 | A2-order-dyn | `cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_random20.yaml` | 1 | `htv_random20_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_random20_a2_order_dyn_seed42_w4_60ep_bs16.log` | 后台运行中，待完成 | 待填 | 待填 |

进度检查命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

ps -ef | grep "htv_" | grep -v grep
nvidia-smi

for f in logs/vr_htv/*.pid; do echo "$f $(cat "$f")"; done

tail -n 80 logs/vr_htv/htv_burst_drop_a2_order_dyn_seed42_w4_60ep_bs16.log

grep -iE "error|exception|traceback|cuda out|nan|killed" logs/vr_htv/*.log
grep -E "Epoch|precision/test|success/test|Validation|loss" logs/vr_htv/*.log | tail -n 120
```

跑完后立刻补充：

- [ ] 每组 final success / final precision。
- [ ] 每组 best success / best precision。
- [ ] 同协议差值表：`A2-order-dyn - A1-order`。
- [ ] 判断 A2 的收益是否集中在 `gap1124` / `burst_drop` 这种强 variable-dt protocol。
- [ ] 将完整结果归档到 `done.md` 和 `compare_results/reports/virtual_rate_protocol_comparison.md`。

## 2. 跑完 6 组后优先做的结果整理

- [ ] 新增或完善 `tools/summarize_virtual_rate_protocols.py`。
- [ ] 输出 `compare_results/reports/virtual_rate_protocol_comparison.md`。
- [ ] 输出 protocol-level summary CSV：
  - [ ] `num_tracklets_before / after`
  - [ ] `num_frames_before / after`
  - [ ] `dropped_frame_ratio`
  - [ ] `mean_tracklet_len_before / after`
- [ ] 输出 time diagnostics CSV：
  - [ ] `delta_t mean / std / p50 / p75 / p95 / max`
  - [ ] `current_delta_t mean / p95 / max`
  - [ ] `gap_bins count`
  - [ ] `delta_t coefficient of variation`
- [ ] 输出 metric bins：
  - [ ] short / medium / long `delta_t`
  - [ ] sparse / medium / dense point count
  - [ ] small / large target displacement
  - [ ] valid-history vs incomplete-history

结果解读模板：

```text
protocol:
  gap1124 / burst_drop / random20

same-protocol comparison:
  A1-order final success / precision
  A2-order-dyn final success / precision
  A2 - A1 delta

claim boundary:
  如果只在 long-gap / sparse / large-displacement bin 提升，也可作为 timestamp-native dynamics 的证据；
  如果 overall final 不升，不要写成 full model 全面优于 SeqTrack3D。
```

## 3. 下一轮模型与配置待办

### 3.1 A2-residual-dyn

- [ ] 实现 `dynamics_motion_mode=residual_limited` 或等价配置。
- [ ] 增加 residual 限制参数：
  - [ ] `dynamics_residual_scale`
  - [ ] `dynamics_max_residual_norm`
  - [ ] `dynamics_warmup_epoch`
  - [ ] `dynamics_long_gap_only`
- [ ] 增加可选 residual 可靠性参数：
  - [ ] `dynamics_min_delta_t`
  - [ ] `dynamics_sparse_only`
  - [ ] `dynamics_min_obs_confidence`
  - [ ] `dynamics_max_alpha`
- [ ] 第一版不要再把 `z_dyn` concat 到 `motion_feature`；改成：

```text
obs_motion = motion_mlp(point_feature)
dyn_disp = velocity_pred * current_delta_t
dyn_disp = clamp_norm(dyn_disp, dynamics_max_residual_norm)
alpha_dyn = reliability_gate(...)
final_center = obs_center + dynamics_residual_scale * alpha_dyn * dyn_disp
```

- [ ] `alpha_dyn` 第一版可以先用规则或很小 MLP，不要复杂化：

```text
输入候选:
  current_delta_t / time_scale
  log1p(num_points_in_search)
  mean_fg_score
  dynamics_valid
  ||obs_motion[:3] - dyn_disp||
  valid_history_ratio

约束:
  0 <= alpha_dyn <= dynamics_max_alpha
  dynamics_valid=0 时 alpha_dyn=0
  warmup 前 alpha_dyn=0 或 scale=0
```

- [ ] 先做 2-step smoke test，确认 loss finite、residual norm 受限。
- [ ] 新增 3 个 HTV 配置：
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml`
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml`
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_random20.yaml`
- [ ] 如果 A2 feature-dyn 6 组结果不稳，优先跑 residual-dyn 的 gap1124 / burst_drop 对照。

初始网格建议：

```text
dynamics_residual_scale: 0.05 / 0.1 / 0.2
dynamics_max_residual_norm: 0.5 / 1.0
dynamics_warmup_epoch: 5 / 10
dynamics_max_alpha: 0.2
```

验收标准：

- [ ] 相比当前 `A2-order-dyn`，seed collapse 缓解，late mean 不再大幅低于 best。
- [ ] 即使 overall final 只持平，也要检查 long-delta_t / sparse / large-displacement bins 是否提升。
- [ ] 如果 residual 仍崩，优先怀疑 dynamics 监督质量、candidate history、速度量级或当前 `delta_t` 信号不足。

### 3.1b TrajTrack-style dyn proposal / refinement 备选

目标：把 dyn 从“特征增强分支”降级为“低维 trajectory proposal / fallback”，更接近 TrajTrack 的 `local proposal + global trajectory proposal + reliability refinement` 思路。

- [ ] 保留 observation branch 的 `obs_motion` 作为主预测。
- [ ] 新增 `dyn_motion_proposal`，只预测当前中心位移，不参与主特征 concat。
- [ ] 记录 `obs_dyn_center_gap = ||obs_motion[:3] - dyn_motion_proposal||`。
- [ ] 第一版 refinement 不用真值 IoU；用可训练时可得、推理时也可得的可靠性统计：
  - [ ] `num_points_in_search`
  - [ ] `mean_fg_score`
  - [ ] `current_delta_t`
  - [ ] `valid_history_ratio`
  - [ ] `obs_dyn_center_gap`
- [ ] 如果观测可靠且 `obs_dyn_center_gap` 小，信 `obs_motion`；如果 sparse / low-confidence / long-gap 且 dyn 合理，再加小幅 residual。
- [ ] 暂不实现完整 TrajFormer；如需轨迹模块，先做小 MLP / GRU / Transformer-lite 的 bbox-center-only 对照。是否引入轨迹模块！？
- [ ] 如果该方向有效，再考虑将 `DynamicsEncoder` 扩展为 timestamp-conditioned trajectory prior，而不是直接引入 TrajTrack 的完整 IMM。

### 3.1c Uncertainty / confidence-aware residual dynamics

目标：不要把 CT-SeqTrack 改成完整概率跟踪器，而是把不确定度/置信度作为“什么时候相信 observation、什么时候允许 dyn 小幅修正”的稳定器。最适合接在 `A2-residual-dyn` / `A2-dyn-proposal` 后面做，不放入第一批 HTV 主表。

外部依据：

- [ ] RLE / probabilistic regression 的启发：回归头可以预测误差分布或 `sigma/logvar`，用 NLL 类辅助损失让模型知道“自己不确定”。
- [ ] Probabilistic detection / bbox uncertainty 的启发：定位框不应只有一个 deterministic center，空间不确定度可以用于避免过度自信。
- [ ] UncTrack 的启发：localization uncertainty 可以作为可靠性信号，用于筛选高置信样本或控制更新。
- [ ] UA-Track 的启发：uncertainty-aware 机制更适合放在困难样本、噪声样本、遮挡/稀疏/小目标场景，而不是全样本强行接管主预测。

第一阶段 U0：先做“无结构改动”的置信度诊断。

- [ ] 在 `compute_loss` 里只记录诊断，不改模型输出：
  - [ ] `obs_center_err = ||obs_motion[:3] - center_label_motion||`
  - [ ] `dyn_center_err = ||dynamics_displacement_pred - center_label_motion||`
  - [ ] `obs_dyn_center_gap = ||obs_motion[:3] - dynamics_displacement_pred||`
  - [ ] `dyn_better_rate = mean(dyn_center_err + margin < obs_center_err)`
  - [ ] 按 `current_delta_t_ratio / num_points_in_search / mean_fg_score / valid_history_ratio` 分桶。
- [ ] 如果 dyn 只在 long-gap、sparse、low-fg-score 桶里更好，说明 residual/gate 方向成立；如果 dyn 在大部分桶都更差，先修 dynamics label / candidate 污染，不急着上 uncertainty head。

第二阶段 U1：新增轻量 `obs_uncertainty_head`，只作为 observation 可靠性估计。

- [ ] 在 `SEQTRACK3D.__init__` 增加独立 head，输入优先用 `point_feature`，不要用 concat 后的 dyn feature：

```text
obs_log_sigma = obs_uncertainty_head(point_feature)  # B,3
obs_log_sigma = clamp(obs_log_sigma, min=-5, max=2)
obs_uncertainty = exp(mean(obs_log_sigma))
obs_confidence = sigmoid(-mean(obs_log_sigma))
```

- [ ] 在 loss 里加很小权重的 heteroscedastic NLL 辅助项，第一版不替换原来的 `smooth_l1_loss`：

```text
err = obs_motion[:3] - center_label_motion
obs_log_var = 2 * obs_log_sigma
loss_obs_unc = 0.5 * err^2 * exp(-obs_log_var) + 0.5 * obs_log_var
loss_total += obs_uncertainty_loss_weight * mean(loss_obs_unc)
```

- [ ] 初始配置建议：

```text
use_obs_uncertainty: true
obs_uncertainty_loss_weight: 0.01 / 0.03 / 0.05
obs_uncertainty_warmup_epoch: 5
obs_uncertainty_log_sigma_min: -5
obs_uncertainty_log_sigma_max: 2
obs_uncertainty_detach_for_gate: true
```

- [ ] 验收：`obs_log_sigma` 不能全贴边，`obs_confidence` 与 `obs_center_err` 应该负相关；如果只学到常数 sigma，先保留日志，不进入 gate。

第三阶段 U2：先用 dynamics 不确定度 proxy，再考虑 dyn uncertainty head。

- [ ] 第一版不要急着给 `DynamicsEncoder` 加复杂概率头，先用可解释 proxy：

```text
dyn_unc_proxy =
  norm(obs_motion[:3] - dynamics_displacement_pred)
  + long_gap_penalty(current_delta_t_ratio)
  + invalid_history_penalty(1 - valid_history_ratio)
dyn_confidence = exp(-normalized_dyn_unc_proxy)
```

- [ ] 如果 U0 诊断证明 dyn 在某些桶里确实更准，再给 `DynamicsEncoder` 加 `dynamics_log_sigma_head(z_dyn)`：
  - [ ] 输出 `dyn_log_sigma`，监督目标仍是 `center_label_motion`。
  - [ ] 权重小于 observation uncertainty，避免 dynamics 为了降 NLL 只放大 sigma。
  - [ ] `dynamics_valid=0` 时不计算 dyn uncertainty loss。

第四阶段 U3：把 uncertainty / confidence 接入现有 `confidence_residual` gate。

- [ ] 保留当前 `obs_gate_fusion_mode=confidence_residual` 思路，不回到 `feature` 融合。
- [ ] 将 `obs_gate_num_stats` 从 5 扩到 8 或 9：

```text
已有 5 维:
  log1p(num_points_in_search)
  log1p(estimated_fg_points)
  mean_fg_score
  valid_history_ratio
  current_delta_t_ratio

新增候选:
  obs_confidence 或 mean(obs_log_sigma)
  dyn_confidence 或 dyn_unc_proxy
  obs_dyn_center_gap
  motion_cls_confidence
```

- [ ] gate 约束保持保守：

```text
0 <= alpha_dyn <= obs_gate_max_dyn_alpha
obs_gate_max_dyn_alpha: 0.1 / 0.2
obs_gate_residual_scale: 0.05 / 0.1
obs_gate_init_obs_bias: 3.0
dynamics_valid=0 -> alpha_dyn=0
warmup 前 alpha_dyn=0 或 residual_scale=0
```

- [ ] 预期行为：`obs_uncertainty` 高、`dyn_uncertainty` 低、`delta_t` 长、前景点稀疏时，`alpha_dyn` 可以略高；正常密集样本中 observation 仍占主导。

第五阶段 U4：实验顺序和论文表述边界。

- [ ] 实验顺序：
  - [ ] `A1-order`
  - [ ] `A2-order-dyn`
  - [ ] `A2-residual-dyn`
  - [ ] `A2-residual-dyn + obs_uncertainty_aux`
  - [ ] `A2-residual-dyn + uncertainty-gate-stats`
- [ ] 只在 `gap1124 / burst_drop` 先跑，不先扩到所有 protocol。
- [ ] 如果 uncertainty 只改善 calibration / 分桶解释，但 overall metric 不涨，论文中只写成“stability analysis / reliability-aware diagnostic”，不要写成主贡献。
- [ ] 如果 uncertainty-gate 在 long-gap / sparse bins 明显提升且 late checkpoint 更稳，才把它升级为 CT-SeqTrack 的稳定性模块。

实现优先级：

```text
P0: U0 diagnostics，最低风险，先证明 dyn 什么时候值得信。
P1: U1 obs_uncertainty_head，只做辅助 NLL 和日志。
P2: U3 gate stats 扩维，把 uncertainty 接进 confidence_residual。
P3: U2 dynamics_log_sigma_head，仅在 proxy 有效果后再加。
```

### 3.1d TrajTrack 的其他启发（非第一优先级）

这些启发不替代 `A2-residual-dyn` 主线，只作为后续诊断、轻量模块和论文叙事的参考。

- [ ] 困难子集评测优先级：TrajTrack 强调 sparse / occlusion 场景收益；CT-SeqTrack 后续结果也要优先汇报 long-gap、sparse point count、large displacement、re-appearance 子集，而不是只看 overall。
- [ ] 置信度来源不要依赖真值：TrajTrack 论文里用 local/global proposal 一致性做 refinement 依据；CT-SeqTrack 可借鉴 `obs_dyn_center_gap`、`mean_fg_score`、`num_points_in_search`、`motion_cls_confidence`、`obs_uncertainty`，但不要使用当前帧 GT IoU 做选择。
- [ ] bbox-only trajectory prior 可作为轻量备选：如果 `A2-residual-dyn` 稳定但收益有限，再尝试只用历史 bbox center + real `delta_t` 的小 MLP / GRU / Transformer-lite；不要一开始上完整 TrajFormer-VAE。
- [ ] proposal/refinement 要做成可插拔模块：保持 `A1-order` observation branch 可单独运行；dyn / trajectory prior 只通过 residual、fallback 或 confidence gate 接入，方便做 ablation。
- [ ] 不确定度可先从 head 诊断做起：TrajTrack 的 RLE head 输出 `sigma`，说明定位不确定度可以辅助判断 prediction 是否可靠；CT-SeqTrack 第一版只记录/辅助监督 `obs_uncertainty`，不要直接替换主损失。
- [ ] fair comparison 警惕：本地 TrajTrack 代码的 refinement 路径存在 GT-assisted 选择风险；引用它时只借鉴思想，不把当前代码结果直接作为公平强 baseline。
- [ ] 论文表述边界：TrajTrack 可作为“低维轨迹先验有效”的动机，但 CT-SeqTrack 的区别要写清楚：TrajTrack 是 frame-index trajectory prior，CT-SeqTrack 主打 real `delta_t/current_delta_t` 的 variable-rate dynamics prior。

### 3.2 timestamp negative controls

- [ ] `true-dt`：使用原始 timestamp，CT dynamics 能看到真实 gap。
- [ ] `fixed-dt`：同一 virtual-rate 帧序列上强制 `delta_t=dt_ref`。
- [ ] `jittered-dt`：对 timestamp 加小扰动，检查测量噪声鲁棒性。
- [ ] `shuffled-dt`：batch 内打乱 `delta_t/current_delta_t`，检查收益是否真的依赖时间。

### 3.3 formal manifest

- [ ] 为正式评测生成 train / val manifest，而不是只依赖 config seed 在线生成。
- [ ] manifest 文件名包含 split、category、mode、seed、max_gap。
- [ ] 所有模型复用同一 manifest，保证 protocol 完全一致。

## 4. 诊断日志待办

- [ ] dynamics candidate 诊断：
  - [ ] candidate0 与 nonzero candidate 的 velocity label 分布。
  - [ ] candidate0 与 nonzero candidate 的 residual / displacement error。
  - [ ] 判断 `num_candidates=4` 是否污染 dynamics 监督。
- [ ] dynamics 数值日志：
  - [ ] `dynamics_valid_ratio`
  - [ ] `velocity_label_norm`
  - [ ] `velocity_pred_norm`
  - [ ] `dynamics_displacement_norm`
  - [ ] `residual_norm`
  - [ ] `residual_scale_effective`
  - [ ] `alpha_dyn_mean / min / max`
  - [ ] `obs_dyn_center_gap_mean / p75 / p95`
  - [ ] `dyn_disp_clamp_ratio`
  - [ ] `dyn_residual_applied_ratio`
- [ ] search difficulty 日志：
  - [ ] `target_center_displacement mean / p75 / p95`
  - [ ] `out_of_search_ratio`
  - [ ] `num_points_in_search` sparse bins
  - [ ] re-appearance 片段统计
- [ ] TrajTrack-style proposal 诊断：
  - [ ] observation-only prediction 与 dyn proposal 的中心距离。
  - [ ] sparse / long-gap / low-foreground-confidence 样本中的 `alpha_dyn` 是否更高。
  - [ ] dyn residual 是否主要在困难样本启用，而不是全样本平均接管。
  - [ ] best checkpoint 与 final checkpoint 的 `alpha_dyn / residual_norm / obs_dyn_gap` 是否发生漂移。
- [ ] uncertainty / confidence 诊断：
  - [ ] `obs_log_sigma_mean / min / max`
  - [ ] `obs_confidence_mean`
  - [ ] `obs_center_err_mean`
  - [ ] `dyn_center_err_mean`
  - [ ] `dyn_better_rate`
  - [ ] `obs_uncertainty_error_corr`
  - [ ] `dyn_unc_proxy_mean`
  - [ ] long-gap / sparse / low-fg-score 桶内的 `alpha_dyn` 与 error 变化。

## 5. 后续决策树

### 5.1 如果 A2-order-dyn 在 HTV 上明显优于 A1-order

- [ ] 先补 best / final / late mean，确认不是单 checkpoint 偶然。
- [ ] 做 `fixed-dt` / `shuffled-dt` negative control。
- [ ] 做 long-gap / sparse / displacement 分桶，支撑论文叙事。
- [ ] 再上 `A2-residual-dyn`，检查是否能保留收益并提升稳定性；不要直接叠 TWC / gate。
- [ ] 如果 residual 版本稳定，继续做 `A2-dyn-proposal` 对照，确认是否比 feature-concat 更可控。

### 5.2 如果 A2-order-dyn 只在 random20 或普通整体上持平

- [ ] 优先看 burst_drop / gap1124 的 long-gap bins。
- [ ] 检查 search crop 是否过小导致所有方法一起退化。
- [ ] 不急着否定真实时间；先实现 residual-limited dynamics。
- [ ] 对 residual 版本优先启用 `long_gap_only` 或 `sparse_only`，用困难子集收益支撑 timestamp-native 叙事。

### 5.3 如果 A2-order-dyn 在三种 protocol 都明显退化

- [ ] 检查 dynamics 输入是否受 nonzero candidates 污染。
- [ ] 检查 velocity label / prediction norm 是否异常。
- [ ] 尝试 residual-only、long-gap-only 或 candidate0-only dynamics。
- [ ] 暂停 feature-concat dynamics；优先做 TrajTrack-style 低维 proposal / fallback。
- [ ] 若 proposal 仍无效，再判断真实 `delta_t` 在当前 nuScenes-mini 设置下是否信号不足。
- [ ] 暂停 TWC / gate 叠加。

## 6. 暂缓方向

- [ ] 暂不把 TWC / gate 放入第一批 HTV 主表。
- [ ] 暂不切换到 MambaTrack3D / TrackM3D / TrajTrack 作为主 baseline；TrajTrack 仅作为 dyn 设计参考。
- [ ] 暂不上 Neural ODE / SDE / CDE。
- [ ] 暂不主打任意时间查询或多传感器异步融合。
- [ ] 频域 / 谱域方向只保留为后续诊断候选。
