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

术语：

| 名称 | 含义 |
| --- | --- |
| `A1-order` | 主干 order-time，无 dynamics / TWC / gate |
| `A2-order-dyn` | 主干 order-time，真实时间只进入 `DynamicsEncoder` |
| `A2-residual-dyn` | 待实现的保守 residual dynamics 版本 |
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
- [ ] 先做 2-step smoke test，确认 loss finite、residual norm 受限。
- [ ] 新增 3 个 HTV 配置：
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml`
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml`
  - [ ] `cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_random20.yaml`
- [ ] 如果 A2 feature-dyn 6 组结果不稳，优先跑 residual-dyn 的 gap1124 / burst_drop 对照。

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
- [ ] search difficulty 日志：
  - [ ] `target_center_displacement mean / p75 / p95`
  - [ ] `out_of_search_ratio`
  - [ ] `num_points_in_search` sparse bins
  - [ ] re-appearance 片段统计

## 5. 后续决策树

### 5.1 如果 A2-order-dyn 在 HTV 上明显优于 A1-order

- [ ] 先补 best / final / late mean，确认不是单 checkpoint 偶然。
- [ ] 做 `fixed-dt` / `shuffled-dt` negative control。
- [ ] 做 long-gap / sparse / displacement 分桶，支撑论文叙事。
- [ ] 再上 `A2-residual-dyn`，检查是否能保留收益并提升稳定性。

### 5.2 如果 A2-order-dyn 只在 random20 或普通整体上持平

- [ ] 优先看 burst_drop / gap1124 的 long-gap bins。
- [ ] 检查 search crop 是否过小导致所有方法一起退化。
- [ ] 不急着否定真实时间；先实现 residual-limited dynamics。

### 5.3 如果 A2-order-dyn 在三种 protocol 都明显退化

- [ ] 检查 dynamics 输入是否受 nonzero candidates 污染。
- [ ] 检查 velocity label / prediction norm 是否异常。
- [ ] 尝试 residual-only、long-gap-only 或 candidate0-only dynamics。
- [ ] 暂停 TWC / gate 叠加。

## 6. 暂缓方向

- [ ] 暂不把 TWC / gate 放入第一批 HTV 主表。
- [ ] 暂不切换到 MambaTrack3D / TrackM3D / TrajTrack 作为主 baseline。
- [ ] 暂不上 Neural ODE / SDE / CDE。
- [ ] 暂不主打任意时间查询或多传感器异步融合。
- [ ] 频域 / 谱域方向只保留为后续诊断候选。
