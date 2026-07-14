# CT-SeqTrack 当前执行清单

更新时间：2026-07-11

本文只维护尚未完成的任务。已完成工程改动见 `done.md`，历史与当前结果口径见 `sum_results.md`，研究定位和论文边界见 `refined_plan.md`。

## 0. 当前主线与结论边界

```text
P0 先完成 corrected-TWC 与 bounded residual 的服务器真实数据验收；
P1 再做最小、配对、可归因的 corrected-TWC 和 residual 实验；
P2 用 true-dt / fixed-dt / shuffled-dt、多 seed 和困难分桶决定是否进入 full data；
P3 只有因果证据成立后，才考虑 gate、uncertainty 或更复杂 trajectory prior。
```

当前必须遵守：

- 旧 active-TWC 的 nonzero candidate 两路坐标系不同，旧正向与负向效果归因全部撤回；不能把旧 A1 precision 增益或 A2 崩坏写进新结论。
- `A2-residual-dyn` 已实现，但只有本地纯逻辑 smoke test；不能写成“已有效”。
- mini 数据只用于筛选假设和排错，不能作为正式论文主结果。
- 同一对照必须使用相同 protocol、manifest、candidate 数、optimizer steps、checkpoint 选择规则和 seed。
- 不再同时叠加 TWC、residual 和 gate。一次只回答一个因果问题。
- 先检查 target-in-crop recall / out-of-search ratio；目标已离开固定 search crop 时，dynamics 再强也无法恢复。

术语：

| 名称 | 当前含义 |
| --- | --- |
| `A1-order` | order-time 主干，无 dynamics / TWC / gate |
| `A2-order-dyn` | 旧 feature-concat dynamics，真实时间只进入 `DynamicsEncoder` |
| `A2-residual-dyn` | observation-only motion head + 有界、近零初始化、真实时间驱动的中心 residual |
| `corrected-TWC` | 共享绝对 frame candidate offset、crop、`coordinate_anchor` 和训练步数的 TWC |
| `gap1124` | gap pattern `[1,1,2,4]` |
| `burst_drop` | 连续保留与连续丢帧交替的强 variable-rate 协议 |
| `random20` | 固定 seed 随机丢 20%，相对温和 |

## 1. P0：服务器工程验收

本机缺少 `easydict / nuscenes-devkit`，以下真实数据检查必须在服务器 `/home/lishengjie/study/lcyu/CT-SeqTrack` 环境完成。任何正式训练都以本节全部通过为前提。

### 1.1 Corrected-TWC paired batch

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

python tools/check_twc_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 4 \
  --workers 0 \
  --require-full-history
```

不要传 `--candidate-zero-only`。验收项：

- [ ] batch 明确覆盖 `candidate_id=0/1/2/3`，A/B candidate id 完全一致。
- [ ] `sampler.num_candidates == cfg.num_candidates == 4`。
- [ ] `len(dataset) == base_frames * num_candidates`，开启 TWC 不减少 steps per epoch。
- [ ] `coordinate_anchor` A/B gap 不超过 `twc_anchor_eps=1e-4`。
- [ ] A/B 共同绝对历史帧的 `candidate_offsets` 完全共享。
- [ ] A/B 共同绝对历史帧和当前帧的 point-sampling seed 共享，最终 XYZ gap 不超过 `1e-6`。
- [ ] candidate 1/2/3 的 anchor offset 非零，但 A/B gap 为零。
- [ ] `num_points_in_search` A/B 相同；`current_timestamp` 相同；历史路径确实不同。
- [ ] 所有 full-history 样本 `twc_valid=True`。

负例保护：

- [ ] 在临时测试中给 view B 的 `coordinate_anchor[:, 0] += 0.01`，`compute_twc_loss()` 必须报错。
- [ ] 临时删除 `coordinate_anchor`，默认配置必须报 `KeyError`，不能退回旧的归一化 `ref_boxs[0]` 检查。

### 1.2 Corrected-TWC forward/loss/backward

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history

CUDA_VISIBLE_DEVICES=0 python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --seed 42 \
  --grad-clip 1.0 \
  --tag corrected_twc_smoke
```

- [ ] forward、supervised loss、`loss_twc`、gradient 全部 finite。
- [ ] `twc_valid_ratio=1`（full-history smoke batch）。
- [ ] `twc_anchor_gap_max <= 1e-4`。
- [ ] `twc_current_point_gap_max <= 1e-6`。
- [ ] 两步训练没有 OOM、NaN、递归重采样异常。

### 1.3 A2-residual-dyn forward/loss/backward

先验收 standard 和最强 variable-rate 两个配置：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history

CUDA_VISIBLE_DEVICES=0 python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history

CUDA_VISIBLE_DEVICES=0 python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --seed 42 \
  --grad-clip 1.0 \
  --tag a2_residual_gap1124_smoke
```

- [ ] `motion_mlp` 输入仍为 256 维，residual 模式没有 concat `z_dyn`。
- [ ] epoch 0 warmup 时 `dynamics_residual_scale_effective=0`。
- [ ] 临时令 `current_epoch >= 5` 后，alpha 有界于 `[0, 0.2]`。
- [ ] `dynamics_residual_clamped_norm <= 1.0`。
- [ ] 最终 residual norm 不超过 `scale * max_alpha * max_norm = 0.02`。
- [ ] `dynamics_valid=0` 时 residual 严格为零。
- [ ] loss、gradient 和全部诊断量 finite。

## 2. P1：最小可归因实验

### 2.1 Corrected A1-order+TWC

只有 P0 全部通过后才启动：

- [ ] 用修复后的代码原样重跑 `A1-order` 与 `A1-order+TWC`，保持 candidate4、总 optimizer steps、seed 和 checkpoint 规则一致。
- [ ] 第一轮只用 `twc_weight=0.05`，不要先扫权重。
- [ ] 至少 seed 42/43/44 三个配对 seed，报告 mean±std 和 tracklet-level bootstrap CI。
- [ ] 除 Success/Precision 外，直接报告同一 endpoint 下 A/B `center_gap / angle_gap / prediction variance`。
- [ ] 若 corrected-TWC 不能显著降低路径方差，不再做 A2+TWC。
- [ ] 若降低方差但不涨 tracking metric，只把 TWC 写成稳定性分析，不写成主增益模块。

### 2.2 Residual mini 筛选矩阵

先只跑 `gap1124` 与 `burst_drop`，每个协议使用完全相同的 virtual-rate manifest：

| 模型/时间条件 | seed | 目的 |
| --- | --- | --- |
| A1-order | 42/43/44 | observation baseline |
| A2 feature-concat true-dt | 42/43/44 | 旧 dynamics 方式 |
| A2 residual true-dt | 42/43/44 | 主假设 |
| A2 residual fixed-dt | 42/43/44 | 同容量时间负对照 |
| A2 residual shuffled-dt | 42/43/44 | 检查物理时间是否真正被使用 |

待新增配置/开关：

- [ ] `fixed-dt`：帧序列保持不变，只把 dynamics 的 `delta_t/current_delta_t` 固定为 reference step。
- [ ] `shuffled-dt`：使用固定 manifest 离线打乱时间字段，不能让每个 epoch 随机变化。
- [ ] `frame-index-dt`：可选，使用离散 offset 作为时间，区分帧序与物理秒数。
- [ ] 所有对照保持模型参数量、训练帧、crop 和 optimizer steps 完全相同。

第一轮不要扫大网格。默认只测：

```yaml
dynamics_residual_scale: 0.1
dynamics_max_residual_norm: 1.0
dynamics_max_alpha: 0.2
dynamics_warmup_epoch: 5
dynamics_long_gap_only: false
dynamics_sparse_only: false
```

只有默认设置出现跨 seed 正信号，才补 `scale=0.05/0.2` 或 `long_gap_only/sparse_only`。

## 3. P2：协议、诊断与公平性

### 3.1 核对历史 HTV 任务状态

旧文档记录过 6 个后台任务，但本地无法确认 2026-07-11 的服务器进程状态。不要沿用旧 PID 推断“仍在运行”。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
ps -ef | grep "htv_" | grep -v grep
nvidia-smi
grep -iE "error|exception|traceback|cuda out|nan|killed" logs/vr_htv/*.log
```

- [ ] 核对 gap1124/burst_drop/random20 的 A1/A2 六组是否完成、失败或只有 partial checkpoint。
- [ ] 为每个 run 记录 git commit、cfg hash、manifest hash、seed、样本数、steps/epoch、best/final checkpoint。
- [ ] 完成的历史 run 只作为筛选证据；新 residual 因果矩阵必须使用冻结 manifest 重跑。

### 3.2 必须补的诊断

- [ ] 时间分布：`delta_t/current_delta_t mean, std, p50, p75, p95, max`。
- [ ] 分桶指标：short/medium/long gap、sparse/medium/dense points、small/large displacement、full/incomplete history、re-appearance。
- [ ] crop 可达性：target-in-crop recall、out-of-search ratio，并按 gap 分桶。
- [ ] residual：alpha mean/min/max、raw/clamped norm、clamp/applied ratio、`obs_dyn_center_gap`。
- [ ] oracle 诊断：`obs_center_err`、`dyn_center_err`、`dyn_better_rate`；oracle 只能分析，不能进入推理 gate。
- [ ] candidate 诊断：candidate0 与 nonzero candidate 的 velocity label、dyn error 和 residual error。
- [ ] 稳定性：best/final/late-mean、3 seed mean±std、tracklet bootstrap CI。
- [ ] 效率：参数量、FLOPs、FPS、residual 分支额外开销。

### 3.3 Formal manifest 与 checkpoint 规则

- [ ] 为 train/val/test 生成 split-specific manifest，文件名包含 dataset、split、category、protocol、seed、max-gap。
- [ ] A1、feature dyn、residual dyn 和 TWC 复用同一份 manifest。
- [ ] 预先规定 checkpoint 选择规则，不能看 test 后选择 epoch。
- [ ] mini_val 只用于开发；正式数据需要独立 val 选 checkpoint、held-out test 一次评测。

## 4. Go / Stop 门槛

### 4.1 Go 到完整数据

mini 阶段需同时满足：

- [ ] residual true-dt 在 `gap1124`、`burst_drop` 至少两项中，相对配对 A1 和 fixed/shuffled-dt 平均约提升 `>= +1.5 Success / +2 Precision`。
- [ ] 3 seed 无单 seed 大崩溃，配对置信区间不跨 0。
- [ ] regular fixed-step 退化不超过约 `0.5 Success / 1 Precision`。
- [ ] shuffled/fixed 时间后收益明显消失。
- [ ] 收益不能由更大 crop、更多参数、更多 optimizer steps 或不同 checkpoint 规则解释。

通过后再做：

- [ ] 完整 nuScenes trainval，至少两个类别，最好官方全部类别。
- [ ] 第二数据集或官方 KITTI-HV/nuScenes-HTV 协议。
- [ ] 与 SeqTrack3D、HVTrack、StreamTrack、Motion-to-Matching/trajectory baseline 和简单 constant-velocity/Kalman baseline 公平比较。
- [ ] 测试未见过的 cadence/drop schedule，证明一个模型跨采样率泛化，而非每个 stride 单独训练。

### 4.2 Stop / Pivot

满足任一项则停止把真实 `delta_t` 作为主算法贡献：

- 连续两轮配对 3-seed 中，`true-dt - fixed/shuffled-dt < 0.5 Success / 1 Precision`。
- 收益只来自扩大 crop、额外参数、更多训练步或 protocol-specific retraining。
- full data 的方向与 mini 正信号相反。
- corrected-TWC 不降低同 endpoint 的路径预测方差。

Pivot 选项：

- 把项目转成 variable-rate 3D SOT benchmark / diagnosis 工作。
- 把 dynamics 降级为可靠 trajectory proposal 或 constant-velocity/Kalman fallback。
- 只有 residual 明确有效后，才尝试 bbox-only MLP/GRU/Transformer-lite；不上完整 TrajFormer、ODE/SDE/CDE。

## 5. 暂缓项

- [ ] Observability gate、uncertainty head 和 residual 不同时开发；等待 residual 因果结果。
- [ ] 不把 corrected-TWC 接入 A2/residual，直到 corrected A1+TWC 有三 seed 干净证据。
- [ ] 不切换到 Mamba/SSM、Neural ODE/SDE/CDE 或多传感器异步融合。
- [ ] 不使用无法核验的 `TrackM3D` 名称；相关思路统一写作 Kalman/continuous-time state estimation。
- [ ] 不把 TrajTrack-style GT-assisted proposal selection 当作公平 baseline；推理 gate 只能使用测试时可得统计量。
