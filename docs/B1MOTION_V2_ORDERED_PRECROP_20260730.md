# B1motion-v2：有序轨迹、crop 前搜索与零初始化接入

> **TrajTrack 参考状态（2026-08-16）**：自本标注起，TrajTrack 不再作为
> CT-SeqTrack 后续方法设计、Gate/proposal 机制选择、超参数设定或性能有效性的
> 参考依据；仅保留为必须引用的相关工作、历史审计对象和 GT-free 评测警示。
> 下文既有 TrajTrack 内容均为历史记录，不再驱动当前或未来方案。

日期：2026-07-30

## 60-epoch 实验结论（已完成）

**当前 B1motion-v2 不涨点，判定为
`NO_GO_CURRENT_B1MOTION_V2 / INCONCLUSIVE_MOTION_DIRECTION`。**

seed42、nuScenes-mini Car、scratch 60 epoch 的 normal validation：

| arm | final Success | final Precision | best Success/Precision | late-3 Success/Precision |
|---|---:|---:|---:|---:|
| B0 baseline（历史） | 53.360 | 64.382 | 54.135 / 64.382 | 52.905 / 63.104 |
| legacy motion α=0 | 47.049 | 49.184 | 49.876 / 58.691 | 46.828 / 49.669 |
| legacy motion α=0.25 | 29.581 | 28.862 | 35.027 / 41.130 | 29.472 / 28.849 |
| **B1motion-v2** | **20.618** | **19.830** | **30.196 / 34.990（epoch5）** | **21.777 / 21.195** |

B1motion-v2 相对 B0 final 下降 **32.742 Success / 44.551
Precision**；最佳 checkpoint 也未通过 normal 守门线。训练有
75,720/75,720 个 scalar step、12/12 个验证点，`last.ckpt` 为 epoch60 /
global step75,720，因此不是运行截断或坏 final checkpoint。

本轮目录中没有 B1motion-v2 的 random20/gap1124 输出，不能声称 irregular
协议涨点。即使后续诊断发现某个 irregular subgroup 有收益，当前 normal
退化幅度也不允许模型晋级。

完整曲线、训练诊断、代码合同和后续 kill-test 见
[`B1motion-v2 seed42 分析`](../compare_results/reports/b1motion_v2_seed42_20260730.md)。

## 决策

`cfgs/ct_v2/02_ct_motion.yaml` 已从被否决的固定
`proposal_innovation(alpha)` 改为：

1. recent-to-old 历史框经过 oldest-to-newest GRU；
2. 因果运动估计在当前帧 crop 之前生成第二搜索区域；
3. B0 原始 1024 个 current tokens 完整保留，第二 crop 使用独立 128 点分支；
4. motion 只通过最后一层全零的 feature adapter 接入；
5. 不再用固定 alpha 改写 observation displacement；
6. 训练由 65% 连续 cadence 和 35% 真实跳帧 cadence 组成；
7. 非 candidate0 历史使用随年龄增长的 correlated drift，近似递归预测误差；
8. ordered history、endpoint/velocity target 与主干点云统一表达在实际
   candidate crop anchor 中，而不是不可用的最新 GT anchor 中。

上述工程目标均已接入并通过初始化/梯度 smoke test，但 60-epoch 结果证明：
**局部合同正确不等于整个训练合同正确。** 当前实现不应原样复跑。

旧 B1 的可复现实验配置冻结为
`cfgs/ct_v2/02_ct_motion_legacy_fixed.yaml`，历史 alpha0/0.25 和
B2/B3 配置继续继承 legacy 文件。

## 失败原因复核

### 1. mixed cadence 先破坏了 B0 主路径

`trajectory_adapter_warmup_epoch=2`，epoch1–2 adapter correction 精确为
0；但 epoch1 total loss 已是 7.721，B0 为 4.804，center loss 为
0.491 vs 0.182。zero-init 确实生效，退化却在 adapter 打开前出现。

原因是 `trajectory_training_irregular_probability=0.35` 改变的不只是
trajectory branch，而是整个主干的历史点云、历史框、motion label 和
Transformer 序列。主干仍使用 `main_time_source=order`，不会看到真实
frame gap；相同 order token 因而对应不同物理运动尺度。这违反了“保留 B0
identity path”的训练分布前提。

### 2. candidate-anchor target 对 trajectory-only encoder 不可识别

ordered encoder 只接收以最新 candidate anchor 归一化的相对历史框，而
`anchor_relative_trajectory_targets()` 把 `current GT − candidate anchor`
作为 displacement/velocity target，包含最新 anchor 的绝对平移误差。

对全部历史框与 anchor 加相同平移时，相对历史输入保持不变，target 却改变。
因此这个 correction 项不可能由 trajectory-only head 唯一恢复。它把应由
当前点云 observation/refinement 处理的定位误差混入“物理速度”，使
velocity/displacement 辅助监督长期平台化，并把有噪表示送入 adapter。

### 3. zero-init 只保护 step-0，normal scale 不是范数上限

adapter 从 epoch3 启用后 correction L2 立即达到 1.859，epoch5 为 2.619，
epoch60 仍为 2.072。`trajectory_adapter_normal_scale=0.1` 只是 MLP 输出
的乘数，不限制最终 correction norm。epoch60 raw norm² penalty 为 19.017，
乘 `trajectory_adapter_l2_weight=1e-4` 后只贡献约 0.0019，无法把模型约束
在 B0 feature 邻域。

### 4. pre-crop search 实际有效率过低

训练期 `trajectory_search_valid` 均值仅 3.93%，远低于 35% irregular
sampling。多数样本未获得至少 16 个 extension points，第二 crop 很少提供
额外观测，既不能补偿主干分布破坏，也不足以证明 random/gap 鲁棒性。

### 5. 次要数值卫生问题

无有效 transition 的样本会把 `nominal_gap` clamp 到 0.001，使 epoch60
记录的 `trajectory_gap_ratio` 均值达到 119.8。最终 correction 会由
`trajectory_valid=0` 清零，所以它不是主崩溃来源；但应将 invalid ratio
置 1、对 valid ratio 设置上限，并分别记录 trigger/available/applied。

`trajectory_nll` 为负数本身不是 bug，Gaussian NLL 的 `log_sigma` 项允许
负值；真正应检查的是 endpoint RMSE、velocity error 与 uncertainty
calibration。

## 修正后的下一步

不再直接启动另一个 60-epoch B1。先做：

1. 当前最终代码上的 same-code B0 seed42 scratch；
2. `irregular_probability=0/0.35 × adapter off/on` 的 10–15 epoch
   factorial kill-test，search 先关闭；
3. 连续 B0 view 保持主监督，irregular history 改成 paired auxiliary view，
   不再替换主路径；
4. trajectory head 只预测 canonical physical motion，candidate-anchor
   correction 交给读取当前点云证据的 refinement/correction head；
5. normal adapter 永久 exact identity；irregular residual 使用相对 feature
   norm 硬上限和 GT-free evidence gate；
6. search 日志拆分为 geometric trigger、extension availability、foreground
   evidence 和 applied，并先做离线 target-recall 审计。

只有 normal 相对 same-code B0 达到 Success ≥ −0.3、Precision ≥ −0.5，
且 core loss 不再从早期分叉，才允许继续 60 epoch、random20 和 gap1124。

## 为什么不是直接复刻外部代码

M²-Track 的可复用原则是“点云证据产生粗 motion proposal，再由第二阶段
refinement”，而不是单独把历史速度加到最终框。官方实现见：

- https://github.com/Ghostish/Open3DSOT/blob/master/models/m2track.py
- https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.pdf

TrajTrack 可复用的是有序 trajectory encoder 和 local/global proposal
互补思想。公开代码的 `pre_w_refine` 路径在评测中使用当前 GT IoU 触发
refinement，并用 GT IoU 在多个预测中选最大项，因此该段不能作为在线
GT-free tracker 直接移植：

- https://github.com/FiBonaCci225/TrajTrack/blob/main/models/trackers/trajtrack.py
- https://arxiv.org/abs/2509.11453

本实现没有读取当前 GT 来触发、扩区或选择预测。

## 代码路径

### 有序神经编码

`models/ct_v2/motion.py::OrderedTrajectoryEncoder`

- 输入：实际最新 candidate/predicted crop anchor 坐标中的
  `[x,y,z,yaw]`、真实 `delta_t` 和 valid mask；
- transition 特征：位移、速度、`sin/cos(dyaw)`、角速度、gap 和 query ratio；
- 顺序：输入框是 recent-to-old，GRU 显式翻转为 oldest-to-newest；
- 冷启动：最近有效速度提供 kinematic base；
- 学习项：零初始化 rate residual；
- 输出：`xyz+yaw` displacement、`log_sigma`、ordered feature 和 valid。

因此 `[1,3]` 与 `[3,1]` 的速度序列不再得到相同表示。

### crop 前轨迹搜索

`utils/ct_search.py::build_ordered_trajectory_search_box`

CPU/dataloader 与递归评测共同使用同一个无学习、无 GT 的因果估计：

`recent velocity + bounded acceleration -> displacement + uncertainty proxy`

触发条件默认是：

- `current_dt >= 0.75 s`，或
- `current_dt / mean(history_dt) >= 1.5`。

正常 0.5 s cadence 默认不扩区。random20 的 1.0 s gap 和 gap1124 长 gap
会触发。搜索长度/宽度同时受位移和 uncertainty 控制，并有硬上限。

`sample_search_extension` 只采第二 crop 相对 baseline crop 的新增点。原始
B0 current points 仍按原路径采样 1024 点，避免旧 Search 分走 25% baseline
tokens。

### 零初始化残差

`models/ct_v2/motion.py::ZeroInitTrajectoryAdapter`

输入包括：

- observation feature；
- ordered trajectory feature；
- 第二 crop 点特征；
- 四维 `log_sigma`；
- gap ratio 与第二 crop valid。

最后一层 weight/bias 全零，因此 step-0 输出严格等于 observation feature。
normal cadence 最大相对 scale 默认 0.1，irregular gap 可逐步升到 1.0。
不存在固定 proposal alpha。

### 训练 cadence 与历史误差

`datasets/sampler.py::MotionTrackingSamplerMF`

- 概率 0.65：历史 offsets `[1,2,3]`；
- 概率 0.35：query gap 从 `[2,4]` 采样，历史 transition 按
  `[1,1,2,4]` 循环；
- 使用实际跳过帧及其真实 timestamp，不伪造时间；
- `recursive_candidate` 将 correlated candidate error 按历史年龄放大；
- ordered history 与 `xyz+yaw` endpoint/velocity target 全部位于实际 crop
  anchor 坐标；因此 candidate1–3 会监督“物理运动 + 最新 anchor 误差修正”，
  与递归推理一致；
- canonical GT 只产生训练 target，不参与评测 crop/trigger。

它仍是递归误差模拟，不等价于完整 frozen-B0 rollout。首轮通过后，下一步应
导出训练集 B0 recursive histories 替换模拟分布。

## 结构性检查

运行：

```bash
python -m pytest tests -q
python tools/ct_v2/check_motion_v2_initialization.py --seed 42
```

当前结果：

- 43 个测试通过（含实际 crop-anchor 位移/旋转 target 合同）；
- B0 的 320 个共享 state tensors 与 B1motion-v2 step-0 完全相同；
- 新增 24 个 tensors 仅属于 dynamics encoder、trajectory search encoder
  和 trajectory adapter；
- adapter 初始 correction 精确为 0；
- ordered residual/uncertainty head 与 adapter 零初始化末层均可收到 finite、
  non-zero first-update gradient。

## 训练与评测

先做初始化检查和 5 epoch kill-test：

```bash
python tools/ct_v2/check_motion_v2_initialization.py --seed 42
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python tools/ct_v2/run.py train --variant motion --seed 42 \
  --epochs 5 --batch-size 16 --workers 4 --preloading \
  --check-val-every-n-epoch 5 --save-top-k 5 \
  --tag b1motion-v2-kill5-seed42
```

kill-test 通过后从头运行正式 60 epoch（不要续训 kill-test）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python tools/ct_v2/run.py train --variant motion --seed 42 \
  --epochs 60 --batch-size 16 --workers 4 --preloading \
  --check-val-every-n-epoch 5 --save-top-k 5 \
  --tag b1motion-v2-60ep-seed42
```

GPU 2 单卡后台正式运行：

```bash
mkdir -p run_logs
TAG=ctv2_b1_motion_v2_car_seed42_60ep_bs16_gpu2_thread1_scratch
LOG=run_logs/${TAG}_$(date +%Y%m%d_%H%M%S).log
nohup env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  python -u tools/ct_v2/run.py train --variant motion --seed 42 \
  --epochs 60 --batch-size 16 --workers 4 --preloading \
  --check-val-every-n-epoch 5 --save-top-k 5 \
  --path /home/lishengjie/data/nuscenes-mini --tag "$TAG" \
  >"$LOG" 2>&1 &
PID=$!
echo "$PID" | tee "${LOG%.log}.pid"
echo "PID=$PID LOG=$LOG"
```

训练完成后同一 final checkpoint 评测：

```bash
python tools/ct_v2/run.py test --variant motion --seed 42 \
  --checkpoint <last.ckpt> --protocol normal
python tools/ct_v2/run.py test --variant motion --seed 42 \
  --checkpoint <last.ckpt> --protocol random20
python tools/ct_v2/run.py test --variant motion --seed 42 \
  --checkpoint <last.ckpt> --protocol gap1124
```

建议先通过 5 epoch kill-test，再启动 60 epoch。第一轮不扫 alpha，因为该参数
已经不参与新路径；也不同时修改点数、gap trigger、loss weight。

## 不能提前保证的内容

代码可以保证 baseline identity、无 GT 在线路径和 motion 在 crop 前可用，
但不能在训练前保证 normal 指标一定上涨。晋级仍要求同一 checkpoint：

- normal 不低于 B0 `-0.3 Success / -0.5 Precision`，目标为持平或为正；
- random20 与 gap1124 各至少 `+1 / +2`；
- 至少一个 irregular 协议达到 `+2 / +4`；
- seed42 通过后才运行 seed43/44。
