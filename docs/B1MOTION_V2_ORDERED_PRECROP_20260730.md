# B1motion-v2：有序轨迹、crop 前搜索与零初始化接入

日期：2026-07-30

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

旧 B1 的可复现实验配置冻结为
`cfgs/ct_v2/02_ct_motion_legacy_fixed.yaml`，历史 alpha0/0.25 和
B2/B3 配置继续继承 legacy 文件。

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
