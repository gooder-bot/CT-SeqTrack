# v31 mini 三组实验：实现状态与启动命令

2026-09-19。当前活动版本是 v31；本页覆盖旧 v30 启动排程。服务器本轮只读，修改只在本地，尚未上传或启动训练。

本轮核对记录见 [就绪报告](CTSEQTRACK_V31_READINESS.md)：89 passed/1 skipped，三臂均通过真实Lightning2.0.2的本地合成帧3epoch训练与最终窗口评测。上传包为 `artifacts/ct_checks/ct_v31_mini_20260919_upload.zip`，在服务器项目根按相对路径同步后再运行以下命令。

## 实现与验收边界

- 独立入口：`main.py` 按 `net_model: ctseqtrackv31` 分流到 `models/ct_v31/entry.py`；历史 host/标定不进入新路径。
- B0 预测当前锚点坐标下的框，XYZ 与 sin/cos；软前景只加权特征，唯一真实点参与统计。
- B1 负责物理先验、获取和可微时序 context；时序表示有效性与速度对有效性分开。按本次要求增加 GRU 对照，与 CfC 使用相同时间输入、公共头、损失和预算。
- B2 固定 768→256 新增点、L/R 分区、36 槽原始身份记忆、三模式 XYZ 投票；最终四假设共用定位和质量头。不存在旧 0.75 m 动作上限或未标定就全退 B0 的策略。
- 四条独立状态分支每端点共四次暴露；真实帧 worker 读取，裁剪/获取在主进程；同一窗口状态串行提交，跨帧 detach。
- Adam 一次联合更新，FP32，lr=1e-4、betas=(0.5,0.999)、eps=1e-6，StepLR 每20轮×0.1，60轮/batch16/workers4/seed42/val5。
- 每两轮保存 epoch 边界 checkpoint，最后三轮全部保存；训练后自动闭环评测58/59/60，写入 `results.json`，固定 final60 与 late-3。只允许相同身份的完整 epoch 边界续训。

本地合同检查和 Lightning 2.0.2 合成原始帧集成检查不能替代服务器真实 mini/CUDA 100步与峰值显存测量。该项及正式60轮、指标提升尚未完成；本次按用户要求不给服务器写文件或启动任务，不增加长预检门。

## 服务器只读核对

项目 `/home/lishengjie/study/lcyu/CT-SeqTrack`，检查时 HEAD=`1c078a7`，只有部分 v31 模型代码，缺本轮新入口/配置。必须先上传本地补齐文件。

环境 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`：Python3.9.19，PyTorch2.0.1+cu118，Lightning2.0.2，CUDA可用，nuscenes/pointnet2_ops/tensorboard存在。数据根 `/home/lishengjie/data/nuscenes-mini`。

已在服务器现有代码上只读执行一次真实raw样本的准备、CPU先验与获取接口：官方mini_val为106轨迹、2285帧，点张量 `[1,4,1024,5]`，该样本唯一点数 `[0,0,1,0]`、extension17点、真实dt=0.499874秒。没有创建optimizer或checkpoint；这项只证明真实数据接口，不代表新版完整CUDA训练通过。

GPU0/1均 A40 46068MiB；核对时占用约1716/4245MiB。两Full按要求共用GPU0，运行成本受竞争影响，不能作为独占GPU速度比较。尚未实测新网络两Full并行峰值。

## 参数说明

三份配置是 `31_b0_mini.yaml`、`31_full_cfc_mini.yaml`、`31_full_gru_mini.yaml`，共用 `31_formal_base.yaml`。不能使用用户示例中的旧 `seqtrack3d_nuscenes_mini.yaml`，该文件当前仓库不存在。

省略旧 `--preloading`：v31 worker 读原始帧，使用每worker256MiB有界点云缓存；旧预处理缓存不符合当前按接受状态裁剪的流程。其余所列训练参数保持不变。每个进程只用一张可见卡；`CUDA_VISIBLE_DEVICES`指定物理卡号，`trainer_devices=1`指设备数量。

下列命令由用户在上传后运行。它们各自立即后台启动，三条依次执行即可并行，不需要等待上一条训练结束。

### B0 → GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_b0-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_b0_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### Full-CfC → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_full_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_full_cfc_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### Full-GRU → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_full_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_full_gru_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

## 新终端查看进度

先连接并进入项目：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

分别选一条；自动选取对应模块最新日期目录，Ctrl+C仅结束tail：

```bash
tail -n 80 -f "$(ls -dt output/*-31_b0-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
tail -n 80 -f "$(ls -dt output/*-31_full_cfc-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
tail -n 80 -f "$(ls -dt output/*-31_full_gru-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

日志开头的 `[v31]` 会记录 arm、temporal_backend、B1/B2/B3 启用情况、参数量及 calibration_required=false；后续 `[v31 train]` 每50个batch打印进度，每次验证打印完整统计。`run_manifest.json`、`resolved_config.yaml`、CSV/TensorBoard和最终评测按run分开保存。

Full走联合模型、CfC/GRU走不同循环单元，这是防止旧机械同分的实现保证；不能据此预先保证两组测量分数一定不同或一定上涨。结果仍按相同数据、预算与指标比较，保留历史40.473742/47.840262参照。
