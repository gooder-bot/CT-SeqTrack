# v36 五组启动、验证与恢复

本轮只有A/C/D/E/F五次scratch：mini Car、seed42、LR1.5e-4、batch16、FP32、60轮、20/50各乘0.1、无warmup。每次71700更新，总预算300epoch/358500更新。结构与监督见[实现说明](CTSEQTRACK_V36_DESIGN.md)，配置与源码登记见[36_registration.json](../cfgs/ct_seqtrack/36_registration.json)。没有正式v36结果，不承诺涨分。

代理只在本地活动项目实施；服务器不写入、不上传、不启动任务。以下服务器命令均由用户执行。旧配置、权重、output/artifacts原样保留；不能从旧权重初始化新实验。

## 上传后的一次真实CUDA检查

先同步本轮源码、五配置及共享 `36_formal_base.yaml`、登记JSON与工具。服务器目录和环境沿用项目SERVER_PATHS；不需要更换依赖。10月4日20:04只读快照：GPU0已有其他用户任务，空闲约16GiB；GPU1空闲约44GiB。因此首次检查放GPU1，下面给GPU1的A/D/F、GPU0的C/E命令。同卡可分批运行，不能根据旧显存快照一次启动全部；按这一次新Full实测峰值安排并发即可，不改变batch或其他训练参数。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u tools/check_v36_full_batch.py \
  --cfg cfgs/ct_seqtrack/36_a_full_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --device cuda
```

该检查做真实batch16的forward/backward/Adam/commit、候选台账JSON和显存记录，不保存权重。报告成功后按实际显存决定并发，保持batch16，不预设五组同时运行。工程权重不能作为正式初始化。

此前报错的二维NLL、整数排序/计数累计及确定性池化修复已逐项核对，v36没有恢复旧故障路径；命令显式使用native allocator和CUBLAS设置。保留现有环境，不传旧的 `--preloading`、`--gpus`、`--devices` 或 `--precision`。辅助loss异常时的检查报告已修复为保留原始错误，避免JSON序列化再次报错。

## 五组独立启动

| 组 | 配置 | 比较含义 |
|---|---|---|
| A | 36_a_full_mini.yaml | 完整闭环 |
| C | 36_c_no_b2_mini.yaml | 无扩展/身份memory |
| D | 36_d_event_cv_mini.yaml | 事件CV均值 |
| E | 36_e_legacy_b1_mini.yaml | 适配后的旧B1整体对照 |
| F | 36_f_window_reset_mini.yaml | 长分支八步重置 |

CUDA检查输出 `status: passed` 后开始正式训练。每段各执行一次，均为独立scratch，不传checkpoint。`CT_RUN`只用于记录本次目录，之后重设不会影响已后台运行的进程。目录采用日期、完整组名、mini/seed/epoch/batch及随机后缀，保留原来的train.log/train.pid格式。

### A：完整v36，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-36_a_full-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_a_full_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### C：无B2/extension/memory，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-36_c_no_b2-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_c_no_b2_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### D：事件CV均值，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-36_d_event_cv-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_d_event_cv_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### E：适配旧B1，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-36_e_legacy_b1-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_e_legacy_b1_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### F：长分支八步重置，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-36_f_window_reset-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_f_window_reset_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

也可用 `bash tools/launch_v36.sh A 1` 等价启动对应组；脚本与上面命令采用相同命名与参数。`CT_PYTHON`和`CT_DATA_ROOT`只用于更换环境/数据路径，不将nuscenes包目录当数据根。

## 新终端查看进度

新终端先进入项目目录，然后选对应的一条执行。每条自动查找该组最新目录，不依赖旧终端中的 `CT_RUN`；`Ctrl+C`只退出tail，不停止训练。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

A：
```bash
tail -n 80 -f "$(ls -d output/*-36_a_full-mini_car_seed42_60ep_bs16-* | sort -r | head -n 1)/train.log"
```
C：
```bash
tail -n 80 -f "$(ls -d output/*-36_c_no_b2-mini_car_seed42_60ep_bs16-* | sort -r | head -n 1)/train.log"
```
D：
```bash
tail -n 80 -f "$(ls -d output/*-36_d_event_cv-mini_car_seed42_60ep_bs16-* | sort -r | head -n 1)/train.log"
```
E：
```bash
tail -n 80 -f "$(ls -d output/*-36_e_legacy_b1-mini_car_seed42_60ep_bs16-* | sort -r | head -n 1)/train.log"
```
F：
```bash
tail -n 80 -f "$(ls -d output/*-36_f_window_reset-mini_car_seed42_60ep_bs16-* | sort -r | head -n 1)/train.log"
```

## 完整epoch恢复

从原RUN下完整checkpoint恢复，保留原manifest与resolved_config。使用同一组配置、原RUN和原源码；恢复不是新实验。训练中断在epoch内部时只恢复上一个完整epoch，不能保存/接续半个状态链。

```bash
RUN=output/实际原运行目录
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_a_full_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --log_dir "$RUN" \
  --checkpoint "$RUN/formal_checkpoints/epoch=010.ckpt" --accelerator gpu
```

第10轮结束恢复后，第11轮会进入连续长分支；恢复测试覆盖该边界。同run、config、source、数据/注释时间、排程和RNG不匹配均拒绝。旧v35与更早checkpoint需要其原登记源码，不能关闭身份校验迁移到v36。

单独评测使用新空目录；自动late-3和独立评测的端点采样epoch均固定0：

```bash
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/36_a_full_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --checkpoint "$RUN/formal_checkpoints/epoch=060.ckpt" --test --accelerator gpu
```

## 固定比较与结论边界

五组完成后运行`python tools/compare_v36.py --help`取得参数。传入五个新RUN、固定R/C/old-B0/old-W参考及登记的强scaled B0/B1。工具从58/59/60原帧重算，核验60轮审计、源代码、配置、端点覆盖及A/F顺序；支持将报告写入新的artifacts/ct_checks文件，拒绝覆盖。

下例中的变量均指向各自原运行目录；旧强参考的建议目录及帧SHA见登记JSON，不能换成别的seed或较弱参考：

```bash
python tools/compare_v36.py \
  --registered-reference "$R_RUN" --baseline "$OLD_C_RUN" \
  --old-b0 "$OLD_B0_RUN" --old-w-quarter "$OLD_W_RUN" \
  --strong-b0 "$STRONG_B0_RUN" --strong-b1 "$STRONG_B1_RUN" \
  --a "$A_RUN" --c "$C_RUN" --d "$D_RUN" --e "$E_RUN" --f "$F_RUN" \
  --output artifacts/ct_checks/v36_five_runs_comparison.json
```

正式比较固定final60和late-3；原总体门、31条移动轨迹门和漂移风险规则保持。validation目录保存每个已验证epoch的指标；最高validation Success和Precision分别连同其同轮配对指标/epoch报告，不取代正式门，不在五组中按最高分重新命名主实验。

相对强B0/B1、控制臂及风险分别报告。D/C/E/F不支持相应机制时收窄结论，不自动追加LR、seed、模块、门控或完整训练。多项改动的综合收益不能直接归因于每一项。

初次本地全仓验证728 passed、12 skipped；详细证据见[实施报告](../artifacts/ct_checks/20261004-194559_v36_implementation/REPORT.md)。上传前又复用本地封存Lightning2.0.2完成两项真实CPU Trainer测试：完整入口/保存/评测以及epoch10→11逐位恢复，均通过；没有安装或升级环境。最新工具与命令核查见[上传就绪报告](../artifacts/ct_checks/20261004-200727_v36_upload_readiness/REPORT.md)。CPU使用torch2.3.0，仍需上述服务器torch2.0.1真实CUDA检查，不追加其他启动前训练。
