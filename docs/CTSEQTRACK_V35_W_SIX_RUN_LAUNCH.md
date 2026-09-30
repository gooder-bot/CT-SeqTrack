# v35 W 新增六组：配置与启动命令

2026-09-30。用户已确认本页六组新训练，覆盖此前“只补胜出配方seed52与参考seed52”的两组建议和条件预算。旧五组seed42均已完成，仅复用其结果；本页六组全部从epoch0随机初始化，不恢复或覆盖旧运行。代理只在本地准备配置和命令，服务器保持只读，上传与启动由用户执行。

本轮不新增网络、loss、数据或恢复迁移逻辑。已有五份seed52配置原样复用，只新增double配方及其正式LR白名单。无需把重复真实CUDA batch检查设为启动前置；本页没有S窗口或v34新训练。

## 六组登记

| 顺序 | 配方 | seed | 物理GPU | 配置 | 第1–20 / 21–50 / 51–60轮LR |
|---|---|---:|---:|---|---|
| 1 | W scaled | 52 | 0 | `35_b0_w_scaled_lr_mini_seed52.yaml` | 1.5e-4 / 1.5e-5 / 1.5e-6 |
| 2 | W piecewise | 52 | 1 | `35_b0_w_piecewise_lr_mini_seed52.yaml` | 5e-5 / 1e-5 / 5e-6 |
| 3 | 独立SeqTrack原正常参考 | 52 | 1 | `35_seqtrack_ref_seed52_mini.yaml` | 保持原StepLR，见下文 |
| 4 | W double | 42 | 1 | `35_b0_w_double_lr_mini.yaml` | 2e-4 / 2e-5 / 2e-6 |
| 5 | W half | 52 | 0 | `35_b0_w_half_lr_mini_seed52.yaml` | 5e-5 / 5e-6 / 5e-7 |
| 6 | W quarter | 52 | 0 | `35_b0_w_quarter_lr_mini_seed52.yaml` | 2.5e-5 / 2.5e-6 / 2.5e-7 |

配置均位于`cfgs/ct_seqtrack/`。W五组保持v35身份、W=1/4/4/8、10轮课程和112 reserve；多步调度完成20/50轮后降档，无warmup。double只是原正常LR的2倍，不能替代已有scaled的跨seed核验。

参考组虽然文件名前缀为35，其模型和配置身份仍为原v33独立SeqTrack，非W。它保留原正常初始LR=1e-4及StepLR(step_size=20, gamma=0.1)：第1–20轮1e-4、第21–40轮1e-5、第41–60轮1e-6；不得改成W的20/50里程碑。

六组共同保持mini Car、scratch60、batch16、workers4、单设备FP32和固定seed42数据划分。训练随机seed依上表；seed52不改变固定训练/验证划分。每组71,700次更新，本次新增共360epoch、430,200次更新。旧五组300epoch、358,500次更新不重复执行；本页授权不自动追加其他实验。

## 更新代码

本地修改提交并推送完成后，由用户在服务器活动项目执行普通同步命令。保留服务器现有`output/`、`artifacts/`、数据与环境，不覆盖旧配置快照、manifest、日志或checkpoint。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
git -c core.autocrlf=false pull --ff-only origin main
```

下面六段各执行一次，可逐组复制。每段用日期加`mktemp`创建新输出目录，记录`train.pid`与`train.log`；每个进程只用一张可见卡，GPU0/1各三个进程。Python使用现有环境绝对路径，保留native allocator、CUBLAS和线程设置，无需重新安装环境。入口使用`--trainer_devices 1`，不添加旧入口参数`--preloading`、`--gpus`或`--devices`。FP32由正式配置固定。

所有命令均不传`--checkpoint`或`--init_checkpoint`。运行首批日志应显示`epoch=1/60`；新目录名中的seed必须与上表一致。每组命令最后打印的目录就是本次准确运行目录。

## 1. scaled seed52，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_scaled_lr-mini_car_seed52_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_scaled_lr_mini_seed52.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 2. piecewise seed52，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_piecewise_lr-mini_car_seed52_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini_seed52.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 3. 独立SeqTrack原正常seed52，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_seqtrack_ref-mini_car_seed52_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_seqtrack_ref_seed52_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 4. double seed42，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_double_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_double_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 5. half seed52，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_half_lr-mini_car_seed52_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_half_lr_mini_seed52.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 6. quarter seed52，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_quarter_lr-mini_car_seed52_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_quarter_lr_mini_seed52.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 新开终端查看日志

以下六段分别查找同配方、同seed的最新目录，顺序与启动命令一致；Ctrl+C只结束tail。若意外重复启动过同一组，直接使用启动命令打印的准确目录，避免把两次运行混在一起。

1. scaled seed52，GPU0：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_b0_w_scaled_lr-mini_car_seed52_60ep_bs16-* | head -n 1)/train.log"
```

2. piecewise seed52，GPU1：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_b0_w_piecewise_lr-mini_car_seed52_60ep_bs16-* | head -n 1)/train.log"
```

3. 独立SeqTrack seed52，GPU1：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_seqtrack_ref-mini_car_seed52_60ep_bs16-* | head -n 1)/train.log"
```

4. double seed42，GPU1：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_b0_w_double_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

5. half seed52，GPU0：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_b0_w_half_lr-mini_car_seed52_60ep_bs16-* | head -n 1)/train.log"
```

6. quarter seed52，GPU0：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-35_b0_w_quarter_lr-mini_car_seed52_60ep_bs16-* | head -n 1)/train.log"
```

## 完成后的比较口径

每组固定报告final60和late-3（58/59/60）Success/Precision，不按best epoch挑结果。保留固定seed42参考R、旧C、旧W-quarter的原始证据和分组：总体四项与固定R比较，固定31条曾移动轨迹全程四项与旧C比较；seed52各配方另外报告相对本次独立SeqTrack seed52的同seed差值，不以同seed参考替换固定R/C登记。

继续报告>5m、>10m与未恢复失跟。final60或late-3的>10m计数较旧W-quarter增加时单列风险并暂停Full复核，不修改原得分硬门，也不增加所有切片必须获胜的要求。

本次把scaled、piecewise、half、quarter各自已有seed42与新seed52配对，提供四种完整配方的有限跨seed稳定性证据。两个seed不足以证明广泛稳定；double仅为seed42的更高LR探索，不能单凭它的结果证明跨seed收益。本轮没有匹配的S训练，不能据此声称W优于S；也不能独立归因于某个模块、某一个调度阶段或推导Full收益。

现有`tools/compare_v35_b0.py`和`tools/compare_v35_piecewise_b0.py`继续服务原四组/五组及原登记身份、源码和条件第二seed协议。本次新增六组超出旧入口的完整实验集合，不得替换旧参数、改标签或借旧源码登记伪装为原协议；新结果汇总必须明确本页六组身份及其原始manifest、配置和逐帧证据。旧五组结论见[9月30日正式复盘](../artifacts/ct_checks/20260930-175409_v35_five_run_review/REPORT.md)，一般规则见[实验协议](EXPERIMENT_PROTOCOL.md)。
