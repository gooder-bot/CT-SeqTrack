# v35 最新启动说明：四组 mini 单 seed，GPU 0 / 0 / 1 / 1

> 2026-09-29更新：本页四组已由用户暂停，均保留完整036 checkpoint；最新安排为更新同一活动项目后恢复四组、第37轮继续，并在GPU1从头追加第5组piecewise。恢复命令、新组命令及五组比较见[新增第5组操作页](CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)。本页原四组scratch命令保留为历史，不应重跑；当前首阶段五次、条件总上限七次覆盖下方旧四次/六次总数，不重复要求CUDA batch。

2026-09-28更新。用户本次明确将原三档扩为四档，新增1.5e-4；四组使用同一v35综合结构和W=1/4/4/8。全部seed42、随机初始化scratch60、batch16、workers4、FP32、20/50衰减、无warmup，每组71,700次更新。四组合计240epoch、286,800更新；原固定R/C门槛不变。当前不运行Full或seed52。

## 上传前结论与本次文件

原v35计划的代码、接口、诊断和恢复路径已实现；本次补齐四档配置及四组比较入口。原三份v35配方和旧v33/v34配置身份保持。第四档不改变网络、loss、课程或单组预算。

最新完整本地回归为515 passed、3 skipped，无失败；compileall和diff检查通过。两项CUDA测试留待真实环境，另一项“未安装Lightning”分支在本地已安装Lightning时跳过。当前结论是可上传并进行一次真实CUDA batch，再开始四组正式训练，不代表已有涨分或真实CUDA峰值结果。

最新文件为[四组增量上传包](../artifacts/ct_checks/20260928-185917_v35_four_run_readiness/upload_code.zip)、[SHA清单](../artifacts/ct_checks/20260928-185917_v35_four_run_readiness/upload_manifest.json)及[就绪报告](../artifacts/ct_checks/20260928-185917_v35_four_run_readiness/REPORT.md)。用本次新包更新现有v34项目；包中包含完整的本轮v35修改，而不只是新增第四档。18:25的三组包保留为历史，不用于本次四组启动。

9月28日18:59只读确认：服务器项目和mini路径有效，Python3.9.19、torch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9；未发现CT-SeqTrack训练进程，服务器尚无v35文件。两张A40各46,068MiB，GPU0/1空闲约34,402/38,808MiB；其他任务仍在运行。每卡两组旧B0占用估算有余量，v35及并发实际峰值尚未测，不把参数增量当显存增量。

代理只读服务器，未上传、安装或启动/停止任务。用户上传时保留服务器output、artifacts、数据和环境，不上传旧权重作为初始化。

## 一次新的真实CUDA batch

只执行一次，不按四个LR重复检查。真实batch16完成forward/backward/Adam/accepted commit，记录诊断、耗时和峰值显存，不保存checkpoint。输出`status=passed`后即可启动下面四组。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_v35_batch.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_quarter_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --device cuda
```

此前报错相关路径已复核：二维NLL、确定性pooling、int64排序键和计数累计修复保留；新局部读取不使用bool排序或浮点cumsum。命令显式设置native allocator，覆盖旧shell环境，并保留CUBLAS确定性设置。**不要沿用旧`cfgs/seqtrack3d_nuscenes_mini.yaml`、`--preloading`、`--gpus`或`--devices`。** 物理卡用`CUDA_VISIBLE_DEVICES`，每个进程用`--trainer_devices 1`。

## 四组独立后台命令

每段各执行一次，日期由运行当天自动生成，随机后缀避免目录重名。四个进程可分别启动，无需等待上一组完成；启动后打印准确目录，日志用`python -u`实时刷新。

### 1. quarter：LR=2.5e-5，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_quarter_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_quarter_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 2. half：LR=5e-5，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_half_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_half_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 3. normal：LR=1e-4，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_normal_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_normal_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 4. scaled：LR=1.5e-4，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_scaled_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 新开终端查看进度

先进入项目目录，再选择对应的一条。每条跟随该配方最新目录；如已保存启动时打印的准确目录，也可直接使用。Ctrl+C只退出tail，不停止后台训练。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

quarter，2.5e-5：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b0_w_quarter_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

half，5e-5：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b0_w_half_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

normal，1e-4：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b0_w_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

scaled，1.5e-4：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

每5轮validation用于观察；完整结果看自动完成的58/59/60评测、`results.json`、日志`[v35 complete]`及训练预算。训练统计在`training_diagnostics/epochNNN.json`。不要加`--no_late3`或按最佳epoch挑选结果。

## 同身份恢复与比较

恢复时使用原配置、原准确`--log_dir`和该运行完整epoch的`formal_checkpoints/epoch=NNN.ckpt`，在原训练命令加`--checkpoint`；不要重新执行mktemp，不使用`--init_checkpoint`。LR/seed/版本不得交叉续训。

四组全部完成后，把下面四个变量替换为准确目录。只读工具强制四组完整，重算原始帧，按固定R/C原硬门评选，不接受较弱参考替换。

```bash
CT_Q=output/这里替换为quarter准确目录
CT_H=output/这里替换为half准确目录
CT_N=output/这里替换为normal准确目录
CT_S=output/这里替换为scaled准确目录
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/compare_v35_b0.py \
  --registered-reference output/20260924-193210-33_seqtrack_ref-mini_car_seed42_60ep_bs16 \
  --baseline output/20260924-193242-33_b0_half_lr-mini_car_seed42_60ep_bs16 \
  --old-b0 output/20260922-191248-32_b0-mini_car_seed42_60ep_bs16 \
  --old-w-quarter output/20260926-174406-34_b0_context_w4_quarter_lr-mini_car_seed42_60ep_bs16-4xyGiB \
  --quarter "$CT_Q" --half "$CT_H" --normal "$CT_N" --scaled "$CT_S"
```

默认stdout JSON，可用`--output artifacts/ct_checks/全新目录/report.json`独占保存。返回0/1/2表示原得分门有通过者/无通过者/证据无效；返回0不代表已允许Full，还需查看漂移风险与后续seed稳定性。

四档失败即停止，不扩大搜索。通过者按final60 Success、Precision、低LR排序；只有通过后才保留原计划的胜出配方seed52与正常独立SeqTrack seed52两次复验。四份`35_b0_w_*_lr_mini_seed52.yaml`只是预备配置，只择一；参考为`35_seqtrack_ref_seed52_mini.yaml`，原算法及20/40衰减保持，划分仍seed42。比较时增加`--selected-seed52 RUN --reference-seed52 RUN`，不重新选LR。

当前首阶段四次、286,800更新；若执行原条件第二阶段，合计上限为六次、360epoch、430,200更新。final60或late-3的>10m帧数高于旧W-quarter时单列风险、暂停Full讨论，不改原得分门结果。单seed结果不等于已经稳定超越；不会自动追加训练。
