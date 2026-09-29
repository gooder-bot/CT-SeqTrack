# v35 第5组piecewise与原四组036恢复

2026-09-29。用户明确追加一组分段学习率，并选择更新同一活动项目后恢复原四组。13:33只读复查确认用户已停止原四组，原PID均不存在、日志记录SIGTERM，各自完整checkpoint均为`formal_checkpoints/epoch=036.ckpt`；恢复后从第37轮继续。此页覆盖此前“保持四组运行并另建代码目录”的部署建议，不重复发送停止命令，也不重新执行四组scratch命令。

停止与checkpoint证据位于[本次实施目录](../artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/)，包括`process_snapshot_2.json`及逐一加载四份权重核验的`stopped_checkpoint_snapshot.json`。四组均完成36轮、累计43,020更新，optimizer和scheduler状态完整。此前第35轮及历史对照见[状态快照](../artifacts/ct_checks/20260929-130405_v35_four_run_status/server_snapshot.json)与[历史比较](../artifacts/ct_checks/20260929-130405_v35_four_run_status/historical_comparison.json)；这些是中途validation，尚非正式final60/late-3结果。

## 本次唯一新增配方

新组为`35_b0_w_piecewise_lr_mini.yaml`，seed42、mini Car B0、W=1/4/4/8、十轮课程、112 reserve、batch16、workers4、FP32、scratch60、无warmup，每轮19,108行/1,195次更新，总71,700次。网络、loss、采样、诊断与递推合同保持，物理GPU1。不得读取原half或其他权重初始化。

| 训练轮次（含端点） | 原half | 新piecewise | 相对原half |
|---|---:|---:|---:|
| 1–20 | 5e-5 | 5e-5 | 1倍 |
| 21–50 | 5e-6 | 1e-5 | 2倍 |
| 51–60 | 5e-7 | 5e-6 | 10倍 |

登记为`lr_schedule=piecewise`、`lr_stage_values=[5e-5,1e-5,5e-6]`、`lr_milestones=[20,50]`、`lr_warmup_steps=0`。完成20/50轮后切换，分别从第21/51轮首次优化起使用下一阶段LR。与原half比较的是整段调度，不能归因于一个固定LR倍数；前20轮配方相同也不授权复用原half checkpoint。

原四份配置身份保持；piecewise的完整调度进入新配置身份。模型仍为`ctseqtrackv35`，schema仍为`ct_seqtrack.joint_identity.v35`，runtime仍为`ct_v35_runtime`。预备`35_b0_w_piecewise_lr_mini_seed52.yaml`只在该配方最终胜出并满足既定条件时使用，不在本次启动。

## 更新与恢复顺序

用户暂停四组已完成。接下来由用户将本次核验后的源码更新到同一活动项目；保留服务器的`output/`、`artifacts/`、数据和环境，不覆盖旧配置快照、manifest、日志或checkpoint。此次没有网络、loss或数据改动，不把重复真实CUDA batch检查设为前置。

1. 用户将本地本次修改提交并推送GitHub，再按下方命令在已暂停的服务器项目拉取；必须一并提交新增配置及源码登记JSON。
2. 原quarter、half、normal、scaled分别使用原cfg、原准确`--log_dir`和各自`epoch=036.ckpt`恢复，物理GPU0/0/1/1。`--epoch 60`仍指总计60轮，不是再训练60轮；恢复从第37轮继续，不重建随机输出目录。
3. 恢复时保持旧`run_manifest.json`不变，在`resume_manifests/`追加此次checkpoint与源码来源；只放行预登记的精确before/after源码关系，不关闭源文件校验，不改写原始证据。
4. 旧训练日志用追加重定向`>>`，保留旧`train.pid`；恢复PID使用新的`resume.pid`或带日期文件，不覆盖原PID记录。中断后未完成轮次从完整checkpoint重新计算，分析训练曲线时按恢复来源去重，不把重复日志当额外有效训练。
5. 在GPU1为piecewise新建唯一输出目录，从epoch0开始；不传`--checkpoint`或`--init_checkpoint`。第五组与原四组恢复是不同操作。

代理只读服务器并在本地准备修改与命令，不替用户停止、上传、恢复或启动任务。原四组源码更新只登记本次准确的before/after对应关系：旧配方的模型计算和调度不变，新增piecewise调度不应污染旧身份。固定登记细节与测试结果以本次实施证据为准。

## 本次交付与可复制命令

本地更新已完成：全仓**554 passed、3 skipped**；compileall、diff、五条Bash命令语法及入口解析检查通过。25份旧配置文件及身份SHA不变；新调度在Lightning2.0.2下跨第20/50轮恢复与连续训练逐位一致。见[验证报告](../artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/REPORT.md)。本次沿用v35 W结构，核心只修改调度、配方校验、宿主调度选择及恢复来源记录。

按用户最新选择，通过GitHub同步，不需要ZIP。提交本次源码、文档、比较工具和测试，尤其不要漏掉三个新增配置文件：`35_b0_w_piecewise_lr_mini.yaml`、`35_b0_w_piecewise_lr_mini_seed52.yaml`、`35_piecewise_source_registration.json`。服务器与本地分支均为`main`；本地推送完成后在服务器执行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
git -c core.autocrlf=false pull --ff-only origin main
```

本次最后的服务器只读网络检查：github.com的A记录能解析，但GitHub SSH 22端口连接超时，443入口直连也在SSH banner阶段超时。因此本地代码已就绪，GitHub拉取尚未验证成功；先成功拉取新提交，再执行下方训练命令。此网络状态不构成checkpoint或调度失败，代理没有修改DNS、代理或SSH配置。

已通过临时Git index模拟提交当前20个文件，63份运行源码blob与恢复登记after逐文件完全一致。本次Git拉取保留原运行的配置快照、manifest和checkpoint；无需重新安装依赖。下面使用环境内Python绝对路径，可以直接在当前shell执行。

下面五段各执行一次。使用环境内Python绝对路径，无需修改环境或安装包。物理卡由`CUDA_VISIBLE_DEVICES`指定，每个进程`--trainer_devices 1`；不要加旧入口的`--preloading`、`--gpus`或`--devices`。原四组日志追加，新PID另存。GPU1同时三个进程共享算力，耗时可能增加。

### 1. 恢复quarter，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="output/20260928-193449-35_b0_w_quarter_lr-mini_car_seed42_60ep_bs16-RQzvPp"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_quarter_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  --checkpoint "$CT_RUN/formal_checkpoints/epoch=036.ckpt" \
  >> "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/resume-$(date +%Y%m%d-%H%M%S).pid"
echo "$CT_RUN"
```

### 2. 恢复half，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="output/20260928-193454-35_b0_w_half_lr-mini_car_seed42_60ep_bs16-Aw4kx1"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_half_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  --checkpoint "$CT_RUN/formal_checkpoints/epoch=036.ckpt" \
  >> "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/resume-$(date +%Y%m%d-%H%M%S).pid"
echo "$CT_RUN"
```

### 3. 恢复normal，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="output/20260928-193459-35_b0_w_normal_lr-mini_car_seed42_60ep_bs16-GXfluF"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_normal_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  --checkpoint "$CT_RUN/formal_checkpoints/epoch=036.ckpt" \
  >> "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/resume-$(date +%Y%m%d-%H%M%S).pid"
echo "$CT_RUN"
```

### 4. 恢复scaled，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="output/20260928-193506-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-bJhtMt"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_scaled_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  --checkpoint "$CT_RUN/formal_checkpoints/epoch=036.ckpt" \
  >> "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/resume-$(date +%Y%m%d-%H%M%S).pid"
echo "$CT_RUN"
```

### 5. 新piecewise从头训练，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_b0_w_piecewise_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 新开终端查看日志

进入项目后分别选择对应命令；Ctrl+C只退出tail。恢复日志首批应显示`epoch=37/60`、`step=43021`，第五组应从`epoch=1/60`开始。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

quarter：

```bash
tail -n 80 -f output/20260928-193449-35_b0_w_quarter_lr-mini_car_seed42_60ep_bs16-RQzvPp/train.log
```

half：

```bash
tail -n 80 -f output/20260928-193454-35_b0_w_half_lr-mini_car_seed42_60ep_bs16-Aw4kx1/train.log
```

normal：

```bash
tail -n 80 -f output/20260928-193459-35_b0_w_normal_lr-mini_car_seed42_60ep_bs16-GXfluF/train.log
```

scaled：

```bash
tail -n 80 -f output/20260928-193506-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-bJhtMt/train.log
```

piecewise（本次唯一新组）：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b0_w_piecewise_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

## 五组比较、预算与停止

首阶段共五次、300epoch、358,500更新；恢复原四组不重复登记为新实验。五组全部完成60轮及58/59/60正式评测后，再使用`tools/compare_v35_piecewise_b0.py`，其参数在原入口基础上增加必填`--piecewise`：

| 参数 | 必须对应的证据 |
|---|---|
| `--registered-reference` | 固定旧R，原始逐帧指纹不变 |
| `--baseline` | 固定旧C，原始逐帧指纹不变 |
| `--old-b0` | 固定旧v32 B0 |
| `--old-w-quarter` | 固定旧v34 W-quarter风险对照 |
| `--quarter --half --normal --scaled` | 原四份v35 seed42完整运行，保留恢复来源 |
| `--piecewise` | 新分段配方seed42完整运行 |

原`tools/compare_v35_b0.py`独立保留，继续支持原四组复核；不能用四组局部排名替代本次五组最终选择。新入口核对原manifest、追加resume来源及固定源码迁移登记，重算原始帧，要求五组完整再排序。默认stdout JSON，`--output`只写`artifacts/ct_checks/`内新文件。

总体final60/late-3的S/P四项≥固定R，且固定31条曾移动轨迹全程657预测帧的四项≥旧C，才通过原硬门。通过者依次按final60 Success、Precision降序、初始LR升序；完全同分且初始LR也同为5e-5时，预登记原half优先piecewise。不得挑best epoch或换较弱参考。

继续报告>5m、>10m与未恢复失跟。final60或late-3的>10m预测帧数高于旧W-quarter时单列风险、暂停Full复核，不改原得分门；不要求所有切片获胜。没有通过者即在这五次后停止，不自动扩大搜索。

仅通过后锁定胜出完整配方补seed52，并补独立正常SeqTrack seed52；划分始终seed42，锁定调度、不重选LR，分别报告同seed差值与固定R/C门。条件总上限为七次、420epoch、501,900更新，不自动启动。完整规则见[实验协议](EXPERIMENT_PROTOCOL.md)。
