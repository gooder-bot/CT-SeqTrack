# v37 B2 重心权重梯度修正：三组配置、检查与启动

2026-10-21。本轮结构改动只改变重心计算中 identity × reliability 的反向路径：对 mass 权重 detach，分子和分母共用该 mass；vote、live center、精修及两头显式监督保留。相同参数和输入的前向不变，需重新训练检验效果。没有新增参数、损失、阈值或 GT 推理输入，没有修改 B3。

本地代码与三组配方已实施；正式训练和真实 CUDA 检查由用户在服务器执行。当前无新三组正式成绩，不宣称涨分、提速或完全消除共享梯度副作用。修改前源码和配置保存在 `artifacts/ct_checks/20261021-163831_b2_centroid_implementation/baseline_source/`。

默认三组为每轮 validation（`check_val_every_n_epoch=1`，val1）、物理GPU1、单进程单卡、scratch60、batch16、workers4，LR和固定final60/late3保持。用户最新要求支持参数覆盖：CLI优先于子YAML、base YAML和代码默认；未传的参数沿用配置，未知参数报错。当前三个B2配方可覆盖验证频率、workers、batch_size、epoch、lr、seed，也开放GPU选择、YAML、数据根和输出路径，分类和例子见 [参数覆盖说明](CTSEQTRACK_V37_CONFIG_OVERRIDES.md)。默认三份YAML及下方完整命令不改。

本次需上传更新后的完整源码和新独立登记，三份默认YAML随版本保持一致。修改前版本封存于 `artifacts/ct_checks/20261021_v37_cli_overrides/previous_source/`；旧已运行任务继续使用原源码。以后仅修改六个允许字段的合法值，不再需要修改代码或重新登记模板，但实际配置哈希仍严格记录；恢复不能改变任何实际覆盖值或源码。

此前val1固定配方版本的配置、入口、恢复及旧配方聚焦回归 **133项通过（7.52秒）**，当时三条完整CLI通过配置解析和登记核验；两份修改的生产Python编译及diff检查通过。历史证据见 [val1固定配方检查](../artifacts/ct_checks/20261021_b2_every_epoch_validation/REPORT.md)。该结果不表示本次参数覆盖改动已重跑同一套测试；下面1164项结果仅属更早的val5版本。

此前val5版本的本地完整CPU回归为1164通过、24跳过、2个预期失败；编译和启动脚本语法检查通过。本机缺少pytorch_lightning，真实Trainer集成测试被跳过，不能把模型/宿主合同测试通过写成真实Lightning训练已验证；真实nuScenes/CUDA检查仍需下述一次服务器检查。

此前val5版本另有三种新配方的CPU宿主恢复测试，使用多帧窗口和实际提交历史，验证完整epoch恢复与连续训练的下一轮loss、提交框、参数、Adam、scheduler和RNG逐位一致；补充文件7项通过，其中3项为新增恢复合同。这仍不替代真实Trainer/CUDA验证。

17:14上传前复查是历史val5快照：当时三条启动命令经配置解析和正式登记核验通过，SSH只读确认Python3.9.19/PyTorch2.0.1+cu118/Lightning2.0.2及mini数据可用，GPU1约42.7GiB空闲、无CT训练进程，新版尚待上传；不据此推断当前服务器状态。历史二维NLL、整数排序/计数及确定性池化修复保留，命令保留native allocator与CUBLAS设置，不使用旧CLI参数。历史证据见 [17:14就绪复查](../artifacts/ct_checks/20261021_b2_launch_readiness/REPORT.md)，原val5源码/配置/登记封存于 `artifacts/ct_checks/20261021_b2_every_epoch_validation/previous_val5_source/`。旧其他正式配方的验证频率不改。

## 三组默认配方

| 配方 | 配置文件（位于 cfgs/ct_seqtrack） | LR milestones | 用途 |
|---|---|---|---|
| B1+B2 shadow | `37_b1_b2_mass_detach_mini.yaml` | `[20,50]` | 与旧 shadow 比较 |
| Full（主结果） | `37_full_mass_detach_mini.yaml` | `[20,50]` | 与旧 Full、强 B1 比较 |
| Full hold_lr | `37_full_mass_detach_hold_lr_mini.yaml` | `[20]` | 与本轮标准 Full 比较 LR |

三组都是 mini Car、seed42、scratch60、batch16、FP32、concat、初始 LR=1.5e-4；无 warmup，每轮验证，训练后评测58/59/60。W=1/4/4/8、10轮课程、112 reserve、1024 B0点、768→256扩展、36 memory slots及既有损失保持。标准组第21轮起1.5e-5、第51轮起1.5e-6；hold_lr第51轮继续1.5e-5。每组71,700次更新，三组共180 epoch/215,100次更新，以完整sampler审计核验；每轮验证增加评测次数和耗时，不增加训练更新预算，也不改为挑选最佳轮。

仅覆盖验证频率或workers时标记 `runtime_override`；batch_size、epoch、lr、seed任一偏离默认时标记 `recipe_variant`，属于研究变体，不计入上述默认三组或替代主结果。所有覆盖均写入实际配置身份；结构、单卡、scratch、时间协议不放宽，仍核验基础模板与源码。

新身份字段为 `v37_centroid_recipe`，登记文件为 `37_b2_centroid_registration.json`。它与旧 B1 专用 `v37_training_recipe` 互斥，执行 options 显式为空；不混入 Full 五项去重或分解投影。B1+B2 和 Full 完整训练同一候选与核验网络，分别执行 shadow/Full 选择与提交策略，不能描述为纯粹有无 B3 网络。

旧其他配方的 YAML、登记和已有 output 保持。新增源码改变完整 source hash：旧 checkpoint 必须继续使用原登记源码恢复/评测；不能覆盖仍在运行的部署目录。将本版部署到独立目录，或在原任务结束后更新；以下命令均从该版本 CT-SeqTrack 根目录运行，不升级环境。

## 一次真实 Full batch 检查

沿用现有 Python 环境与真实数据根；本轮检查和三组训练统一使用物理 GPU1（`CUDA_VISIBLE_DEVICES=1`）。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_v37_full_batch.py \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --device cuda
```

检查应返回 `status=passed`，覆盖 batch16 的 forward、loss、backward、Adam、唯一状态提交和候选台账。报告在新的 `artifacts/ct_checks/`，不保存工程权重。正式训练另开进程从零开始。无需增加 BF16、分解或额外 seed 检查。

## 分别启动

三组全部使用物理 GPU1，每组为独立单卡进程（`trainer_devices=1`）。下面每段按默认参数只启动对应一组，日期加 `mktemp` 随机后缀生成新输出目录；`main.py` 正式入口会核验源码和基础配方模板，记录完整实际配置。保存打印的 `RUN` 路径，不能混用旧输出或从其他组第50轮 checkpoint 分叉。

### B1+B2 shadow

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN=$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_b2_mass_detach-mini_car_seed42_60ep_bs16-XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_b2_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'RUN=%s\nPID=%s\n' "$CT_RUN" "$CT_PID"
```

新终端查看该配方日期最新的一次运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(printf '%s\n' output/*-37_b1_b2_mass_detach-mini_car_seed42_60ep_bs16-*/train.log | sort -r | head -n 1)"
```

### Full（主结果）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN=$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_full_mass_detach-mini_car_seed42_60ep_bs16-XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'RUN=%s\nPID=%s\n' "$CT_RUN" "$CT_PID"
```

新终端查看该配方日期最新的一次运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(printf '%s\n' output/*-37_full_mass_detach-mini_car_seed42_60ep_bs16-*/train.log | sort -r | head -n 1)"
```

### Full hold_lr

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN=$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_full_mass_detach_hold_lr-mini_car_seed42_60ep_bs16-XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_hold_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'RUN=%s\nPID=%s\n' "$CT_RUN" "$CT_PID"
```

新终端查看该配方日期最新的一次运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(printf '%s\n' output/*-37_full_mass_detach_hold_lr-mini_car_seed42_60ep_bs16-*/train.log | sort -r | head -n 1)"
```

也可选择等价的启动器命令；与上面的完整命令二选一，避免重复启动同一组：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
bash tools/launch_v37_centroid.sh b1_b2_mass_detach 1
bash tools/launch_v37_centroid.sh full_mass_detach 1
bash tools/launch_v37_centroid.sh full_mass_detach_hold_lr 1
```

启动器先核验实际配置再创建唯一输出目录，默认Python可通过 `CT_PYTHON` 指定；未设置 `CT_DATA_ROOT` 时使用所选YAML的path，显式设置时作为基础 `--path`，额外CLI `--path` 最后优先。支持透传参数，例如 `bash tools/launch_v37_centroid.sh full_mass_detach 1 --check_val_every_n_epoch 5 --workers 2`，未传项沿用YAML；自动目录中的seed/epoch/batch按实际值生成。以上三个日志命令匹配默认seed42/60ep/bs16，不依赖启动终端的 `CT_RUN`；研究变体或多次尝试请用保存的实际 `RUN` 路径。

还可传 `--devices 1` 选择单个物理GPU，优先于YAML的devices和环境GPU；`--trainer_devices 1` 仍表示只用一张卡。启动器的位置GPU提供默认CLI devices，额外 `--devices` 最后优先；第一个recipe提供默认cfg，可用 `--cfg` 覆盖为同recipe的YAML。`--path` 改数据根，`--log_dir` 指定新/空输出目录，未指定输出目录才自动命名。默认三条完整nohup继续使用环境GPU1。设备编号写入manifest，不参与算法配置哈希，真正多卡训练仍不支持。

## 同一运行恢复

同一运行完整 epoch 边界恢复，使用同一配置、原 RUN、该 RUN 内的完整 checkpoint 和原部署源码。例如 Full：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --accelerator gpu --trainer_devices 1 \
  --checkpoint <RUN>/formal_checkpoints/epoch=020.ckpt --log_dir <RUN>
```

`<RUN>` 是需替换的同一实际目录；epoch=020仅为完整边界示例。上述例子按默认实际配置恢复。若启动时覆盖为val5、workers2，恢复命令也必须带 `--check_val_every_n_epoch 5 --workers 2`；未传会沿用YAML的val1、workers4并触发实际配置身份不匹配。其他覆盖项及源码也必须与原运行一致。两组 Full 的配置、运行身份和调度不同，禁止互相恢复。未启用 `--init_checkpoint`、参数冻结或梯度累积。

## 结果与机制检查

主结论固定 final60 与 late-3（58/59/60），不按每组最佳轮挑结果。标准 Full 是预先指定主结果；hold_lr 不替代标准组的结构对照。两组 Full 应在相同环境下具有相同初始化、采样和前50轮训练/共同验证；第51轮才因 LR 分叉。提前分叉先核查运行环境、源码与确定性记录。

登记内的初始化数值来自本地CPU探针，环境见 `initialization_probe_environment`，不作为跨PyTorch版本硬门。服务器启动后比较新三组彼此及旧同环境运行的manifest初始化哈希；本地探针通过不等于已验证服务器新旧权重相同。

| 对照 | 旧 final60 S/P | 旧 late-3 S/P |
|---|---|---|
| 原 v37 shadow | 51.4376 / 62.4376 | 50.3158 / 60.6831 |
| 原 v37 Full FP32 | 49.9136 / 59.5274 | 50.0715 / 59.6327 |
| 强 B1 | 53.7133 / 66.2746 | 52.8541 / 65.3082 |

精确复用证据见新登记的 `reused` 字段及 [既有研究审查](CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md)。保持 [实验协议](EXPERIMENT_PROTOCOL.md) 的原 R/C 门与移动轨迹口径。另复核 >5m/>10m、持续失跟、首测缺测、真实恢复和错误接管，不新增“所有切片必须全胜”的硬门。旧 `compare_v37.py` 专用于原五组，不能直接把这三组代入。

两类新增被动摘要只在新配方启用，位于训练诊断的 statistics，字段前缀为 `v37_centroid_`：

- `identity_{fg,bg}_probability_{sum,count}`：全部有效768点的身份概率。计算类别点均值时，用 probability_sum 的累计 sum 除以 probability_count 的累计 sum；不要用端点统计自带的 mean 代替点均值。
- `reliability_{fg,bg}_abs_error_{sum,count}`：selected有效点对既有 `y/(1+d/r)` 软目标的绝对误差，同样用对应累计sum/count。无该类时分母为零，结果缺失，不填零均值。

这些摘要只描述显式目标匹配，不能称概率已校准。vote误差、seed→final、正mode与好候选供给复用已有日志；更细的权重和Huber分析离线读取候选台账，不增加训练前向。

若语义与候选、成绩均改善，支持保留；若仅语义改善而vote仍不准，下一步研究几何/精修；若候选改善但Full未改善，再查B3选择与递推；shadow仍受损则继续查共享训练通路。只有hold_lr上涨属于组合配方证据。此次仍保留共享编码器梯度，以及正mode中背景成员的vote梯度，不能声称完全解耦或必然涨分。
