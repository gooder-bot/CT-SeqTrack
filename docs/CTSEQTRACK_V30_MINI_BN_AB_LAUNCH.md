# v30 mini：两组 Full 与 B0 有效BN反向重算对照（2026-09-15）

2026-09-17：本页四组已完成训练，见[结果分析](../artifacts/ct_checks/reports/20260917_v30_mini_four_arm/REPORT.md)。
下方命令仅训练，不自动拟合Full策略；后处理见[补校准与评测脚本](../artifacts/ct_checks/reports/20260917_v30_mini_four_arm/mechanism/fit_and_eval_full.sh)。
当前尚无校准后Full闭环分数，不能用训练验证中回退B0的同分选择时间后端。

本页替代此前 GPU0 B0 / GPU1 GRU / GPU2 CfC 的三任务安排。
这里的 Full 指完整模型；四个任务都使用 **nuScenes-mini / Car**。
仅本地补充配置及命令，没有同步或启动服务器任务。

| 物理GPU | 任务 | 配置（位于 `cfgs/ct_seqtrack/`） | `ct_b0_masked_bn_recompute` |
|---|---|---|---|
| 0 | Full-GRU | `30_full_gru_mini.yaml` | false |
| 2 | Full-CfC | `30_full_cfc_mini.yaml` | false |
| 1 | B0，BN反向重算 false | `30_b0_mini_bn_recompute_false.yaml` | false |
| 1 | B0，BN反向重算 true | `30_b0_mini_bn_recompute_true.yaml` | true |

两份 B0 配置继承 `30_b0_mini.yaml`，解析后仅实验名称和重算开关不同。
false 使用有效BN直接计算；true 在反向局部重算同一公式，running statistics只在前向更新一次。
两者都启用有效测量mask、BN和池化；false不是关闭有效BN。实现见[预算取舍](CTSEQTRACK_V30_MEMORY_TRADEOFF.md)。

## 参数与启动位置

四组保持 Car、seed42、scratch60、batch16、workers4、每5轮验证、FP32、strict deterministic。
Adam lr=1e-4、betas=(.5,.999)、eps=1e-6、weight_decay=0、foreach/fused=false；StepLR20×.1。
每进程 `trainer_devices=1`；`CUDA_VISIBLE_DEVICES`选择物理卡，各进程内部显示cuda:0属于正常映射。
使用每worker256MiB缓存，`preloading=false`；命令不加 `--preloading`。
保持 `PYTORCH_CUDA_ALLOC_CONF=backend:native`，用于本次默认与重算两种执行方式的相同分配器条件。

先同步本地最新代码、基础配置和两份新增YAML，在服务器原来的seqtrack3d环境与项目根目录执行。
历史路径为 `~/study/lcyu/CT-SeqTrack`；数据根为 `/home/lishengjie/data/nuscenes-mini`。
`main.py`没有 `--ct_b0_masked_bn_recompute` 参数，因此用已登记的YAML切换。
四段均可独立执行，日期目录、日志和PID分开；两份B0都放GPU1，无需等待上一份结束。

## GPU 0：Full-GRU

```bash
CT30_GRU_RUN="output/$(date +%Y%m%d-%H%M%S)-30_full_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_GRU_RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=backend:native \
python -u main.py \
  --cfg cfgs/ct_seqtrack/30_full_gru_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 \
  --tag ct30_full_gru_mini_car_seed42_60ep_bs16 \
  --log_dir "$CT30_GRU_RUN" \
  > "$CT30_GRU_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT30_GRU_RUN/train.pid"
printf '日志：%s/train.log\n' "$CT30_GRU_RUN"
```

## GPU 2：Full-CfC

```bash
CT30_CFC_RUN="output/$(date +%Y%m%d-%H%M%S)-30_full_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_CFC_RUN"
nohup env CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=backend:native \
python -u main.py \
  --cfg cfgs/ct_seqtrack/30_full_cfc_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 \
  --tag ct30_full_cfc_mini_car_seed42_60ep_bs16 \
  --log_dir "$CT30_CFC_RUN" \
  > "$CT30_CFC_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT30_CFC_RUN/train.pid"
printf '日志：%s/train.log\n' "$CT30_CFC_RUN"
```

## GPU 1：B0，BN反向重算 false

```bash
CT30_B0_FALSE_RUN="output/$(date +%Y%m%d-%H%M%S)-30_b0_bn_recompute_false-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_B0_FALSE_RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=backend:native \
python -u main.py \
  --cfg cfgs/ct_seqtrack/30_b0_mini_bn_recompute_false.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 \
  --tag ct30_b0_bn_recompute_false_mini_car_seed42_60ep_bs16 \
  --log_dir "$CT30_B0_FALSE_RUN" \
  > "$CT30_B0_FALSE_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT30_B0_FALSE_RUN/train.pid"
printf '日志：%s/train.log\n' "$CT30_B0_FALSE_RUN"
```

## GPU 1：B0，BN反向重算 true

```bash
CT30_B0_TRUE_RUN="output/$(date +%Y%m%d-%H%M%S)-30_b0_bn_recompute_true-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_B0_TRUE_RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=backend:native \
python -u main.py \
  --cfg cfgs/ct_seqtrack/30_b0_mini_bn_recompute_true.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 \
  --tag ct30_b0_bn_recompute_true_mini_car_seed42_60ep_bs16 \
  --log_dir "$CT30_B0_TRUE_RUN" \
  > "$CT30_B0_TRUE_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT30_B0_TRUE_RUN/train.pid"
printf '日志：%s/train.log\n' "$CT30_B0_TRUE_RUN"
```

## 新终端查看日志

新终端先进入项目根目录，再任选一条；以下命令查找各组最近日期目录，无需沿用启动终端的变量。
每个 `tail -f` 占用当前终端，Ctrl+C只退出日志查看。

Full-GRU（GPU 0）：

```bash
tail -n 100 -f "$(ls -dt output/*-30_full_gru-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

Full-CfC（GPU 2）：

```bash
tail -n 100 -f "$(ls -dt output/*-30_full_cfc-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

B0，BN反向重算 false（GPU 1）：

```bash
tail -n 100 -f "$(ls -dt output/*-30_b0_bn_recompute_false-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

B0，BN反向重算 true（GPU 1）：

```bash
tail -n 100 -f "$(ls -dt output/*-30_b0_bn_recompute_true-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

## 对照与结果使用

Full主比较使用B0 false，三臂的有效BN执行路径相同；B0 true作为额外的显存/计算取舍对照。
两份B0共用GPU1时，GPU整卡显存是任务占用的总和，迭代耗时也会互相影响；比较显存用每进程值或训练中的allocated/reserved计数。
这种并行安排的s/it不用于宣称重算的独立速度差异。

四组保留058/059/060；final固定60，late-3为58/59/60。
Full每个checkpoint训练后独立拟合并评测策略，训练过程中的未校准验证分数不能代替最终Full策略结果。
至少一个Full的final60 S/P同时超过同版本B0 false后进入full/KITTI，维持原实验顺序。

本轮配置合同测试52 passed；两份B0解析差异与四组CLI参数已核对。
此前两执行路径的定向数值回归为102 passed/2 skipped（CUDA），详见[显存预算说明](CTSEQTRACK_V30_MEMORY_TRADEOFF.md)。
配置检查允许按本页启动；修复后的GPU峰值、耗时及闭环S/P仍由这次服务器实跑取得。
