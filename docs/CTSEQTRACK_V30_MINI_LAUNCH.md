# v30 mini 最新三组后台命令（2026-09-15）

本轮轻量复核未发现新的本地启动阻断。v30本地实现已完成；此前完整回归868 passed/15 skipped。
三组真实CLI解析及正式配置合同通过，历史CE/AP/池化修复保留。实际服务器CUDA与60轮成绩尚未执行验证。
本次定向复核74 passed/1 skipped（12.04秒），仅CUDA运行测试跳过；本页6个Bash代码块均通过语法检查。
本次仅交付本地修改和命令，不自动同步服务器、启动训练或停止任何任务。

## 配置与参数

| 物理GPU | 实验 | 最新mini配置 |
|---|---|---|
| 0 | B0 | `cfgs/ct_seqtrack/30_b0_mini.yaml` |
| 1 | Full-GRU | `cfgs/ct_seqtrack/30_full_gru_mini.yaml` |
| 2 | Full-CfC | `cfgs/ct_seqtrack/30_full_cfc_mini.yaml` |

共同继承`30_formal_base.yaml`及历史基础配置：mini / Car、seed42、scratch60、batch16、workers4、val5，
Adam lr=1e-4、betas=(.5,.999)、eps=1e-6、wd=0、StepLR20×.1、FP32、strict deterministic。
每进程`trainer_devices=1`，`CUDA_VISIBLE_DEVICES`选择物理卡；日志中各自显示`cuda:0`是正常映射。
保留058/059/060 checkpoint。B0、Full-GRU和Full-CfC都从随机初始化开始。

**不要带旧示例中的`--preloading`**：v30使用每worker256MiB有界点云缓存，传该参数会直接报错。
数据根是`/home/lishengjie/data/nuscenes-mini`，不使用旧`seqtrack3d_nuscenes_mini.yaml`或v29 full配置。

先将本地最新项目代码和新增文件同步到服务器，在原训练环境、CT-SeqTrack项目根目录执行下列命令。
不能只同步三份小YAML：它们还依赖新的models/utils/datasets及基础配置。无需额外工程报告门。
历史记录的项目位置是`~/study/lcyu/CT-SeqTrack`，若当前路径不同，用实际项目目录。

## GPU 0：B0

```bash
CT30_B0_RUN="output/$(date +%Y%m%d-%H%M%S)-30_b0-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_B0_RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
python -u main.py \
  --cfg cfgs/ct_seqtrack/30_b0_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 \
  --tag ct30_b0_mini_car_seed42_60ep_bs16 \
  --log_dir "$CT30_B0_RUN" \
  > "$CT30_B0_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT30_B0_RUN/train.pid"
printf '日志：%s/train.log\n' "$CT30_B0_RUN"
```

新开终端、进入同一项目目录后查看最近一次B0：

```bash
tail -n 100 -f "$(ls -dt output/*-30_b0-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

## GPU 1：Full-GRU

```bash
CT30_GRU_RUN="output/$(date +%Y%m%d-%H%M%S)-30_full_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_GRU_RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
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

新终端查看最近一次Full-GRU：

```bash
tail -n 100 -f "$(ls -dt output/*-30_full_gru-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

## GPU 2：Full-CfC

```bash
CT30_CFC_RUN="output/$(date +%Y%m%d-%H%M%S)-30_full_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$CT30_CFC_RUN"
nohup env CUDA_VISIBLE_DEVICES=2 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
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

新终端查看最近一次Full-CfC：

```bash
tail -n 100 -f "$(ls -dt output/*-30_full_cfc-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

`Ctrl+C`只结束tail查看，不会停止nohup训练。每组独立目录含日期、模块、mini、Car、seed、轮数和batch；
`train.log`包含stdout/stderr，`python -u`及时写出Python日志，`train.pid`保存对应进程PID。

## 历史错误复核及训练后安排

- `nll_loss2d_forward_out_cuda_template`：原三维log_softmax→二维NLL路径保留；v30 mask只控制有效标签与类别权重分母。
- `cumsum_cuda_kernel`：AP二值计数保留int64 cumsum；不恢复浮点累计。
- 固定槽max、首个最大值梯度、FP32、严格确定性及Adam配置保留；新共识不用浮点scatter累加。
- 旧`tqdm.__del__`是主RuntimeError后的清理异常；本轮没有用修改tqdm掩盖主错误。

这些是代码及本地检查结论，不代表新v30已在服务器通过CUDA整轮训练。
训练期间Full没有已拟合策略时，验证可能回退observation；这不能直接判为Full最终无收益。
训练后分别对两Full的058/059/060独立校准再正式评测，见[完整运行手册](V30_DATA_AND_RUNBOOK.md)。
mini最终晋级只看至少一个Full的final60 S/P是否同时严格超过同版本B0，late-3只报告。
