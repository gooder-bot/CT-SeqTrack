# v38 三组 mini：检查、启动、恢复与比较

2026-10-22 上传前复核版。新方法见 [设计说明](CTSEQTRACK_V38_DESIGN.md)。本页是当前配置和命令入口：两组 B1 使用物理 GPU1，Full 使用物理 GPU0；均为单进程单卡。已按授权只读检查服务器，未上传、安装或启停任务；上传、一次真实 CUDA 检查和正式训练由用户执行。

| 配方 | 配置 | 完成哪些轮后LR乘0.1 | 物理 GPU |
|---|---|---|---|
| B1 standard | `cfgs/ct_seqtrack/38_b1_standard_mini.yaml` | 20、50 | 1 |
| B1 hold | `cfgs/ct_seqtrack/38_b1_hold_mini.yaml` | 20 | 1 |
| Full standard（主结果） | `cfgs/ct_seqtrack/38_full_standard_mini.yaml` | 20、50 | 0 |

全部为concat、mini Car、seed42、scratch60、batch16、FP32、workers4、每轮验证、LR1.5e-4、原Adam/W课程。每组71,700更新，合计180epoch/215,100更新。hold与standard前50轮实际LR相同，第51轮开始不同。每组独立随机初始化，不能由旧checkpoint或第50轮分叉。本次hold不是历史factorized hold。

`--devices` 是物理GPU编号，`--trainer_devices 1` 是使用一张卡，Full也必须保持后者为1。验证频率保持批准方案的 `--check_val_every_n_epoch 1`，不使用旧v35示例的5。GPU和数据/输出路径可选，以上正式训练参量保持固定；旧v37的六字段研究变体不属于本轮三组。

20:28只读快照确认 Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、nuscenes-devkit1.1.9及mini数据路径正常；服务器尚无v38文件，没有CT-SeqTrack训练进程。A40 GPU0/1分别空闲16,495/27,386MiB，另有其他任务，资源会变化；这不是新Full或两组B1并发显存的实测保证。无需升级环境。记录见 [上传前复核](../artifacts/ct_checks/20261022_v38_upload_readiness/REPORT.md)。

## 准备与一次真实 CUDA 检查

将完整新版源码、三份 `38_*_mini.yaml` 和 `38_registration.json` 一起上传到 `/home/lishengjie/study/lcyu/CT-SeqTrack`，不要只上传YAML或模型文件。旧输出保留；旧checkpoint恢复仍使用其对应的封存源码。以下命令均从该项目根目录执行，Python与数据路径已只读核实，详见 [服务器路径](SERVER_PATHS.md)。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
export CT_PYTHON=/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python
export CT_DATA_ROOT=/home/lishengjie/data/nuscenes-mini
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 "$CT_PYTHON" tools/check_v38_full_batch.py \
  --cfg cfgs/ct_seqtrack/38_full_standard_mini.yaml \
  --path "$CT_DATA_ROOT" --device cuda
```

检查覆盖真实batch16×4×1024点、forward/loss/backward/Adam/commit、梯度和三候选台账，报告写到新的 `artifacts/ct_checks/` 文件。它不保存工程checkpoint；正式训练重新启动scratch进程。若失败，应修复后更新新v38源码登记再启动，不绕过检查或修改旧source身份。

## 启动

以下三块分别执行，每块都可以在独立SSH终端使用，自动创建日期、时分秒和随机后缀的新目录。两个B1都放在GPU1；若同时运行显存不足，让其中一组完成后再启动另一组，batch16和其余训练参量保持不变。

### B1 standard：GPU1，20/50轮后衰减

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-38_b1_standard-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/38_b1_standard_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 1 --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### B1 hold：GPU1，仅20轮后衰减

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-38_b1_hold-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/38_b1_hold_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 1 --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### Full：GPU0，20/50轮后衰减

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-38_full_standard-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/38_full_standard_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

已有的 `tools/launch_v38.sh` 是等价简写，参数分别为 `b1_standard 1`、`b1_hold 1`、`full_standard 0`。使用上面nohup后无需再次执行启动器。

## 新终端查看进度

在本地新终端先连接：

```bash
ssh lishengjie@10.109.253.86
```

然后选择对应一块；自动找到该配方日期最新的目录。`Ctrl+C` 只退出tail，不停止训练。若要看之前某次运行，将CT_RUN设为那次启动时打印的路径。

B1 standard：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(ls -dt output/*-38_b1_standard-mini_car_seed42_60ep_bs16-* | head -n 1)"
tail -n 100 -f "$CT_RUN/train.log"
```

B1 hold：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(ls -dt output/*-38_b1_hold-mini_car_seed42_60ep_bs16-* | head -n 1)"
tail -n 100 -f "$CT_RUN/train.log"
```

Full：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN="$(ls -dt output/*-38_full_standard-mini_car_seed42_60ep_bs16-* | head -n 1)"
tail -n 100 -f "$CT_RUN/train.log"
```

## 同run完整epoch恢复与单独评测

确认原进程已结束，使用原RUN、同配置和完全匹配源码。完整epoch检查同时核验run_id、实际配置、source、执行路径、sampler、预算与RNG。中断在epoch内则从最近完整checkpoint重新执行下一轮，不跳过剩余数据。

```bash
RUN=/absolute/path/to/original/run
CHECKPOINT="$RUN/formal_checkpoints/epoch=020.ckpt"
bash tools/resume_v38.sh full_standard 0 "$RUN" "$CHECKPOINT" --path "$CT_DATA_ROOT"
```

恢复另建resume日志，不覆盖train.log；按恢复命令打印的新日志路径tail。其他两组替换recipe并使用GPU1。单独评测需要新目录：

```bash
"$CT_PYTHON" main.py --cfg cfgs/ct_seqtrack/38_full_standard_mini.yaml \
  --path "$CT_DATA_ROOT" --devices 0 --checkpoint "$RUN/formal_checkpoints/epoch=060.ckpt" \
  --test --log_dir /absolute/path/to/a/new/evaluation
```

## 固定比较

三组都完成后运行 `"$CT_PYTHON" tools/compare_v38.py --help`，提供三组RUN与原固定R/C参考。工具严格核对三组独立运行、60轮审计、71700更新、source/config和58/59/60原帧，重新计算final60/late-3及固定R/C门。Full是主结果；hold结果不能替换Full的standard B1对照。

```bash
"$CT_PYTHON" tools/compare_v38.py \
  --registered-reference /absolute/path/to/original/R \
  --baseline /absolute/path/to/original/C \
  --b1-standard /absolute/path/to/new/B1-standard \
  --b1-hold /absolute/path/to/new/B1-hold \
  --full-standard /absolute/path/to/new/Full \
  --output artifacts/ct_checks/v38_final_review/comparison.json \
  --reliability-output artifacts/ct_checks/v38_final_review/trusted_reliability.png
```

输出路径须未存在。可靠性图为58/59/60各自独立面板，附同名 `.data.json`，空箱不伪造成0事件率。退出码0表示Full通过原门，1表示正式结果未通过，2表示证据缺失或身份无效。

同时检查三候选接管与漂移、真实恢复、可信写入错误与覆盖、速度恢复、运动/低运动/启停、T的Brier与固定十分箱可靠性。单步oracle是同状态被动诊断，不是闭环成绩。保留负结果，不追加训练或阈值选择。
