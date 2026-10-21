# v37 五组启动、恢复、评测与同卡测速

**2026-10-21 状态：本页原五组已完成。** 下方 scratch 命令保留作历史操作与复现，不表示需要再次启动。结果及剩余问题见 [当前研究审查](CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md)；Full FP32 未过原门。旧 checkpoint 必须使用其原登记源码/配置/精度，不能套用后来的 B1 执行配方。正式训练完成不自动证明独立 CUDA 配对检查或测速已完成，本次没有操作服务器或扩充预算。

2026-10-09。用户在服务器自行执行，代理本轮只修改本地。工作目录始终为活动 `CT-SeqTrack/`；无需安装或升级服务器环境。源码/五组身份见 `cfgs/ct_seqtrack/37_registration.json`，结构见 [v37 设计](CTSEQTRACK_V37_DESIGN.md)。旧配置、权重、output 和 artifacts 保留。

## 配方与两项短工程检查

五组均 mini Car、seed42、batch16、60 epoch scratch、LR1.5e-4、20/50 衰减、无 warmup，W=1/4/4/8。每组71,700次更新，总358,500次；BF16 是独立 Full 执行对照，不能挑它替代 Full FP32 主结果。

```bash
export CT_PYTHON=/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python
export CT_DATA_ROOT=/home/lishengjie/data/nuscenes-mini
```

上传本地最新版后，先做一次 Full FP32 的真实 CUDA batch 生命周期，再做 Full FP32/BF16 的短数值检查。两项都使用真实 batch16，不保存权重；报告自动写入新的 `artifacts/ct_checks/`。不需要在服务器重跑全仓测试。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_v37_full_batch.py \
  --cfg cfgs/ct_seqtrack/37_full_mini.yaml --path /home/lishengjie/data/nuscenes-mini --device cuda
```

短 BF16 检查分别覆盖初始窗口和成熟窗口阶段（零基 epoch 0/10，各 8 个训练步），记录有限 loss/梯度/Adam、支持掩码和闭环数值差异，不要求逐位相等。通过表示局部 BF16 工程可运行，最终分数仍由该组正式60轮验证。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_execution_performance.py \
  --cfg cfgs/ct_seqtrack/37_full_mini.yaml --path /home/lishengjie/data/nuscenes-mini --device cuda \
  --candidate_profile optimized_bf16 --baseline_profile optimized_fp32 \
  --mode bf16 --check_steps 8 --check_epochs 0 10
```

## 五组分别启动与查看日志

每组下面都是完整独立命令，可在单独终端复制。每次 `mktemp` 创建新的 RUN，`train.pid` 保存进程号；记下最后打印的目录。当前示例 GPU 分配为 B0/B1→0、B1+B2/Full FP32/Full BF16→1，按实测显存分批启动，五组无需同时运行。

新终端的日志命令按组名和目录时间戳选择最新一次该组运行。若需要查看较早运行，直接把 `CT_RUN` 设为启动时打印的原目录，再执行 `tail -n 80 -f "$CT_RUN/train.log"`。

### B0（GPU 0）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b0-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b0_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$CT_RUN/train.pid"
echo "$CT_RUN"
```

新终端查看 B0 最新运行日志：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b0-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### B1（GPU 0）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$CT_RUN/train.pid"
echo "$CT_RUN"
```

新终端查看 B1 最新运行日志：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### B1+B2（GPU 1）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_b2-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_b2_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$CT_RUN/train.pid"
echo "$CT_RUN"
```

新终端查看 B1+B2 最新运行日志：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_b2-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### Full FP32（GPU 1）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_full-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$CT_RUN/train.pid"
echo "$CT_RUN"
```

新终端查看 Full FP32 最新运行日志：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_full-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### Full BF16（GPU 1）

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_full_bf16-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_bf16_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"$CT_RUN/train.pid"
echo "$CT_RUN"
```

新终端查看 Full BF16 最新运行日志：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_full_bf16-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

启动前 entry 核对注册源码、完整配置与 execution profile；禁止 `--init_checkpoint`。metadata 保存共同参数初始化摘要、run_id、源码内容哈希、配置和数值执行身份。正式 BF16 不能用于旧版或 v37 B0/B1/B1+B2；设备不支持时失败，不静默回退后冒充 BF16。

## 同一运行完整 epoch 恢复

使用原源码、原配置、原 RUN 的完整 epoch checkpoint。示例恢复 Full FP32 第10轮，继续第11轮；实际把 RUN/epoch 换成该组真实记录。BF16 组必须保持 `37_full_bf16_mini.yaml` 和原精度身份。

```bash
RUN=output/实际原Full运行目录
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$CT_PYTHON" -u main.py --cfg cfgs/ct_seqtrack/37_full_mini.yaml \
  --path "$CT_DATA_ROOT" --log_dir "$RUN" \
  --checkpoint "$RUN/formal_checkpoints/epoch=010.ckpt" --accelerator gpu \
  >"$RUN/resume-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &
```

保留最初 manifest/resolved config，恢复事件追加到 `resume_manifests/`。run、源码、配置、精度、数据、初框政策、W 排程、完整计数及 RNG 不匹配都拒绝；不迁移旧 v35/v36 权重，不接续半个状态链。

其余各组也须使用各自原 RUN 和原配置。需要恢复多种臂时可先在 shell 定义一次下方函数，再仅执行实际中断的那一组调用；它不扫描、停止或自动批量恢复进程。最后的 `010` 是完整 checkpoint 编号，按实际记录更改。

```bash
ct37_resume() {
  local arm="$1" gpu="$2" run="$3" epoch="$4"
  local cfg="cfgs/ct_seqtrack/37_${arm}_mini.yaml"
  local ckpt="$run/formal_checkpoints/epoch=${epoch}.ckpt"
  test -f "$cfg" && test -f "$ckpt" || return 2
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$CT_PYTHON" -u main.py --cfg "$cfg" --path "$CT_DATA_ROOT" --log_dir "$run" \
    --checkpoint "$ckpt" --accelerator gpu \
    >"$run/resume-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &
  printf 'resume PID=%s RUN=%s\n' "$!" "$run"
}
```

B0（原配置 `37_b0_mini.yaml`）：

```bash
ct37_resume b0 0 "$B0_RUN" 010
```

B1（原配置 `37_b1_mini.yaml`）：

```bash
ct37_resume b1 0 "$B1_RUN" 010
```

B1+B2（原配置 `37_b1_b2_mini.yaml`）：

```bash
ct37_resume b1_b2 1 "$B1_B2_RUN" 010
```

Full FP32（原配置 `37_full_mini.yaml`；若已执行上方直接恢复示例，不再执行此行）：

```bash
ct37_resume full 1 "$FULL_RUN" 010
```

Full BF16（原配置 `37_full_bf16_mini.yaml`）：

```bash
ct37_resume full_bf16 1 "$FULL_BF16_RUN" 010
```

独立评测输出到新的评测目录；训练完成已自动评测58/59/60，以下用于按原配置重做单个 checkpoint 的只读评测：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
"$CT_PYTHON" -u main.py --cfg cfgs/ct_seqtrack/37_full_mini.yaml \
  --path "$CT_DATA_ROOT" --checkpoint "$RUN/formal_checkpoints/epoch=060.ckpt" --test --accelerator gpu
```

## 同卡、同样本配对测速

这是独立的速度证据采集，可在空闲卡/空闲时段单独执行，不作为五组正式训练的共同启动前置。

在同一空闲 GPU 上配对 Full optimized_fp32 与 Full optimized_bf16，保留真实 batch16、诊断、训练 logger 和闭环提交。训练及单独闭环推理均20预热+100计时、3次重复；只报告实测加速与显存，不把旧历史速度当本次对照。不用于初始化正式训练。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
"$CT_PYTHON" tools/check_execution_performance.py \
  --cfg cfgs/ct_seqtrack/37_full_mini.yaml --path "$CT_DATA_ROOT" --device cuda \
  --candidate_profile optimized_bf16 --baseline_profile optimized_fp32 \
  --inference --mode benchmark --warmup 20 --steps 100 --repeats 3
```

BF16 工程比较报告数值偏差、有限 loss/梯度/Adam、支持掩码/状态变化与速度，不宣称逐位等价。最终精度影响由独立60轮 BF16 对照的 final60/late-3 检验。工程权重丢弃，正式五组从 epoch0 重新初始化。

## 五组固定比较

以下变量是原 RUN 的实际路径，不能替换 seed、LR 或较弱参考。强旧 B0/B1 的建议路径已在 registration `reused` 中钉住，其 config/source 和末三轮帧 SHA 必须一致。

```bash
R_RUN=output/20260924-193210-33_seqtrack_ref-mini_car_seed42_60ep_bs16
OLD_C_RUN=output/20260924-193242-33_b0_half_lr-mini_car_seed42_60ep_bs16
OLD_B0_RUN=output/20260922-191248-32_b0-mini_car_seed42_60ep_bs16
OLD_W_RUN=output/20260926-174406-34_b0_context_w4_quarter_lr-mini_car_seed42_60ep_bs16-4xyGiB
STRONG_B0_RUN=output/20260928-193506-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-bJhtMt
STRONG_B1_RUN=output/20261002-214602-35_b1_w_scaled_lr-mini_car_seed42_60ep_bs16-6kP66o
# B0_RUN / B1_RUN / B1_B2_RUN / FULL_RUN / FULL_BF16_RUN：上面五次启动分别打印的 CT_RUN。
"$CT_PYTHON" tools/compare_v37.py \
  --registered-reference "$R_RUN" --baseline "$OLD_C_RUN" \
  --old-b0 "$OLD_B0_RUN" --old-w-quarter "$OLD_W_RUN" \
  --strong-b0 "$STRONG_B0_RUN" --strong-b1 "$STRONG_B1_RUN" \
  --b0 "$B0_RUN" --b1 "$B1_RUN" --b1-b2 "$B1_B2_RUN" \
  --full "$FULL_RUN" --full-bf16 "$FULL_BF16_RUN" \
  --output "artifacts/ct_checks/$(date +%Y%m%d-%H%M%S)_v37_comparison/report.json"
```

退出码0表示 Full FP32 通过原固定 R/C 门，1表示正式主模型未通过，2表示身份/预算/原帧证据无效或缺失。工具核验五个独立run、60轮审计、358,500更新、共同W排程与原帧，重算final60/late-3，输出新B1对旧强B0/B1、各级候选链、>5m/>10m与失跟风险。通过平均指标不自动代表漂移问题解决；BF16或其他组较高分不改变预登记主模型。

目前只有实现和本地工程验证，尚无 v37 正式五组成绩；真实 CUDA 检查、同卡测速和五组训练由用户执行后才形成新证据。
