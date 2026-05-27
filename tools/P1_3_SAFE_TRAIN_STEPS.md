# P1-3 Safe Train-Step Check

This note records the low-interference command for checking that a few
training steps can run with finite loss.

## Why Not `main.py`

`main.py` uses Lightning with `devices=-1`, so it may consume all visible GPUs.
On a shared server, use `tools/check_train_steps.py` instead. It runs only a
small number of forward/backward/optimizer steps and writes loss after every
step.

## Real Timestamp Check On GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
timeout 20m nice -n 19 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 8 \
  --require-full-history \
  --memory-fraction 0.25 \
  --sleep-seconds 2 \
  --grad-clip 1.0 \
  --log-file output/p1_3_real_time_gpu0_loss.jsonl \
  --checkpoint-dir output/p1_3_real_time_gpu0_ckpt \
  --tag p1_3_real_time_gpu0
```

If GPU0 is still very busy, start with:

```bash
--max-steps 2 --memory-fraction 0.20
```

If that OOMs, the current shared load is not suitable for even a tiny training
check.

## Fixed Pseudo-Time Control

Run this only after the real timestamp check passes.

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
timeout 20m nice -n 19 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 8 \
  --require-full-history \
  --pseudo-time \
  --memory-fraction 0.25 \
  --sleep-seconds 2 \
  --grad-clip 1.0 \
  --log-file output/p1_3_pseudo_time_gpu0_loss.jsonl \
  --checkpoint-dir output/p1_3_pseudo_time_gpu0_ckpt \
  --tag p1_3_pseudo_time_gpu0
```

## Output Files

Each completed step appends one JSON object:

```text
output/p1_3_real_time_gpu0_loss.jsonl
output/p1_3_pseudo_time_gpu0_loss.jsonl
```

Each completed step also overwrites:

```text
output/p1_3_real_time_gpu0_ckpt/last.pt
output/p1_3_pseudo_time_gpu0_ckpt/last.pt
```

So if the run is interrupted, the finished loss records are still preserved.
