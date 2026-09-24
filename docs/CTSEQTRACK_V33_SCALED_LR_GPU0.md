# v33 追加 D：学习率放大 1.5 倍，GPU 0

2026-09-25，用户希望再运行一组适度增大学习率的综合 B0。本组相对 B 配方全程学习率乘1.5，保留20/50轮衰减，原A/B/C配置保持。

| 训练轮次 | B | D：本组 |
|---|---:|---:|
| 1–20 | 1e-4 | 1.5e-4 |
| 21–50 | 1e-5 | 1.5e-5 |
| 51–60 | 1e-6 | 1.5e-6 |

采用1.5倍作为适度向上探索的配方，不预判优于B。现有训练审查没有证明1e-4过大；保留第20轮降档和最后10轮低学习率收尾，让D与B直接比较步长尺度。具体1.5倍是本轮实验选择，不是从total loss比例推导的最优值。Adam依据梯度矩估计自适应更新，不能把跨模型total loss比值直接当作学习率换算依据，见[Adam原论文](https://arxiv.org/abs/1412.6980)。历史证据见[训练审查](../artifacts/ct_checks/20260924_v32_four_arm_final/training/TRAINING_REVIEW.md)。

配置为`cfgs/ct_seqtrack/33_b0_scaled_lr_mini.yaml`，继承B配置；模型、loss、数据、seed42、batch16、workers4、FP32、Adam其余参数与60轮/71,700次更新预算保持一致，每5轮验证，结束自动评测58/59/60。与B比较final60和late-3的S/P，保留移动目标和失跟诊断。

本地验证：`tests/test_ct_v33_recipes.py tests/test_ct_v33_entry_metadata.py`共20 passed，覆盖60轮LR序列、epoch40恢复、同seed初始化、配方身份以及reference/Full不误用D；不追加模型训练探针。

## 上传与启动

本轮只修改本地，未操作服务器。服务器已具备当前v33时，至少同步以下两个文件：

- `cfgs/ct_seqtrack/33_b0_scaled_lr_mini.yaml`
- `models/ct_v31/config.py`

第二个文件登记了新增正式配方；只上传YAML会被旧入口以`unregistered formal v33 learning-rate recipe`拒绝。基础依赖为已有`33_b0_late_decay_mini.yaml`与`33_formal_base.yaml`。保持现有环境，从头训练。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_b0_scaled_lr_x1p5-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_b0_scaled_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

## 新终端看进度

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-33_b0_scaled_lr_x1p5-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

Ctrl+C只退出tail，后台训练继续。最终结果位于该run的`results.json`，`[v33 complete]`表示训练与58–60评测完成。本组GPU0通过环境变量指定，`trainer_devices=1`仍表示单卡。
