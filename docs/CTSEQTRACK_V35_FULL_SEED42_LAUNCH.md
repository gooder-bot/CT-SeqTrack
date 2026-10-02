# v35 完整链六组：seed42、两档学习率

2026-10-02。当前启动页；替代旧W六组的启动安排，旧实验全部保留。用户确认本轮在GPU0/1各同时运行三组，上传后先执行一次真实Full CUDA batch检查。代理只准备本地代码、配置和文档，对服务器只读；以下命令均由用户执行。

本地就绪结果：全仓644 passed、3 skipped，最终工具增量另行复核通过；旧配置、Full CPU batch16、完整epoch恢复和命令检查通过。[交付报告](../artifacts/ct_checks/20261002-211032_v35_full_readiness/REPORT.md)。真实CUDA检查尚待上传后由用户执行。

## 配置与预算

| 启动键 | 正式臂 | GPU | 配置（cfgs/ct_seqtrack/） | 第1–20 / 21–50 / 51–60轮LR |
|---|---|---:|---|---|
| scaled_b1 | B0+B1 | 0 | 35_b1_w_scaled_lr_mini.yaml | 1.5e-4 / 1.5e-5 / 1.5e-6 |
| scaled_b1_b2 | B0+B1+B2 | 0 | 35_b1_b2_w_scaled_lr_mini.yaml | 1.5e-4 / 1.5e-5 / 1.5e-6 |
| scaled_full | Full | 0 | 35_full_w_scaled_lr_mini.yaml | 1.5e-4 / 1.5e-5 / 1.5e-6 |
| normal_b1 | B0+B1 | 1 | 35_b1_w_normal_lr_mini.yaml | 1e-4 / 1e-5 / 1e-6 |
| normal_b1_b2 | B0+B1+B2 | 1 | 35_b1_b2_w_normal_lr_mini.yaml | 1e-4 / 1e-5 / 1e-6 |
| normal_full | Full | 1 | 35_full_w_normal_lr_mini.yaml | 1e-4 / 1e-5 / 1e-6 |

六组均训练seed42、固定数据划分seed42、mini Car、W=1/4/4/8、10轮课程、112 reserve、batch16、workers4、FP32、scratch60、CfC真实时间、无warmup、20/50各乘0.1。每组71,700次更新，总计360epoch、430,200更新。B0及启用模块共同从随机初始化训练，不加载旧B0权重，不冻结参数。本轮不改变网络、loss、采样、优化器或状态计算。

已有normal42/scaled42 B0结果复用；先按相同LR比较完整模块链，再单独选择Full配方。不重跑SeqTrack，不以各臂不同LR的最高分拼接模块消融表。原final60/late-3、固定R/C、移动及漂移风险口径保持。静态来源与六份配置身份见[登记JSON](../cfgs/ct_seqtrack/35_full_seed42_registration.json)，研究依据见[方案报告](../artifacts/ct_checks/20261002-205829_v35_seed42_full_plan/REPORT.md)。

## 同步与一次真实CUDA检查

本地由用户提交并推送全部本轮改动，包括`.gitattributes`、新源码、六配置、登记JSON、工具和文档。无需上传旧output、工程产物或任何初始化权重。按正常git status提交本轮文件即可，不做全仓换行转换或`git add --renormalize`；少量历史文件的原始CRLF字节属于既有source身份，保持原样。服务器活动项目：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
git -c core.autocrlf=false pull --ff-only origin main
```

若仍是此前GitHub DNS解析故障，保留已验证host key，用历史可用地址临时拉取：

```bash
git -c core.autocrlf=false -c core.sshCommand='ssh -o HostName=20.205.243.166 -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o UpdateHostKeys=no' pull --ff-only origin main
```

上面IP是此前可用的连接方式，并非承诺长期固定。不要改DNS、关闭host-key检查、修改代理或重装环境来运行本轮实验。

当前旧B0 CUDA检查工具明确只接收B0，因此新增下面的Full检查入口。它读取一个真实batch16，执行获取、前向、损失、反向、Adam和commit，验证候选记录JSON，并报告显存/耗时；自动使用独立ct_checks目录，**不保存checkpoint，不计入正式训练初始化**。

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u tools/check_v35_full_batch.py \
  --cfg cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --device cuda
```

终端返回`status: passed`及报告路径后，查看报告中的`cuda_memory`和`wall_seconds`。本地CPU验证不能替代服务器这次真实CUDA检查。单批峰值不是完整训练显存上限；先逐组启动并观察显存，再增加到每卡三组，维持batch16不变。如果单个Full已经占用接近15GiB，不能仅凭当前45GB空闲就认为三个进程一定容得下，应减少并发而不改变实验参数。

```bash
nvidia-smi
```

10月2日只读快照：两张A40各46,068MiB总显存、45,415MiB空闲，无计算进程；数据路径存在，Python3.9.19。该快照不保证之后仍空闲。

## 六条独立后台启动命令

下面脚本每次仅启动一组，内部就是`nohup env ... python -u main.py`，不执行循环或自动加实验。每条只执行一次；每次用日期＋随机后缀创建新output目录，保存`train.log`和`train.pid`，打印准确RUN路径。重复执行会产生新的独立训练，不会自动恢复旧任务。

先进入项目目录：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

1. B0+B1，1.5e-4，GPU0：

```bash
bash tools/launch_v35_full_seed42.sh scaled_b1 0
```

2. B0+B1+B2，1.5e-4，GPU0：

```bash
bash tools/launch_v35_full_seed42.sh scaled_b1_b2 0
```

3. Full，1.5e-4，GPU0：

```bash
bash tools/launch_v35_full_seed42.sh scaled_full 0
```

4. B0+B1，1e-4，GPU1：

```bash
bash tools/launch_v35_full_seed42.sh normal_b1 1
```

5. B0+B1+B2，1e-4，GPU1：

```bash
bash tools/launch_v35_full_seed42.sh normal_b1_b2 1
```

6. Full，1e-4，GPU1：

```bash
bash tools/launch_v35_full_seed42.sh normal_full 1
```

脚本固定现有seqtrack3d解释器、mini数据根、线程数1、native allocator、CUBLAS确定性配置、`--batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1`。配置已经给出seed42和tag；不传不受支持的`--preloading/--gpus/--devices`，不传checkpoint，不要求重新安装依赖。

## 新终端分别查看进度

进入`/home/lishengjie/study/lcyu/CT-SeqTrack`后，各执行相应一条。Ctrl+C仅结束tail。以下选择同配方最新目录；若曾重复启动，优先使用启动时打印的准确路径。

1. scaled B1：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b1_w_scaled_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

2. scaled B1+B2：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b1_b2_w_scaled_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

3. scaled Full：

```bash
tail -n 80 -f "$(ls -dt output/*-35_full_w_scaled_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

4. normal B1：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b1_w_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

5. normal B1+B2：

```bash
tail -n 80 -f "$(ls -dt output/*-35_b1_b2_w_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

6. normal Full：

```bash
tail -n 80 -f "$(ls -dt output/*-35_full_w_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

每5轮验证并保存逐帧诊断；训练结束自动评估58/59/60，生成`results.json`、`evaluation/epoch=058|059|060/frames.jsonl`。完整候选字段为`diagnostic_candidates`，格式见[候选记录说明](CTSEQTRACK_V35_FULL_CANDIDATES.md)。不要用中途最佳epoch替代正式结果。

## 同一实验恢复

启动脚本只负责新scratch。若任务中断，用原cfg、原log_dir及该运行最新**完整epoch**checkpoint恢复，保持本次源码，不跨臂、学习率、seed、版本或旧B0权重。下面是scaled Full示例，先把`CT_RUN`与checkpoint轮数改为实际值：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_RUN='output/替换为该次scaled_Full的准确目录'
CT_CKPT="$CT_RUN/formal_checkpoints/epoch=030.ckpt"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" --checkpoint "$CT_CKPT" \
  >> "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
```

不覆盖原manifest，不截断旧日志；旧B0在本轮源码下跨源码续训没有获得授权，也没有增加迁移白名单。新Full运行只允许同身份同源码的epoch恢复。

## 六组完成后的只读比较

以下在活动项目根执行；六个`CT_*`变量填写启动时保存的准确目录（无需为了比较恢复模型）。固定旧参考路径已列好；若迁移了output目录，仅相应调整路径，禁止替换参考内容。这个工具读取原帧、核对来源，不训练、不推理。`--output`只接受ct_checks下尚不存在的新文件；省略则只输出终端JSON。

```bash
python tools/compare_v35_full.py \
  --registered-reference output/20260924-193210-33_seqtrack_ref-mini_car_seed42_60ep_bs16 \
  --baseline output/20260924-193242-33_b0_half_lr-mini_car_seed42_60ep_bs16 \
  --old-b0 output/20260922-191248-32_b0-mini_car_seed42_60ep_bs16 \
  --old-w-quarter output/20260926-174406-34_b0_context_w4_quarter_lr-mini_car_seed42_60ep_bs16-4xyGiB \
  --normal-b0 output/20260928-193459-35_b0_w_normal_lr-mini_car_seed42_60ep_bs16-GXfluF \
  --scaled-b0 output/20260928-193506-35_b0_w_scaled_lr-mini_car_seed42_60ep_bs16-bJhtMt \
  --scaled-b1 "$CT_SCALED_B1" --scaled-b1-b2 "$CT_SCALED_B1_B2" --scaled-full "$CT_SCALED_FULL" \
  --normal-b1 "$CT_NORMAL_B1" --normal-b1-b2 "$CT_NORMAL_B1_B2" --normal-full "$CT_NORMAL_FULL" \
  --output "artifacts/ct_checks/$(date +%Y%m%d-%H%M%S)_v35_full_comparison/report.json"
```

返回0表示至少一个Full通过原固定门，1表示完整实验没有Full通过，2表示证据不完整/不匹配；是否超过最强scaled B0及漂移风险另列，不由返回0自动推断稳定性或全量数据收益。
