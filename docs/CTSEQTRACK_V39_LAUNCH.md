# v39：E＋B1＋M＋G＋自适应跟踪

2026-10-28 部署修正：`git pull`若出现`Aborting`，只完成下载，工作目录仍可能是混合旧版。本次用户已授权代理备份服务器源码并更新到匹配的最新版；不自动启动GPU任务。`ctseqtrack3d`已由用户创建，无需再次克隆。真实batch和profile现在先校验源码登记，失败会列出具体差异文件；不得在服务器`--refresh`掩盖混合版本。

2026-10-28 最新实施版。基础是本地 v38 clean1；所有新模块、损失和闭环策略从第一个 epoch 启用。1/4/4/8 与前10轮窗口课程保留，M 的2/4跨度独立于窗口长度。用户最新要求：环境名`ctseqtrack3d`，两组同GPU0并行，OMP/MKL/OPENBLAS各2，workers4及其他训练配方不变。用户已创建ctseqtrack3d，本次授权代理备份并更新服务器源码；GPU任务由用户执行。详细参数见[配置说明](CTSEQTRACK_V39_CONFIG.md)。

## 代码和事务

- `models/ct_v39/data.py`：metadata endpoint、state-only 输入、GPU crop/采样/seg/BC、模板和唯一原子提交。
- `raw_service.py`：一个主进程 owner 的 CPU 512MiB/GPU 256MiB raw LRU，四个解码线程；frame-local FP32，原点差 FP64；目标标签不进共享 raw cache。
- `efficient_observation.py`、`attention.py`：有效点 packing、保留 masked BN/首次最大值池化梯度，4×64 attention、FA2 varlen/deterministic，空 query 清零。正式后端缺依赖时直接失败。
- `model.py`：state-only B1 → 当前 G → route/search → 按需 B0 → 七槽统一核验 → 提交证据。G 不读取 B0 特征/前景，也不读取 GT。
- `g_observation.py`、`g_losses.py`、`tensor_geometry.py`：首模板＋两个可信模板、局部编码、5轮 dustbin Sinkhorn、有界 yaw/平移纠偏、目标中心投票及两种 GPU IoU。
- `prior.py`、`multistep.py`、`losses.py`：删除 acquisition head/21维统计/9×9标签/损失；保留一步物理损失，增加 h=2/4 metadata 多跨度监督，未来 GT 仅进入损失。
- `models/ctseqtrackv39.py`、`runtime.py`：Adam 后提交、严格恢复身份、七槽和恢复逐帧诊断。

训练每个 endpoint 仍完整计算 B0；稳定 hash 选中25%整窗按实际预测轻重路由提交，其余整窗按完整路径提交。训练中的 teacher B0 计算不刷新轻路径的上次完整观测时间。G 路由前向不保留计算图，纠偏监督在相同原始观测池重算；匹配内用 checkpoint 控制显存，语义不变。

正常 G 双中心各256点、去重后最多512点；触发恢复时增加最多两个256点精化区，联合池最多1024点。每帧仍最多一轮粗搜索和一次 B0 前向。G(q0) 在已有当前证据中核验，不额外从 q0 裁剪更多观测。固定模板不覆盖；推理不执行 M 辅助头。

这次针对原 profile 的51%路径，既删除 acquisition 整条辅助任务，也把动态 crop、采样、标签和几何搬到 GPU，并合并诊断回传、缓存原云和延迟历史物化。它不等同于“只优化 GPU/CPU 传输”；实际消除多少耗时需配对 profile，不能从删除代码直接推算速度。

## 环境及验收

最新环境目标：`/home/lishengjie/miniconda3/envs/ctseqtrack3d`。服务器已有环境来源`ctseqtrack_v39`，实测Python3.9.19 / torch2.0.1+cu118 / Lightning2.0.2 / flash-attn2.5.8，扩展可导入但尚未执行CUDA。用户现已创建新名称；下面仅保留首次尚未创建时的克隆命令，无需重复执行，不直接`mv`含绝对前缀的Conda环境：

```bash
/home/lishengjie/miniconda3/bin/conda create -n ctseqtrack3d \
  --clone /home/lishengjie/miniconda3/envs/ctseqtrack_v39 -y
```

公共`seqtrack3d`不升级。正式特征BF16，主参数/Adam/B1/BN统计/几何/回归及损失FP32。上传本地完整代码与`39_registration.json`后，首次启动前完成一次真实CUDA检查及profile：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=backend:native
PYTHON=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
$PYTHON tools/register_v39.py --verify && \
$PYTHON tools/check_v39_full_batch.py --cfg cfgs/ct_seqtrack/39_full_a_mini.yaml --path /home/lishengjie/data/nuscenes-mini --device cuda:0 --threads 2 && \
$PYTHON tools/benchmark_v39.py --path /home/lishengjie/data/nuscenes-mini --training --steps 12 --warmup 2 --output "artifacts/ct_checks/$(date +%Y%m%d-%H%M%S)-v39-paired-profile"
```

工程产物只写`artifacts/ct_checks/`。完整batch检查会确认实际FA2函数调用、QKV BF16、有限梯度、FP32出口、Adam后提交和GPU/CPU几何审计。FA2数值参考与packing专项可运行`python -m pytest -q tests/test_ct_v39_efficiency.py`，CUDA项skip不能算已通过。配对profile使用相同初始化与端点顺序；BF16及递推可能改变crop工作量，因此记录端点和工作量hash，不宣称同张量因果提速。plain计时与单独trace分离；包含原云、准备、搜索、模型、损失、Adam和提交。正式权重另测推理可传`--checkpoint`，速度档加`--inference-mode adaptive`。

**当前可以上传，尚不能声称已通过GPU验收。** 两进程同卡是否放得下需看真实batch/profile的峰值显存；当前尚无该测量。共享GPU时间必须标注共享状态。工具不自动改batch、学习率或网络来迁就显存。

CPU 合成工具只验证生命周期，不提供真实 nuScenes 精度或 FA2 速度证据。旧 v38 源码封存于 `artifacts/ct_checks/20261028_v39_implementation/baseline_v38_clean1.zip`，旧产物和旧登记不改；旧 checkpoint 必须配对应源代码。

## 两组正式运行

共同 mini Car、seed42、batch16、scratch60、每轮精度档验证，Adam betas=(.5,.999)、eps=1e-6、wd=0，没有 warmup。

| epoch | Full-A | Full-B |
|---|---:|---:|
| 1–20 | 1.5e-4 | 1e-4 |
| 21–40 | 1.5e-5 | 2e-5 |
| 41–50 | 1.5e-5 | 4e-6 |
| 51–60 | 1.5e-6 | 4e-6 |

两组初始参数hash一致；源码和配方已独立登记，跨配方/跨源码恢复拒绝。下面两条命令可分别执行，均后台运行在物理GPU0。`check_val_every_n_epoch`保持v39的1，不沿用旧示例的5。环境线程上限改为2，`workers=4`不变。

Full-A：1.5e-4，完成20/50轮后×0.1。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_A="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-39_full_a_mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/39_full_a_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_A" \
  > "$CT_A/train.log" 2>&1 < /dev/null &
echo $! > "$CT_A/train.pid"
echo "$CT_A"
```

Full-B：1e-4，每完成20轮×0.2。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_B="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-39_full_b_mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/39_full_b_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_B" \
  > "$CT_B/train.log" 2>&1 < /dev/null &
echo $! > "$CT_B/train.pid"
echo "$CT_B"
```

新开终端看A（自动选择最新A目录）：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_A="$(ls -dt output/*-39_full_a_mini_car_seed42_60ep_bs16-* | head -n 1)"
tail -n 60 -f "$CT_A/train.log"
```

新开另一终端看B：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT_B="$(ls -dt output/*-39_full_b_mini_car_seed42_60ep_bs16-* | head -n 1)"
tail -n 60 -f "$CT_B/train.log"
```

`launch_v39_sequential.sh`仅保留作可选串行辅助，不用于上述并行启动。

入口仍是 `main.py`；若独立运行，只使用 `39_full_a_mini.yaml` 或 `39_full_b_mini.yaml`。训练结束自动评价58–60精度档和默认速度档；60轮额外扫描速度阈值.8/.95，.9复用默认结果。输出完整 `frames.jsonl`、七候选、当前唯一支持、错误可信写入、>10m漂移、恢复及 B0 调用统计；最终 `results.json` 保留两档和阈值曲线，不挑各自最佳轮。

```bash
python tools/compare_v39.py output/实际A目录 output/实际B目录 --output artifacts/ct_checks/v39_compare/comparison.json
```

主比较 final60 和 late3（58–60）。先按原总体/运动门判断，再比较 late3 的 S/P 均值，final60 与严重漂移作次级判断。这两组只能比较完整学习率方案，不能归因单独 E/G/M 贡献。
