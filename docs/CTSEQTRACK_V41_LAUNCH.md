# V41 KITTI Car 运行入口与历史操作记录

## 2026-11-18 当前运行

v3 的训练、同 run 恢复和 q0 开发集比较使用 [V41 v3 运行流程](CTSEQTRACK_V41_RUN_WORKFLOW.md)。`uniform` 与 `inverse_lwh` 两组已分别从头启动。北京时间 11 月 18 日 18:26:18 的日志记录首轮 1,880/4,771 和 1,913/4,771 步。两臂的固定 dev 结果待齐备后配对。[正式运行快照](../artifacts/ct_checks/20261118_v41_live_speed_1814/REPORT.md)

下方命令供复现 v1 Full 与原始 SeqTrack3D 方法的 KITTI 适配基线。该次两系统比较已完成，固定 58/59/60 轮使用同一组 test 0019–0020 原始帧。读取旧 checkpoint 时使用其原 run 的源码和配置。[v1 固定测试结果](../artifacts/ct_checks/20261117_kitti_final_research_review/test_review_overview.md)

## 历史操作 · 2026-11-16 exec v2

当时的 `exec v2` 引入新的训练、恢复、独立 dev 评估和输出规则。默认每轮 test 与完整 checkpoint，best Precision/last 和最近 K 个评估轮取代新 run 的固定60轮报告约束。本轮只完成源码/文档与静态审阅，未部署、未访问服务器、未运行 pytest/compileall 或模型。

下方保留 **v1 两组首次启动与历史快照**，不作为 v2 新 run 命令依据。旧 run 不改配方，旧 checkpoint 继续使用与其 manifest 匹配的原源码；v2 不能续训 v1。

## 历史操作 · 2026-11-08 V41 v1

当时可引用的运行快照为 **2026-11-04 10:46:53（北京时间）**，不是今日实时状态：Full完成9/60轮、第10轮2901/4771 batch；原始SeqTrack3D方法＋KITTI适配已完成60轮。Full第9轮dev为9.4719/7.7493；baseline dev final60为38.8072/50.7109、late3为35.7262/46.2808。尚未找到两组固定test19–20的最终结果。不同训练轮次不能据此宣布最终胜负。见 [项目状态](PROJECT_STATUS.md) 与 [原始审计](../artifacts/ct_checks/20261104_kitti_readonly_audit/REPORT.md)。

本页上传、验收与scratch命令保留原首次启动和复现用途，**不要求对已有run重新上传、重复scratch或补跑全部检查**。旧快照不能作为今天停止、恢复或重启的依据；已有checkpoint继续遵守同run、同源码/配置/数据和完整双流epoch规则。实际训练已发生也不能倒推每项独立工程验收都已单独留存通过记录。本次仅更新文档，不操作服务器。

当前已取消CT启动/真实batch工具对预先登记的依赖。**不用生成、同步或刷新41_registration.json，也不需要先运行verify。** 训练自动保存本次配置与源码信息；源码指纹归一化CRLF/LF，避免上传换行转换导致误拦截。恢复链、损失及学习率不变。

2026-11-02 21:39历史失败：两次均停在旧登记门，29个文件全部仅有CRLF→LF变化，代码内容相同，并非CUDA或模型错误。当时修复涉及6个文件：`models/ct_v41/entry.py`、`models/ct_v41/config.py`、新增`models/ct_v41/provenance.py`，以及 `tools/check_v41_server.py`、`tools/verify_ct_v41.py`、`tools/compare_ct_v41.py`。旧登记文件保留历史但不参与启动。后续11月4日审计确认运行版本为`5c92b1c`，manifest的162个源码文件与服务器源码归一化后匹配；原“待上传、真实batch尚未通过”仅为21:39时的记录，不是当前部署结论，也不用于证明未单独核实的验收项已通过。

2026-11-02 21:05上传前复核：服务器仅只读检查，baseline全部登记文件SHA仍吻合。两套环境均为torch2.0.1+cu118、Lightning2.0.2、numpy1.23.0；CT的FA2为2.5.8，nvidia-ml-py已安装。两张A40当时free分别44,159/43,757MiB，未发现main.py训练进程；这只是当时快照，不替代新Full实测。最新分卡为 **Full物理GPU0，SeqTrack物理GPU1**，线程上限均2。

## 路径与原首次上传范围

- CT：`/home/lishengjie/study/lcyu/CT-SeqTrack`；Python：`/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python`。
- baseline：`/home/lishengjie/study/lcyu/seqtrack`；Python：`/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`。
- KITTI：`/home/lishengjie/data/cxtrack/training`。

上传CT时保留旧源码与output，并同步V41代码、数据适配、配置、工具和测试，以及共享 `main.py`、`utils/execution_profile.py`。CT启动不再要求`cfgs/seqtrack_baseline/`里的外部回执。不要以当前CT目录运行旧checkpoint；V40封存见 `artifacts/ct_checks/v41_implementation_20261102-201312/v40_frozen_source.zip` 与 `snapshot_manifest.json`。

## 原首次启动的验收入口

上传后直接做一次实际Full batch检查；无需登记核对或重复旧V40的六阶段并发容量流程。本次两组分卡运行。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
PY=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 \
  $PY tools/check_v41_server.py --path /home/lishengjie/data/cxtrack/training --devices 0 --steps 2
```

原首次启动流程在成功后使用下方scratch命令。其余计划验收保留以下工具入口；本次未重新核实其独立执行记录，不将“代码已实现”或“已训练若干轮”解释为“验收全部完成”：

```bash
$PY tools/audit_v41_kitti.py --path /home/lishengjie/data/cxtrack/training
$PY -m pytest -q tests/test_ct_v41_recovery.py tests/test_ct_v41_training.py tests/test_ct_v41_protocol.py tests/test_ct_v41_comparison.py tests/test_kitti_v41_dataset.py tests/test_kitti_input_audit.py
$PY -m pytest -q
$PY -m compileall -q models/ datasets/ utils/ tools/
```

`tools/verify_ct_v41.py verify --resolved`现在只是可选的源码语法/配置读取工具，不读取旧登记，不是训练前置步骤。audit只核对metadata和文件清单，不证明点框几何或训练正确。

smoke检查真实主/辅助前向、辅助梯度隔离、BN/mode/RNG保留、同一次Adam和两流commit、FA2调用；不保存checkpoint。该工具为充分触发辅助流按每主batch一辅助batch执行，**不是正式稀疏插入比例的速度测试**。两batch不能证明完整epoch/同run恢复稳定。正式开跑前仍应完成完整辅助轨迹、整epoch结束与续训一致性检查，工程权重丢弃。开发集验收应查看：

1. q0/prior/fallback与R好候选覆盖、已有好候选却错选、R连续接管与正确trusted写入。
2. 长失跟段、删失率、trusted年龄，以及目标重新进入当前采样后的可用监督数量；辅助轨迹长不等于有效覆盖。
3. 误trusted写入、大幅重定位后速度合法性、原云/GT缓存的预测ROI点集合一致。
4. 实际wall time与显存峰值；两组配方不同，不把成本差异简单归于195参数。

开发评估不用于挑选测试最佳轮。原首次启动要求发现错误时先修复未启动版本再正式训练；已有运行的源码/配置与恢复身份仍保留，不能因本页更新覆盖运行中源码或改变配方。

## 原两组scratch命令与固定评测

启动器**只打印nohup、PID、日期目录与新终端tail命令**，不创建目录、不提交训练。分别生成：

```bash
$PY tools/launch_ct_v41.py --arm full --gpu 0
$PY tools/launch_ct_v41.py --arm seqtrack --gpu 1
```

执行输出中的对应命令即可。Full会在60轮结束后自动评测固定58/59/60。baseline先训练保存 `run/fixed_checkpoints/epoch=058.ckpt`、059、060及来源hash；训练完成后再执行打印出的 `tools/evaluate_kitti_v41_baseline.py` 命令，结果在日期目录下 `fixed-test/`，不选best。每次scratch使用全新输出目录，禁止 `--init_checkpoint` 或继承工程权重。

**Full / GPU0：**

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
mkdir -p output
CT_FULL="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-41_full_kitti_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/41_full_kitti.yaml \
  --path /home/lishengjie/data/cxtrack/training \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_FULL" \
  > "$CT_FULL/train.log" 2>&1 < /dev/null &
echo $! > "$CT_FULL/train.pid"
echo "$CT_FULL"
```

**原始SeqTrack方法＋KITTI适配 / GPU1：**

```bash
cd /home/lishengjie/study/lcyu/seqtrack
mkdir -p output
SEQ_BASE="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-seqtrack_original_kitti_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/seqtrack3d_kitti_v41_baseline.yaml \
  --path /home/lishengjie/data/cxtrack/training \
  --batch_size 16 --epoch 60 --workers 10 --seed 42 --log_dir "$SEQ_BASE/run" \
  > "$SEQ_BASE/train.log" 2>&1 < /dev/null &
echo $! > "$SEQ_BASE/train.pid"
echo "$SEQ_BASE"
```

baseline原入口不接受 `--devices`，只暴露GPU1即可实现单卡运行，不传CT专有CLI。它要求scratch模型目录为空，因此日期父目录保存日志/PID，模型输出另放 `run/`；直接将 `train.log` 放进baseline的模型目录会触发已有的非空目录保护。Full原入口已允许这两个启动文件，直接用日期目录即可。Full allocator保留 `roundup_power2_divisions:2`，不能照旧v39命令删掉该项。

**新终端登录服务器后，查看Full最新日期目录：**

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="$(ls -td output/[0-9]*-41_full_kitti_car_seed42_60ep_bs16-*/ | head -n 1)"
echo "$RUN"
tail -n 60 -f "${RUN%/}/train.log"
```

**另一个终端查看SeqTrack：**

```bash
cd /home/lishengjie/study/lcyu/seqtrack
RUN="$(ls -td output/[0-9]*-seqtrack_original_kitti_car_seed42_60ep_bs16-*/ | head -n 1)"
echo "$RUN"
tail -n 60 -f "${RUN%/}/train.log"
```

这些tail命令无需之前终端的变量；若同时存在多个同组运行，改为启动时打印的具体目录。Ctrl+C仅退出tail，不停止训练。

**SeqTrack完成60轮后固定评测：**

```bash
cd /home/lishengjie/study/lcyu/seqtrack
RUN="$(ls -td output/[0-9]*-seqtrack_original_kitti_car_seed42_60ep_bs16-*/ | head -n 1)"
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u tools/evaluate_kitti_v41_baseline.py \
  --run-dir "${RUN%/}/run" --output-dir "${RUN%/}/fixed-test" \
  --path /home/lishengjie/data/cxtrack/training --device 1 \
  > "${RUN%/}/fixed-test.log" 2>&1 < /dev/null &
echo $! > "${RUN%/}/fixed-test.pid"
```

固定评测每轮详细日志在 `fixed-test/epoch=058.log`、059、060；最终汇总在 `fixed-test/fixed_comparison.json`。

baseline保持入口默认的非 `--preloading` 模式，训练GT10宽裁剪仍逐帧生效。不要额外开启持久pickle预加载：历史适配器的该缓存仅按轨迹数量校验，不能保证标注/标定变化后旧缓存失效；本轮命令不使用该缓存。

Full的有效LR为epoch1–20：1.5e-4，21–40：3e-5，41–60：6e-6。baseline维持1e-4、1e-5、1e-6，Adam betas=(0.5,0.999)、eps=1e-6、wd=0及原采样/teacher。两组bs16、60epoch、seed42；Full workers4，baseline workers10。

Full同run epoch边界续训用原cfg、原log_dir和该run完整checkpoint，例如已完成的第20轮：

```bash
$PY main.py --cfg cfgs/ct_seqtrack/41_full_kitti.yaml \
  --path /home/lishengjie/data/cxtrack/training --log_dir /absolute/original/full-run \
  --checkpoint /absolute/original/full-run/formal_checkpoints/epoch=020.ckpt
```

上例checkpoint必须替换为该run实际保存的完整epoch文件路径。默认每2轮及最后3轮保存。恢复检查包含源码、配置、输入metadata、主采样计划、辅助完整轨迹计划及RNG身份；不支持跨版本/跨run加载或半epoch续训。

## 固定结果比较

```bash
$PY tools/compare_ct_v41.py \
  --full-run /absolute/full-run \
  --seqtrack-evaluation /absolute/seqtrack-date-directory/fixed-test \
  --output artifacts/ct_checks/v41_two_arm_results.json
```

比较器读取两组实际运行配置和数据身份，核对三轮scene/track/raw_frame_id集合、初始化帧和逐帧S/P与总分，不要求CT事先登记baseline的源码。Full必须完整60轮且辅助流完成，baseline三权重必须来自同一run、使用同一份实际源码。结果给出final60、late3与差值；late3不是三个独立seed，两组比较不能单独证明某个模块的因果收益。

源码/部署细节见 [设计](CTSEQTRACK_V41_DESIGN.md) 和 [baseline实施回执](../artifacts/ct_checks/v41_implementation_20261102-201312/seqtrack_baseline/REPORT.md)。
