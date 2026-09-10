# v28 服务器验收与 mini 三组重启（2026-09-09）

**9月10日最新安排：** 三组mini完成后，用户决定先做一组Full/Car/seed42完整nuScenes、
从头60轮诊断，再集中分析。现有full配置已核查，完整数据启动、关闭预加载的原因及
每个checkpoint的后处理命令见 [完整数据单组诊断](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。
下文mini重启命令保留历史用途；不能再用旧“首轮仅B0”顺序阻止这次已授权诊断。

所有命令从 CT-SeqTrack 仓库内运行，使用 [SERVER_PATHS.md](SERVER_PATHS.md) 的数据根。工程输出指定 `artifacts/ct_checks/` 新目录；按当前确认的安排，三组正式重启使用带日期的 `output/` 新目录，不覆盖历史结果。下面命令是待执行流程，不代表已通过。

9月10日状态更新：三组已完成上述60轮训练并同步结果，见[三组分析](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。本页三组启动命令保留为运行记录，当前下一步是补58/59评测、Full校准及B0/B2定位；不要因旧的“重启”标题重复训练。B0/42未达恢复目标，尚未进入完整nuScenes大矩阵。

遇到 `nll_loss2d_forward_out_cuda_template`、`cumsum_cuda_kernel` 或紧随其后的 tqdm 异常，先读 [CUDA 故障记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)：内含两次根因、修复提交、同步文件、专项测试和未验收项。下面保留可直接使用的三组命令；后文矩阵工具只执行单 B0 的限制属于工具行为，不取消用户已确认通过 `main.py` 并行运行三组的安排。

## 2026-09-09：Full 的 AP 累计统计修复

随后只有 Full 报 `cumsum_cuda_kernel`，原因是 B2 relation AP/AUPRC 指标把二值标签转成浮点后累计；B0 未启用 B2，因此不进入该路径。该指标仅记录日志，不参与 `loss_ct_plugin_total`。`models/seqtrack3d.py` 的 relation AP 和旧 binary rank AP 两处均改为 `cumsum(..., dtype=torch.int64)`，再转回原浮点类型计算 precision。二值计数及排序/分母保持不变，严格确定性继续开启，没有 CPU 传输或训练参数变化。此次增量运行文件只有 `models/seqtrack3d.py`，先同步服务器，再执行下方三组命令。

定向回归在实际 host loss 上强制整数累计，与旧浮点二值计数逐位比较，并覆盖非空/空支持及 B2/B3 梯度归属；另运行实际 v28 B0/Full 训练与 Adam hooks 回归。本次结果为 **15 passed, 1 skipped**，跳过项是之前 CE 修复的 CUDA 测试。本地没有 CUDA，此处通过不表示完整 CUDA 训练已验收。重启三组前先结束仍在运行的旧 B0 任务，避免同卡重复启动。

## 2026-09-09：分割 CE 的 CUDA 确定性修复与并行重启

三组启动时的 `nll_loss2d_forward_out_cuda_template` 异常来自同一个 B0 分割目标：原 `[B,2,4096]` CE 被分派为空间 NLL，其 CUDA mean 归约使用 atomicAdd，严格确定性直接拒绝。之前的池化替换没有覆盖该损失算子；tqdm 的析构异常是退出后的次生错误。本次启动错误不能用于解释历史得分下降。

修复在 `models/ct_v2/observation_reference.py`：保留原三维 `log_softmax(dim=1)`，仅将 log-probability 和标签按对应点顺序整理为 `[B*4096,2]`、`[B*4096]`，交给二维 `nll_loss`。类别权重仍为 `[0.5,2.0]`，分母仍为整个 batch 的目标类别权重之和。其余损失和优化器不变，严格 deterministic 不关闭，也不使用 warn_only。旧 CUDA 原子归约与新路径可能有末位舍入差异，不宣称两者 CUDA 逐位相同。

`ct_b0_ce_contract=class_axis_logsoftmax_flat_nll_v1` 已加入 v28 公共配置、训练校验及 resume 身份，也随配置进入校准身份。同步以下三个运行文件后再启动：

- `models/ct_v2/observation_reference.py`
- `cfgs/ct_seqtrack/28_formal_base.yaml`
- `utils/online_contract.py`

现有 `28_b0.yaml`、`28_b0_seed52.yaml`、`28_full.yaml` 自动继承新合同。按照当前已确认的并行安排，GPU1 B0/42、GPU2 B0/52、GPU3 Full/42 均重新 scratch；不恢复本次失败运行的 checkpoint，保留失败日志。此安排不代表 B0 的分数恢复已验收。

```bash
STAMP=$(date +%Y%m%d-%H%M%S)
B0_42_DIR="output/${STAMP}-28_b0-mini_car_seed42_60ep_bs16"
B0_52_DIR="output/${STAMP}-28_b0-mini_car_seed52_60ep_bs16"
FULL_DIR="output/${STAMP}-28_full-mini_car_seed42_60ep_bs16"
mkdir -p "$B0_42_DIR" "$B0_52_DIR" "$FULL_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

nohup env CUDA_VISIBLE_DEVICES=1 python -u main.py \
  --cfg cfgs/ct_seqtrack/28_b0.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 12 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$B0_42_DIR" \
  > "$B0_42_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$B0_42_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=2 python -u main.py \
  --cfg cfgs/ct_seqtrack/28_b0_seed52.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 12 --seed 52 \
  --preloading --check_val_every_n_epoch 5 \
  --tag mini_car_seed52_60ep_bs16 --log_dir "$B0_52_DIR" \
  > "$B0_52_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$B0_52_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=3 python -u main.py \
  --cfg cfgs/ct_seqtrack/28_full.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 12 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$FULL_DIR" \
  > "$FULL_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$FULL_DIR/train.pid"

printf '输出目录：\n%s\n%s\n%s\n' "$B0_42_DIR" "$B0_52_DIR" "$FULL_DIR"
tail -f "$B0_42_DIR/train.log" "$B0_52_DIR/train.log" "$FULL_DIR/train.log"
```

回归覆盖原 CE 整批 loss/梯度（类别不均衡、全背景/全前景、忽略槽）、完整 B0 多种 moving 比例的原目标/更新、B0/Full 训练宿主隔离及新合同身份；另提供严格 CUDA loss/梯度/Adam 重复测试。此次针对性回归为 **47 passed, 1 skipped**，修改文件 compileall 与 diff 检查通过。本地只有 CPU，跳过项为 CUDA 重复测试，实际服务器 CUDA 验收仍待执行。

## 1. 固定单卡环境与真实数据

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python tools/preflight_ct_v28.py --cfg cfgs/ct_seqtrack/28_full.yaml --path /home/lishengjie/data/nuscenes-mini --output artifacts/ct_checks/v28_preflight/mini_full.json
```

不传 `--manifest-only` 才实际构造数据和遍历索引调度。应确认8训练场景、2官方验证场景、完整候选总体shuffle及mechanism每个非首帧endpoint恰好一次。这里验证数据与调度覆盖；模块实际获得梯度和更新由下面的训练工程验收检查。

mini/Car 必须从真实 `len(dataset)//16` 得到1262步/epoch，60轮共75720步。main/preflight均核对这个配置绑定的预期；不符则停下检查数据总体，禁止重复/补齐样本凑数。full配置的此项预期为null，不套用mini预算。

## 2. 同卡顺序 100-step 与逐位比较

```bash
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/28_b0.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --numerical-audit --artifact-dir artifacts/ct_checks/v28_100step/b0_a
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/28_b0.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --numerical-audit --artifact-dir artifacts/ct_checks/v28_100step/b0_b
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/28_b1_gru.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --numerical-audit --artifact-dir artifacts/ct_checks/v28_100step/gru
python tools/compare_ct_v28_audits.py artifacts/ct_checks/v28_100step/b0_a/numerical_audit artifacts/ct_checks/v28_100step/b0_b/numerical_audit --output artifacts/ct_checks/v28_100step/same_arm.json
python tools/compare_ct_v28_audits.py artifacts/ct_checks/v28_100step/b0_a/numerical_audit artifacts/ct_checks/v28_100step/gru/numerical_audit --output artifacts/ct_checks/v28_100step/cross_arm.json
```

工程入口保留真实main/Trainer事务，使用 `--ct_engineering_check`；限定1..3个工程epoch、每轮1..100个batch，log_dir必须位于本仓 `artifacts/ct_checks/`。checkpoint不得用作正式初始化。默认workers0仅用于数值工程定位；正式配置workers12。需要首个差异的完整层张量时加 `--audit-activations`，它会明显增大磁盘占用。

保存初始化及step1/2/3/4/5/10/100的输入、前向、BN、梯度、Adam状态和实际更新。验收要求同臂及跨臂B0逐位一致；数值容差只帮助定位，不得把strict失败改写成通过。若同臂已不同，先查执行路径；若同臂通过而跨臂不同，再查隔离边界。100步通过不能替代完整训练或跟踪质量验收。

若输入/前向/梯度都一致而更新首次不同，可以在原GPU和同一环境重放该步Adam：

```bash
python tools/replay_ct_v28_adam.py artifacts/ct_checks/v28_100step/b0_a/numerical_audit/step_001.pt --device cuda:0 --output artifacts/ct_checks/v28_100step/adam_replay_step1.json
```

重放只消费审计中的参数、梯度、Adam状态与参数组，不调用模型、不修改训练checkpoint。CPU重放可定位状态问题，但不能代替CUDA路径的等价结论。

## 3. Full 真实 epoch 边界恢复

```bash
python tools/check_ct_v28_resume.py --cfg cfgs/ct_seqtrack/28_full.yaml --path /home/lishengjie/data/nuscenes-mini --output artifacts/ct_checks/v28_resume_full --steps 16 --gpu 0
```

连续两工程epoch，与同一split运行在epoch1边界停止、用其自身checkpoint恢复至epoch2比较。工程配置相同，参数/BN/Adam/scheduler/更新计数/全局及loader RNG逐位核对；不能通过跨run checkpoint加载规避resume身份。输出工程报告，checkpoint丢弃。

## 4. 只生成并启动首轮 B0

先在服务器生成绝对路径正确的命令：

```bash
python tools/run_ct_v28_matrix.py --stage initial --path /home/lishengjie/data/nuscenes-mini --output artifacts/ct_checks/v28_initial
```

默认只有B0一臂，且不启动训练。验收通过后在同一命令增加 `--execute` 才从epoch0启动首轮正式B0。该函数拒绝自动执行mini/full后续矩阵。训练不会自动校准或评测58/59/60；对应命令位于 `next_commands.txt`。

正式60轮结束后，使用同一resolved config和各自58/59/60 checkpoint补评官方mini_val。固定报告final60及三者平均，不选择不同best epoch。

v28 评测报告同时导出 [失跟、稀疏点桶和空段恢复统计](CTSEQTRACK_V28_RECOVERY_REPORTING.md)。这些是固定定义的事后诊断，原始点数缺失不会算成空 crop，未观察到恢复会标记右删失。

健康参考Success/Precision为 **50.9857788 / 59.9617119**，恢复目标为 **S ≥49.986 且 P ≥58.962**。同时保留独立的matched reference条件：相同协议与固定报告轮次的S/P差距各自≤1个百分点。不能用两个低分运行相近来宣告B0恢复；不能用100-step数值一致性代替这项跟踪质量验收。

## 5. 后续矩阵与 Full 策略

```bash
python tools/run_ct_v28_matrix.py --stage mini --path /home/lishengjie/data/nuscenes-mini --output artifacts/ct_checks/v28_mini_pending
python tools/run_ct_v28_matrix.py --stage full --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ --output artifacts/ct_checks/v28_full_pending
```

mini含五个CT臂与同共享实现reference共6次；full五类共30次。这里只生成计划，在B0验收后另行决定启动，不视作已经验证的实验结果。

Full每个checkpoint先单独运行 `tools/calibrate_ct_actions.py --config CFG --checkpoint CKPT --path DATA --output POLICY`，再将POLICY传入main的 `--ct_action_calibration_path` 评官方val。v28按配置自动进入真实闭环runner，也兼容 `--v27` 或 `--v28` 显式参数。calibration/dev与参数训练重叠；官方val只评测。更换源码/观测或动作定义必须重新导出、拟合，不能复用旧策略。
