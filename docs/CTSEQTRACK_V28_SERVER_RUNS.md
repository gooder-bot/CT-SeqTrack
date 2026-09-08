# v28 服务器验收与首轮 B0（2026-09-08）

所有命令从 CT-SeqTrack 仓库内运行，使用 [SERVER_PATHS.md](SERVER_PATHS.md) 的数据根。所有工程/新运行输出指定 `artifacts/ct_checks/` 新目录，不覆盖历史 `output/`。下面命令是待执行流程，不代表已通过。

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
