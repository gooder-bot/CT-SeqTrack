# V41 v2 新运行流程

2026-11-16：本页适用于执行身份 **`v41_execution_revision: v2` 的新 run**。本轮已实施配置、日志、评估与保存流程并做静态审阅，**尚未运行 pytest、compileall、真实 batch、完整 epoch 或恢复验收，也未部署或访问服务器**。这些工程改动不构成新精度、速度或显存结论。旧运行证据仍见 [项目现状](PROJECT_STATUS.md)。

## 配置与身份

优先级为 **显式具名 CLI > YAML > 默认值**；未传的 CLI 不覆盖 YAML。没有 `--set`、参数别名或新增调度器。可调整 epoch、batch、seed、workers、LR、weight decay、已有调度、缓存预算和执行/日志频率，不必预先登记每种配方。模型结构、损失系数、可信更新规则、KITTI 划分与几何合同保持；正式 run 仍为 Full、scratch、单卡和实际 FA2。

默认值由 `models/ct_v41/config.py` 定义，所选 YAML 可覆盖：

| 项目 | 默认值 |
|---|---|
| 训练 | `epoch=60`、`batch_size=16`、`workers=4`、`seed=42` |
| 优化 | `lr=0.00015`、`wd=0`、`lr_schedule=multistep`、`lr_milestones=[20,40]`、`lr_decay_rate=0.2`、`lr_warmup_steps=0` |
| CPU 点云缓存 | `ct_pointcloud_cache_bytes=268435456` |
| 控制台 | `console_mode=auto`、`progress_refresh_seconds=1`、`plain_log_interval_seconds=30` |
| 标量/诊断 | `log_every_n_steps=50`、`diagnostics_level=summary`、`full_diagnostic_epochs=[]` |
| 可选日志 | `tensorboard=false`、`v41_memory_logging=false` |
| 评估/保存 | `eval_split=test`、`check_val_every_n_epoch=1`、`checkpoint_every_n_epochs=1` |
| 报告选择 | `report_checkpoint=best_precision`、`report_last_k=3` |

调度器仍只有 `step`、`multistep`、`piecewise`。从含 milestones 的 YAML 切换到 step 时，显式传空列表：`--lr_schedule step --lr_milestones`。布尔开关采用 `--tensorboard` / `--no-tensorboard`、`--v41_memory_logging` / `--no-v41_memory_logging`；列表可用 `--full_diagnostic_epochs 1 10 40`，空列表用单独的 `--full_diagnostic_epochs`。旧 `v41_diagnostics`、`v35_train_diagnostics` 保留读取兼容，新诊断档位优先；旧 `v31_evaluate_late3` / `--no_late3` 不再触发训练后的自动 late3 复测。

运行自动保存 `resolved_config.yaml`、`run_manifest.json`，记录实际配方、命令、源码/数据身份及评估政策；无需 `41_registration.json` 或 baseline receipt。启动器 `tools/launch_ct_v41.py --arm full` 只打印命令，按实际 tag、seed、epoch、batch 命名并仅转发显式覆盖项；兄弟 baseline 配方保持原状。

v2 只接受同执行版本的 checkpoint。训练续跑必须使用原目录、run_id、源码、训练配置/数据及完整主辅双流 epoch；不把改 LR、epoch、seed 的新配方当作续训。纯显示/诊断项不参与训练配置摘要；独立 `--test` 可改 `eval_split` 与诊断档位，但必须使用新的空输出目录。训练恢复继续核对原评估/选择政策。

v1 的 final60/late3 协议、checkpoint 和结果仍按原源码解释，不能用 v2 接续或重写。此次变更前源码在 [source_before](../artifacts/ct_checks/v41_run_workflow_20261116-c1065b45/source_before/)；旧 run 应使用与其 manifest 匹配的原源码，不能假定任意快照都匹配。

## 输出与结果含义

`console_mode=auto` 在交互终端使用单行进度条，重定向/nohup 使用 plain；也可明确选 `bar` 或 `plain`。每轮结束输出三行，以下仅为格式示意，尖括号不是已测数值：

```text
E<轮>/<总轮>  Loss: opt=<优化目标> | main_stream=<主流loss> | aux_stream=<辅助流loss>
Test S=<Success> P=<Precision> | lost=<失跟段数> recovered=<恢复段数> | >10m=<帧数>/<预测帧数>
Time train=<时长> eval=<时长> save=<时长> | ckpt=<轮号> | remaining=~<预计剩余>
```

| 文件 | 内容 |
|---|---|
| `<run>/metrics.csv` | 一份追加式长表：`session_id,epoch,step,metric,value`。恢复追加并换 session_id，不重写全历史；新指标直接加行。step 与 epoch loss 使用原指标名区分，稀疏辅助 loss 只作 epoch 汇总。 |
| `epochs/epoch=NNN.json` | 该轮实际记录的全部 epoch loss、评估摘要、计时与 checkpoint 路径；full 评估包含详细表。 |
| `frame_streams/<split>/<session>/epoch=NNN.jsonl` | 每次评估的唯一逐帧流，summary 保留计算/复核指标所需字段，full 增加诊断；恢复不会覆盖旧 session 文件。 |
| `training_audits/epoch=NNN.json` | 既有主辅采样/端点/更新预算与完整性审计。full 训练分组诊断仍写既有 `training_diagnostics/epochNNN.json`。 |
| `formal_checkpoints/epoch=NNN.ckpt` | 默认每轮保存一份完整训练 checkpoint，暂不清理历史。改变保存间隔时仍保存新 best 和最后一轮。 |
| `results.json` | 训练 run 持续记录 `best`、`last`、`main`、`lateK` 与 checkpoint 路径；独立评估目录记录该 checkpoint 的单次评估结果。 |

没有新增 `metrics.jsonl`。TensorBoard 仅在显式开启时写入 `tensorboard/`。`summary` 跳过昂贵详细诊断，`full` 或 `full_diagnostic_epochs` 指定轮启用详细诊断；必要指标与逐帧证据两档都保留。

默认每个完整训练轮都评估 test19–20；间隔可调，但最后一轮始终评估。`best_precision` 按 Precision 严格改善选轮，同分保留较早轮；`last` 选择最后完成轮。`main` 指向选定报告，`lateK` 是**最近 K 个已评估轮**的均值，并记录实际轮号和数量；不足 K 轮时 `complete=false`。这与 v1 固定58/59/60不同。默认 best 是 **在 test 上选轮的结果**，不能写成未参与选择的独立测试结论，也不能与旧 final60/late3 混用。

评估使用 `isolated_split_seed_v2`，按公开 split 固定种子并在结束/异常时还原训练 RNG、loader generator 与各层 train/eval 状态；dev 对应内部 val loader。该隔离机制已实现，运行一致性仍待验收。

## 服务器命令与待验收项

以下供用户后续在服务器执行，本轮未执行。先在正确源码与独立环境中运行针对性测试和实际 batch 检查：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
PY=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
$PY -m pytest -q tests/test_ct_v41_run_config.py tests/test_launch_ct_v41_config.py tests/test_ct_v41_monitoring.py tests/test_ct_v41_summary.py tests/test_ct_v41_protocol.py tests/test_ct_v41_training.py tests/test_ct_v41_recovery.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 \
  $PY tools/check_v41_server.py --path /home/lishengjie/data/cxtrack/training --devices 0 --steps 2
```

还需以工程 run 验证完整主辅 epoch、实际 Lightning 回调顺序、评估 RNG 隔离、完整 checkpoint/恢复、CSV 追加、summary/full 同权重指标与逐帧覆盖。两 batch smoke 不替代这些验收；工程权重不用于正式初始化。

新 run 示例（40轮/seed7/bs8 为参数覆盖示例，不是性能建议或已运行实验；命令会启动训练）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=7 \
  $PY -u main.py --cfg cfgs/ct_seqtrack/41_full_kitti.yaml \
  --path /home/lishengjie/data/cxtrack/training --devices 0 \
  --tag v41_v2_e40_s7 --epoch 40 --batch_size 8 --seed 7 \
  --lr 0.0001 --lr_milestones 20 30 --diagnostics_level summary
```

同 run 恢复使用已保存的实际配置；将 `RUN` 和 checkpoint 轮号替换为目标 v2 run，不改配方：

```bash
RUN=/absolute/path/to/the/v2/run
$PY -u main.py --cfg "$RUN/resolved_config.yaml" --log_dir "$RUN" \
  --checkpoint "$RUN/formal_checkpoints/epoch=020.ckpt" --devices 0
```

独立 dev17–18 诊断使用同一 v2 checkpoint，写到新的空目录，不进入训练 run 的 best/lateK：

```bash
EVAL="$(mktemp -d artifacts/ct_checks/v41-v2-dev-XXXXXX)"
$PY -u main.py --cfg "$RUN/resolved_config.yaml" \
  --checkpoint "$RUN/formal_checkpoints/epoch=020.ckpt" --devices 0 \
  --test --eval_split dev --diagnostics_level full --log_dir "$EVAL"
```

旧两组操作与结果约束保留在 [v1 操作记录](CTSEQTRACK_V41_LAUNCH.md)。`tools/compare_ct_v41.py` 仍服务该固定两组历史协议，不是任意 v2 配方的验收器。
