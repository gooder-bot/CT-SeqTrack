# CT-SeqTrack 正式工具面

## v29 性能工具补充（2026-09-11）

- `tools/profile_ct_v29_training.py`：真实`main.py`通路的同卡ABBA、三卡并行ABBA、
  分段profile或100步紧凑逐位对照。前20步预热、后80步统计；测速与数值快照分开。
- `tools/benchmark_ct_v29_h3.py`：两个Full分别寻找真实合法H3事件，固定输入/权重做独立ABBA微基准。
  没有合法事件、没有valid收益标签或逐位不一致均不得当作通过；普通100步不能代替此覆盖。
- 两者只生成`artifacts/ct_checks/`内的可丢弃工程产物，不加载旧checkpoint，不改变正式训练入口，
  不运行全套哈希。启动子进程时清理相互污染的审计/微基准环境变量。

配置、诊断统计语义、验收边界及三个正式后台命令见[性能记录](CTSEQTRACK_V29_PERFORMANCE.md)。

## v29 当前工具协议（2026-09-10）

训练和评测仍仅从`main.py`进入。新增`tools/run_ct_v29_checks.py`串行调度真实数据preflight、
四组同卡100步（B0两次、两种Full）和epoch恢复检查，逐位比较失败即停止；不启动正式训练。
沿用的v28命名数值审计/恢复工具识别新配置身份，工程产物只写`artifacts/ct_checks/`。
`preflight_ct_v28.py`按v29配置执行新的raw窗口→host roll-in→B0数据通路，不把窗口交给普通tensor collate。
动作导出/拟合仍使用已有工具，按配置分派v29 schema与完整方法身份，不使用旧v28策略冒充新策略。
具体参数和正式三臂命令见[服务器运行说明](CTSEQTRACK_V29_SERVER_RUNS.md)。
旧`run_ct_v28_matrix.py`保留历史用途，不用于当前三臂。

## v28 历史工具协议（2026-09-10）

最新已授权一组Full/Car/seed42完整数据60轮诊断。沿用`main.py`及已有v28 preflight、
短检和校准工具，命令见[完整数据单组诊断](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。
preflight缺少`preloading`时现在补False，保留显式YAML设置；默认按需读取完整数据。
无需修改或绕过整矩阵工具的执行限制来运行这一组。

以 [v28服务器流程](CTSEQTRACK_V28_SERVER_RUNS.md) 为当前可执行顺序。旧v27工具
行为保留；新28配置进入reference_batch观测、严格确定性与官方val路由。

当前用户授权的三组mini直接从`main.py`后台启动：GPU1 B0 seed42、GPU2 B0 seed52
（`28_b0_seed52.yaml`）、GPU3 Full seed42，各自scratch60epoch、batch16、workers12、
每5轮官方mini_val验证；这组命令不经过下面只允许执行首轮B0的矩阵工具。
三组CE与Full AP的CUDA报错、修复及验证边界见 [CUDA排错记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。
必须保留`class_axis_logsoftmax_flat_nll_v1`的CE合同和AP整数cumsum统计修复，不能改为warn_only绕过。

- main.py仍为训练/评估唯一入口。正式28配置不允许epoch/workers/数值设置漂移。
  --ct_engineering_check仅供真实事务短验收，限制1..3epoch、每轮1..100batch，
  log_dir必须是本仓artifacts/ct_checks子目录，工程checkpoint不可正式初始化。
- tools/preflight_ct_v28.py检查v28场景、observation总体索引和mechanism完整覆盖；
  --manifest-only不代表真实数据验收。
- tools/check_train_steps.py --numerical-audit [--audit-activations]导出实际B0数值快照；
  tools/compare_ct_v28_audits.py逐位比较同臂与跨臂，不用宽容差替代严格通过。
- tools/replay_ct_v28_adam.py重放审计中的单步参数/梯度/Adam状态；仅用于首次更新分叉定位。
- tools/check_ct_v28_resume.py使用真实main/Trainer比较Full连续两工程epoch与同run边界恢复。
- tools/run_ct_v28_matrix.py默认--stage initial仅生成首轮B0；显式--execute只能执行该B0。
  --stage mini生成6次计划，--stage full生成30次计划，工具拒绝自动执行后续整矩阵。
  final/58/59/60评估及Full逐checkpoint校准都是待执行命令，不自动运行。
- tools/calibrate_ct_actions.py与export_ct_action_rows.py按28配置自动选择真实闭环runner，
  兼容旧--v27与新--v28；v28策略/rows/scene schema与旧版本区分，源码hash覆盖新输入/主干。

旧output为只读历史证据；工程/诊断工具输出置于artifacts/ct_checks新目录。
当前服务器正式训练通过`--log_dir`指定新建的
`output/YYYYMMDD-HHMMSS-28_模块-mini_car_seedXX_60ep_bs16/`，各目录保存`train.log`、
`train.pid`和训练产物；这项授权不允许覆盖历史输出，也不改变工程验收目录的限制。

## v27 当前工具协议（2026-09-05）

训练/评估仍只有`main.py`。五臂配置为`cfgs/ct_seqtrack/27_*.yaml`，外部架构
参考为`cfgs/27_seqtrack_reference{_nuscenes_full}.yaml`。方法、场景用途、指标和
校准定义见[CTSEQTRACK_V27_METHOD.md](CTSEQTRACK_V27_METHOD.md)。

在训练服务器导出运行矩阵，默认只写可审阅配置和命令；mini是五个CT臂加外部参考共6次，full是5类共30次：

```bash
python tools/run_ct_v27_matrix.py --stage mini --path MINI_DATA_ROOT --output artifacts/ct_checks/v27_mini_matrix
python tools/run_ct_v27_matrix.py --stage full --path FULL_DATA_ROOT --output artifacts/ct_checks/v27_full_matrix
```

确认命令后，在相同命令末尾加`--execute`才顺序启动训练。工具为每类保存resolved config，训练固定
scratch/seed42/60epoch/batch16。`manifest.json`与`next_commands.txt`列出各运行epoch58/59/60的待执行命令；
Full每个checkpoint先独立校准，再评估官方split。该工具不自动执行这些后续步骤，不以静态命令清单充当完成结果。
运行日志也进入指定新目录，禁止使用历史`output/`。有执行记录的矩阵目录不能被重新覆盖。

`python tools/preflight_ct_v27.py --cfg cfgs/ct_seqtrack/27_full.yaml --path DATA_ROOT --output artifacts/ct_checks/v27_preflight.json`
实际遍历机制索引，报告endpoint完整性、逐轨迹顺序、partial slot batch及每个观测事务最大tick数量；
`--manifest-only`只核对场景划分，不可解释为真实数据遍历通过。

现有action工具通过显式`--v27`路由到`tools/ct_action_v27_runtime.py`：

```bash
python tools/export_ct_action_rows.py --v27 --config cfgs/ct_seqtrack/27_full.yaml --checkpoint FINAL_CKPT --path MINI_DATA_ROOT --partition calibration --output artifacts/ct_checks/v27/full_calibration.csv
python tools/calibrate_ct_actions.py --v27 --config cfgs/ct_seqtrack/27_full.yaml --checkpoint FINAL_CKPT --path MINI_DATA_ROOT --output artifacts/ct_checks/v27/full_policy.json
```

第二条会重放真实calibration闭环候选策略，再记录锁定dev策略结果，不复用旧v26
promotion门或静态rows替代闭环。每个checkpoint、类别和resolved config分别生成artifact。
工具产物写入新`artifacts/ct_checks/`路径；历史`output/`不变。

本地必检：`python -m pytest -q`、`python -m compileall -q models/ datasets/ utils/ tools/`。
瘦身verify仍绑定旧HEAD，后续commit失败必须与代码失败区分。服务器真实batch、B0更新
一致性、resume和阶段耗时须另外验收；一次CPU测试不能代替真实数据通路检查。

旧报告工具仅在确认支持当前schema后用于v27，不能将v26 presence/风险阈值或不完整分母
强套到v27。下面旧命令和“当前工具”描述保留历史用途，不覆盖本节。

> v26 新增两个正式只读入口：`export_ct_action_rows.py` 按稳定且互斥的
> calibration/dev tracklet 分区导出 action rows；`report_ct_b2_v26.py`
> 验证 schema-v3 漏斗与反事实指标。`calibrate_ct_actions.py` 现在要求两份
> rows 和两份 manifest，阈值只在 calibration 拟合并在 dev 锁定验证。

当前工具只覆盖八类正式任务。报告工具只读取实验产物并写入用户指定的新路径，不作为训练门禁，也不生成跨实验初始化 checkpoint。

| 类别 | 工具 | 职责 |
|---|---|---|
| B1 报告 | `report_ct_b1.py`、`export_b1_calibration.py`、`calibrate_b1_uncertainty.py` | learned mean 与 CV、NLL、coverage、support 和校准残差 |
| B2 acquisition 报告 | `report_ct_b2.py` | supply、retention、presence、raw gain、oracle headroom、harm |
| B3 校准 | `calibrate_ct_actions.py` | 在独立 calibration tracklets 上生成绑定 checkpoint/config 的阈值 |
| risk--coverage | `report_ct_risk_coverage.py` | action coverage、harmful rate、gain 和风险曲线 |
| memory 对照 | `report_ct_memory.py` | real/empty/time-misaligned 配对报告，不阻断训练 |
| 共享参数审计 | `compare_ct_module_audits.py` | 比较 matched-scratch 共享前缀参数 hash |
| 运行检查 | `verify_ct_slimming.py`、`check_candidate_shared_se2.py`、`check_forward_batch.py`、`check_time_batch.py`、`check_train_steps.py` | 配置/保护区、candidate 几何、真实 batch、时间链路和有限训练步 |
| 点云和预测框 | `visualize_pointcloud_sample.py`、`visualize_model_predictions.py` | 检查非空点云、有限框、逐帧输出和同序列模型对照 |

所有接收 `--cfg` 的保留工具都通过 `utils.config.load_yaml_config` 解析 `_base_`，与 `main.py` 的 resolved-config 语义一致。

`check_train_steps.py` 同时识别只读 v24 和 Safe-SeqTrack v25。v25 manifest 会记录 runtime protocol、单优化器拓扑、observation RNG 和精确的 B0 candidate 权重。用 `--steps 100` 分别运行四臂后，checkpoint 中的 `ct_b0_prefix_hashes` 与 `ct_b0_optimizer_state_hashes` 必须在 initial、step1、step100 一致；每个启用参数组还必须出现过有限非零梯度且无 frozen 参数。`ct_cuda_stage_audit` 会保存 batch transfer、forward、loss、backward、step 的 allocated/reserved/peak。一次短跑仍不能替代 epoch-boundary resume 等价证明。

服务器真实 batch forward 与 B1 后端 disposable smoke 示例：

```bash
python tools/check_forward_batch.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --require-full-history --b1-backend gru
python tools/check_forward_batch.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --require-full-history --b1-backend cfc
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 8 --b1-backend gru --artifact-dir artifacts/ct_checks/v25_b1_gru_smoke
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 8 --b1-backend cfc --artifact-dir artifacts/ct_checks/v25_b1_cfc_smoke
```

正式比较前必须再做 matched-seed 100-step 公平性审计；三个目录都必须是新目录：

```bash
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b0.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --artifact-dir artifacts/ct_checks/v25_b0_100
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --b1-backend gru --artifact-dir artifacts/ct_checks/v25_b1_gru_100
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --b1-backend cfc --artifact-dir artifacts/ct_checks/v25_b1_cfc_100
python tools/compare_ct_module_audits.py B0_100_LAST_CKPT B1_GRU_100_LAST_CKPT B1_CFC_100_LAST_CKPT --modules b0
```

三个 checkpoint 的 B0 参数和 Adam `step/exp_avg/exp_avg_sq` hash 必须在
initial、step1、step100 全部一致；任一点不一致就停止正式比较。实际 checkpoint
路径读取对应目录的 `check_manifest.json`；smoke checkpoint 只用于审计，之后删除，
不能初始化正式实验。胜出后端确定后，Full-B3/Full 也按同一方式从 epoch 0
另跑 100-step，并与 matched B0/B1 审计，不能迁移上述 checkpoint。

旧 M/TWC/CRPA/Search/Gate/replay、旧报告生成器、preflight 和 promotion 脚本不属于当前工具闭包。需要审计原实现时，从 `001951a:<path>` 只读恢复。

本地 `output/` 是保护区。工具默认临时产物写入 `artifacts/ct_checks/`；服务器正式实验使用 `main.py` 自动生成的新 `output/YYYYMMDD-HHMM-*` 目录。

## B1-GRU/CfC 与独立校准命令

两个后端复用同一份 `25_b1.yaml`，都从 epoch 0 训练；不要把 B1-only 权重
迁移到 Full-B3：

```bash
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --tag b1_gru_seed42 --b1-backend gru
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --tag b1_cfc_seed42 --b1-backend cfc
```

每个 scratch checkpoint 分别导出 calibration 与 dev tracklets。两份 artifact
必须来自相同 checkpoint、resolved config、seed 和 split：

```bash
python tools/export_b1_calibration.py --config cfgs/ct_seqtrack/25_b1.yaml --checkpoint B1_LAST_CKPT --path /home/lishengjie/data/nuscenes-mini --b1-backend BACKEND --partition calibration --output artifacts/ct_checks/b1_calibration_fit.npz
python tools/export_b1_calibration.py --config cfgs/ct_seqtrack/25_b1.yaml --checkpoint B1_LAST_CKPT --path /home/lishengjie/data/nuscenes-mini --b1-backend BACKEND --partition dev --output artifacts/ct_checks/b1_calibration_dev.npz
python tools/calibrate_b1_uncertainty.py --fit-input artifacts/ct_checks/b1_calibration_fit.npz --eval-input artifacts/ct_checks/b1_calibration_dev.npz --output artifacts/ct_checks/b1_calibration.json --checkpoint B1_LAST_CKPT --output-checkpoint artifacts/ct_checks/B1_EVAL_ONLY.ckpt
```

`B1_EVAL_ONLY.ckpt` 只能用于评估，不能续训。GRU/CfC 的逐 endpoint 配对报告
以 CfC 为 candidate、GRU 为 reference：

```bash
python tools/report_ct_b1.py --rows CFC_ROWS.jsonl --reference-rows GRU_ROWS.jsonl --output artifacts/ct_checks/b1_cfc_vs_gru.json
```

只有 `candidate-minus-reference RMSE` 的 tracklet paired-bootstrap 95% CI 上界
小于 0，且 NLL 与 coverage ECE 均不劣于 GRU，CfC 才可晋升。

真实序列可视化比较两个后端时，不新建重复 YAML；为每个 checkpoint 显式绑定
其训练后端：

```bash
python tools/visualize_model_predictions.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --model GRU cfgs/ct_seqtrack/25_b1.yaml B1_GRU_LAST_CKPT --model CFC cfgs/ct_seqtrack/25_b1.yaml B1_CFC_LAST_CKPT --model-backend GRU gru --model-backend CFC cfc --output-dir artifacts/ct_checks/b1_backend_boxes
```
