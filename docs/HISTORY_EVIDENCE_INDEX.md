# CT-SeqTrack 历史证据与 Git 恢复索引

2026-09-22。工作树只维护v31，旧源码、配置、工具与说明不另建归档目录。主要恢复提交为 **`64ad0560bd3037956370a92ee58e95f79e1ebb55`**（下文简称`64ad056`）。本次清理前含未提交状态的完整只读副本为 `artifacts/ct_checks/20260922_v31_slimming/before/`。

## 当前事实与保留证据

| 阶段 | 已确认结论 | 原始报告 |
|---|---|---|
| v28 mini，2026-09-10 | 三组60轮完成；B0/42=45.200/47.243，B0/52=52.876/64.478；Full未校准、动作0，等于B0/42；late-3当时未补 | [三组结果](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md) |
| v30 mini，2026-09-17 | 四组60轮均40.473742/47.840262；两个Full未拟合策略、验证回退B0；BN重算数值等价、省约1.84GiB抽样张量峰值 | [四组结果](../artifacts/ct_checks/reports/20260917_v30_mini_four_arm/REPORT.md) |
| v30局部根因 | 真实稀疏首预测中motion先产生大跳跃；两个端点干预不能当完整闭环消融 | [真实权重探针](../artifacts/ct_checks/reports/20260917_v30_root_cause/REPORT.md) |
| v31 mini，2026-09-21 | 三臂60轮与58–60独立评测完成；旧同分/标定回退消除，但Full final Success仍低于B0，性能未通过 | [完整结果与修订方向](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md) |

历史较好的SeqTrack mini成绩50.985779/59.961712并非稳定、完全匹配的唯一基线；另有31.683805/31.336981与27.996719/26.483589。论文完整nuScenes Car的62.55/71.46不能作为mini分数。当前结论以 [待办](../need_to_do.md) 和 [最新问题](../最新问题.md) 为准。

`output/` 与既有 `artifacts/` 全部保留原地，不删除或覆盖，包括报告脚本、逐帧记录、图、权重和日志。它们部分被Git忽略，**不能依赖Git恢复全部实验数据**。历史脚本若导入旧host或测试fixture，使用对应Git版本复跑，不为了新树导入而改写原证据。

## 恢复方法

只读查看某文件：

```bash
git show 64ad056:models/seqtrack3d.py
git show 64ad056:docs/CTSEQTRACK_V30_IMPLEMENTATION.md
git show 64ad056:tools/check_train_steps.py
git ls-tree -r --name-only 64ad056 -- models datasets utils tests cfgs tools docs
```

需要运行旧版时使用独立checkout/worktree及匹配环境，不将旧实现重新混入当前v31树，不改写Git历史，不覆盖历史输出。文件恢复不意味着其历史命令在当前环境仍可直接运行。

9/21新增的两段状态没有提交到`64ad056`；其完整清理前文件另保存在 [need_to_do副本](../artifacts/ct_checks/20260922_v31_slimming/before/need_to_do.md) 和 [最新问题副本](../artifacts/ct_checks/20260922_v31_slimming/before/最新问题.md)。当前根文档已保留这些事实、数字、待办与限制，不能仅恢复旧HEAD而丢掉最新结论。

## 配置、工具和源码

- 当前仅保留四份`cfgs/ct_seqtrack/31_*.yaml`，字节内容不变。旧102份YAML和`cfgs/ct_v2/README.md`均从`64ad056:cfgs/`按原路径恢复。
- 旧配置包括v24–v30全链、B4、旧SeqTrack参考入口。v25→v30存在逐代继承，复跑应恢复对应完整树，不只取单个叶子YAML。
- 当前v31仍保留nuScenes full、KITTI、模块臂、CfC/GRU和时间控制接口；删除旧配置不代表删除这些能力，也不代表已经运行对应实验。
- 旧模型/数据宿主、训练事务、标定、采样和测试按`64ad056:models/`、`datasets/`、`utils/`、`tests/`的原路径恢复；本次具体源码裁剪与等价检查见 [精简报告](../artifacts/ct_checks/20260922_v31_slimming/REPORT.md)。
- 旧`research_handoff.json`为v24摘要，通过`64ad056:research_handoff.json`查看，不再充当当前方法说明。

移出的32个工具，恢复对象均为`64ad056:原路径`：

```text
tools/benchmark_ct_v29_h3.py
tools/calibrate_b1_uncertainty.py
tools/calibrate_ct_actions.py
tools/check_candidate_shared_se2.py
tools/check_ct_v28_resume.py
tools/check_forward_batch.py
tools/check_time_batch.py
tools/check_train_steps.py
tools/compare_ct_module_audits.py
tools/compare_ct_v28_audits.py
tools/ct_action_v27_runtime.py
tools/ct_v30_batch_runtime.py
tools/export_b1_calibration.py
tools/export_ct_action_rows.py
tools/preflight_ct_v27.py
tools/preflight_ct_v28.py
tools/preflight_v26_full.py
tools/profile_ct_v29_training.py
tools/replay_ct_v28_adam.py
tools/report_ct_b1.py
tools/report_ct_b2.py
tools/report_ct_b2_v26.py
tools/report_ct_memory.py
tools/report_ct_risk_coverage.py
tools/run_ct_v27_matrix.py
tools/run_ct_v28_matrix.py
tools/run_ct_v29_checks.py
tools/run_ct_v30_server.py
tools/summarize_ct_v30_mini.py
tools/verify_ct_slimming.py
tools/visualize_model_predictions.py
tools/visualize_pointcloud_sample.py
```

这些工具面向旧模型/schema，包括两个旧可视化工具；不是v31正式入口。当前操作统一走[正式工具面](FORMAL_TOOLING.md)。

## 移出工作树的旧文档

以下39份已提交说明、诊断与基线清单仅移出活动树，原文仍在`64ad056:原路径`。它们保留各自日期的证据边界，不以新状态改写历史：

```text
docs/CTSEQTRACK_B0_B3_METHOD.md
docs/CTSEQTRACK_V26_METHOD.md
docs/CTSEQTRACK_V27_IMPLEMENTATION.md
docs/CTSEQTRACK_V27_METHOD.md
docs/CTSEQTRACK_V27_MINI_LAUNCH.md
docs/CTSEQTRACK_V27_TRAINING_READINESS.md
docs/CTSEQTRACK_V28_CHANGE_AUDIT.md
docs/CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md
docs/CTSEQTRACK_V28_FULL_DIAGNOSTIC.md
docs/CTSEQTRACK_V28_IMPLEMENTATION.md
docs/CTSEQTRACK_V28_LOCAL_VALIDATION.md
docs/CTSEQTRACK_V28_RECOVERY_REPORTING.md
docs/CTSEQTRACK_V28_SERVER_RUNS.md
docs/CTSEQTRACK_V29_CHANGE_AUDIT.md
docs/CTSEQTRACK_V29_IMPLEMENTATION.md
docs/CTSEQTRACK_V29_LOCAL_VALIDATION.md
docs/CTSEQTRACK_V29_PERFORMANCE.md
docs/CTSEQTRACK_V29_SERVER_RUNS.md
docs/CTSEQTRACK_V30_CUDA_MEMORY_FIX.md
docs/CTSEQTRACK_V30_IMPLEMENTATION.md
docs/CTSEQTRACK_V30_MEMORY_TRADEOFF.md
docs/CTSEQTRACK_V30_MINI_BN_AB_LAUNCH.md
docs/CTSEQTRACK_V30_MINI_LAUNCH.md
docs/SAFE_SEQTRACK_V25_PROTOCOL.md
docs/SOURCE_SLIMMING_GATE.md
docs/V30_DATA_AND_RUNBOOK.md
docs/analysis/20260823_B0_B1_DUAL_STREAM_DIAGNOSIS.md
docs/analysis/20260823_D86990C_B0_DATAPATH_MEMORY_ROOT_CAUSE.md
docs/analysis/20260823_FOUR_ARM_DUAL_STREAM_DIAGNOSIS.md
docs/analysis/ct24_b0_b1_diagnostic.ipynb
docs/slimming_baseline/README.md
docs/slimming_baseline/active_configs_resolved.json
docs/slimming_baseline/baseline_summary.json
docs/slimming_baseline/closures.json
docs/slimming_baseline/environment.json
docs/slimming_baseline/model_initialization.json
docs/slimming_baseline/output_protection.json
docs/slimming_baseline/pytest_baseline.txt
docs/slimming_baseline/tracked_files.jsonl
```

保留的六份当前文档在`64ad056`中也有旧版本；需要9/19原始就绪/启动文字时使用相应`commit:path`。旧固定HEAD瘦身门禁、旧22配置全等与共享B0逐位一致要求不适用于当前v31清理。

## 更早的 v24 前证据

2026-08早期瘦身前对象位于 `001951a3aee15fdad6e5d5e32ca02d87bf083f3a`：

| 对象 | 允许的历史结论 | 恢复路径 |
|---|---|---|
| CT22 matched B0/B1 | B0 52.196/64.707、B1 50.141/68.038；不能证明当前方法增益 | `001951a:compare_results/data/ct22_ablation_summary_20260811.json` |
| CT22 B2 | 1,986行仅17行新增extension，target-bearing为0，旧恢复证据不足 | `001951a:compare_results/data/ct22_minival_test_diagnosis_20260809.json` |
| CT22 B3 | 有效校准3/704，无合规Full结果 | `001951a:compare_results/data/ct22_ablation_summary_20260811.json` |
| 历史B4 | final51.189/60.886且特征塌缩、成本较高，不进入当前主线 | `001951a:compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md` |

更早文件的原路径、大小、SHA256和恢复对象可从 `64ad056:docs/slimming_baseline/tracked_files.jsonl` 读取。历史低分、负结果与未完成条件持续约束论文主张；本次清理不构成模型涨分或因果机制证据。
