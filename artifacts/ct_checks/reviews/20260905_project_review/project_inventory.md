**CT-SeqTrack 项目清单｜2026-09-05**

审阅对象为 `HEAD=b445ecd04fcdc41474c29d829b2626f3780f759d`。仅审阅 CT-SeqTrack；没有访问用户排除的参考模型。已有未跟踪的 `generate_20260905_v26_mini_five_arm_report.py` 和对应报告目录保持原样。本次新增文件均在本目录，未修改模型、正式配置、git 历史或 `output/`。

| 范围 | 主要来源 | 审阅用途 |
|---|---|---|
| 现行设计 | README.md、docs/CTSEQTRACK_V26_METHOD.md | v26 设计、模块责任与注册实验 |
| 协议和状态 | docs/EXPERIMENT_PROTOCOL.md、SAFE_SEQTRACK_V25_PROTOCOL.md、FORMAL_TOOLING.md、need_to_do.md、research_handoff.json | 识别版本冲突、scratch/候选/校准/比较规则 |
| 模型与优化 | models/seqtrack3d.py、ctseqtrack.py、ct_variant.py、ct_v2/pipeline.py、motion.py、evidence_memory.py、pipeline_contracts.py | B0–B3、prepass、损失、detach、状态提交、参数分组 |
| 数据与几何 | main.py、datasets/sampler.py、nuscenes_lidar_mf.py、temporal_protocol.py、utils/recursive_state.py、ct_search.py、box_membership.py | 双数据流、历史预测、物理时间、wlh/xyz、extension 集合差 |
| 评估与校准 | models/base_model.py、utils/metrics.py、action_calibration.py、tools/export_ct_action_rows.py、calibrate_ct_actions.py、report_ct_b2_v26.py | 全轨迹递归、AUC 分母、漏斗、独立校准和部署匹配 |
| 实验原始证据 | output/20260903-2301-26_* 的 TensorBoard、checkpoint、provenance、hparams、CSV | 重算五臂进展、最终成绩、参数/Adam hash、B1/B2/B3 实际有效性 |
| 历史 | git log/show、docs/HISTORY_EVIDENCE_INDEX.md、历史 output 与 artifacts/ct_checks/reports | 解释 baseline 反复波动、历史负结果及不能跨版本计算的收益 |
| 文献 | literature.md 中 10 篇原始论文及作者实现入口 | 新颖性定位、最近邻、可迁移设计与协议差异 |

当前核心文件仍较大：`seqtrack3d.py` 8,769 行，`base_model.py` 3,489 行，`sampler.py` 2,664 行，`ct_search.py` 1,768 行。此处是维护风险证据，不意味着应立即再次大规模重构。优先用共享函数/类型合同修正已复现缺陷，再在真实服务器对等验证后拆分活动路径。

本次本地验证：`python -m pytest -q` 为 **176 passed, 1 skipped**；`python -m compileall -q models/ datasets/ utils/ tools/` 成功。`python tools/verify_ct_slimming.py verify` 在 HEAD 必须等于 `001951a` 的检查处退出，属于该历史基线工具的设计限制，未继续执行后续核验，不能将其记为本轮输出指纹保护检查已通过。

独立复现脚本为 `reproduce_code_audit.py` 和 `root_contract_probe.py`；实验重算证据为 `recomputed_experiment_evidence.json`。它们不是正式训练，也不产生可初始化模型的 checkpoint。

尚缺的关键材料：修复后的服务器真实 batch/100-step/resume 对等结果；完整五臂 final 与 e58/e59/e60 测试；完整 nuScenes 结果；B3 calibration/dev artifact 和冻结策略闭环评估；严格匹配的外部 SeqTrack 参考；跨 seed、时间/记忆因果对照；可用于跨模型公平比较的统一数据清单与运行成本。当前没有足够证据形成涨分论文主表。
