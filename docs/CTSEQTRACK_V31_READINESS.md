# v31 验证记录与当前边界

2026-09-22整理。9/19的本地准备状态已被9/21真实实验覆盖；本页分别记录两阶段证据，不把历史测试数量当作本次清理验收。

## 2026-09-21：正式三臂已完成，性能未通过

代码版本 `64ad056` 的 B0/Full-CfC/Full-GRU 均完成真实mini Car/seed42 scratch60，58/59/60独立评测齐全。每臂71,911次Adam、1,146,480次端点曝光；官方mini_val为106轨迹/2285帧，九份JSONL重算与汇总一致。

| v31 | final60 S/P | late-3 S/P | peak allocated MiB |
|---|---:|---:|---:|
| B0 | 23.959519 / 27.834792 | 23.812546 / 28.955142 | 7034 |
| Full-CfC | 22.491247 / 29.352297 | 24.213348 / 33.709701 | 7329 |
| Full-GRU | 22.839169 / 31.844639 | 25.303793 / 37.146243 | 7330 |

三组分数和轨迹已不同，无旧标定回退；两个Full的final Success低于同版B0。总时长约19.23/30.16/30.54小时，含训练、验证与三次最终评测；两Full共GPU0，不能用来比较独占推理速度。完整权重、运行身份与诊断见 [正式结果报告](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)。

已完成正式运行不等于原计划的独立100步、同卡重复/恢复检查均单独执行。报告中的真实数据CPU短探针、BN交换和候选oracle仅用于诊断，不能当新正式闭环成绩。当前帧身份、稀疏token、尾批BN、模式重复与状态污染的修订仍未实施，见 [最新问题](../最新问题.md)。

## 2026-09-19：本地准备阶段历史证据

- 当时五个v31测试文件为 **89 passed、1 skipped**，使用本地独立target的Lightning2.0.2；skip为已安装Lightning时不适用的缺包测试。
- 三臂通过真实 `JointTracker + CTSEQTRACKV31 + entry.run` 的合成原始点3epoch训练与逐checkpoint闭环评测。工程设置N16/batch4/workers0，分数只说明流程。
- Lightning2.0.2的3epoch与epoch1中断恢复对照中，最终参数、Adam张量、global_step和LR逐位相同；覆盖Dropout/RNG，worker2预取另有专项。
- 当时Python3.9语法检查15个v31/host文件、compileall和diff检查通过。
- 原始记录：[三臂入口结果](../artifacts/ct_checks/v31_entry_integration_latest.json)、[运行日志](../artifacts/ct_checks/v31_entry_integration_20260919.log)。旧上传包仅代表该日期快照，不作为清理后源码发布包。
- 当时服务器只读检查过raw→prepare→CPUprior→acquire单行：dt=0.499874秒、当前零点/历史1点、extension17点。这项接口检查不替代后来的正式训练结果。

## 2026-09-22：行为等价清理

本轮仅收敛工作树，不修改已登记算法和四份v31 YAML，不宣称已解决性能问题。生产Python由163文件/57,233行减至39文件/6,153行；清理前源码及两份未提交状态文档保存在 `artifacts/ct_checks/20260922_v31_slimming/before/`。

- 保留测试：**152 passed、2 skipped**；跳过本机不可用的CUDA检查和已安装Lightning时不适用的缺包检查。
- 三正式臂的初始化、参数顺序、前向、全部损失、梯度、BN与Adam对照：**81,380项记录，0差异**；正常、0/1/2点、全空及混合有效性均独立执行两步更新。四份配置身份一致，三份原epoch60正式权重严格加载通过。
- 三臂合成3epoch训练、末轮评测与跨源码epoch1恢复：**14,333项记录，0差异**；连续/恢复的最终参数、Adam、LR、RNG、端点预算相同。
- 最终`entry.run`三臂合成集成通过：各3轮、3份末轮checkpoint及评测，训练预算、结果JSON/JSONL字段、CSV/TensorBoard输出均核对。
- 数据对照：12组SDK构造参数、3份完整场景协议、6组true/fixed/shuffled原始帧序列一致；覆盖原始点ID、变换、物理时间、端点和manifest哈希。
- 原有output/artifacts共20,670项的路径、大小、修改时间不变；本轮新检查只写独立审计目录。四份YAML字节不变，9/21两份状态正文的数字、结论与待办已核对保留。

本次环境为Windows CPU/PyTorch2.3.0，使用既有独立target的Lightning2.0.2；数据检查为合成KITTI与nuScenes SDK stub。未安装或升级包，未执行真实数据/CUDA/服务器验证。完整方法、记录、限制与复现入口见 [精简报告](../artifacts/ct_checks/20260922_v31_slimming/REPORT.md)。

旧源文件与协议通过 [历史索引](HISTORY_EVIDENCE_INDEX.md) 复现，实验结果/权重/诊断证据原样保留。当前服务器仍仅只读；命令见 [运行说明](CTSEQTRACK_V31_MINI_LAUNCH.md)。
