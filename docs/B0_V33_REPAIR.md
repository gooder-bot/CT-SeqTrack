# v33 综合 B0：实现与验证

本版执行用户批准的综合方案：一次整合合理兼容修改，再比较三种训练配方；不以逐项消融为前置条件。活动代码保留原物理路径，模型身份升级为v33。

## 实现

- decoder的角点query由4维改为7维，显式附带detached输出参考中心，仅增加192个参数。q0中心直接回归；history/mode仍为seed残差；yaw不改，mode输出中心live梯度保留，query几何detach。
- BC监督全部真实点，包括无GT前景帧中的背景点；current/history按组等权，缺组重新归一化。history定位先端点内平均，再端点平均。系数不变，中心/角度、BC分组及seg/quality加权贡献作为日志，不重复累加。
- 可信状态依据accepted框内至少3个唯一预测FG及原质量阈值，不再依赖memory写入结果。框外高FG不写记忆，也不强制改为BG。换可信锚点而无相邻合格速度对时清零速度/invalid；无新可信框时保留原可信运动。
- seed合法pose与速度初始化保持，强观测时间依据真实点；未有strong的年龄从窗口初始化时刻起算。B1历史运动边资格不改。
- 三配方A原StepLR，B在20/50轮衰减，C为B全程半LR；schema/config SHA/checkpoint均隔离旧版。公开框和四候选接口保持。
- 正常评测附带raw/crop/sampled目标点数、移动GT位移、同帧coarse/fine几何、支持/速度状态、失跟段及右删失统计；新旧结果用固定帧键比较。独立参考网络、teacher和评分数学不改。

## 实验解释

用户最新选择重新训练R，与A/B/C一起运行四组mini单seed，GPU依次0/0/1/1。既有v32正式SeqTrack仍可作为历史对照：final60为51.823851/61.341356，late-3为51.884391/61.966448。旧B0为49.650985/59.803063与49.175420/59.365062。新R使用项目内独立参考网络，不套用B0修改。

A−旧B0检验整个综合包；B−A检验第二次降档；C−B检验步长尺度。三组同seed，不代表多seed证据。选择和达标依据[实验协议](EXPERIMENT_PROTOCOL.md)，不能把固定权重中心替换探针视为新模型成绩。

## 验证记录

本地首次完整检查：285 passed、7 skipped（当前Windows Python缺Lightning且torch为CPU）；compileall和git diff --check通过。涉及非零anchor yaw、query参考中心碰撞、mode梯度、no-FG BC、padding梯度、history端点权重、可信速度重置、初始化语义、三配方逐轮LR及optimizer恢复、被动评测和结果汇总。

此前已在服务器原环境完成最终验证：Python3.9.19、torch2.0.1+cu118、pytorch_lightning2.0.2，完整测试 **314 passed、1 skipped**；唯一跳过项是已安装Lightning时不适用的“缺包”测试。最初测试fixture的`any(dim=tuple)`不兼容torch2.0.1，已改为`flatten(1).any(dim=1)`后完整通过。

真实nuScenes-mini CUDA batch已通过一次forward/backward/Adam/commit：输入16×4×1024×5 FP32，总loss7.467861、梯度全有限（L2范数67.73209），更新后无pending事务；峰值allocated6994.92MiB、reserved7054MiB，未保存checkpoint。完整证据：[服务器测试](../artifacts/ct_checks/20260924-190116_v33_implementation/server_pytest_final_tail.log)、[真实batch](../artifacts/ct_checks/20260924-190116_v33_implementation/real_batch.json)。正式运行另起进程从epoch0开始。

2026-09-24本轮复核开始时，本地58个运行源码SHA与真实batch证据全部一致。随后仅修正一处被动汇总：独立SeqTrack未提供的四种目标点统计改为`None`并记录各项覆盖帧数，真实观测的零仍保留为0；相应比率在未记录时不产生伪零。不改变模型、loss、优化器、S/P或参考实现。定点验证`tests/test_ct_v33_evaluation.py tests/test_ct_v31_runtime.py`为 **40 passed、2 skipped**（本地缺Lightning），compileall与diff检查通过。既有服务器训练链路验证继续适用，无需重复真实batch。

此次仅只读查看服务器，未上传、启动或停止任务。服务器主目录仍是v32；先前已验证的v33位于`artifacts/ct_checks/20260924-190116_v33_deployment/source`。本地当前文档修复了中文编码损坏，并统一为四组启动安排；四条命令已逐条通过当前入口的参数解析。上传当前本地版本后可开始实验，独立后台命令和tail见[最新启动页](CTSEQTRACK_V33_MINI_LAUNCH.md)。

v32配置与已完成output/artifacts保留。综合修改与训练前工程验证已完成，四组正式训练及结果判定尚未完成；不声称涨分或已达到SeqTrack。
