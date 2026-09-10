# v28 实施验收记录（2026-09-10 更新）

## 9月10日：完整数据单组诊断就绪审计

用户最新安排是一组Full/Car/seed42完整nuScenes、从头60轮后集中分析。现有full配置和
B0—B3链路已核查，补齐preflight缺省preloading以及val/test导出元信息；未修改网络、loss、
采样随机流或优化预算。命令与边界见[单组完整数据诊断](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。

| 本次检查 | 实际结果 |
| --- | --- |
| 修复前当前代码全套pytest | 466 passed / 13 skipped，181.52秒 |
| 两处修复后全套pytest | **481 passed / 13 skipped，177.90秒** |
| `python -m compileall -q models/ datasets/ utils/ tools/` | 通过 |
| `git diff --check` | 通过；Git另提示工作副本LF/CRLF转换，不是diff错误 |
| 文档两段命令及启动脚本Git Bash语法 | 通过；16步/12workers工程命令与60轮正式预算合同通过 |
| 已分析mini的90份原始文件SHA256 | 全部匹配，未改变旧checkpoint/日志/指标 |
| 旧slimming工具 | 因固定HEAD=001951a、当前749bc13失败；未修改该历史门禁 |

新增15个测试覆盖真实get_dataset控制入口的缺省/显式preloading，及生产导出hook的
scene/tracklet/角色/重叠、嵌套索引、未知元信息、前向输入/RNG/指标/监控tag不变。
测试用元数据替代nuScenes数据对象；并非真实点云训练或CUDA通过。

本地没有完整nuScenes、CUDA和实际Lightning执行条件。13个跳过项仍未验收；服务器真实
full preflight、16步事务、整个epoch资源峰值与epoch恢复没有在本地执行。独立同卡数值
重复验收及分数恢复也不能由CPU合同通过推导。机器可读记录和源码hash见
[validation.json](../artifacts/ct_checks/reports/20260910_v28_full_readiness/validation.json)。

## 9月10日：已同步的服务器运行证据

三组已在服务器完成60轮/75720次B0更新，两个显式CUDA阻断不再出现。
B0/42 final=45.200/47.243，B0/52=52.876/64.478；Full/42未校准且动作0，分数与
B0/42相同。跨臂全部观测loss、末轮模型/BN/Adam一致。seed42尚未达健康目标；
58/59评测、独立同卡重复、完整epoch边界恢复及Full策略验收不能由此自动视为完成。
只读数据分析与复核见[三组报告](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。
以下9月8–9日条目保留各自当时的检查范围，不与本次服务器证据混计。

## 9月9日：服务器报错后的增量验证

用户已实际启动 B0/42、B0/52、Full/42 三组 mini。先后三组共同遇到空间 NLL 的严格
确定性报错、仅 Full 遇到 AP 浮点 cumsum 报错；对应本地修复已完成，修复后重启命令
已提供。**尚未收到修复后完整 CUDA 验收、epoch 边界恢复或60轮正式得分。**
根因、提交、修改位置及复核命令集中记录在 [CUDA 故障说明](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。

| 增量检查 | 当时实际结果 | 范围 |
| --- | --- | --- |
| CE 修复专项（`3b8cc57`） | **47 passed / 1 skipped** | 原 CE 整批 loss/梯度、完整 B0 的 BC 与不同 moving 比例、Adam 更新、v28 配置身份、B0/Full host。 |
| AP 修复专项（`749bc13`） | **15 passed / 1 skipped** | 真实 host 的非空/空支持、整数累计与旧 CPU 计数逐位比较、B2/B3 梯度归属、v28 B0/Full 训练与 Adam hooks，另复核 CE。 |
| 两次修复的修改文件 compileall 与 diff 检查 | 通过 | 不是全量测试或服务器验收。 |

两次专项覆盖重叠，不能相加。两次唯一跳过项均是 CE CUDA 重复测试；AP 整数累计的
真实 CUDA 执行同样未在本地验证。以下453/12及旧HEAD信息保留9月8日历史快照，
不宣称是修复后的当前HEAD全量结果。此次文档更新未重新运行训练测试。

## 9月8日：实施全量验证历史快照

当时 v28 代码与本地回归完成，尚未运行服务器工程验收或正式训练；当时没有分数恢复结论。

| 检查 | 实际结果 |
| --- | --- |
| `python -m pytest -q` | **453 passed，12 skipped，192.87秒** |
| `python -m compileall -q models/ datasets/ utils/ tools/` | 通过 |
| `git diff --check` | 通过 |
| 首轮矩阵工具 | 生成1个B0计划；training_executed=false，followups_executed=false |
| 旧slimming校验 | 未通过：固定要求HEAD=`001951a`，当前HEAD=`8b8b8d9`。未修改该门禁。 |

本地环境为 Python 3.12.7、PyTorch 2.3.0+cpu，CUDA不可用，未安装 pytorch-lightning。12个跳过项涉及CUDA、真实Lightning及既有可选运行依赖；不能计为真实训练通过。

## 已取得的数值证据

- 独立冻结的 SeqTrack 前向/损失算式与 v28 真实网络逐位相等：候选不均衡，moving数量0/1/3/4；覆盖完整batch各项loss、BC梯度系数、所有参数梯度与Adam更新。
- B0、B1-GRU、B1-CfC、Full−B3、Full、reference 六配置两次更新后，B0梯度、参数、BN、Adam和RNG逐位一致；插件loss对B0参数无梯度路径。
- 实际 `training_step`、`on_before_optimizer_step`、`on_train_batch_end` 接线覆盖B0与Full；审计开关不改变loss、参数、梯度、BN、Adam或RNG，Full四个启用组首个合法事务均有更新。
- 实际B0与B1-GRU的初始化及step1审计文件经比较工具逐位通过；比较只使用原B0网络模式，排除CT指标/合同/RNG容器的存在性差异。
- Adam快照重放覆盖第1步与已有状态的第2步，修改梯度后能检测失败。池化CPU前向/反传、Seg导出和B2特征对应、稀疏ID/原重抽、结构候选、物理诊断、场景/版本/恢复身份、失跟恢复报告均有回归覆盖。

冻结参考算式来源 SHA256 为 `d0bfa7805681fda53a7fb6354719c9cd911c1e29ce61051034d63f05c478f163`。正式reference配置与B0共用兼容实现，不是独立架构消融；独立算式仅用于本地等价验证。

## 历史资产检查

本轮没有修改旧YAML或`output/`；相对当前HEAD的旧配置diff为空，最终检查期间受保护输出清单指纹保持一致。项目相对更早的`001951a`清单已存在演进：20/22个旧resolved配置hash及历史output清单不同，因此该固定历史门禁不能当作v28通过证据。没有重置HEAD、改写历史或更新其基准。

该历史检查的机器可读记录及当时修改文件指纹见 [validation.json](../artifacts/ct_checks/v28_local_validation/validation.json)。本地生成的示例矩阵含Windows绝对路径；服务器应按说明重新生成。

## 服务器仍须验收

实际数据preflight、同卡B0-A/B0-B/B1-GRU各100步、Full两工程epoch连续/恢复检查仍需服务器证据。用户后续已确认GPU1 B0/42、GPU2 B0/52、GPU3 Full/42三组mini并行，从头60轮，结果写入带日期的`output/`新目录；不再将历史“只B0”排程当作当前限制。完整记录官方mini_val的final60和58/59/60均值，并附失跟/稀疏恢复统计。初步健康目标分别为S≥49.986、P≥58.962；严格1个百分点条件以同协议比较验收。完整nuScenes和其余模块大规模扩展仍待B0验收。

完整命令见 [服务器流程](CTSEQTRACK_V28_SERVER_RUNS.md)，修改原因见 [变更审计](CTSEQTRACK_V28_CHANGE_AUDIT.md)。本地结果没有定位历史首个异常CUDA算子，也没有证明历史得分差全部由随机性引起。
