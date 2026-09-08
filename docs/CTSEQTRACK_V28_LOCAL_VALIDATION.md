# v28 本地实施验收（2026-09-08）

v28 代码与本地回归完成；**尚未运行服务器工程验收或 v28 正式训练，尚无分数恢复结论**。

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

机器可读记录及当前修改文件指纹见 [validation.json](../artifacts/ct_checks/v28_local_validation/validation.json)。本地生成的示例矩阵含Windows绝对路径；服务器应按说明重新生成。

## 服务器仍须验收

先做实际数据preflight、同卡B0-A/B0-B/B1-GRU各100步、Full两工程epoch连续/恢复检查，再只正式训练B0 mini Car seed42 60轮。记录官方mini_val的final60和58/59/60均值，并附失跟/稀疏恢复统计。初步健康目标分别为S≥49.986、P≥58.962；严格1个百分点条件以同协议比较验收。未恢复不启动模块长跑。

完整命令见 [服务器流程](CTSEQTRACK_V28_SERVER_RUNS.md)，修改原因见 [变更审计](CTSEQTRACK_V28_CHANGE_AUDIT.md)。本地结果没有定位历史首个异常CUDA算子，也没有证明历史得分差全部由随机性引起。
