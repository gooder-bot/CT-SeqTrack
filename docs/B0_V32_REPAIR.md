# v32 B0修复实现记录

## 依据与范围

用户批准B0恢复SeqTrack局部序列观测基础，保留递推1/3/3/8和必要接口。2026-09-22 最新安排先跑独立SeqTrack、B0、Full-GRU、Full-CfC，各seed42，物理GPU依次0/0/1/1；原计划的基线seed52复验留待后续。v31证据见[9/21报告](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)。不修B1/B2/B3机制，不读取TrajTrack，不修改冻结参考或既有输出。

## 实现对应

| 子系统 | 修正 | 理由与边界 |
|---|---|---|
| B0几何 | anchor-local XYZ/相对yaw，公共框还原世界轴 | 减少绝对朝向负担，B1/B2保持世界几何 |
| 语义门控 | foreground detach，coarse260→256，quality摘要detach | 去除定位→分类概率捷径，BC/共享特征仍训练 |
| 稀疏token | 最多128组，稀疏时每真实点一个token | 不复制点，不加FPS/KNN |
| 历史观测 | sequence_valid与current_valid分开 | 历史可支持定位，不伪装当前测量 |
| B0损失 | local pose监督，CE current/history各50%，BC按帧端点 | 缺组归一化，保留原类权重/总系数 |
| 递推初始化 | 共用±.3m、±1.5°seed偏差，估计hint=.2/.8 | 不转轨迹中心，不改GT物理位移，不GT重置 |
| 尾批 | reserve112，drain/partial使用B0 running BN | 避免单轨迹和末尾teacher集中污染EMA |
| 独立参考 | 原网络/loss/teacher4/重采样及SHA来源 | 共用合法数据和总预算，方法独立 |

保留sin/cos角度、fine residual、正确帧布局、唯一点mask、严格身份标签和decoder quality头。B2输入/vote/协方差/损失、B3 mode/quality目标保持；decoder只临时换几何坐标，不旋转学习特征或物体系记忆。

参考对照额外记录了三项运行适配：数学等价的确定性pooling/flat-NLL、Windows离散标签转int64、每个正式checkpoint评测前重置共同seed。网络bbox_size在训练及推理均取首帧，原teacher标签保持。原forward/梯度/BN/Adam核对一致，CE标量有归约次序导致的浮点舍入差；真实CUDA尚未验证。

## 实验与版本

采用v32 schema与32_*配置，原位演进、不复制旧CT实现。四份31_* YAML原样保留，旧checkpoint/config不能进入新实验。参考原始重采样可能改变实际曝光分布，不声称逐端点配对。预算与验收见[协议](EXPERIMENT_PROTOCOL.md)，命令见[工具面](FORMAL_TOOLING.md)。

只读工具 `tools/compare_v32_baselines.py` 留作后续 reference/B0 × seed42/52 的完整验收；它不是本轮四臂汇总器。补齐两个seed后核验四次训练身份/预算及12份逐帧记录，重算final60与late-3，按每seed双指标不低于reference超过2pp判定；证据缺失时输出invalid_evidence，不生成通过结论。当前四组命令见 [v32 mini 运行说明](CTSEQTRACK_V32_MINI_LAUNCH.md)。

## 验证状态

实施前本地CPU检查150 passed、4 skipped；首轮修复后246 passed、3 skipped。本次按四臂启动要求补齐检查，使用已有Lightning2.0.2运行全量测试：**249 passed、3 skipped**（80.81秒）。跳过项为一个“缺Lightning”分支及两个真实CUDA测试；没有把CUDA跳过解释为通过。compileall与git diff --check通过。

- B0局部几何、梯度隔离、稀疏/空观测、B2/B3原监督与模式梯度合同通过。
- 真实采样器两seed共120份计划均为19,108行、1,195步，无漏重或因果顺序冲突；Windows workers0/2的production准备张量一致。
- 合成原始点云通过B0、reference、Full-GRU、Full-CfC完整entry训练、checkpoint重载和连续评测；自动收尾与单独test逐帧文件一致。独立test结果中的checkpoint轮次取已校验权重的元数据，不再写null。
- Full-GRU/CfC另完成含非零锚框yaw和B2有效模式的CPU前向/反传/Adam，以及连续3步raw host获取和状态提交，loss/梯度有限且strict deterministic保持开启；不改B1/B2/B3机制。
- 真实Lightning生命周期验证B0中断恢复；reference真实网络中断恢复的参数、Adam状态和曝光审计与连续训练一致。
- 独立参考与冻结源码数学核对、重采样审计、首帧尺寸合同通过；验收工具使用临时合成证据检验通过/未达标/缺证据三类分支。

检查使用Windows/Python3.12.7/PyTorch2.3.0+cpu，复用只读的 `artifacts/ct_checks/pl202_runtime`（Lightning2.0.2），未安装或升级环境。复现命令：

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath artifacts/ct_checks/pl202_runtime).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ tests/ main.py
git diff --check
```

检查原始记录：[首次实现](../artifacts/ct_checks/20260922_v32_b0_implementation/pytest.xml)、[最新四臂入口复核](../artifacts/ct_checks/20260922_v32_four_arm_readiness/pytest.xml)。合成入口产物在独立 `artifacts/ct_checks/v32_integration/`，不能作为正式权重或分数。

四份v31 YAML及prior/evidence/memory/acquisition源文件无diff。真实nuScenes/CUDA和四次scratch60尚未执行，尚不能判断得分验收；服务器仍只读，未上传、安装或启动任务。

## 本次启动复核补齐

当前协议、README、工具面、服务器路径和Full配置注释已统一为seed42四臂；v31启动/就绪页增加历史标记和v32链接，防止误用旧配置。独立参考/B0的seed52是后续复验，双seed比较工具保留明确用途。后台命令显式固定native allocator、CUBLAS workspace和GPU 0/0/1/1，保留原训练预算。旧空间NLL、adaptive pooling、bool排序/浮点cumsum路径已复核；CPU验证不能代替PyTorch2.0.1的真实CUDA验证。

另以Python3.9语法模式解析58个活动Python文件，并直接提取新版运行说明的四条命令交给实际CLI解析，配置/seed/预算/物理GPU映射均符合本轮安排；没有执行这些服务器训练命令。
