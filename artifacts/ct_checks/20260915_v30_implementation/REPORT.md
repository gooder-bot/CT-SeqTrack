# v30 本地实施与验收记录（2026-09-15）

已在本地实施用户批准的v30方案。没有连接服务器、同步文件、启动训练、停止旧任务或改写历史实验输出。
本次代码基于`525eb543a321a36b5ae65545da967d53d3eb00c3`工作区；未创建提交。

后续mini启动复核：按用户最新选择登记GPU0 B0、GPU1 Full-GRU、GPU2 Full-CfC。
定向74 passed/1 skipped；三组实际CLI/config解析通过，6个后台/tail代码块通过Bash语法检查。
补齐启动器OPENBLAS线程与无缓冲日志设置，旧示例`--preloading`在v30必须省略。
记录见[mini_launch_review.json](mini_launch_review.json)，命令见[最新mini启动页](../../../docs/CTSEQTRACK_V30_MINI_LAUNCH.md)。
本轮不重跑全量测试或全套哈希，不添加工程报告启动门；没有服务器操作。

## 交付内容

| 部分 | 已落地行为 |
|---|---|
| 共同B0 | 有效测量Seg/BC/历史ref监督、mask-aware BN和max/token mask、真实速度、概率moving、teacher端点保留、短/长roll-in |
| B1 | 实际B0 crop外获取带、21维上下文、crop一次复用、actual/max/9×9一致、需求分组平衡、不可达样本不伪装负例 |
| B2 | 确定性K=3加权支持模式、统一EvidenceModeSet、前景vote、模式质量软标签、support尺度及独立log sigma |
| B3 | 六动作共享收益网络、同状态全动作标签、行内/行间归约、轨迹行为混合、唯一最终状态提交 |
| 校准 | never行max-action-q、仅数值去重、最多10个拟合闭环、锁定dev、新动作/数据/源码/权重身份 |
| 数据与效率 | nuScenes/KITTI通用manifest、KITTI完整标定与真实帧时间、无GT预裁剪、metadata复用、二分索引、256MiB LRU |
| 实验与统计 | 9个主臂配置+6项消融+紧范围；全量训练/评测漏斗、unique/事件/条件分母、分组S/P和mini汇总工具 |

实现细节见[方法说明](../../../docs/CTSEQTRACK_V30_IMPLEMENTATION.md)，
检查、校准、评测及启动命令见[运行手册](../../../docs/V30_DATA_AND_RUNBOOK.md)。

## 验证状态

最终完整`python -m pytest -q --tb=short`：**868 passed、15 skipped，358.73秒，退出码0**。
完整输出保存在[pytest_final.log](pytest_final.log)。

- 16个可运行配置均经过`configure_ct_variant`及正式scratch合同校验。
- `python -m compileall -q models/ datasets/ utils/ tools/`通过。
- `git diff --check`通过；`output/`和历史24/29正式配置无差异。
- 静态检查原始结果见[local_static_checks.json](local_static_checks.json)。
- Python/torch版本、配置清单、72个方法源码文件哈希见[implementation_identity.json](implementation_identity.json)。

新增13个v30测试文件覆盖实际CPU host前向/反向/Adam、三臂B0参数/梯度/BN/RNG一致，
空/1/2测量、固定token桶、3/8步roll-in与teacher端点、获取/标签81格一致、两个共识反例、
次模式动作选择、世界坐标六动作监督、B3无跨模块梯度、接受状态、合法H3及epoch边界恢复，
KITTI标定、原帧时间、尾批完整覆盖、LRU副本语义、校准及mini晋级身份。
外部数据/CUDA依赖由CPU测试夹具替代，这些结果不冒充服务器真实数据训练。

## 集成期间查出的具体问题及修正

1. 旧获取消融误把v30 band参数传给旧严格签名；在几何分派处移除v30专属参数，真实host/sampler覆盖。
2. 全空测量时旧部署分支丢弃合法历史query、强制零位移；v30跳过该覆盖，保持历史预测与训练语义一致。
3. KITTI导出仍依赖`source.nusc`；改用通用scene/sequence元数据，完整端点人口按metadata核验。
4. 评测不运行81格监督，默认监督零值不能充当最大可达计数；改读实际raw-ID sidecar。
5. 机制行为改变动作后旧标量诊断仍可能指向预选候选；统一选择输出，标签和执行框一致。
6. 旧H3/恢复工具的验证周期与v30工程合同不匹配；新增配置分派并用计划级测试核验生产CLI。
7. full全轮计数可能超过float32精确整数范围；设备端double累计，epoch JSON保留完整计数。
8. 两个旧AST提取测试的夹具未包含拆出的metadata/profiling依赖；接入实际辅助函数，原隔离断言保留。

## 尚需实际执行的实验

服务器CUDA真实稀疏帧前向/反向/Adam、合法H3、校准/评测及同卡/epoch恢复仍待执行；
本地`torch.cuda.is_available()`为False。没有新checkpoint、mini S/P或吞吐结果。

先mini Car/seed42/B0、Full-CfC、Full-GRU，scratch60，保留58/59/60，每个Full权重独立校准。
**至少一个Full的final60 S、P同时高于同版本B0才进入full/KITTI。** late-3只报告，无额外多seed前置门。
六项mini消融与时间间隔实验按[正式协议](../../../docs/EXPERIMENT_PROTOCOL.md)登记执行；
不能以loss、presence AP或本地测试数量代替闭环涨分证据。
