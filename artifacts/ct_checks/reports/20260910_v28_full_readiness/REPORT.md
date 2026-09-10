# v28 完整 nuScenes 单组诊断就绪结论

审计日期：2026-09-10。范围：当前代码的 **Full模型、Car、seed42、完整nuScenes、scratch60epoch**；不将“完整数据”误解为已经启动全部五类多臂矩阵。本地没有完整数据/CUDA，未远程启动训练。

## 结论

**支持运行这一组完整数据诊断。** 已核对配置、观测/机制数据流、B0—B3所有权、保存恢复及训练后校准评测，没有发现mini切换full时必然出现的模型接口阻断。真实数据预检和短事务检查仍须在服务器通过；具体命令已整理到 [完整数据诊断说明](../../../../docs/CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。

用户明确希望先完成一次完整数据训练再集中分析，该安排可以执行。它不依赖“mini已经恢复”的错误前提，也不等于耦合最优、分数必涨或完整数据论文对照已完成。

## 基线证据与本次价值

| 已完成mini实验 | final60 Success | Precision | 判断 |
| --- | ---: | ---: | --- |
| B0 seed42 | 45.200219 | 47.242888 | 未达mini历史恢复目标49.986/58.962；后期闭环退化仍需解释 |
| B0 seed52 | 52.876368 | 64.478118 | 达到历史健康区间，但不能单选此seed宣布稳定恢复 |
| Full seed42，未校准 | 45.200219 | 47.242888 | 动作为0、回退B0；不是已校准Full的收益测量 |

Full/B0 seed42在75,720步观测loss、末轮B0参数/BN/Adam及观测前缀指纹上一致；Full各插件已参与18,000次机制事务。当前证据不支持“插件污染B0导致这次低分”，也不支持“模块被冻住所以Full相同”。详细原始证据见 [mini三组分析](../20260910_v28_mini_three_arm/REPORT.md)。

完整数据诊断可以继续观察seed42后期退化、点证据获取和动作选择。mini只有2帧/4个新增目标点进入support，分母22,715是有机会帧的crop外目标点总量，其中可能有当前有限获取范围不可达的点；不能只用这个比例断言坐标错误，也不能预言更大数据一定修复它。完整数据S/P必须与完整数据同协议对照比较，mini数值阈值不适用于full。

## 数据和模块链路

1. **完整数据配置已经存在。** [解析结果](resolved_full_config.json)和[mini→full逐键差异](mini_full_config_diff.json)只有实验名、版本、路径、三个split及mini步数约束这7项变化；B0/B1/B2/B3网络、loss与优化预算沿用mini。没有新增或改写正式YAML。
2. **350/150协议正确。** train_track350全训练，官方val150只评测；内部17/18为参数训练内子集，用于阈值拟合和锁定诊断。某类别有效轨迹可能只覆盖其中部分场景，应同时报告manifest与实际轨迹/帧数量。
3. **更新与端点自然计算。** full解除1262预期，B0按实际四候选总体长度shuffle/drop_last；机制覆盖非首帧完整端点，尾部partial slot不丢弃。每个观测事务的mechanism tick上限依实际调度计算。报告中的合成长度测试不是full真实样本量。
4. **B0—B3接入一致。** B0原目标和严格数值路径保留；B1物理时间/预测历史和实际获取记录一致；B2真实Seg第二层64特征与slot/frame/ID对齐并detach，新证据使用整个raw crop差集；当前B3为效用策略实现，共享结构合法候选，host唯一提交训练递归状态。
5. **全部启用参数从头训练。** 一个Adam四组，BN/RNG/梯度所有权隔离；机制eval/no_grad只用于隔离B0的重复前向，不冻结其观测训练，也不构成插件的延迟解冻。
6. **保存与后处理完整。** last在完整epoch边界保存，58/59/60另存。每checkpoint单独校准，再用同run resolved config评测官方val；同一Full权重另跑observation闭环控制，有助于分析B0，但不是独立训练的正式B0或原始SeqTrack臂。

独立审计：[数据/控制面](data_control_audit.md)、[耦合](coupling_audit.md)、[评测与校准](evaluation_audit.md)。审计中记录的“待修复preloading默认值/元信息”已由本轮下述补丁处理。

## 本轮实际修复及边界

| 文件 | 问题 | 处理 |
| --- | --- | --- |
| `tools/preflight_ct_v28.py` | YAML没有preloading，真实get_dataset访问时AttributeError；旧提示写B0-only | 缺省False、保留显式配置；提示支持单组完整数据诊断，保留scratch/确定性校验 |
| `models/base_model.py` | CSV场景unknown、验证partition硬编码dev、full官方val被误标参数训练重叠 | 仅在前向完成后，从真实dataset及wrapper索引解析导出scene/tracklet/role/split/overlap；失败明确unknown/None及metadata_error警告 |
| 两个新回归测试文件 | 先前测试没有覆盖真实工厂默认字段及导出身份 | 执行真实get_dataset控制入口与生产val/test hooks，覆盖full/mini、嵌套索引、缺失元信息、输入/RNG/指标/监控tag不变 |

没有改变`evaluate_one_sequence`输入、递归tracklet key、网络、损失、学习率、动作半径、点采样或冻结策略。旧`precision/dev`及`dev_diagnostics/`名称为兼容监控保留，行内角色已修正。只改日志也会更新base_model源码hash，必须同步代码后再训练/校准，不能篡改旧policy身份来复用。

## 资源与剩余服务器边界

- **完整集首跑关闭预加载。** 当前按轨迹重复保存整幅点云，观测/机制/验证各持对象；按需读取不改变帧数据和目标，增加磁盘IO以控制缓存RAM。
- 完整epoch的CPU递归状态与诊断行仍会累积；多个机制tick统一反向可能产生额外峰值。16步只验真实入口和首批事务，不能替代整epoch内存、尾部调度和恢复验收。
- 训练时Full尚无策略，验证会回退B0。最终分析必须包含逐58/59/60内部校准及官方val选择性评测；不能只收集训练日志就断言模块没有效果。
- 不固定full硬编码步数，不削减数据/端点，不换seed挑best或延长60轮掩盖退化。完整五类矩阵及独立匹配SeqTrack对照另行完成。

两处修复后本地全套pytest **481 passed / 13 skipped（177.90秒）**；compileall、diff和命令语法检查通过，90份mini原始文件hash保持一致。详见 [validation.json](validation.json) 和 [项目验收记录](../../../../docs/CTSEQTRACK_V28_LOCAL_VALIDATION.md)。旧slimming工具仍固定要求HEAD=001951a，在当前749bc13因历史基点限制失败，未改该门禁。
