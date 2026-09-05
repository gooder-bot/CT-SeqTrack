**校准与协议补充审计｜2026-09-05**

**1. 已复现：物理时间 CLI 对照会被覆盖。** `main.py:540` 接受 `--dynamics_time_mode`，在 `main.py:654` 合并参数；`models/ct_variant.py:195` 左右优先读取已经存在的 `ct_time_mode`，随后写回 `dynamics_time_mode`。v26 继承的 `ct_time_mode: true` 因而把 fixed 和 shuffled 请求均归一化为 true。运行本目录 `root_contract_probe.py` 已复现三种请求全部得到 true。

影响：按文档 CLI 运行的时间消融可能根本没有改变时间模式。此结论不表示历史显式同时设置两个字段的专用 YAML 一定有问题；也不声称尚未执行的实验已被污染。建议统一唯一规范字段，在 CLI 层显式同步并拒绝冲突；日志同时写请求值、最终值和实际 delta_t 摘要，检查真正的数据时间张量。

**2. 已确认：校准隔离的范围仅覆盖 mechanism 训练。** `main.py:811` observation 数据来自整个 train_split；`main.py:850` 起才对 mechanism sampler 应用 train 分区。`utils/recursive_state.py:31` 用全轨迹 hash 做 70/15/15 切分。`tools/export_ct_action_rows.py:38` 默认读取 train_track，再在 `:98` 选 calibration/dev。于是 B1/B2/B3 的拟合轨迹与校准轨迹可不重叠，但 B0 在 observation 训练中已见过这些轨迹。

这不是“官方测试集进入训练”的证据，也不能直接否定所有经验校准结果；它意味着“整个预测器未见过校准集”不成立。新实验若要严格 held-out，可在训练场景内预留独立 calibration/dev scene，所有正式臂的 observation/mechanism 都遵守同一清单，并重新从头训练。保留现有设计时，论文应准确称 mechanism-heldout 并做独立场景敏感性验证。同场景不同 tracklet 共享点云背景，也应在统计分析中考虑 scene 聚类。

**3. 已确认：静态 dev rows 不是闭环策略 promotion。** `export_ct_action_rows.py:67` 强制 observation 模式；calibration 与 dev 均由该模式递归导出。`utils/action_calibration.py:228` 起在固定 rows 上选择/评估阈值。部署在 `models/base_model.py:1950` 左右追加当前实际 candidate_box，并在 `:1957` 将其写入下一帧递归状态。因此被接受的动作会改变后续输入分布。

现有校准可说明 observation 访问状态上的单步动作表现，不能单独证明 selective 全轨迹风险/收益。应先锁定 calibration 阈值，再在不相交 dev 上运行完整 selective rollout，比较相同初始框下两种策略的整轨迹 S/P、累计 gain、恢复时延和伤害，promotion 后才做未见测试。若根据 dev 结果再次调阈值，需要新的验证数据或明确记录 dev 的调参角色。

**4. 统计限制：零伤害 percentile bootstrap 退化。** `utils/action_calibration.py:135` 按 selected tracklets 重采样。30 条轨迹、120 个正收益动作、零观测伤害的合成例子得到 harmful upper95=0。本目录脚本复现这一边界情况。它是经验分位数行为，不能解释为总体零风险；30 个独立二项单位零失败的一侧精确上界约 9.50%，仅作说明，不是本项目 action 加权风险的替代公式。

短期保持经验校准措辞，报告 selected tracklets/actions、风险和覆盖、置信区间及零事件限制。若要做理论风险主张，需重新定义独立单位、轨迹损失、有效单侧检验及阈值族校正。另 `action_calibration.py:218` 用“平均 center gain（米）+平均 IoU gain”挑阈值，量纲和覆盖目标混合；以后可预注册以全体 endpoint 的期望 IoU/跟踪效用为主、center 为约束的选择目标，而不是临时降低伤害阈值。

**5. 已排除的疑似问题。** 对 v26 Full 默认配置，纯 YAML 经 configure 与 main 默认 CLI 评估参数经 configure 的 calibration config identity 一致，本次没有发现默认导出/评估 hash 因这些 CLI 字段而错配。兼容 self-check artifact 虽可由低层函数构造，但 `models/ctseqtrack.py:45` 在正式部署明确拒绝，不能将低层兼容路径误报为已能绕过正式校准。

**6. 已确认的多 seed 隐患。** mechanism 的分区来源是 `ct_partition_seed`（`main.py:855`，正式继承值 42），而 action exporter 的分区来源是 `args.seed/config.seed`（`export_ct_action_rows.py:60`、`:98`）。训练 seed 改成 52/62、分区 seed 仍为 42 时，两边的 train/calibration/dev 归属可能交叉。当前 seed42 不受此差异影响。建议 exporter 使用与训练相同的分区种子并写入 manifest；训练随机种子与数据划分种子应分开管理。

本文件只提出审阅和下轮修复建议，不改变正式协议和已有实验身份。
