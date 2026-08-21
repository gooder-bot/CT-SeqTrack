# CT-SeqTrack v24 当前状态

本文件只记录状态和下一步；实验定义以
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) 为唯一来源。

## 已完成

- [x] 固定 `main@001951a` 的 tracked-file、22 配置、测试和 `output/` 保护基线。
- [x] 正式 v24 配置脱离历史 21/22/23 继承链，resolved-config 逐键一致。
- [x] 删除 93 份非活动 YAML；保留 20 个正式入口、2 个 B4 入口和必要基础配置。
- [x] 删除非活动且可由 `001951a:<path>` 恢复的历史资产/文档；`output/` 指纹未变。
- [x] 删除 73 个被忽略的旧服务器输出、旧协议 manifest 和 run log（67,986,460 字节）；这些文件原本不在 Git 中。
- [x] 工具面收敛为八类正式任务，删除 73 个旧 M/TWC/CRPA/Search/Gate/replay/preflight/promotion 工具。
- [x] 删除旧 replay 专用测试并移除 preflight/promotion 门禁测试，保留当前耦合与行为合同测试。
- [x] B0--B3 所有权、candidate0、extension-only、calibration 和 scratch 合同保持不变。
- [x] 历史数值压缩为证据索引，负结果和 claim boundary 未被隐藏。
- [x] 当前测试为 `105 passed, 1 skipped`；瘦身前基线为 `122 passed, 1 skipped`。

## 服务器验收（用户本轮跳过）

- [ ] 在完整 Lightning/nuScenes 环境补齐 seed42 参数/state_dict 初始化快照。
- [ ] 固定真实 mini batch 比较 B0/B1/B2/B3 输出、loss、梯度和 optimizer 分组。
- [ ] 完成 scratch 100-step、epoch-boundary resume 与连续运行等价检查。
- [ ] 可视化真实序列点云与逐帧预测框，确认非空、有限且帧数一致。

本机有可用 GPU，但没有发现正式 nuScenes 数据根，且当前 Python 环境缺少
`easydict`、Lightning、torchmetrics 和 nuScenes devkit。上述真实门禁完成前，
不得物理删除 `SEQTRACK3D` 中仍承载兼容行为的 dormant source branches。

用户已明确选择跳过本轮服务器 smoke/等价检查。因此源码兼容分支按安全合同保留，
本地瘦身和正式配置可以交付，但这些服务器检查不得标记为“已通过”。

## 瘦身完成后实验

- [x] 固定 candidate 协议：B0 为 4 views，B2 只读取 canonical candidate0。
- [ ] B0、B1、Full-B3、Full mini seed42 分别从 epoch0 跑满 60 epoch。
- [ ] 完成 B1、B2、B3 机制指标和 matched-prefix hash 审计。
- [ ] mini 通过后运行完整 nuScenes 四臂 seed42。
- [ ] 补完整 nuScenes B0/Full seeds43、44。

## Claim guardrails

在注册实验完成前，不得声称稳定涨点、SOTA、物理时间因果贡献、memory 贡献或
分布无关风险保证。B4 默认关闭，不进入当前主结果。
