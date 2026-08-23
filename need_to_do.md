# CT-SeqTrack Safe-SeqTrack v25 当前状态

本文件只记录状态和下一步；实验定义以
[docs/SAFE_SEQTRACK_V25_PROTOCOL.md](docs/SAFE_SEQTRACK_V25_PROTOCOL.md) 为当前协议来源；
v24 仅作为只读失败证据。

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
- [x] 新增 `25_b0/b1/full_minus_b3/full` 及 full-nuScenes 配置，不覆盖 v24。
- [x] B0 observation 改为无状态 shuffle/candidate/点采样，四候选损失固定为 `0.5, 1/6, 1/6, 1/6`。
- [x] 四臂统一 `ct_seqtrack.train.v2`，mechanism iterator 惰性创建并隔离 observation RNG。
- [x] v25 使用单 Adam 自动优化和 B0→B1→B2→B3 互斥参数组；重复 B0 mechanism 前向为 no-grad、BN 隔离。
- [x] checkpoint 增加 v7 runtime/RNG/参数组/update/hash 审计和 CUDA 分阶段审计。
- [x] 本地合同测试、全量 pytest 和 compileall 已通过；准确数量以本次交付记录为准。

## 服务器验收（待执行）

- [ ] 固定同一安全 batch，对比 SeqTrack control 与 CT B0 的输入、输出、loss、梯度和一次 Adam 更新。
- [ ] 四臂从相同 seed42 完成 disposable 100-step，比较 initial/step1/step100 B0 hash。
- [ ] 对比 validation cadence 1/5 的相同 observation-step B0 hash。
- [ ] 完成 scratch 连续训练与 epoch-boundary resume 等价检查。
- [ ] 对比各阶段 CUDA allocated/reserved/peak，并执行 B0 显存阈值门禁。
- [ ] 可视化真实序列点云与逐帧预测框，确认非空、有限且帧数一致。

本机有可用 GPU，但没有发现正式 nuScenes 数据根，且当前 Python 环境缺少
`easydict`、Lightning、torchmetrics 和 nuScenes devkit。上述真实门禁完成前，
不得物理删除 `SEQTRACK3D` 中仍承载兼容行为的 dormant source branches。

当前本地机器不具备真实 nuScenes/完整 Lightning 环境，因此这些服务器项目不得标记为“已通过”。任何 disposable smoke checkpoint 都不能用于正式实验初始化。

## 瘦身完成后实验

- [x] 固定 candidate 协议：B0 为 4 views，candidate0 权重 0.5；B2 只读取 canonical candidate0。
- [ ] v25 B0、B1、Full-B3、Full mini seed42 分别从 epoch0 跑满 60 epoch。
- [ ] 完成 B1、B2、B3 机制指标和 matched-prefix hash 审计。
- [ ] mini 通过后运行完整 nuScenes 四臂 seed42。
- [ ] 补完整 nuScenes B0/Full seeds43、44。

## Claim guardrails

在注册实验完成前，不得声称稳定涨点、SOTA、物理时间因果贡献、memory 贡献或
分布无关风险保证。B4 默认关闭，不进入当前主结果。
