# CT-SeqTrack v26 当前状态

本文件只记录状态和下一步；实验定义以
[docs/CTSEQTRACK_V26_METHOD.md](docs/CTSEQTRACK_V26_METHOD.md) 为当前协议来源；
v24/v25 仅作为只读历史证据。

## 已完成

- [x] 实现 observation-anchored bounded adaptive shell、三帧 causal backup corridor 与 B1-invalid 固定 `2m/1m` CV fallback。
- [x] 实现 deterministic 768-point novel pre-pool、三位 source bitmask、relation/FPS/stateless 256-point selection 与 K=3 robust consensus voting。
- [x] counterfactual schema 升级为 v3，修复 nuScenes wlh/local-axis 和 raw-vs-novel 支持统计，加入逐层集合不变量与 v26 报告工具。
- [x] 恢复 B3 helpful/harmful action-confidence gate，加入 consensus/covariance/inlier/margin 特征与 calibration/dev promotion artifact v2。
- [x] 新增 SeqTrack-strict、B0、B1-GRU、B1-CfC、Full-B3、Full 的 v26 full-nuScenes seed42/60-epoch scratch 配置；集成主臂固定 GRU，CfC 仅作为 B1 backend 诊断；所有启用模块从第一步参与 unified Adam。
- [x] 修复 v26 B1 acquisition-margin 事务复算遗漏、corridor 长度截断中心偏移、Full action-export 非完整 checkpoint 误载入风险。
- [x] 正式训练显式固定 `min_epochs=max_epochs=60`、`max_steps=-1`、单卡、全量 train/val batch，并保存 `last.ckpt` 与 epoch 58/59/60。
- [x] 新增零训练步 `preflight_v26_full.py`，检查完整数据根、CUDA/PointNet++/Lightning、模型构造、优化器参数组和冻结状态。

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
- [x] checkpoint 增加 v8 runtime/RNG/B1-backend/loss/参数组/update/hash 审计和 CUDA 分阶段审计。
- [x] 修复 B1 gap2/gap4 transaction、归一化残差长尾监督、detached-mean beta-NLL 与独立 calibration/dev 校准；固定 B2 geometry 不变。
- [x] 新增参数匹配、无外部依赖的 CfC 插件与 `--b1-backend gru|cfc`，默认仍为 GRU。
- [x] 本地合同测试、全量 pytest 和 compileall 已通过；准确数量以本次交付记录为准。

## 服务器验收（待执行）

- [ ] 分别对本轮五臂运行零训练步 preflight；它不运行 mini、不读训练样本、不产生 checkpoint。
- [ ] 直接从 epoch 0 独立运行 v26 B0、B1-GRU、B1-CfC、Full-B3、Full 完整 nuScenes Car seed42；本轮不先跑 mini，也不从 smoke/其他 arm checkpoint 初始化；SeqTrack-strict 保持为单独登记的外部参考。
- [ ] 对 final 与 late-3 的每个 Full checkpoint 分别导出互斥 calibration/dev rows、完成 promotion，并生成不可跨 checkpoint 复用的 artifact。
- [ ] 报告 schema-v3 geometry/sampling/voting/B3 漏斗、final/late-3 tracking、耗时、显存及 tracklet-paired bootstrap CI。

本机有可用 GPU，但没有发现正式 nuScenes 数据根，且当前 Python 环境缺少
`easydict`、Lightning、torchmetrics 和 nuScenes devkit。上述真实门禁完成前，
不得物理删除 `SEQTRACK3D` 中仍承载兼容行为的 dormant source branches。

当前本地机器不具备真实 nuScenes/完整 Lightning 环境，因此服务器 preflight 未运行前只能给出条件式 GO。任何工程或其他 arm checkpoint 都不能用于正式实验初始化。

## Claim guardrails

在注册实验完成前，不得声称稳定涨点、SOTA、物理时间因果贡献、memory 贡献或
分布无关风险保证。B4 默认关闭，不进入当前主结果。
