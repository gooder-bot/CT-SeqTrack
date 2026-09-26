# CT-SeqTrack 工作约定（v34 B0 上下文）

## 2026-09-26 当前安排（覆盖下方 v33 历史启动安排）

- 用户最新明确扩为八组mini单seed：SeqTrack正常1e-4/减半5e-5；S（1/3/3/8）和W（1/4/4/8）各1e-4/5e-5/2.5e-5。SeqTrack保留StepLR20/40，S/W保留20/50，无warmup；均seed42、scratch60、71,700更新。见 [八组协议](docs/B0_V34_LR_GRID.md) 和 [最新启动说明](docs/CTSEQTRACK_V34_MINI_LAUNCH.md)。该请求覆盖此前“两组、不重跑参考、不扩LR”。
- 活动源码继续沿用 `models/ct_v31/` 和 `models/ctseqtrackv31.py`。v34 按显式版本身份启用新结构；v33/reference 的原配置、schema、checkpoint 键和行为保持。不得只替换全局 v33 身份常量。
- 旧R/C/旧v32 B0仍为固定证据，另由用户启动新R正常/减半；GPU0跑R正常＋S三组，GPU1跑R减半＋W三组，各四个独立单卡进程。代理服务器权限只读，不得上传、安装、启动/停止任务。不得把可复用旧R写成用户无需运行本轮新R。
- 总体 final60/late-3 的 S/P 四项均达到 R，且固定 31 条曾移动轨迹全程（657 预测帧）的四项均不低于 C，才允许进入 Full 规划。未完成正式训练前不宣称涨分。
- v34 窗口4是本轮明确授权的协议变化，十轮课程、112 reserve、BN 策略及总预算不变。历史残差、loss、pooling、quality 目标与原 Full 梯度边界保持。
- 原v34实现证据为 `artifacts/ct_checks/20260926-154028_v34_implementation/`；本轮八组就绪复查与服务器只读证据为 `artifacts/ct_checks/20260926-161231_v34_eight_run_readiness/`。上传后仅一次新的v34真实CUDA batch由用户执行；旧v33服务器检查不构成v34验证。启动显式保留native allocator，避免继承旧max_split_size_mb:64。

## 当前范围与授权

- 唯一活动项目为本仓库，所有命令从本仓库根目录执行。兄弟基线只读，不修改；不读取 TrajTrack。
- 下方为 v33 实验历史约定；当前 v34 范围以上节为准。v32 通过 Git `ddcb1a1`、v31 通过 `b1d886e` 复现；不改写历史。
- 用户最新要求：服务器 `lishengjie@10.109.253.86` 本轮只读；允许本地修复，由用户自行上传和启动。不得自行上传、安装、启动训练、停止进程或改写服务器文件。旧部署授权不延续到本轮。
- 本轮由用户从头运行四组 mini Car、seed42：R 独立 SeqTrack、A 综合 B0 原配方、B 综合 B0 延后第二次衰减、C 为 B 全程半学习率。物理 GPU 依次为 0/0/1/1，每组单卡。
- 2026-09-25用户追加D：综合B0的B配方全程学习率×1.5，初始1.5e-4、milestones=[20,50]，物理GPU0；原A/B/C保留。只登记新增B0配方，不改变预算或参考模型配方，见`docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md`。
- 2026-09-25用户追加E：继承B，峰值LR=3e-4、前2000次更新线性warmup（从峰值/2000升至峰值），完成20/50轮后降至3e-5/3e-6，物理GPU1；warmup计入原71,700次更新。原R/A/B/C/D不改。E相对B同时改变LR与warmup，不能独立归因，见[追加E说明](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。本轮未访问或修改服务器，由用户上传并启动。
- 已完成 v32 R 仍是可复用的历史证据；用户当前选择重跑 R。不得把“可复用”写成“本轮无需启动 R”。本轮不新增 Full 独立诊断。
- 协议以 [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) 为准；实现见 [B0_V33_REPAIR.md](docs/B0_V33_REPAIR.md)，用户启动说明见 [CTSEQTRACK_V33_MINI_LAUNCH.md](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)，工具范围见 [FORMAL_TOOLING.md](docs/FORMAL_TOOLING.md)。

## 实验与实现不变量

- scratch-only：所有正式运行从 epoch0 随机初始化。`--init_checkpoint` 禁止；`--checkpoint` 仅用于相同配置身份的完整 epoch 边界恢复或评测。工程检查权重不得作为正式初始化。
- 启用参数不冻结；detach 和 BN 统计隔离是耦合合同，不是参数冻结。每 batch 一次 forward/backward/Adam，随后提交 detached accepted 状态。
- B0 保持四分支；v33/v34 S 为1/3/3/8，v34 W 为1/4/4/8。保持10轮课程、112个尾部预留窗口和既定BN策略；不靠缩小batch、截断端点、追加历史forward或冻结模块降低预算。
- 公开框使用 anchor 平移后的世界轴 XYZ 和 absolute yaw；B0 几何视图换到 anchor-local。q0 直接回归中心，history/mode 输出 seed 残差，共享最终头；mode 输出 live center 和 query detach 保持。
- 可信状态、缺测、padding、唯一 raw ID、裁剪和记忆写入按现有合同执行。GT 仅用于合法初始化、监督和被动诊断，不得用于当前预测裁剪、候选选择或 accepted 提交。
- B1 历史运动边资格保持原定义；保留 nuScenes full、KITTI、CfC/GRU、模块臂及时间控制能力。接口保留不代表对应实验已完成。
- 比较固定 final60 与 late-3（58/59/60），不得跨实验挑选不同最佳轮次；不得在四组完成前声称 v33 涨分、达到 SeqTrack、SOTA 或时间/记忆因果收益。

## 保护与验证

`output/` 和既有 `artifacts/` 的数据、日志、权重、报告、脚本及图片不得删除或覆盖。新工程产物写入 `artifacts/ct_checks/` 的独立目录；不得用 `git clean` 清理 ignored 证据。项目注释和文档使用 UTF-8 中文。

此前服务器快照验证：Python3.9.19、torch2.0.1+cu118、pytorch_lightning2.0.2 下 **314 passed、1 skipped**，一次真实 CUDA batch 的 forward/backward/Adam/commit 已通过且未保存 checkpoint。证据见 `artifacts/ct_checks/20260924-190116_v33_implementation/`。后续被动汇总增量单独记录本地验证，不声称最新源码已逐文件完成服务器复验。这些历史检查不授予本轮服务器写权限，也不是四组正式训练完成证明；不要反复堆成用户启动前置步骤。

本地代码修改后按改动范围验证：

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
```

缺 Lightning/CUDA 的本地跳过必须如实说明。旧固定 HEAD 精简门禁仅是历史工具，不作为 v33 验收。当前网络使用 PyTorch 路径，不要求旧 PointNet++ CUDA 扩展；不擅自升级服务器环境。
