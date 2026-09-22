# CT-SeqTrack

CT-SeqTrack 研究真实时间先验条件下的目标证据获取、身份保持与递归定位，任务为 3D 点云单目标跟踪。当前工作树维护 **v32 B0 修正版与独立 SeqTrack 对照**，在原活动实现中演进，保留 `ct_v31` 物理模块路径；v31 及更早实验按对应 Git 版本复现。

## 当前方法

| 模块 | 职责 |
|---|---|
| B0 | 锚框局部坐标中的四帧唯一点观测；分割门控停止梯度、有效点 token、历史支持定位 |
| B1 | CfC 或 GRU 物理时间先验、获取范围与可微时序 context |
| B2 | B0 裁剪外新增点的 768→256 固定预算获取，原始身份记忆与三个投票模式 |
| B3 | 主观测与三个模式共用定位/质量头，直接选择四个框假设之一 |

每个训练 batch 执行一次联合前向和一次 Adam 更新；窗口内各端点仅提交一次 accepted 状态并继续递推，跨帧 detach。sigma、离散几何、标签与跨帧状态的梯度边界保留。Full 不依赖旧版策略标定，也不要求不同臂的 B0 参数逐位相同。

实现入口为 [main.py](main.py) → [entry](models/ct_v31/entry.py) → [Lightning host](models/ctseqtrackv31.py) 与 [联合模型](models/ct_v31/model.py)。数据准备、先验、获取、证据、记忆、解码、损失与评测保留 `models/ct_v31/` 路径，独立对照在 `models/seqtrack_reference/`。当前网络使用 PyTorch 算子，不要求安装旧 PointNet++ CUDA 扩展。

## 已完成的实验

2026-09-21 已核实代码版本 `64ad056` 的 mini Car/seed42 三臂完成 scratch60 与独立 58–60 评测；各 71,911 次 Adam、1,146,480 次端点曝光，官方 mini_val 为 106 轨迹/2,285 帧。

| v31 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| B0 | 23.959519 / 27.834792 | 23.812546 / 28.955142 |
| Full-CfC | 22.491247 / 29.352297 | 24.213348 / 33.709701 |
| Full-GRU | 22.839169 / 31.844639 | 25.303793 / 37.146243 |

工程训练与评测已跑通，**性能验收未通过**：两个 Full 的 final Success 均低于同版 B0，B0 也低于 v30 历史参照 40.473742/47.840262。不能用 Precision 单升、late-3 或中途最好轮次替代预定 final60 S/P 双升条件。

[完整结果与真实数据探针](artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md) 定位了当前帧近全前景、稀疏 token 压缩、尾批 BN、候选重复和记忆错写等问题。本轮只修 B0 与递推训练及必要连接，B1/B2/B3 的机制问题留待基线验收后处理；[v32 实现说明](docs/B0_V32_REPAIR.md) 记录改动与验证边界。v32 尚无正式训练成绩，不声称稳定涨分或时间/记忆因果收益。

## 运行

从本仓库根目录执行：

```bash
python main.py --cfg cfgs/ct_seqtrack/32_b0_mini.yaml --path DATA_ROOT --seed 42 --tag b0_seed42
python main.py --cfg cfgs/ct_seqtrack/32_seqtrack_ref_mini.yaml --path DATA_ROOT --seed 42 --tag ref_seed42
python main.py --cfg cfgs/ct_seqtrack/32_b0_mini.yaml --checkpoint RUN/formal_checkpoints/epoch=060.ckpt --test --log_dir NEW_EVAL_DIR
```

新配置共用 `32_formal_base.yaml`。本轮按用户最新安排运行 **SeqTrack reference、B0、Full-GRU、Full-CfC，均为 seed42，物理 GPU 依次 0/0/1/1**。scratch60、batch16、workers4、FP32、每 5 轮验证；mini 每轮 19,108 行、1,195 次更新。训练结束自动评测 58/59/60，报告 final60 和 late-3。四份 `31_*` YAML 原样保留，只能在历史 Git 版本中使用，不在当前算法下解释。

联合模型配置接口保留 `b0/b1/b1_b2/full`、CfC/GRU、nuScenes mini/full、KITTI 和 `true/fixed/shuffled` 时间控制。SeqTrack 对照保留原伪时间，不接受这些时间消融。原计划的 reference/B0 seed52 复验留待后续，本轮四组不能代替双 seed 验收。四条后台命令与日志查看见 [v32 mini 运行说明](docs/CTSEQTRACK_V32_MINI_LAUNCH.md)；配置和评价见 [协议](docs/EXPERIMENT_PROTOCOL.md)、[工具面](docs/FORMAL_TOOLING.md)，环境和数据根见 [服务器路径](docs/SERVER_PATHS.md)。

## 验证与保护范围

v32最新本地检查 **249 passed、3 skipped**，包含四组模型在真实Lightning2.0.2下的合成数据训练、checkpoint重载、自动/独立评测与epoch恢复；compileall、diff检查通过。两个CUDA用例因CPU环境跳过，正式nuScenes训练尚未执行。完整说明见 [v32修复记录](docs/B0_V32_REPAIR.md)。

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

2026-09-22 精简后，生产 Python 从 163 个文件、57,233 行缩为 39 个文件、6,153 行；配置从 106 份缩为 4 份。保留测试 **152 passed、2 skipped**；清理前后数值与训练恢复共 **95,713 项记录一致**，三臂既有 epoch60 checkpoint 严格加载通过。检查使用本地 CPU 和合成数据，未重跑真实数据或 CUDA。完整范围、对照与历史证据见 [精简报告](artifacts/ct_checks/20260922_v31_slimming/REPORT.md) 和 [验证记录](docs/CTSEQTRACK_V31_READINESS.md)。

`output/` 和既有 `artifacts/` 是受保护实验结果、权重与诊断证据，不清理、不覆盖。其他兄弟项目为冻结参考，不修改。当前服务器授权仅只读；本地清理不包含上传、安装或启动/停止任务。更多约定见 [AGENTS.md](AGENTS.md) 与 [正式工具面](docs/FORMAL_TOOLING.md)。
