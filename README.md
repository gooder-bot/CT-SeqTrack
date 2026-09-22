# CT-SeqTrack

CT-SeqTrack 研究真实时间先验条件下的目标证据获取、身份保持与递归定位，任务为 3D 点云单目标跟踪。当前工作树只维护 **v31 联合模型**；旧版本源码、配置、工具与文档通过 [历史索引](docs/HISTORY_EVIDENCE_INDEX.md) 中的 Git 对象复现。

## 当前方法

| 模块 | 职责 |
|---|---|
| B0 | 三帧历史与当前帧的唯一真实点观测，分割、点框特征和粗定位 |
| B1 | CfC 或 GRU 物理时间先验、获取范围与可微时序 context |
| B2 | B0 裁剪外新增点的 768→256 固定预算获取，原始身份记忆与三个投票模式 |
| B3 | 主观测与三个模式共用定位/质量头，直接选择四个框假设之一 |

每个训练 batch 执行一次联合前向和一次 Adam 更新；窗口内各端点仅提交一次 accepted 状态并继续递推，跨帧 detach。sigma、离散几何、标签与跨帧状态的梯度边界保留。Full 不依赖旧版策略标定，也不要求不同臂的 B0 参数逐位相同。

实现入口为 [main.py](main.py) → [v31 entry](models/ct_v31/entry.py) → [Lightning host](models/ctseqtrackv31.py) 与 [联合模型](models/ct_v31/model.py)。数据准备、先验、获取、证据、记忆、解码、损失与评测位于 `models/ct_v31/`。当前网络使用 PyTorch 算子，不要求安装旧 PointNet++ CUDA 扩展。

## 已完成的实验

2026-09-21 已核实代码版本 `64ad056` 的 mini Car/seed42 三臂完成 scratch60 与独立 58–60 评测；各 71,911 次 Adam、1,146,480 次端点曝光，官方 mini_val 为 106 轨迹/2,285 帧。

| v31 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| B0 | 23.959519 / 27.834792 | 23.812546 / 28.955142 |
| Full-CfC | 22.491247 / 29.352297 | 24.213348 / 33.709701 |
| Full-GRU | 22.839169 / 31.844639 | 25.303793 / 37.146243 |

工程训练与评测已跑通，**性能验收未通过**：两个 Full 的 final Success 均低于同版 B0，B0 也低于 v30 历史参照 40.473742/47.840262。不能用 Precision 单升、late-3 或中途最好轮次替代预定 final60 S/P 双升条件。

[完整结果与真实数据探针](artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md) 定位了当前帧近全前景、稀疏 token 压缩、尾批 BN、候选重复和记忆错写等问题。下一步先修新版 B0/Full-GRU，CfC 保留同接口时间机制对照；具体证据和未实施修订见 [最新问题](最新问题.md) 与 [待办](need_to_do.md)。当前不声称稳定涨分、SOTA、连续时间或记忆的因果收益。

## 运行

从本仓库根目录执行：

```bash
python main.py --cfg cfgs/ct_seqtrack/31_b0_mini.yaml --path DATA_ROOT --tag exp_name
python main.py --cfg cfgs/ct_seqtrack/31_full_gru_mini.yaml --checkpoint RUN/formal_checkpoints/epoch=060.ckpt --test --log_dir NEW_EVAL_DIR
```

现有三份 mini 配置共用独立 `31_formal_base.yaml`，正式预算为 60 轮、batch16、workers4、seed42、FP32、每 5 轮验证。训练结束自动评测 58/59/60，报告 final60 和 late-3。不使用旧 `--preloading`；原始云缓存每 worker 256 MiB。

配置接口还保留 `b0/b1/b1_b2/full`、CfC/GRU、nuScenes mini/full、KITTI 和 `true/fixed/shuffled` 时间控制。现有正式实验仅覆盖 mini Car 三臂；保留能力不代表 full/KITTI 或时间消融已完成。shuffled 必须传入匹配 manifest。配置定义见 [协议](docs/EXPERIMENT_PROTOCOL.md)，后台启动命令见 [mini 运行说明](docs/CTSEQTRACK_V31_MINI_LAUNCH.md)，环境和数据根见 [服务器路径](docs/SERVER_PATHS.md)。

## 验证与保护范围

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

2026-09-22 精简后，生产 Python 从 163 个文件、57,233 行缩为 39 个文件、6,153 行；配置从 106 份缩为 4 份。保留测试 **152 passed、2 skipped**；清理前后数值与训练恢复共 **95,713 项记录一致**，三臂既有 epoch60 checkpoint 严格加载通过。检查使用本地 CPU 和合成数据，未重跑真实数据或 CUDA。完整范围、对照与历史证据见 [精简报告](artifacts/ct_checks/20260922_v31_slimming/REPORT.md) 和 [验证记录](docs/CTSEQTRACK_V31_READINESS.md)。

`output/` 和既有 `artifacts/` 是受保护实验结果、权重与诊断证据，不清理、不覆盖。其他兄弟项目为冻结参考，不修改。当前服务器授权仅只读；本地清理不包含上传、安装或启动/停止任务。更多约定见 [AGENTS.md](AGENTS.md) 与 [正式工具面](docs/FORMAL_TOOLING.md)。
