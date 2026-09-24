# CT-SeqTrack

CT-SeqTrack 是面向 3D 点云单目标跟踪的研究项目。当前活动版本为 **v33 综合 B0 修订**，代码保留 `models/ct_v31/` 的物理路径；v32 可通过 Git `ddcb1a1` 复现。

| 模块 | 作用 |
|---|---|
| B0 | 从当前与历史真实点提取观测，生成 coarse query 并通过共享 decoder 定位 |
| B1 | 使用 CfC/GRU 建模物理时间运动先验与 context |
| B2 | 在 B0 原始裁剪之外获取 768 个点槽，选取 256 个证据点槽，结合原始身份记忆 |
| B3 | 让 q0 与三个测量模式共享定位和质量头，选择最终输出 |

v33 整合中心回归、BC 与历史监督归一化、可信状态修复及被动评测诊断。当前先比较三种 B0 学习率配方与独立 SeqTrack；Full 接口保留，本轮不新增 Full 独立诊断。具体改动见 [v33 实现说明](docs/B0_V33_REPAIR.md)。

## 当前运行安排

用户将在上传本地修订后，自行从头启动四组 nuScenes-mini Car、seed42 实验：

| 组别 | 配置 | 物理 GPU | 学习率配方 |
|---|---|---:|---|
| R | `33_seqtrack_ref_mini.yaml` | 0 | 原 SeqTrack，Adam 1e-4，StepLR 每 20 轮乘 0.1 |
| A | `33_b0_mini.yaml` | 0 | 综合 B0，Adam 1e-4，同 R 的衰减 |
| B | `33_b0_late_decay_mini.yaml` | 1 | 综合 B0，Adam 1e-4，在 20/50 轮后乘 0.1 |
| C | `33_b0_half_lr_mini.yaml` | 1 | 综合 B0，Adam 5e-5，同 B 的衰减 |

四组均 scratch60、batch16、workers4、FP32，每组单卡。A/B/C 的模型、loss、数据、seed 和训练预算相同。R 使用独立原 SeqTrack 网络、teacher 数据处理与原 loss，不继承生产 B0 的修订。

本轮服务器只读；由用户上传和启动，不由代理在服务器写入或运行训练。已有 v32 SeqTrack 成绩可复用，但用户当前选择重新训练 R。四组独立后台命令、日志与恢复方式见 [运行说明](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)。

2026-09-25追加D：B配方学习率全程×1.5，初始1.5e-4，GPU0；原A/B/C保留。新配置、同步文件和后台命令见[放大学习率运行说明](docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md)。

## 已完成的历史结果

v32 nuScenes-mini Car、seed42，全部 scratch60、71,700 次优化；评测 106 条轨迹、2,285 帧：

| v32 模型 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| SeqTrack reference | 51.823851 / 61.341356 | 51.884391 / 61.966448 |
| B0 | 49.650985 / 59.803063 | 49.175420 / 59.365062 |
| Full-GRU | 44.699125 / 55.191466 | 45.132385 / 55.249088 |
| Full-CfC | 43.659738 / 53.991247 | 43.488330 / 53.970824 |

证据见 [四组报告](artifacts/ct_checks/20260924_v32_four_arm_final/REPORT.md) 和 [训练审查](artifacts/ct_checks/20260924_v32_four_arm_final/training/TRAINING_REVIEW.md)。这些成绩不属于 v33，不能用于宣称本次修改已经有效。

## 入口与评测

唯一入口为 `main.py`：

```bash
python main.py --cfg cfgs/ct_seqtrack/33_seqtrack_ref_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_late_decay_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml --path DATA_ROOT
```

以上为入口示例；物理卡绑定与后台运行使用运行说明中的独立命令。默认每 5 轮验证，训练结束自动评测 58/59/60 并保存逐帧记录及 `results.json`。比较固定 final60 和 late-3，不挑选最佳轮次。不要传旧 `--preloading` 参数；当前按需读取原始点云，每 worker 缓存 256MiB。

## 验证与边界

此前服务器快照检查 **314 passed、1 skipped**，真实 CUDA batch 的 forward/backward/Adam/commit 已通过，未保存工程 checkpoint。日志与报告位于 [v33 检查目录](artifacts/ct_checks/20260924-190116_v33_implementation/)。后续被动汇总增量单独记录本地验证；这些历史记录不代表最新源码已逐文件完成服务器复验，也不代表四组正式训练已经完成。无需把同一检查反复作为启动步骤。

本地修改后的验证命令：

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
```

nuScenes full、KITTI、CfC/GRU 和时间控制接口保留；接口存在不等于已经完成相应实验。`output/` 与既有 `artifacts/` 受保护，不删除、不覆盖。详细约定见 [AGENTS.md](AGENTS.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)、[工具范围](docs/FORMAL_TOOLING.md) 和 [服务器路径](docs/SERVER_PATHS.md)。
