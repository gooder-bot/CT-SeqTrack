# CT-SeqTrack

2026-09-29本次piecewise调度与恢复增量已完成本地验证：**554 passed、3 skipped**，25份旧配置文件及身份SHA不变。由用户提交GitHub、服务器拉取后按[最新五组命令](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)恢复原四组并启动新组；[验证报告](artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/REPORT.md)。

CT-SeqTrack 是面向 3D 点云单目标跟踪的研究项目。当前新增 **v35 B0 综合优化**：在 v34 W=1/4/4/8 的基础上，对齐真实首测初始化、为 coarse 提供显式合法历史条件、为 fine 提供候选局部真实点证据。活动代码仍在 `models/ct_v31/`；v33/v34/reference 原行为与权重身份继续兼容。

**2026-09-29追加第5组piecewise，覆盖此前四组总数。** 原四组LR=2.5e-5/5e-5/1e-4/1.5e-4已完成36轮；13:33只读复查确认用户已暂停四组，各自完整checkpoint均为`epoch=036.ckpt`。共同最新validation35，尚无正式final60/late-3。本次仅新增一组：第1–20轮5e-5、第21–50轮1e-5、第51–60轮5e-6；仍seed42、scratch60、W=1/4/4/8、无warmup，网络、loss、数据及其他单组参数不变。五组合计300epoch、358,500更新；仅达标后补胜出完整配方seed52与独立SeqTrack seed52，条件上限七次、420epoch、501,900更新，不自动启动。

保留固定R/C门槛和旧W-quarter风险对照。用户暂停已完成，接下来更新同一活动项目，从各自原036 checkpoint恢复四组、第37轮继续，GPU仍0/0/1/1；然后在GPU1从头启动piecewise并使用新输出目录。原四组恢复保持各自配置与运行目录，保留最初manifest，另记`resume_manifests/`；仅接受固定before/after源码关系，旧训练日志追加写入，不覆盖旧output/artifacts。新组不读取旧权重初始化。新增配置、调度解释及精确操作顺序见 [piecewise操作说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)，结构见 [v35说明](docs/B0_V35_INTEGRATED.md)。[原四组操作页](docs/CTSEQTRACK_V35_MINI_LAUNCH.md)继续用于查询已有运行；不要重新执行四组scratch命令。此次只改调度，不重复要求CUDA batch检查，代理服务器权限仍为只读。

**截至2026-09-28，v34八组已全部完成。** W-quarter final60 S/P=51.4650/62.8435，late-3=51.2527/62.6765；Precision高于R，Success及移动守底未全通过，不能晋级Full。[八组正式复盘](artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)。v35阶段状态见[9月29日只读快照](artifacts/ct_checks/20260929-130405_v35_four_run_status/server_snapshot.json)，同轮次历史对照见[逐项证据](artifacts/ct_checks/20260929-130405_v35_four_run_status/historical_comparison.json)；中途validation不替代正式验收。下方v33数字和启动文字为历史记录，不覆盖本段当前状态。

**本轮明确优化严重漂移与持续失跟。** 方案1只表示保留原总体/移动硬门、单列漂移风险，不新增漂移数值硬门，不表示仅记录或推迟修复。综合改法及其成因对应见[v35结构说明](docs/B0_V35_INTEGRATED.md)；>10m计数增加时暂停Full待复核，小幅孤立波动不自动否决方案，明显持续退化必须处理。原硬门结果与风险结论分开报告，统一口径见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。

| 模块 | 作用 |
|---|---|
| B0 | 从当前与历史真实点提取观测，生成 coarse query 并通过共享 decoder 定位 |
| B1 | 使用 CfC/GRU 建模物理时间运动先验与 context |
| B2 | 在 B0 原始裁剪之外获取 768 个点槽，选取 256 个证据点槽，结合原始身份记忆 |
| B3 | 让 q0 与三个测量模式共享定位和质量头，选择最终输出 |

v33 整合中心回归、BC 与历史监督归一化、可信状态修复及被动评测诊断。R/A/B/C/D/E 六次 mini 运行已完成并拉回本地；Full 接口保留，本轮不新增 Full 独立诊断。具体改动见 [v33 实现说明](docs/B0_V33_REPAIR.md)。

## v33 历史结果（2026-09-26）

六次运行是 1 组 SeqTrack + 5 组 B0，均 seed42、scratch60、71,700 次更新；已核验全部 58–60 轮正式逐帧结果。

| 模型 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| R：SeqTrack | 51.8239 / 61.3414 | 51.8844 / 61.9664 |
| A：原配方 | 49.1214 / 57.6499 | 47.4271 / 54.5514 |
| B：延后降档 | 47.5525 / 55.9179 | 46.6116 / 53.9143 |
| C：半学习率 | **50.4562 / 59.2560** | **49.3738 / 58.0744** |
| D：1.5 倍 LR | 49.3993 / 57.6214 | 48.1601 / 55.8435 |
| E：3 倍 LR + warmup | 44.5766 / 53.8600 | 44.0835 / 53.2586 |

**C 是已测最佳 B0，仍未达到 SeqTrack。** 后续采用 C 的 5e-5、20/50 轮衰减、无 warmup；不继续扩大 LR。C 相对旧 v32 B0 的 Success 略升、Precision 下降，不能登记综合修订全面有效。主要剩余问题是缺测时历史利用、fine 无稳定定位增益，以及完整预测历史的训练覆盖；具体建议尚未实施。完整结果、曲线与代码依据见 [六组复盘](artifacts/ct_checks/20260926_v33_six_recipe_review/REPORT.md)。

## 已完成运行的配置

以下为已完成运行的配置与物理 GPU 安排，保留用于复现：

| 组别 | 配置 | 物理 GPU | 学习率配方 |
|---|---|---:|---|
| R | `33_seqtrack_ref_mini.yaml` | 0 | 原 SeqTrack，Adam 1e-4，StepLR 每 20 轮乘 0.1 |
| A | `33_b0_mini.yaml` | 0 | 综合 B0，Adam 1e-4，同 R 的衰减 |
| B | `33_b0_late_decay_mini.yaml` | 1 | 综合 B0，Adam 1e-4，在 20/50 轮后乘 0.1 |
| C | `33_b0_half_lr_mini.yaml` | 1 | 综合 B0，Adam 5e-5，同 B 的衰减 |

四组均 scratch60、batch16、workers4、FP32，每组单卡。A/B/C 的模型、loss、数据、seed 和训练预算相同。R 使用独立原 SeqTrack 网络、teacher 数据处理与原 loss，不继承生产 B0 的修订。

本轮服务器只读；由用户上传和启动，不由代理在服务器写入或运行训练。已有 v32 SeqTrack 成绩可复用，但用户当前选择重新训练 R。四组独立后台命令、日志与恢复方式见 [运行说明](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)。

2026-09-25追加D：B配方学习率全程×1.5，初始1.5e-4，GPU0；原A/B/C保留。新配置、同步文件和后台命令见[放大学习率运行说明](docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md)。

2026-09-25追加E：继承B，峰值学习率3e-4，前2000次更新从1.5e-7线性升至峰值，20/50轮完成后降至3e-5/3e-6，GPU1，warmup计入原71,700次更新。原R/A/B/C/D保持；E相对B同时改变学习率与warmup，不能独立归因。五个同步文件、独立后台命令和新终端tail见[追加E运行说明](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。本轮未访问或修改服务器，E由用户上传启动。

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

唯一入口为`main.py`；本次唯一新增scratch配方的入口示例：

```bash
python main.py --cfg cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini.yaml --path DATA_ROOT
```

原四组须使用各自checkpoint和原log_dir恢复，仅piecewise从头运行，完整命令见[新增第5组说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)。五组完整后用`tools/compare_v35_piecewise_b0.py`比较，保留旧四组工具用于原四组复核。第二seed须等第一阶段通过后再运行，seed42独立SeqTrack参考直接复用。默认每5轮验证，训练结束自动评测58/59/60并保存逐帧记录及`results.json`。比较固定final60和late-3，不挑选最佳轮次。不要传旧`--preloading`参数；当前按需读取原始点云，每worker缓存256MiB。

## 验证与边界

v35原四组版本的本地实现、旧版本兼容、诊断无干预、真实Lightning入口和epoch恢复已验证；该版本完整pytest为515 passed、3 skipped，compileall与diff检查通过，详见[9月28日就绪报告](artifacts/ct_checks/20260928-185917_v35_four_run_readiness/REPORT.md)。两项跳过需要真实CUDA，另一项只测试缺失Lightning的环境；本地已安装Lightning时不适用。新增参数36,608（约0.99%）。第5组仅新增调度，实施与验证证据单列在`artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/`，不把旧测试结果冒充本次验证。此前三组实现过程保留在[历史实施报告](artifacts/ct_checks/20260928-182545_v35_implementation/REPORT.md)。

此前服务器快照检查 **314 passed、1 skipped**，真实 CUDA batch 的 forward/backward/Adam/commit 已通过，未保存工程 checkpoint。日志与报告位于 [v33 检查目录](artifacts/ct_checks/20260924-190116_v33_implementation/)。后续被动汇总增量单独记录本地验证；这些历史检查不替代正式结果验收。六组完成状态以 2026-09-26 复盘为准，无需把同一工程检查反复作为启动步骤。

本地修改后的验证命令：

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
```

nuScenes full、KITTI、CfC/GRU 和时间控制接口保留；接口存在不等于已经完成相应实验。`output/` 与既有 `artifacts/` 受保护，不删除、不覆盖。详细约定见 [AGENTS.md](AGENTS.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)、[工具范围](docs/FORMAL_TOOLING.md) 和 [服务器路径](docs/SERVER_PATHS.md)。
