# CT-SeqTrack

CT-SeqTrack 是基于 SeqTrack3D 的连续时间 3D 单目标跟踪项目。第一版 v2
候选曾从大量互相耦合的实验分支收敛为：

```text
SeqTrack3D
  + Continuous-Time Motion Prior
  + Time-Guided Search Expansion
  + Adaptive Proposal Fusion
```

2026-07-27 的完整 B0–B3 首筛以及后续 Search-only A1 已经否决这套三模块
组合及当前独立搜索设计。2026-07-30 完成的 motion `alpha=0/0.25` scratch
复核进一步确认：`alpha=0.25` 相对精确关闭 innovation 的 `alpha=0` final
下降 `17.468/20.322`，因此问题不只是旧 `alpha=0.75` 太大，当前固定全局
proposal innovation 正式 No-Go。

同日完成的有序/pre-crop B1motion-v2 也未修复 normal：epoch60 仅
`20.618 Success / 19.830 Precision`，相对 B0 下降 `32.742/44.551`；
最佳 epoch5 也只有 `30.196/34.990`。完整训练诊断表明，35% mixed-cadence
在 adapter 仍为零时就先破坏了 gap-blind B0 主路径，epoch3 后无范数硬上限
的 feature residual 又放大退化；pre-crop extension 实际只在 3.93% 训练样本
有效。当前 B1motion-v2 原样配置同样 No-Go。

2026-08-01 拉回的完整日志确认第四模块 Δt-PFTC 已跑满 60 epoch。它的 final
为 `51.189/60.886`，相对 B0 下降 `2.171/3.496`；late-3 也下降
`1.507/2.487`，因此当前 B4 明确没有涨点。canonical yaw 逆变换符号仍与项目
坐标约定相反，前景 feature std 到 epoch60 只剩 epoch1 的 16.4%，训练开销为
B0 的 8.24 倍。当前结论更新为
`NO-GO_CURRENT_B4_IMPLEMENTATION / PFTC_IDEA_NOT_YET_FAIRLY_TESTED`：停止原样
扩展，先修正几何、防坍缩目标与性能，再做 5-epoch 三臂机制测试。

## 已完成首筛的 v2 候选

1. **Continuous-Time Motion Prior**：从历史框和真实 `delta_t` 学习速度，按当前时间间隔生成候选帧位移先验；训练时以 clean/correlated history 混合替代纯 GT history。
2. **Time-Guided Search Expansion**：保留 SeqTrack3D 原搜索区域，额外构造有界轨迹 tube；总点数仍为 1024，其中默认 75% 来自原搜索区域。
3. **Adaptive Proposal Fusion**：根据观测特征、运动特征、proposal disagreement、点云可靠性、时间间隔和扩展比例，预测小幅有界修正。

TWC、旧 Observability Gate、M3 EMA teacher、M4 Kalman/filter 等路线保留为
历史代码，默认配置全部关闭。当前三模块候选没有晋级；第二阶段一致性和记忆
继续暂停。

详细数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，论文计划见
[refined_plan.md](refined_plan.md)。关于真实时间的投入产出、Random-20%
现实性、ChronoTrack/近期工作借鉴顺序和现有模块审计，见
[真实时间价值与模块路线审计](docs/TIME_VALUE_AND_MODULE_ROADMAP_20260728.md)。
服务器上的 KITTI、nuScenes-mini 和 nuScenes Python 包路径及其正确用法见
[服务器路径说明](docs/SERVER_PATHS.md)。

第四模块的完整数据、实现错误、坍缩诊断和恢复路径见
[Δt-PFTC seed42 60-epoch 最终诊断](compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md)。

Motion alpha 的完整性、曲线、机制诊断和后续 2×2 见
[Motion fixed-alpha 复核](compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md)；
可独立打开的图表版见
[便携 HTML 报告](compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.html)。
对 B1motion 的逐层代码链路、训练/推理闭环、时序表达能力、历史 M2 归因和
SeqTrack3D/M²-Track/STTracker/HVTrack/TrajTrack 对照，见
[B1motion 深度审计](compare_results/reports/b1_motion_module_deep_audit_20260730.html)。
有序/pre-crop 修正版的完整 60-epoch 曲线、训练损失、代码合同漏洞与下一步
factorial kill-test 见
[B1motion-v2 seed42 结果](compare_results/reports/b1motion_v2_seed42_20260730.md)。

## B1motion-v2 完成结果（2026-07-30）

| 组别 | final Success | final Precision | best Success | best Precision |
|---|---:|---:|---:|---:|
| B0 baseline | **53.360** | **64.382** | **54.135** | **64.382** |
| legacy motion α=0 | 47.049 | 49.184 | 49.876 | 58.691 |
| legacy motion α=0.25 | 29.581 | 28.862 | 35.027 | 41.130 |
| B1motion-v2 ordered/pre-crop | 20.618 | 19.830 | 30.196 | 34.990 |

本轮 B1motion-v2 训练完整，不是坏 checkpoint。当前没有该 checkpoint 的
random20/gap1124 输出，因此不能声称 irregular 涨点；normal 大幅失败也已
阻止晋级。下一步先重训 current-code B0，再做
`irregular_probability 0/0.35 × adapter off/on` 的 10–15 epoch 归因实验，
不再直接跑另一个 60 epoch。

## Motion fixed-alpha 复核（2026-07-30）

| 组别 | final Success | final Precision | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|
| B0 baseline | **53.360** | **64.382** | **52.905** | **63.104** |
| B1 motion，alpha=0 | 47.049 | 49.184 | 46.828 | 49.669 |
| B1 motion，alpha=0.25 | 29.581 | 28.862 | 29.472 | 28.849 |
| B1 motion，alpha=0.75 | 26.021 | 24.972 | 26.080 | 25.299 |

`alpha=0/0.25` 均为 commit `5f260e7`、seed42、scratch、60 epoch、
75,720 step，resolved config 除 cfg/tag 外只差 alpha。`alpha=0.25`
warmup 后实际平均系数仅 0.184、平均修正约 0.083 m，仍相对 `alpha=0`
大幅退化；epoch25–60 的 8 个验证点两项指标全部更低。与此同时它的 epoch60
training loss 更低，说明主要矛盾是 teacher-forced 训练与 recursive tracking
错位，而不是训练不足。

当前只否定固定全局 proposal innovation，不把结论扩大为所有 motion prior
无效。下一步不再长训更小 alpha；先用两个已有 checkpoint 做推理 alpha
开/关 2×2，并导出逐 endpoint observation/dynamics proposal 归因。

## 第四模块最终状态（2026-08-01）

| 项目 | 结果 | 判断 |
|---|---:|---|
| 训练完整性 | 75,720 step / 12 验证点 / epoch60 checkpoint | 完整 |
| epoch60 final | 51.189 Success / 60.886 Precision | 比 B0 低 2.171 / 3.496 |
| late-3 | 51.398 / 60.618 | 比 B0 低 1.507 / 2.487 |
| 前景 feature std | 0.0947 → 0.0156 | 只剩 16.4%，强坍缩警报 |
| weighted/raw loss 差异中位数 | -0.252% | 没有真实时间增量证据 |
| 单卡 step time | 2.983 s vs B0 0.362 s | 约慢 8.24 倍 |

当前旧公式正式 No-Go，不再补 PFTC-U、seed43/44 或强协议。下一步依次是：把
canonicalization 改为项目一致的 `R(-yaw)` 并修正交叉单测；加入
projector/normalized loss/variance floor 等明确防坍缩机制；把 step time 压到
B0 的 2 倍以内；最后才重新运行同代码 B0/PFTC-U-v2/Δt-PFTC-v2 的 5-epoch
机制筛选。只有机制门槛全部通过，才重新预检 λ 和考虑 60-epoch 正式三臂。

## 当前实验结论（2026-07-27）

seed42、nuScenes-mini、Car、60 epoch 的原始 TensorBoard 标量复核结果如下。
主比较统一采用 epoch60 final checkpoint；best 与 late-3 只作为稳定性诊断。

| 组别 | 模块 | final Success | final Precision | 状态 |
|---|---|---:|---:|---|
| B0 | SeqTrack3D baseline | 53.360 | 64.382 | 完整 |
| A1 | B0 + search expansion only | 27.036 | 25.596 | 完整，当前独立搜索不通过 |
| B1 | B0 + motion prior，alpha=0.75 | 26.021 | 24.972 | 完整，当前设计不通过 |
| B2 | B1 + search expansion | 47.973 | 52.088 | 完整，交互恢复但仍低于 B0 |
| B3 | B2 + adaptive fusion | 25.537 | 24.707 | 完整，adaptive gate 不通过 |

Search-only A1 相对 B0 下降 26.324 Success / 38.786 Precision，late-3
也下降 24.972 / 36.705；12 个验证点均未接近 B0，不是 final checkpoint
选择问题。A1 的末轮训练损失仅比 B0 高约 0.0013，训练 search-used sample
ratio 为 3.460%，与 B2 的 3.458% 几乎一致，因此失败不能解释为 search
未执行或常规训练发散。B2 相对 B1 的恢复应解释为 motion×search 交互，
不能再当作 search 的独立正贡献。

当前只保留 B0。下一步不训练 A2，而是用现有 B0/A1 checkpoint 做 Search
开/关 2×2 评测，并补验证阶段逐 endpoint 的搜索激活、扩展点数和首次漂移
诊断。在原因定位前，不启动多 seed、时间控制、full nuScenes、Random-20%
或第二阶段模块。

完整的可复核结果、完整性检查和机制诊断见
[B0–B3 seed42 消融复核](compare_results/reports/ct_v2_ablation_seed42_20260727.md)。
可用 `python tools/analyze_ct_v2_ablation.py` 从本地 event/checkpoint 重新生成
报告、CSV 和曲线图。

Search-only 的独立复核见
[A1 Search-only 技术复核](compare_results/reports/ct_search_only_seed42_20260727.md)，
可用 `python tools/analyze_ct_search_only.py` 重新生成报告、CSV 和曲线图。

## B0–B3 复现实验入口

下面入口用于复现已经完成的首筛，不代表建议重跑相同配置：

```bash
# 1. 同代码 baseline
python tools/ct_v2/run.py train --variant baseline --path /data/nuscenes-mini

# 2. 连续时间运动先验
python tools/ct_v2/run.py train --variant motion --path /data/nuscenes-mini

# 3. 加时间引导搜索扩展
python tools/ct_v2/run.py train --variant motion_search --path /data/nuscenes-mini

# 4. 完整 CT-SeqTrack v2
python tools/ct_v2/run.py train --variant full --path /data/nuscenes-mini

# 5. Search-only（已完成且未通过；仅用于复现）
python tools/ct_v2/run.py train --variant search_only --path /data/nuscenes-mini
```

正常验证：

```bash
python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini
```

当前 B3 没有通过 mini，所以下列 full nuScenes 命令暂不执行；仅保留为未来
晋级模型的运行模板：

```bash
python tools/ct_v2/run.py train \
  --variant baseline_full \
  --path /data/nuscenes

python tools/ct_v2/run.py train \
  --variant full_dataset \
  --path /data/nuscenes
```

时间负对照必须使用同一晋级 checkpoint。当前 B3 未晋级，以下命令只保留为
实验合同；`fixed` 可直接运行，`shuffled` 先生成冻结 manifest：

> 2026-07-25 以前的 B3 时间控制会通过 observation statistics
> 继续读取真实 `dt`，不能作为 v2 因果证据。修复后的 gate、motion、
> innovation radius 与 search tube 均只消费 `current_delta_t_effective`，
> 因此必须用新代码重新运行三路控制。

```bash
python tools/build_dynamics_time_manifest.py \
  --cfg cfgs/ct_v2/04_ct_seqtrack_v2.yaml \
  --path /data/nuscenes-mini \
  --role test \
  --output protocols/manifests/ct_v2_mini_test_shuffled_seed42.json

python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini \
  --time-mode shuffled \
  --time-manifest protocols/manifests/ct_v2_mini_test_shuffled_seed42.json
```

Random-20% 同样暂不执行；晋级后可按下面模板复测：

```bash
python tools/ct_v2/run.py test \
  --variant full \
  --checkpoint /path/to/last.ckpt \
  --path /data/nuscenes-mini \
  --protocol random20
```

命令可加 `--dry-run` 只打印最终 `main.py` 调用。原有 `main.py --cfg cfgs/<legacy>.yaml ...` 命令保持可用。

## 目录

```text
models/ct_v2/       连续时间运动与自适应融合
utils/ct_search.py  训练/评测共享的时间引导搜索
utils/ct_history.py 轻量的相关历史误差契约
cfgs/ct_v2/         当前唯一活跃的消融配置
tools/ct_v2/run.py  当前唯一推荐运行入口
docs/legacy/        旧阶段计划和运行说明
compare_results/    历史结果与正式分析
```

## 现有证据边界

- B0–B3 与 Search-only A1 均已完成 75,720 个训练 step、12 次验证和
  epoch60 `last.ckpt`；本轮数据足够否决当前设计。
- 当前固定 `alpha=0.75` 的 B1 明显低于 B0；B2 对 B1 有较大恢复，但 final
  Success/Precision 仍分别低 5.387/12.294，不能晋级。
- 新 scratch 对照中 `alpha=0.25` 相对 `alpha=0` final 仍下降
  17.468/20.322；固定全局 innovation 的 No-Go 不再只是“0.75 过大”。
- `alpha=0` 是精确关闭 correction 的 fallback control，不是 motion 正向
  贡献；它与 B0 还存在共享初始化混杂，不能把两者差值解释为 dynamics 净效应。
- B3 gate 从初始 0.25 快速饱和为常数上限 0.75，最终结果基本退回 B1；
  当前 adaptive fusion 不具有可用的条件可靠性。
- A1 相对 B0 final 下降 26.324/38.786，证明当前 search 不能独立涨点；
  B2 的正增量只能表述为对失败 B1 的交互恢复。
- A1/B0 checkpoint 的 320 个 state tensor 名称和 shape 完全一致；但服务器
  初始化等价 preflight 日志未随结果拉回，因此不能声称该 artifact 已审计。
- 可选模块在共享层之前实例化会改变后续层的随机初始化；单 seed 的精确模块
  效应仍有初始化混杂。当前大幅退化足以做 No-Go，机制归因仍需同 checkpoint
  Search 开/关 2×2。
- 正确时间尚未与 fixed/shuffled 做同 checkpoint 的有效比较，因此不能声称
  “真实时间已被证明产生因果收益”。
- B1–B3 使用 candidate0 clean、其余 candidate correlated 的历史混合；
  canonical displacement/velocity 标签不随该输入扰动改变。

历史细节见 [sum_results.md](sum_results.md) 和 [done.md](done.md)。
## 相关工作

- [SeqTrack3D, ICRA 2024](https://arxiv.org/abs/2402.16249)：多帧点云与历史框序列 baseline。
- [StreamTrack, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28196)：流式多帧记忆。
- [HVTrack, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php)：历史视角跟踪与扩展搜索的收益/背景噪声边界。
- [TrajTrack, 2025/2026](https://arxiv.org/abs/2509.11453)：显式运动 proposal 与隐式历史轨迹联合细化。
- [ChronoTrack, CVPR 2026 Findings](https://arxiv.org/abs/2604.13789)：对齐后的紧凑时序记忆与一致性目标。
