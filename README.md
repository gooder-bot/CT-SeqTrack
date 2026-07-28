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
组合及当前独立搜索设计。当前研究主线进入机制诊断：先用现有 B0/A1
checkpoint 做 Search 开/关 2×2 评测，再决定删除还是重构搜索；不继续叠加
motion、gate 或记忆模块。目标仍是在正常 nuScenes 上稳定涨点，再用同
checkpoint 的 `true / fixed / shuffled` 控制检验真实时间。Random-20% 只作为
最终鲁棒性补充，不参与选模。

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

## 当前实验结论（2026-07-27）

seed42、nuScenes-mini、Car、60 epoch 的原始 TensorBoard 标量复核结果如下。
主比较统一采用 epoch60 final checkpoint；best 与 late-3 只作为稳定性诊断。

| 组别 | 模块 | final Success | final Precision | 状态 |
|---|---|---:|---:|---|
| B0 | SeqTrack3D baseline | 53.360 | 64.382 | 完整 |
| A1 | B0 + search expansion only | 27.036 | 25.596 | 完整，当前独立搜索不通过 |
| B1 | B0 + motion prior | 26.021 | 24.972 | 完整，当前设计不通过 |
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
