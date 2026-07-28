# CT-SeqTrack 下一步

更新时间：2026-07-28

真实时间投入产出、Random-20% 现实边界、近期方法借鉴和模块兼容性总审计见
[`docs/TIME_VALUE_AND_MODULE_ROADMAP_20260728.md`](docs/TIME_VALUE_AND_MODULE_ROADMAP_20260728.md)。

旧阶段的细碎任务保留在 `docs/legacy/need_to_do_20260724.md`。当前只维护
一个最小决策链，不再扩展实验树。

## 0. B0–B3 与 Search-only 已完成：当前新增模块均不晋级

| 组别 | final Success | final Precision | 关键比较 | 结论 |
|---|---:|---:|---|---|
| B0 | 53.360 | 64.382 | — | 当前唯一晋级基线 |
| A1 Search-only | 27.036 | 25.596 | −26.324 / −38.786 vs B0 | 当前独立搜索否决 |
| B1 | 26.021 | 24.972 | −27.339 / −39.410 vs B0 | 固定 0.75 motion correction 否决 |
| B2 | 47.973 | 52.088 | +21.952 / +27.116 vs B1 | search 有恢复作用，但仍低于 B0 |
| B3 | 25.537 | 24.707 | −22.435 / −27.381 vs B2 | adaptive gate 否决 |

五组均有 75,720 个训练 step、12 个验证点和 epoch60 `last.ckpt`。A1
late-3 为 27.933 / 26.400，仍比 B0 低 24.972 / 36.705；best 也只有
29.257 / 30.202，因此不是 checkpoint 选择问题。A1 的训练 search 使用率
与 B2 相同，末轮训练 loss 又接近 B0，说明当前问题集中在递归评测语义或
motion×search 交互，而不是 search 未启用或训练不足。

完整证据见 [`Search-only 技术复核`](compare_results/reports/ct_search_only_seed42_20260727.md)。

## 1. 下一步只做同 checkpoint Search 开/关 2×2

不再训练 A2，也不先调 75/25、tube 或 gate。使用现有 B0 与 A1 final
checkpoint，各自执行 baseline crop 和 search-on crop：

| checkpoint | baseline crop | search-on crop |
|---|---|---|
| B0 final | 已有 B0 | 待评测 |
| A1 final | 待评测 | 已有 A1 |

这四格不需要重新训练。评测时同时记录逐 endpoint 的 search 是否启用、
expansion-only 可用点数、实际扩展 token 数、预测 tube 位移和首次明显漂移帧。

- 两个 checkpoint 都只在 search-on 时下降：当前递归 search 路径是主因，
  删除当前实现或改为更严格的 fail-closed 搜索。
- A1 在 search-off 下仍明显低：训练期少量 expansion 暴露已改变模型，需要
  重新设计训练分布，而不是只改推理阈值。
- B0 search-on 不下降而 A1 两路都下降：优先检查跨 commit 初始化和优化路径。

本地已验证 A1/B0 checkpoint 均为 320 个同名同 shape tensor，且 resolved
config 的实质差异仅是 search 与其历史输入。服务器
`search_only_model_equivalence.log` 未拉回，所以初始化 exact-equality
preflight 只能记为“artifact 未审计”，不能补写为已通过。

## 2. 暂停项

在上述 2×2 完成且有新 search 设计通过 normal-mini 前，不运行：

- A2 conservative motion residual；
- seed43/44；
- `true / fixed / shuffled` 因果时间控制；
- full nuScenes；
- Random-20%；
- M3 非对称 endpoint path distillation、ChronoTrack point-feature
  consistency 或紧凑记忆。

若当前 search 被确认失败，先删除或只重构这一个模块，不用 motion、gate 或
memory 掩盖。正常集出现新晋级模型后，顺序仍是：同 checkpoint 时间控制 →
seed43/44 → full nuScenes → Random-20%。

## 3. 第二阶段候选

Search 归因完成后，第二阶段必须区分三种不同设计：

1. **现有 M3 非对称 endpoint path distillation**：canonical EMA teacher 对
   irregular-history student，只约束共同 endpoint；它不是 ChronoTrack 的
   temporal consistency。第一轮应移到纯 B0，保持 motion/search/gate/memory
   全关，并做 single / paired-weight0 / distill A/B/C。
2. **Chrono-lite point-feature consistency**：使用训练 GT box 将不同帧前景点
   变换到 canonical coordinates，匹配对应点并约束 latent feature。只有 M3
   或独立 feature-drift 诊断给出正信号才实现。
3. **紧凑前景记忆 + memory cycle consistency**：只有 point-feature
   consistency 通过后，再加入固定数量 recurrent foreground tokens；不能先用
   memory 掩盖失败的 motion/search。

三个模块不能同时首测；每个候选都必须相对进入该阶段的最佳模型独立涨点。
