# CT-SeqTrack 下一步

更新时间：2026-08-02

当前只维护最小决策链。最新 B2 证据见
[`B2-v2 seed42 技术复核`](compare_results/reports/b2_search_v2_seed42_20260802.html)，
第四模块的完整证据见
[`Δt-PFTC seed42 60-epoch 最终诊断`](compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md)，
最新 motion 证据见
[`B1motion-v3 seed42 60-epoch 技术复核`](compare_results/reports/b1motion_v3_seed42_20260801.html)，
整体时间路线见
[`真实时间价值与模块路线审计`](docs/TIME_VALUE_AND_MODULE_ROADMAP_20260728.md)。

## 0. 当前决定

```text
NO-GO_CURRENT_B4_IMPLEMENTATION
PFTC_IDEA_NOT_YET_FAIRLY_TESTED
V3_PHYSICAL_PRIOR_LEARNS_BUT_GATE_TRANSFER_FAILS
B2V2_NORMAL_SIGNAL_POSITIVE
B2V2_SEARCH_CONTRIBUTION_NOT_ESTABLISHED
HOLD_B2V2_PROMOTION
NO_EVIDENCE_FOR_PHYSICAL_TIME
```

## 0A. B2-v2：先完成归因与 matched B0

B2-v2 full final 为 `54.132/64.755`，相对历史 B0 为
`+0.772/+0.373`；late-3 为 `+1.557/+2.909`。epoch60 Success 通过
`+0.5`，Precision 未达到 `+1.0`，且新 SeqTrack control 异常低至
`31.684/31.337`，不能作为唯一 baseline。当前状态是“正信号但不晋级”。

旧 search-only 是 long-tube + 75/25 token legacy 路径，final 比历史 B0
低 `3.705/7.990`，不是新版 Search Evidence-only 消融。full 内部 search
candidate 有效率只有 `23.29%`，epoch60 argmax 选择率只有 `0.104%`；因此
当前正信号不能归因给 search。

- [ ] 给 `tools/ct_v2/run.py test` 增加/核验四个同 checkpoint 模式：full、
  observation-only、motion-only、search-only；不得改变候选值与点采样。
- [ ] 用 B2-v2 epoch60 `last.ckpt` 在完全相同 mini_val endpoint 上跑四模式。
- [ ] 导出 obs/motion/search/final/GT、candidate validity、confidence、targetness
  mass/entropy、三类 gate probability、oracle 类、previous prediction error 和
  tracklet id。
- [ ] 修正训练/验证诊断：`joint_search_error` 只在 valid search 行聚合，并记录
  oracle search prevalence、valid-search helpful rate、selected-search helpful
  precision 和 correction norm。
- [ ] 对四模式做 tracklet-level paired bootstrap；best checkpoint 只诊断，主判定
  仍使用 epoch60。
- [ ] 在 commit `a486a36`、seed42、batch16、workers4、scratch 60 epoch、相同
  数据 hash 下补 `01_seqtrack3d_baseline.yaml` matched B0。
- [ ] 仅当 full 相对 matched B0 的 epoch60 同时达到 `+0.5 Success / +1.0
  Precision` 且 late-3 不退化，才运行 seed43/44。

若同 checkpoint 证明 search 无净贡献，B2-v2.1 只允许以下定向修改：保留紧凑
endpoint crop 与独立 128 点；允许 endpoint crop 全部点进入 source-aware encoder
并增加 baseline-overlap flag，解决 extension-only 的 77% invalid；将 availability
与 utility 分开监督，让 reliability 拟合 search 相对 observation 的 advantage，
移除未校准的 `log(confidence)` 二次惩罚。禁止退回 long tube、压缩 B0 1024 点或
用固定平均融合。

完成 attribution 和 matched B0 前，暂停 seed43/44、gap1124、random20、
burst-drop、true/fixed/shuffled 与论文 search 消融长训。

## 0B. PFTC：先修正当前实现

这次 `dt_pftc_true_5f260e7_seed42_60ep_bs16_gpu0` 已经完成：

- 75,720 step、12 个验证点和 epoch60 `last.ckpt` 完整；
- final 为 51.189 Success / 60.886 Precision，相对 B0 下降 2.171 / 3.496；
- late-3 相对 B0 下降 1.507 / 2.487；
- canonical yaw 的逆变换符号与项目约定相反；
- feature std 从 epoch1 的 0.0947 降到 epoch60 的 0.0156；
- weighted/raw PFTC loss 差异中位数只有 -0.252%；
- 单卡训练约慢于 B0 8.24 倍。

因此当前 B4 必须登记为“不涨点 / 当前实现 No-Go”。但由于它训练的是错误
canonical geometry，且 raw SmoothL1 出现明显表示收缩，不能把结论扩大为
point consistency 思路本身无效；修正后的版本只允许一次受控 kill-test。

## 0C. B1motion-v3：先做同 checkpoint 归因

v3 的 60-epoch final 为 `52.655/61.835`，相对 current B0 为
`−0.705/−2.547`；late-3 为 `−0.855/−1.898`，当前不涨点。但 v3 相对
v2 恢复 `+32.037/+42.004`，而 learned prior 在 main/gap2/gap4 相对 CV
分别改善 `7.6%/10.9%/16.0%`。因此不重写 prior encoder，先定位 fusion：

补充口径：v3 相对原始 SeqTrack3D plain final 是 `+1.670/+1.873`，但 current
B0 相对原始 SeqTrack 是 `+2.374/+4.420`；论文模块归因不能使用前一组跨代码
版本差值，仍以 current B0 和 same-checkpoint fusion-off 为准。

- [ ] 用 epoch30 checkpoint 跑 standard fusion-on。
- [ ] 用同一 epoch30 checkpoint 加 `--fusion-off` 跑 observation-only。
- [ ] 用 epoch60 `last.ckpt` 跑 standard fusion-on。
- [ ] 用同一 epoch60 `last.ckpt` 加 `--fusion-off` 跑 observation-only。
- [ ] 四次评测固定同一 mini_val selection、seed、endpoint 顺序和时间模式。
- [ ] 导出逐 endpoint 的 observation/prior/final/GT、gate probability/alpha、
  correction、history-valid、previous prediction error、speed、delta_t 和
  tracklet id。
- [ ] 对 on−off 做逐 tracklet bootstrap，并报告 helpful precision、applied
  rate、首次失控帧、连续漂移长度和 disagreement/history-error 分桶。

决策分叉：

- fusion-off 恢复 B0、fusion-on 退化：冻结 observation/prior，只重构 gate；
- fusion-off 也低于 B0：先跑 same-code scratch B0，并审计主视图、RNG 与代码
  版本，禁止用新 gate 掩盖 baseline 回归；
- 只有同 checkpoint fusion-on 在 standard 的 Success/Precision 均超过 off，
  才允许一个 epoch15 的 v3.1 gate kill-test；
- v3.1 应使用 frozen recursive mini_train rollout 学习 bounded correction 的
  实际收益，取消 class-balanced 部署偏置，分开 helpful probability 与 step
  size，并使用渐进 fusion ramp；不再复用当前约 50% 的应用率。

gap1124/random20 与 true/fixed/shuffled 保持锁定，直到 standard on/off 同时为正。
本节只需推理，不阻塞下方 PFTC 修复。

## 0D. Motion legacy：只做无训练归因，不再扫全局 alpha

完整数据见
[`Motion fixed-alpha 复核`](compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md)。
alpha0.25 相对同代码 alpha0 的 final 为 `−17.468/−20.322`，late-3 为
`−17.357/−20.820`；当前 fixed global innovation 已经 No-Go。

- [ ] 用 alpha0 epoch60 checkpoint 分别以推理 alpha0/0.25 评测。
- [ ] 用 alpha0.25 epoch60 checkpoint 分别以推理 alpha0/0.25 评测。
- [ ] 四格评测固定同 endpoint、点采样 seed 和 normal mini_val。
- [ ] 导出 observation proposal、dynamics proposal、GT、previous prediction
  error、disagreement、valid history、foreground points、speed 和 delta_t。
- [ ] 报告 dynamics helpful rate、recursive oracle alpha、GT residual cosine、
  分桶净增益、首次失控帧和连续漂移长度。

判定分叉：

- alpha0 checkpoint 开启 0.25 立即退化，且 alpha0.25 checkpoint 关闭后恢复：
  直接 fusion 永久停止；
- 关闭后仍不恢复：说明还有训练期 coarse-query co-adaptation，同样不继续扫
  alpha，只保留失败机制分析；
- 只有出现跨 split 稳定、可由 GT-free 特征识别的 helpful subgroup，才允许
  重新讨论条件 alpha。P0-B4 已否定过一版 reliability gate，不能复刻原 gate。

本节不占用新训练 GPU，不阻塞下面的 PFTC 修复。禁止新增 alpha0.05/0.1 的
60-epoch 训练、seed43/44、full nuScenes 或 motion+search。

## 1. P1：修正 canonical geometry

- [ ] 将 `canonicalize_points` 从 `R(+yaw)` 改为项目一致的 `R(-yaw)`：

```text
x_local = cos(yaw) * x + sin(yaw) * y
y_local = -sin(yaw) * x + cos(yaw) * y
```

- [ ] 重写测试：用项目 `R(+yaw)` 执行 object-local → shared，再验证逆变换恢复。
- [ ] 覆盖 radians、degrees、非零相对 yaw 和跨帧 correspondence。
- [ ] 用 `datasets.points_utils` 的几何 primitive 做一条交叉测试，不能再让实现
  与单测共享同一个错误假设。

这一步完成前禁止任何新 PFTC 训练。

## 2. P2：移除平凡表示收缩

不要只把 λ 从 1.0 改成 0.1。raw SmoothL1 只有正对应，所有 feature 收缩即可
降低 loss；减小 λ 只会延缓，不会删除这个解。

- [ ] 把 consistency 作用到 train-only projector，隔离主干 64-D feature。
- [ ] positive feature 使用 L2-normalized similarity。
- [ ] 加入 per-frame/per-sample variance floor。
- [ ] 若仍收缩，再加入 canonical 距离足够远的 point negatives，或使用
  stop-gradient teacher/predictor。
- [ ] B0 即使 PFTC loss 关闭，也记录同定义的 backbone foreground feature std。
- [ ] 前 200 batch 记录 PFTC 与 supervised 对 FeaturePointNet 前两层的 gradient
  norm ratio 和 cosine，识别目标冲突。

## 3. P3：把训练开销降下来

完整运行平均 `2.983 s/step`，B0 为 `0.362 s/step`。正式重跑前的硬门槛是 PFTC 不超过
B0 的 2 倍。

- [ ] 消除 sample/frame/pair 循环中的 GPU `.item()` 同步。
- [ ] 优先复用 sampler 的采样索引去重，或在 DataLoader worker 预计算 match
  indices；训练 loss 只做 feature gather。
- [ ] 若仍在线匹配，按 frame pair 批量化并使用 padding/mask。
- [ ] 在 batch16、4×1024 上重新基准 step time 和峰值显存。

## 4. P4：5-epoch 机制 kill-test

新 clean commit、seed42、scratch、batch16、standard cadence，三臂各 5 epoch：

| arm | PFTC | pair 权重 | 用途 |
|---|---:|---|---|
| B0-diagnostic | 关 | — | feature std 与速度对照 |
| PFTC-U-v2 | 开 | 全 1 | consistency 本身 |
| Δt-PFTC-v2 | 开 | effective Δt | 时间加权增量 |

本阶段不按 mini 验证分数选模型，只按机制 gate：

- [ ] 有效样本率 ≥30%；
- [ ] feature std ≥ 同代码 B0 的 50%，且不持续趋近 0；
- [ ] PFTC/supervised gradient 无持续强负 cosine；
- [ ] step time ≤B0 的 2×；
- [ ] correspondence 在 true/fixed/shuffled 下拓扑相同；
- [ ] weighted 与 raw 的 pair-level loss/gradient 差异可测。

任一失败即停止 PFTC。全部通过后重新做 200-batch λ 预检；旧的 λ=1.0 不继承
到改变目标函数后的版本。

## 5. P5：正式三臂消融

只有 P4 全通过，才从 scratch 运行同代码、同初始化的 60-epoch：

1. B0；
2. PFTC-U-v2；
3. Δt-PFTC-v2。

仍使用 final + late-3，不按 best checkpoint 报主结果。先回答 PFTC-U 是否同时
提高 Success/Precision，再回答 Δt 是否在 PFTC-U 上有额外收益。若 weighted 与
unweighted 接近，只能保留 point consistency，不能声称 physical time。

## 6. 继续暂停

在 PFTC-v2 的 normal-mini 三臂完整晋级前，不运行：

- seed43/44；
- full nuScenes；
- Random-20% 或 gap1124；
- true/fixed/shuffled 的完整训练树；
- compact memory、MCC、Mamba；
- 用 motion/search/gate 掩盖 PFTC 失败。

旧 B0–B3、Search-only 与 Search on/off 2×2 归因保留为历史证据；它们不应和
当前 PFTC 修复同时扩展实验树。
