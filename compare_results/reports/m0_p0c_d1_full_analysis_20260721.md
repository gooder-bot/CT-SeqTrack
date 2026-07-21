# M0 P0-C-D1 full：gap1124 true / fixed / shuffled 冻结诊断

日期：2026-07-21

结论：`NO_GO_P0C_A2_TRUE_DT_PROMOTION`（再次确认）
分析对象：同一个 A2-order-dyn seed42 60ep `last.ckpt`，只干预 DynamicsEncoder 读取的 effective time。

## 1. 技术摘要

本次回传的不再是 2-tracklet smoke，而是 gap1124 mini_val 的完整三路逐 endpoint 输出：每路 `91` 个 tracklet、`1257` 个 endpoint、`102` 个字段。三路 endpoint key 与顺序完全一致，checkpoint、source config、selection 和 manifest hash 一致；`true` 的 effective time 与 real time 精确相等，`fixed` 全为 `0.5 s`，`shuffled` 使用与真实 gap 相同的多重集合但打乱了 endpoint 对应关系。因此三路比较满足“同模型、同帧、同预测流程，只改变时间输入”的冻结因果控制要求。

结果没有通过预注册 promotion gate。`true-fixed` 为 `+0.438 Success / +0.523 Precision`，低于要求的 `+0.5 / +1.0`；`true-shuffled` 为 `-0.123 / +0.056`，其中 Success 反而更低。逐 tracklet bootstrap 也没有给出 Success/Precision 的稳定正区间。正确物理时间对应关系对当前 frozen feature-concat A2 没有可推广的因果收益。

模型并非完全忽略时间：`true-fixed` 和 `true-shuffled` 都有 `1079/1257` 个 endpoint 的预测中心发生变化。但“会响应时间”没有转化为“正确时间优于负对照”。这一区分是本次 D1 最重要的机制结论。

## 2. 核心结果与可视证据

### 2.1 三路整体指标

| mode | Success | Precision | mean IoU | mean center error (m) | empty fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| true | 55.2247 | 66.8775 | 0.551270 | 2.270447 | 94 |
| fixed | 54.7872 | 66.3544 | 0.546596 | 2.461409 | 106 |
| shuffled | 55.3481 | 66.8218 | 0.552621 | 2.249559 | 92 |

### 2.2 配对效应与预注册门槛

| comparison | ΔSuccess (pp) | ΔPrecision (pp) | Δmean error (m) | Δfallback | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| true − fixed | +0.4376 | +0.5231 | -0.1910 | -12 | Fail：两项均未达 `+0.5/+1.0` |
| true − shuffled | -0.1233 | +0.0557 | +0.0209 | +2 | Fail：Success 为负，Precision 近零 |

逐 tracklet bootstrap（10000 次，seed42）的 95% CI：

| comparison / metric | tracklet mean | 95% CI | 中位数 |
| --- | ---: | ---: | ---: |
| true − fixed Success (pp) | +0.4664 | [-0.0041, +1.0559] | 0.0000 |
| true − fixed Precision (pp) | +0.5209 | [-0.1035, +1.2862] | 0.0000 |
| true − fixed center-error gain (m) | +0.1468 | [+0.0147, +0.3731] | +0.0005 |
| true − shuffled Success (pp) | -0.1558 | [-0.5989, +0.2901] | 0.0000 |
| true − shuffled Precision (pp) | +0.0223 | [-0.6448, +0.6719] | 0.0000 |
| true − shuffled center-error gain (m) | -0.0598 | [-0.1852, +0.0223] | -0.0008 |

Success/Precision 的区间均跨过 0。唯一不跨 0 的 `true-fixed` center-error gain 也不能单独作为方法收益，因为它主要由一条已彻底失控的长尾 tracklet 驱动，见第 2.4 节。

原生分组柱图、置信区间表和可执行分析单元见：

- `compare_results/notebooks/m0_p0c_d1_full_analysis_20260721.ipynb`
- `compare_results/reports/m0_p0c_d1_full_analysis_20260721.html`（已生成；当前 portable verifier 对生成器的 sticky header 报告轻微水平溢出，Markdown/notebook 结论不受影响）

### 2.3 时间分桶没有显示“越长 gap，真实时间越占优”

以下只统计 `1166` 个非初始化 transition：

| real Δt | N | true−fixed S/P (pp) | true−shuffled S/P (pp) |
| --- | ---: | ---: | ---: |
| `<0.75 s` | 644 | +0.497 / +0.613 | -0.078 / +0.097 |
| `[0.75,1.0) s` | 165 | +0.182 / +0.136 | -0.333 / +0.136 |
| `[1.0,2.0) s` | 257 | +0.623 / +0.506 | -0.195 / -0.263 |
| `≥2.0 s` | 100 | +0.400 / +1.100 | +0.000 / +0.525 |

效应不随 gap 单调增强。尤其对 shuffled 控制，true 在所有 gap 桶的 Success 都没有正优势；`≥2 s` 的 Precision `+0.525 pp` 也低于 `+1 pp` 门槛。

按当前帧 GT 位移分桶后，`<0.5 m` 的 `1191` 个 endpoint 承担了全部 Success 差异；`0.5–1 / 1–2 / 2–4 / ≥4 m` 四个更高运动桶中，true 相对两个控制的 Success 差均为 `0`。因此当前时间注入没有在真正高运动 endpoint 上恢复成功跟踪。

### 2.4 mean error 的表面优势由灾难性长尾主导

`true-fixed` 的平均中心误差看起来改善 `0.191 m`，但 tracklet `0cfdfb5bbe8a41268271e24f2edefb9c` 一条就贡献了约 `9.09 m` 的 tracklet-average gain，并承载全部 `-12` fallback 差异。该序列三种模式从第一个预测帧开始 IoU 都为 0；fixed 后期漂到约 35 m，true 仍漂到约 22 m。它说明 true-time 让一次既有灾难性漂移“没那么坏”，而不是恢复了目标。

移除这条序列后：

```text
true − fixed Success       +0.4450 pp
true − fixed Precision     +0.5320 pp
true − fixed mean error    -0.0397 m
true − fixed fallback       0
```

Success/Precision 判定几乎不变，但 mean-error 优势缩小约 79%。同样，`true-shuffled` 的平均误差差异主要由 tracklet `898c8b3c29d64d80806dd5e5fc6c02d3` 驱动；去除后差值只剩 `-0.0013 m`。因此本次主结论必须以配对 Success/Precision 和 tracklet bootstrap 为主，不能用 overall mean error 讲正向故事。

### 2.5 时间输入确实改变预测，但没有一致提高正确性

- true vs fixed：`1079/1257` 个 endpoint 的预测中心改变；中心差中位数 `0.0162 m`，P95 `0.4739 m`，最大 `13.604 m`。
- true vs shuffled：同样 `1079/1257` 个 endpoint 改变；中位数 `0.0195 m`，P95 `0.3992 m`，最大 `10.428 m`。
- 局部正例存在。例如 tracklet `4a8058d0441a46d381995e1af1086039` 在约 `2.05 s` gap 后，true 的 IoU/error 为 `0.831/0.187 m`，优于 fixed 的 `0.643/0.814 m` 和 shuffled 的 `0.668/0.709 m`，且后续仍保持优势。
- 但逐 tracklet 符号并不集中。true-fixed 的 Success 为 `31` 条正、`35` 条零、`25` 条负；true-shuffled 为 `25/36/30`。局部成功案例不足以支撑整体 physical-time alignment claim。

### 2.6 首次失控、持续失败和 fallback 也没有形成 true-time 优势

| mode | 出现首次失败的 tracklet | 出现持续失败的 tracklet | 最大失败 streak 的 tracklet 均值 | 最大 streak |
| --- | ---: | ---: | ---: | ---: |
| true | 17 | 16 | 1.560 | 20 |
| fixed | 17 | 16 | 1.571 | 20 |
| shuffled | 15 | 15 | 1.418 | 20 |

true 没有减少首次失控或持续失败的 tracklet 数；shuffled 反而各少 2 条。true−fixed 的 `-12` fallback 全部集中在前述 `0cfdf...` 长尾序列，其余 `90/91` 条 tracklet 的 fallback delta 都为 0；true−shuffled 的 `+2` fallback 也只来自一条 tracklet。fallback 差异不是广泛机制改善。

## 3. 范围、数据与指标定义

- 数据：nuScenes `v1.0-mini`，`mini_val`，冻结 gap1124 test manifest。
- selection：`91/106` 个 tracklet、`1257/2285` 帧，drop ratio `0.449891`。
- 模型：standard-trained A2-order-dyn seed42 60ep final checkpoint。
- checkpoint SHA256：`b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad`。
- source config SHA256：`69b801f7949aa2c72d6a30257cea36819cc1150af965ee6254e4fe17edb7d658`。
- selection SHA256：`85e5603c941030b050adab7876a275e654d9da328c859c1101e32940f9649f6f`。
- manifest content SHA256：`48f80593c35290aceba952891516742c48e8d5e65572a3815a917d6fd8b25285`。
- exporter SHA256：`740def7d6ef595a2e7ce25c9d4999d5cdc7ebe04faea41af470aefb664c4de55`。
- Success：逐 endpoint 3D IoU 的 0–1 阈值曲线均值 ×100。
- Precision：逐 endpoint 3D center error 的 0–2 m 阈值曲线均值 ×100。
- failure：center error `>2 m`；persistent failure：连续至少 2 帧失败。
- bootstrap 单位：tracklet，而不是把 endpoint 当独立样本。

三份 CSV SHA256：

```text
true      808c0190267e09891a041c756f52fd776d257ac740ab9674ad7578a0a944e167
fixed     27cb2f1ae7e67f46c2124e3f8537034ad714476c5d9f5ee3618268017200104d
shuffled  e2d71f877a34f8dd2955d118ade56147b45f6b2f5af73229d4af364d2fed63c4
```

## 4. 方法与数据质量审计

1. 解包前验证 `m0_manual_20260721_v2.tar.gz`，实际 SHA256 与 sidecar 均为 `7806ccd3652092aa58aa3047932eed28d1492afed2aaf3e4910fa247b54c45a2`。
2. 三路各有 `1257` 行、`91` 个 tracklet；复合键 `(tracklet_key, source_frame_index, frame_token)` 无重复，key 和行序 exact match。
3. 身份、真实时间、历史索引、GT 位移/框、checkpoint/config/selection/manifest 等不变量逐字段 exact match。
4. IoU、center error、real/effective time 无 NaN/Infinity。
5. `true` effective-real 最大绝对差为 0；`fixed` 精确为 0.5；`shuffled` effective-time 多重集合与 true real-time 多重集合完全一致。三路共 `1166 = 1257 - 91` 个 transition。
6. exporter summary、summarizer summary 与 CSV 使用 round-trip float parser 重算后完全一致。
7. 使用逐 tracklet 配对差和 seed42 的 10000 次 bootstrap；另做 leave-one-tracklet-out、gap bin、GT displacement bin、首次失败、最大连续失败和 fallback 定位。

## 5. 局限性、不确定性与稳健性

- 服务器仓库记录为 commit `343145dd50fa11fb63bbb8b7583a0a267ff5ca0d`，但运行时 `git_dirty=true`，主要因为手动上传的未跟踪工具和输出。精确 exporter/config/checkpoint/manifest/CSV hash 已保存，且结果重复了先前 clean P0-C aggregate 的效应方向与幅度；因此足以关闭本次冻结诊断，但正式论文归档仍应在 clean worktree 中复跑或至少保存 exact source bundle。
- 当前只覆盖 nuScenes mini_val、一个 checkpoint、一个 gap1124 selection；结论是“当前 frozen A2 机制不晋级”，不是“所有真实时间建模都无效”。
- path variance 字段在本次 P0-C-D1 中未启用；它属于后续 A/B/C frozen matrix，不能从本次结果推断 TWC 的多路径稳定性。
- CSV 若用 pandas 默认浮点解析，会把 Precision 提高约 `0.007955 pp`，原因是极少数值恰在阈值附近。使用 `float_precision='round_trip'` 或 Python `csv` 可精确复现原 summary；该差异不影响任何 paired delta 或 verdict。
- 首帧 self-overlap 口径已在 exporter 中修复；本次 full 三路 exporter summary 与 CSV 已一致，旧 true-only smoke 只保留为历史工程记录。

## 6. 决策与下一步

正式决策：

```text
NO_GO_P0C_A2_TRUE_DT_PROMOTION
```

本次 D1 已完成其目标：把旧 aggregate No-Go 下钻到 endpoint/tracklet 层，并确认不是“模型完全没读取时间”，而是“时间响应缺少正确对应关系的稳定优势”。因此：

1. 停止扩展当前 feature-concat A2：不补 burst-drop、unseen fixed-gap、多 seed，也不围绕它调 fixed value、scale 或 gate。
2. M0 下一优先级改为冻结 A/B/C 的 standard/gap1124/burst-drop/fixedgap2 endpoint 与 evaluation-only path variance；这是 TWC 机制收尾，不是继续 P0-C。
3. 并行完成 crop-reachable `d_obs -> d_dyn` convex-blend oracle。只有 oracle 显示 proposal 互补空间，才解锁 M2 innovation。
4. 完成 candidate0/1/2/3 伪速度审计，据此只冻结一种 M1 augmentation（shared SE(2) 或 smooth drift）。
5. M1 只能先做 zero-init dual-clock adapter 的代码与 A1 数值等价性；正式训练仍需新的 true/fixed/shuffled 因果 gate，不能把本次局部正例当作训练授权。

## 7. 尚待回答的问题

- TWC 的 `C-B` 是否能在 strong cadence 降低同 endpoint 的多路径预测方差，且不依赖单一异常 tracklet？
- crop-reachable endpoint 中，`d_dyn` 是否沿 `d_obs -> d_dyn` 方向提供稳定、非灾难长尾驱动的 oracle 改善？
- candidate1/2/3 的独立 jitter 是否制造与真实时间不一致的伪速度，足以解释 feature-concat dynamics 的不稳定？
- 若上述两个机制 gate 都失败，是否直接把论文收敛为 variable-rate 3D SOT benchmark/diagnosis，而不再增加复杂时间模块？

## 8. 可复查产物

- 原始回传：`output/diagnostics/m0_transfers/m0_manual_20260721_v2.tar.gz`
- 解包结果：`output/diagnostics/m0_manual_20260721_v2/`
- 服务器汇总：`output/diagnostics/m0_manual_20260721_v2/analysis/p0c_d1_gap1124_manual_20260721_summary.json`
- 可执行 notebook：`compare_results/notebooks/m0_p0c_d1_full_analysis_20260721.ipynb`
- 便携报告：`compare_results/reports/m0_p0c_d1_full_analysis_20260721.html`（见上方 verifier caveat）
