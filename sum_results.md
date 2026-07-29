# CT-SeqTrack 实验结果简要总结

更新时间：2026-07-30

这份文件只保留实验主线，不展开所有 epoch 数据。完整表格和曲线见 `compare_results/`。

## 2026-07-25 v2 重构说明

CT-SeqTrack 已完成 baseline、CT Motion、CT Motion + Search、完整
Adaptive Fusion 四组正常数据消融。下方 2026-07-24 及更早的 M2/M3/M4
数值只保留为历史证据；当前决策以新的 B0–B3 normal-mini 首筛和后续
Search-only A1 为准。

## 0. 当前总判断

### 2026-07-30 Motion fixed-alpha scratch 复核

两组新运行都完成 75,720 step、12 个验证点和 epoch60 `last.ckpt`。它们来自
同一 commit `5f260e7`，tracked source clean，resolved config 除 cfg/tag
外只差 `dynamics_innovation_alpha`：

| arm | final Success | final Precision | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|
| B0 baseline（历史上下文） | **53.360** | **64.382** | **52.905** | **63.104** |
| motion alpha=0 | 47.049 | 49.184 | 46.828 | 49.669 |
| motion alpha=0.25 | 29.581 | 28.862 | 29.472 | 28.849 |
| motion alpha=0.75（旧 B1） | 26.021 | 24.972 | 26.080 | 25.299 |

`alpha=0.25 − alpha=0` final 为 `−17.468/−20.322`，late-3 为
`−17.357/−20.820`。alpha0.25 warmup 后实际平均系数只有 0.184、平均修正
约 0.083 m，仍在 epoch25–60 的 8/8 个验证点同时低于 alpha0。更高 alpha
使末轮训练 loss 从 0.223 降到 0.217/0.215，但递归验证反向下降，支持
train/recursive-history mismatch 与错误 proposal 闭环放大。

当前判定为 `NO_GO_FIXED_GLOBAL_MOTION_INNOVATION`。alpha0 是精确关闭
correction 的负对照，并不提供 motion 正贡献；它与 B0 还有共享初始化混杂。
不再训练更小全局 alpha。下一步只做已有 alpha0/0.25 checkpoint 的推理
on/off 2×2 和逐 endpoint proposal attribution。完整报告见
`compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md`。

### 2026-07-27 Search-only A1 normal-mini

A1 已完成 75,720 个训练 step、12 个验证点和 epoch60 `last.ckpt`：

| arm | final Success | final Precision | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|
| B0 baseline | **53.360** | **64.382** | **52.905** | **63.104** |
| A1 Search-only | 27.036 | 25.596 | 27.933 | 26.400 |
| B1 motion-only | 26.021 | 24.972 | 26.080 | 25.299 |
| B2 motion + search | 47.973 | 52.088 | 46.437 | 49.818 |

A1 相对 B0 final 为 `−26.324/−38.786`，late-3 为
`−24.972/−36.705`；best 也只有 `29.257/30.202`，当前 Search-only 明确
No-Go。A1 与 B0 的 epoch60 mean training loss 仅差约 0.0013；训练侧
search-used ratio 为 3.460%，与 B2 的 3.458% 一致。失败不是 search 未启用
或常规训练发散，更可能来自训练历史与递归预测历史下的搜索分布错位，或
motion×search 强交互。

B2−B1 的 `+21.952/+27.116` 只能解释为交互恢复，不能作为 search 独立
收益。下一步不训练 A2；先用现有 B0/A1 checkpoint 做 Search 开/关 2×2，
并补验证阶段逐 endpoint search diagnostics。完整报告见
`compare_results/reports/ct_search_only_seed42_20260727.md`。

### 2026-07-27 CT-SeqTrack v2 B0–B3 normal-mini

四组均完成 75,720 个训练 step、12 个验证点和 epoch60 `last.ckpt`：

| arm | final Success | final Precision | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|
| B0 baseline | **53.360** | **64.382** | **52.905** | **63.104** |
| B1 + motion | 26.021 | 24.972 | 26.080 | 25.299 |
| B2 + search | 47.973 | 52.088 | 46.437 | 49.818 |
| B3 + adaptive gate | 25.537 | 24.707 | 26.321 | 25.104 |

B1 相对 B0 为 `−27.339/−39.410`；B2 相对 B1 恢复
`+21.952/+27.116`，但相对 B0 仍低 `−5.387/−12.294`；B3 相对 B2
再次下降 `−22.435/−27.381`，最终基本退回 B1 水平。

B3 gate 在 epoch5 仍为初始 nominal alpha 0.25，epoch6 升到 0.707，
epoch7 已为 0.749；epoch60 的 batch-min mean 仍是 0.749998。它实际
退化成最大权重常数 gate，没有学到条件可靠性。B3 的 epoch60 mean training
loss 还略低于 B2（0.206 vs 0.213），但验证显著更差，更符合
train/recursive-validation mismatch 或 co-adaptation，而不是训练不足。

本节当时的正式决定是先补 Search-only；该实验现已完成并在上一节判定
No-Go。当前决策以 Search-only 技术复核为准，不再训练 A2。B0–B3 完整报告
仍见 `compare_results/reports/ct_v2_ablation_seed42_20260727.md`。

### 2026-07-24 M2 standard / gap1124 八组正式控制

commit `473738f` 的 R1 epoch60 `last.ckpt` 已完成 standard/gap1124 的 `true/fixed/shuffled`，并在相同 endpoint 上导出 matched A1。结果包通过 `89/89` artifact hash；8 份 CSV 的 endpoint key/order、GT 与 real-time 字段在协议内 exact match，原始 CSV 独立复算与服务器 summary 最大绝对误差为 0。

| protocol | comparison | ΔSuccess | ΔPrecision | tracklet 95% CI（S / P） |
| --- | --- | ---: | ---: | --- |
| standard | M2 true − A1 | **+4.133** | **+9.445** | `[1.920,5.454] / [4.486,10.634]` |
| gap1124 | M2 true − A1 | **+2.279** | **+4.143** | `[0.687,3.940] / [1.452,6.022]` |
| standard | true − fixed | +0.031 | −0.010 | 均跨 0 |
| standard | true − shuffled | +0.068 | +0.085 | 均跨 0 |
| gap1124 | true − fixed | −0.127 | +0.014 | 均跨 0 |
| gap1124 | true − shuffled | −0.318 | −0.209 | 均跨 0 |

冻结门槛给出明确分叉：

```text
STANDARD_GUARDRAIL_PASS
GAP1124_COMPLEMENTARITY_PASS
PHYSICAL_TIME_CAUSAL_GATE_FAIL
M2_TRACKING_SIGNAL_POSITIVE
PHYSICAL_TIME_CAUSAL_CLAIM_NO_GO
METHOD_ATTRIBUTION_HOLD
```

时间路径不是失活：gap1124 true 的 effective-`dt` CV 为 `62.2%`，约 `90.8%` 非初始 endpoint 实际应用 innovation，平均 radius 从 fixed 的 `0.750 m` 增至 `0.962 m`；true/fixed 与 true/shuffled 分别有约 `31.6%/32.5%` endpoint 的预测中心改变超过 1 cm。但正确时间没有带来 alignment 优势，gap shuffled 反而在两项主指标上都高于 true。因此当前涨点更可能来自 continuation、联合表征或通用 proposal correction，不能归因于正确 physical time。

gap1124 的 M2−A1 在四个 real-`dt` 桶都为正，而 true−shuffled 的 Success 在四桶都为负；`≥4 m` GT 位移桶只有 28 个 endpoint，M2−A1 的 Success/Precision 增益均为 0。这与既有 crop-reachability 结论一致：大位移时瓶颈常发生在网络 forward 前，末端 correction 无法稳定追回离开 crop 的目标。

正式决定：不为 physical-time claim 补 seed43/44，不用 burst-drop 推翻 gap1124 因果失败，不启动 timestamp-conditioned M3/M4。下一步转为 M2 正信号归因：R1 运行时 2×2、A1-init W0 continuation、current-code legacy W0、proposal target/gradient/recursive-error audit。完整报告见 `compare_results/reports/m2_standard_gap8_analysis_20260724.md`。

### 2026-07-23 M2 三组训练结果

M1/M2 已通过 E0–E5 服务器工程门禁。E6 的参数与配置静态冻结也已完成：mini_train M0-3 的既有 `1311 endpoints / 213 tracklets` 只用于确认一个预声明规则，正式值固定为 `alpha=0.75`、`R(dt)=min(0.5+0.5dt,2.0)`、adapter/innovation 共享 warmup=5。该规则的 endpoint mean gain 为 `0.288 m`，tracklet-equal mean gain 为 `0.263 m`，bootstrap 95% CI `[0.230,0.296] m`；但 clamp rate 为 `34.48%`，属于安全而保守的首轮规则，不声称最优。

commit `473738f` 的三组训练均已完成并通过本地完整性审计：

```text
PASS_M1_M2_E0_E5_ENGINEERING_GATES
FREEZE_M2_ALPHA_RADIUS
E6_STATIC_FREEZE_READY
PASS_E6_SERVER_MANIFESTS_PREFLIGHT_COMMIT_473738f
R1_R2_R3_TRAINING_INTEGRITY_PASS
M2_STANDARD_SIGNAL_POSITIVE
METHOD_ATTRIBUTION_AND_CAUSAL_TIME_HOLD
```

| run | 初始化/结构 | Final Success | Final Precision | Late mean Success | Late mean Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| 历史 A1 | scratch / SeqTrack3D order-time | 51.229 | 57.863 | 50.975 | 58.123 |
| R1 | A1-init / full M2 | **55.303** | **67.182** | **54.677** | **66.514** |
| R2 | scratch / full M2 | 53.318 | 62.503 | 51.894 | 60.254 |
| R3 | scratch / shared-SE(2) W0 | 28.999 | 28.023 | 27.664 | 26.488 |

R1 相对历史 A1 final 为 `+4.074/+9.318`，R2 相对历史 A1 为 `+2.090/+4.640`，R2 相对 R3 为 `+24.319/+34.480`。R1/R2/R3 都是 clean `473738f`、退出码 0、epoch59/global step75720、12 个评测点与 75720 条 loss；R1 的 35 项 artifact manifest 全匹配。

这推进了 standard 性能信号，但没有完成方法归因。R1 在 A1 60 epoch 后又训练 60 epoch，缺少 A1-init W0 continuation；R3 又不是历史 A1，而是当前 shared-SE(2) 下明显塌陷的 W0。后续 2026-07-24 formal controls 已将 physical-time claim 从 HOLD 更新为 No-Go；本节只保留训练阶段证据。完整数据、图表和结论见 `compare_results/reports/m2_three_run_analysis_20260723.md`。

标准 protocol 的本地 `delta_t` 为 `0.4974±0.0228 s`，CV 仅 `4.59%`，物理时间分支在该分布上很难被单独辨识。standard 因此是 normal-cadence guardrail；主要方法证据必须来自 strong/held-out cadence 和同 checkpoint `true/fixed/shuffled`。完整 HTV/丢帧边界与执行计划见 `compare_results/reports/htv_identifiability_and_execution_plan_20260722.md`。

### 2026-07-21 TWC A/B/C 同提交 seed42

同 commit `343145d`、同 seed42、同 mini_train/mini_val selection、batch16、candidate4、60 epoch 和 75720 optimizer steps 的三组控制已经完成并拉回本地：

| run | Success final | Precision final | Success late mean | Precision late mean |
| --- | ---: | ---: | ---: | ---: |
| A: single view | 50.01 | 58.20 | 49.06 | 54.84 |
| B: paired views, `twc_weight=0` | 34.71 | 34.02 | 35.35 | 34.96 |
| C: paired views, `twc_weight=0.05` | 43.01 | 45.76 | 42.70 | 45.60 |

Final 效应分解为：

```text
B-A = -15.30 Success / -24.18 Precision
C-B =  +8.31 Success / +11.74 Precision
C-A =  -7.00 Success / -12.44 Precision
```

这严格确认了 corrected-TWC 相对相同 paired-view control 的净正效应，但也确认 paired-view 训练路径本身造成巨大退化。C 在 Final 只恢复 `54.3%/48.6%` 的 A-B 损失，在 Late mean 恢复 `53.6%/53.5%`，仍未回到 single-view A。B/C 的有效样本序列完全一致，anchor/current-point gap max 全程为 0；C 的末 1000 步 center/angle gap 仅比 B 低 `2.17%/6.13%`，属于温和的训练期一致性改善。

正式边界为：

```text
C_MINUS_B_POSITIVE_ON_STANDARD_SEED42
NO_GO_TWC_MAIN_METHOD_PROMOTION
```

原预注册 gate 还要求 standard 无明显退化、gap1124/burst-drop 的 `C-B` 和 held-out evaluation-only path variance。当前训练包只含 standard mini_val aggregate，没有 per-tracklet/endpoint 或 strong-cadence 终点；因此不补 seed43/44，只允许用冻结 final checkpoint 做一次不重训的 strong-cadence/path-variance 收尾。完整报告、图表和 CSV 见 `compare_results/reports/twc_abc_seed42_comparison_20260721.md`。

### 2026-07-20 P0-B4 independent validation

固定5特征 `observation_v1` calibrator 只在 mini_train standard 上拟合一次，然后在 disjoint mini_val standard/gap1124/burst-drop 上原样评估。gap/burst 的 AUROC 为 `0.680/0.712`，固定阈值 recall 为 `0.568/0.609`，均未通过预注册的 `0.75/0.70` 门槛；正式判定为：

```text
NO_GO_OBSERVATION_RELIABILITY_VALIDATION
```

强协议 AUPRC margin、ECE 和 FPR 虽通过，但 precision 只有 `0.177/0.185`，Brier 还略差于本协议 prevalence 常数基线。mini_train fit prevalence 为 `0.283`，mini_val 强协议只有 `0.089/0.085`，说明存在明显 split/难度漂移；正例还只分布在 10/9 个 tracklet，不能把结果扩写成“所有 observation reliability 都无效”，但当前冻结 calibrator 不能部署，也不能在 mini_val 上重调。

同批 passive raw-CV 第二 crop 在 gap/burst 的 trajectory-only endpoint 都为 0，oracle union gain 都是 `0.00 pp`。因此 P0-B3 的 observation-quality Conditional-Go 未通过独立验证升级，reliability-controlled Kalman/frozen-state anchor、active dual-anchor 和 learned gate 全部停止。完整复核见 `compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。

### 2026-07-20 P0-B3 reliability 三协议结果

standard/gap1124/burst-drop full passive diagnostic 已下载并完成本地复核，endpoint 数为 4246/2127/2098，均与 P0-B/P0-B2 reference exact match，使用同一 A1 checkpoint hash。三份 CSV 无重复 endpoint、关键字段缺失或数值范围异常；现有 summary 的全部核心指标已从原始 CSV 独立复算，最大绝对误差小于 `6e-16`。

预注册 all-13 trigger 在 gap1124/burst-drop 得到 AUROC `0.787/0.785`、AUPRC `0.660/0.671`，达到原定 reliability Go 门槛；但 raw-CV dual crop 的 oracle recall 增益只有 `+2.88/+3.15 pp`，低于两协议均 `+5 pp` 的要求。因此正式判定为：

```text
RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO
```

这个名称不能解释成“timestamp-aware reliability 已验证”。同一 grouped 流程的 post-hoc feature ablation 显示，previous-observation-only trigger 在 gap/burst 反而达到 AUROC `0.867/0.873`、AUPRC `0.778/0.789`；time/CV geometry-only 只有 AUROC `0.553/0.557`。加入 raw `current_delta_t` 是跨协议退化的主要来源：删除它后 AUROC 恢复到 `0.865/0.872`，ECE 从 `0.134/0.155` 降到 `0.061/0.060`。这说明当前可复现的正信号是 **上一 observation 质量预测下一 crop 风险**，真实时间对 reliability 的独立增量仍未成立。

当前 post-crop selector 也不能进入 active tracker：standard/gap/burst AUROC 为 `0.729/0.605/0.433`，强协议 ECE 为 `0.401/0.490`。P0-B3 当时只允许在 P0-B4 通过后测试一次独立 state anchor；P0-B4 现已失败，因此该候选不再实现。完整 P0-B3 验证见 `compare_results/reports/p0b3_reliability_validation_20260720.md`。

### 2026-07-17 P0-A / P0-B 诊断

P0-B oracle 与 P0-B2 recursive predicted-history 均已完成。oracle 表明 GT-history CV 可将强协议 recall 恢复到 98.96%/99.05%，但真实 A1 预测历史下，CV 相对 previous-A1 只提高 2.91/2.65/3.03 pp，未达到预注册的总体 +5 pp 门槛；gap/burst 的 `>4 m` 位移桶提高 8.45/9.96 pp，也未同时达到 +10 pp。结论是 **always-on raw predicted-history CV recenter No-Go**。

bounded residual 的 standard 真实 batch 也已完成第一轮诊断。warmup 内 residual/gate gradient 严格为 0；active 64-batch 中 observation error P50/P75/P95 为 `0.213 / 0.577 / 3.838 m`，默认 2 cm 理论上限已明显偏小。更关键的是 gate alpha 约 `2e-5`，实际 applied residual P50 只有 `7.25e-8 m`，gate gradient P50 只有 `4.00e-10` 且 31/64 batch 为 0。默认路径数值稳定但在功能上接近关闭，未通过“非平凡修正幅度”验收。

这些 observation error 混入了 out-of-crop 失败，不能据此直接把 residual bound 调到几十厘米或数米。递归结果进一步显示：上一预测误差不超过 4 m 时 pred-CV recall 为 97.34%–98.64%，超过 4 m 后只有 0.80%–1.61%。后续 P0-B4 已否定当前 reliability-controlled dual-anchor 入口；若继续 residual，只能在 crop-reachable mini_train subset 上做一次预注册机制收尾。完整报告见 `compare_results/reports/p0_ab_diagnostics_20260717.md` 与 `compare_results/reports/p0b2_recursive_crop_reachability_20260717.md`。

当前 P0-B/P0-B2/P0-B3/P0-B4 已形成完整的 oracle、recursive、开发集与独立验证链；P0-B4 在 active 之前 No-Go，所以不存在 active dual-anchor 跟踪结果。仍不完整的是 P0-A，它只有 standard 64-batch，没有完整 split、强 gap 或真正 2-step optimizer 结果。

### 2026-07-16 corrected-TWC、HTV 与 TrajTrack 结果口径

代码审查发现，旧 active-TWC sampler 在 candidate 1/2/3 下分别为 A/B 两路采样最近历史框扰动，导致两路 current search crop 和局部坐标系不同；旧检查比较归一化后天然接近零的 `ref_boxs[:, 0]`，没有发现这个问题。因此：

- 旧 A1+TWC 的 precision-positive 信号暂时撤回，不能归因给 TWC。
- 旧 A2+TWC 的退化也暂时撤回，不能据此判断 TWC 与 dynamics 冲突。
- 两路各自的 supervised loss 仍有效，但跨 view TWC loss 不是干净的一致性约束。
- 共享 candidate offset、`coordinate_anchor` 和 point-sampling seed 的修复已经完成；修复后的 A1/A2 seed42 训练均已完成，anchor gap max 与 current XYZ gap max 都为 0。
- `A2-residual-dyn` 已完成 standard 真实 batch 的 warmup/active forward-loss-backward 诊断；默认量级近乎为零，尚无完整 split、强 gap、2-step optimizer 或跟踪性能结果。

修复后的 seed42 结果显示：A1+corrected-TWC 相对旧配置对齐 baseline 的 final 为 `+1.49 Success / +5.03 Precision`，late mean 为 `+0.99 / +2.67`；A2+corrected-TWC 的 final 为 `-0.93 / -2.07`。前者是值得复现的单 seed 正信号，后者不支持把 TWC 接入当前 A2 主线。由于 baseline 来自旧 run、没有 git commit，二者仍只是配置级参考，不能视为严格同提交因果结果。

HTV 六组 seed42 筛选也已完成：旧 feature-concat `A2-order-dyn` 在 random20 上相对 A1 final 为 `+9.09 / +14.23`，但在 gap1124 为 `-4.01 / -9.55`、burst-drop 为 `-7.45 / -14.40`。这不支持“时间间隔越不规则，旧 A2 越有效”，而支持继续验证 observation-first bounded residual、candidate 运动监督和 crop 可达性。

TrajTrack aligned seed42 run 虽得到 64.94 / 79.07，但当前本地 evaluator 使用当前帧 GT overlap 触发 refinement，并用 GT overlap 选择 proposal。该数值只能作为 oracle-assisted 实现诊断，不能作为对 SeqTrack3D 或 CT-SeqTrack 的公平在线增益。

目前结果不支持继续把真实时间直接塞进 SeqTrack3D 主干时间 token。更稳的方向是：

```text
主干保留 SeqTrack3D 的 order-time 语义；
真实 delta_t 优先驱动受可靠性约束的第二 trajectory proposal；
当前末端 bounded residual 只保留为待校准 refinement 对照；
用固定 manifest 的 variable-rate / HTV 因果矩阵验证时间因果性；
corrected TWC 先独立复现 A1，gate 暂缓；TrajTrack 先修正为 GT-free evaluator。
```

旧 60ep seed42 汇总里，`A2-order-dyn` 的 final success 基本追平 SeqTrack baseline，final precision 高于 baseline，因此它曾是最清楚的真实时间正向信号。但 2026-07-08 整理的 5 次复核改变了当前判断力度：`A2-order-dyn` seed43 崩坏到 23.64 / 23.77，seed44 只有 46.90 / 52.62；`A2-order-dyn+TWC w0.01` 仍只有 22.88 / 24.27；`A3-conf-res` best-e14 复测只有 28.06 / 37.70，没有复现旧 62.04 / 76.30 高点。现在的主线应从“证明 A2 已稳定有效”改为“先建立 variable-rate 问题设置，再围绕 residual dynamics 做 seed 稳定性、checkpoint 对齐和机制诊断”。

### 0.1 当前能说和不能说

当前能说：

- 真实时间方向没有被否定；失败主要来自不合适的注入方式。
- SeqTrack3D 主干对 order-time token 语义敏感，直接替换为 raw real-time token 不稳定。
- `A2-order-dyn` 是旧 60ep seed42 下最强正向信号，说明真实 `delta_t/current_delta_t` 作为 dynamics prior 有潜力。
- 标准 fixed-step 整体 final 不是当前最有把握的涨点场景；variable-rate、long-gap、sparse / re-appearance 子集更适合证明真实时间的价值。
- `A2-order-dyn-cand1` 在当前 60 epoch 协议下明显退化，不支持简单去掉非 0 candidate。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本持平，final precision 小幅更高，说明小权重 displacement loss 不伤主线，但也不是主要解释。
- 旧 active `A1-order+TWC` 曾观察到 final precision +3.24，但该 run 的 nonzero candidate 坐标不共享，不能作为 TWC 正向证据。
- 旧 active `A2-order-dyn+TWC` 及 `twc_weight=0.01` 曾明显退化，但同样受坐标污染，不能作为 TWC 与 dynamics 冲突的证据。
- `A3-order-gate-safe` 比旧 P5 full 安全，但相对 A2-order-dyn final success / precision 仍下降 -2.64 / -8.45。
- `A3-order-conf-res-gate` 旧汇总 best checkpoint 很高（success 62.04 / precision 76.30），但最新 best-e14 复测只有 28.06 / 37.70，暂时不能把旧 best 当作确认收益。
- corrected A1+TWC 的旧配置级正信号已由同提交 A/B/C 取代：`C-B` final 为 `+8.31/+11.74`，但 `C-A` 为 `-7.00/-12.44`；TWC 只部分修复 paired-view 退化。
- HTV 六组说明旧 feature-concat dynamics 的效果依赖 protocol：random20 为正，gap1124/burst-drop 为负。
- 三协议 crop oracle 说明高速位移失败会发生在模型 forward 前；强 gap/burst 下固定 2x expanded 也明显不足，而 GT-history CV recenter 仍接近 99% recall 且没有额外背景点代价。
- 三协议递归诊断否定了 raw CV 恒开启替换 anchor，但确认可靠预测历史下的 CV recall 可达 97%–99%，支持“可靠性控制的预防性第二锚点”这一更窄假设。
- 默认 bounded residual 的实际修正约为 `1e-7 m`，gate 梯度极小；它目前是稳定但近乎关闭的路径，不是可直接训练的主配置。
- TrajTrack 论文的“历史轨迹 proposal + local/global proposal agreement”值得借鉴，但当前本地 GT-assisted evaluator 的高分不能进入公平主表。

当前不能说：

- 不能说完整 CT-SeqTrack full model 已经稳定超过 SeqTrack3D。
- 不能说 corrected-TWC 已稳定超过 A1；虽然同提交 seed42 的 `C-B` 已确认净正效应，但 C 在 final、best、late mean 上都低于 single-view A，且没有 strong-cadence/path-variance 与多 seed 证据。
- 不能说 gate 已经无效，因为 gate-safe 比旧 P5 full 安全，conf-res 又出现很高 best；但也不能说 gate 已经稳定有效。
- 不能按 `A3-order-conf-res-gate` 旧 best 下正向结论，因为最新复测未复现。
- 不能说 candidate noise 已被彻底排除，因为 `cand1` 只有 `num_candidates=4` 实验约 1/4 的 optimizer step，且还缺少 candidate 分桶日志。
- 不能说 displacement 监督已经是必要模块；目前它只是一个小幅、温和的正向/不伤信号。
- 不能只靠普通 fixed-step benchmark 讲论文成功；如果没有 variable-rate / HTV 协议和分桶收益，真实时间贡献会显得证据不足。
- 不能把 TrajTrack 本地 64.94 / 79.07 写成公平结果，也不能把其与 SeqTrack3D 的算术差值写成方法增益。
- 不能把 GT-history CV 的约 99% recall 或被动 pred-CV 的 2.65–3.03 pp 写成 active 在线增益；当前还没有 dual-anchor 实际跟踪结果。

## 1. 第一轮：Baseline vs P5 full

比较：

```text
SeqTrack baseline
vs
CT-SeqTrack P5 full
```

P5 full 同时打开：

```text
real timestamp + DynamicsEncoder + ObservabilityGate
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
P5 full final:          success 31.19, precision 31.89
```

说明：

- P5 full 明显退化，尤其 final precision 掉得很重。
- 但这不能说明 timestamp-native 思路失败，因为这一轮把 real time、dynamics、gate 一次性混在一起。
- P5 full 的 best precision 曾接近 baseline，说明模型不是完全学不到定位，问题更像后期递归跟踪不稳定。

下一步由此产生：

```text
必须拆开 A1 / A2 / A3 消融，先定位退化来自真实时间主链路、dynamics，还是 gate。
```

## 2. 第二轮：A1-raw 和 A2 Dynamics

比较：

```text
SeqTrack baseline
vs
A1 CT-base: real timestamp, no dynamics, no gate
vs
A2 Dynamics: real timestamp + DynamicsEncoder, no gate
```

关键结果：

```text
A1-raw final: success 28.28, precision 27.43
A2 raw-dyn final: success 45.27, precision 58.83
```

说明：

- `A1-raw` 大幅退化，说明直接把真实秒数输入主干时间通道非常危险。
- `A2 raw-dyn` 相比 `A1-raw` 明显恢复，尤其 precision 接近 baseline final。
- 这支持一个判断：dynamics 分支本身有作用，但 raw real-time 主干路径有明显问题。

下一步由此产生：

```text
先诊断 A1 的时间输入形式，确认是不是 raw 秒数尺度或注入位置导致崩坏。
```

## 3. 第三轮：A1 时间编码诊断

比较：

```text
A1-raw
A1-pseudo
A1-MLP
A1-Fourier
SeqTrack baseline
```

关键结果：

```text
A1-pseudo final:  success 48.34, precision 52.25
A1-MLP final:     success 27.44, precision 26.28
A1-Fourier final: success 30.72, precision 29.82
```

说明：

- `A1-pseudo` 明显接近 baseline，说明 CT 代码主链路不是完全坏掉。
- `A1-MLP` 和 `A1-Fourier` 没有救回 real-time A1。
- 问题不只是“raw 秒数太大”，也不是简单换一个 scalar-preserving time encoding 就能解决。

下一步由此产生：

```text
不要继续在 MLP/Fourier 上堆复杂编码；
需要检查真实时间注入主干的位置和语义。
```

## 4. 第四轮：scaled real time

比较：

```text
SeqTrack baseline
vs
A1-scaled
vs
A2-scaled-dyn
```

关键结果：

```text
A1-scaled final:     success 31.33, precision 31.22
A2-scaled-dyn final: success 29.41, precision 31.51
```

说明：

- 把真实时间缩放回接近伪时间数值范围，并没有修复 A1。
- `A2-scaled-dyn` 也没有带来稳定收益。
- 这进一步说明：问题不只是时间数值尺度，而是主干分支对时间 token 的语义很敏感。

下一步由此产生：

```text
主干不再强行使用 real-time token；
改为恢复 SeqTrack3D 原本的 order-time 语义。
```

## 5. 第五轮：order-time 主干恢复

比较：

```text
SeqTrack baseline
A1-pseudo
A1-order
A2-order-dyn
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
A1-order final:          success 51.23, precision 57.86
A2-order-dyn final:      success 50.96, precision 63.31
```

说明：

- `A1-order` 基本修复了 A1-raw / A1-scaled 的崩坏，说明主干确实需要保留 order-time token 语义。
- `A2-order-dyn` 在 final success 上基本追平 baseline，在 final precision 上超过 baseline。
- 这给当前论文主线提供了最清楚的正向证据：真实时间更适合进入 dynamics prior，而不是直接替换主干时间 token。

当前结论：

```text
A2-order-dyn 是当前最值得作为主线继续推进的配置。
```

## 6. 第六轮：cand1 / displacement 诊断

比较：

```text
SeqTrack baseline
A2-order-dyn
A2-order-dyn-cand1
A2-order-dyn-disp
```

关键结果：

```text
SeqTrack baseline final:    success 50.99, precision 59.96
A2-order-dyn final:         success 50.96, precision 63.31
A2-order-dyn-cand1 final:   success 26.68, precision 24.50
A2-order-dyn-disp final:    success 50.54, precision 63.85
```

best 结果：

```text
A2-order-dyn-cand1 best: success 41.99 epoch 10, precision 54.62 epoch 5
A2-order-dyn-disp best:  success 52.44 epoch 10, precision 64.81 epoch 40
```

重要注意：

```text
cand1 的 num_candidates=1 会把每个 epoch 的训练 batch 数减少约 4 倍。
因此 cand1 60 epoch 的 final_step 只有 18899，
而 A2-order-dyn / disp 60 epoch 的 final_step 是 75719。
所以 cand1 与 A2-order-dyn 不是严格 optimizer-step 对齐。
```

结果解读：

- `A2-order-dyn-cand1` 在前 5-10 epoch 还有可用信号，但随后 success 和 precision 都明显塌到 20 多分。当前 60 epoch 协议下，它不支持“直接移除非 0 candidate 可以清理 dynamics”的假设。
- cand1 的退化至少有两个可能来源：一是移除 multi-candidate 后训练鲁棒性下降；二是 optimizer step 只有原来的约 1/4，学习率按 epoch 衰减导致有效训练不足。若要严格判断 candidate noise，后续需要 `cand1-240ep` 或等 step 版本。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本同水平：final success 低 0.42，final precision 高 0.53；best success 高 0.90，best precision 高 1.24。
- 这说明 `dynamics_displacement_weight=0.01` 没有破坏主线，并且对 precision 有一点温和正向信号；但幅度还不足以把 displacement loss 作为核心贡献。

当前决策：

```text
保留 multi-candidate 训练。
displacement loss 可以作为小权重稳定项或诊断项继续观察，
但当前主线贡献仍应放在 order-time main branch + real-time DynamicsEncoder。
```

图表和完整表格：

```text
compare_results/cand1_disp_dynamics_curves.png
compare_results/cand1_disp_dynamics_success_curve.png
compare_results/cand1_disp_dynamics_precision_curve.png
compare_results/cand1_disp_dynamics_best_final_summary.png
compare_results/cand1_disp_dynamics_metrics_points.csv
compare_results/cand1_disp_dynamics_metrics_summary.csv
compare_results/cand1_disp_dynamics_comparison.md
```

## 7. 第七轮：旧 order+TWC 诊断（inactive）

比较：

```text
SeqTrack baseline
A1-order
A1-order+TWC
A2-order-dyn
A2-order-dyn+TWC
```

关键结果：

```text
A1-order final:          success 51.23, precision 57.86
A1-order+TWC final:      success 45.61, precision 50.77
A2-order-dyn final:      success 50.96, precision 63.31
A2-order-dyn+TWC final:  success 38.27, precision 38.85
```

关键诊断：

```text
两组旧 order+TWC 的 loss_twc / twc_valid_ratio / twc_center_gap / twc_angle_gap
全程为 0。也就是说，这一轮训练没有真正施加 TWC consistency 项。

旧结果也使用 num_candidates=1，60 epoch final_step 只有 18899，
不是 active-TWC，也不是严格 optimizer-step 对齐结果。
```

结果解读：

- 旧 `A1-order+TWC` 和 `A2-order-dyn+TWC` 的下降是真实观察，但不能据此判断 TWC 机制本身有害，因为 TWC loss 没有激活。
- 这轮的价值是暴露了 TWC validity / paired-view 训练协议问题。
- validity 修复后已经重跑，正式 active-TWC 结论见下一节。

## 8. 第八轮：旧 active TWC / gate-safe / conf-res（TWC 归因已失效）

> 2026-07-11 更正：本节 TWC run 的 validity mask 确实激活，但 nonzero candidate 下 A/B 不共享坐标系。指标和 loss 日志只作历史记录，不能再用来判断 TWC 的收益、损害或它与 dynamics 的兼容性；gate 结果不受这个 sampler 问题影响。

比较：

```text
SeqTrack baseline
A1-order
A2-order-dyn
A1-order+TWC
A2-order-dyn+TWC
A3-order-gate-safe
A3-order-conf-res-gate
```

关键结果：

```text
SeqTrack baseline final:      success 50.99, precision 59.96
A1-order final:               success 51.23, precision 57.86
A2-order-dyn final:           success 50.96, precision 63.31
A1-order+TWC final:           success 51.16, precision 61.10
A2-order-dyn+TWC final:       success 28.23, precision 32.04
A3-order-gate-safe final:     success 48.32, precision 54.87
A3-order-conf-res-gate final: success 31.17, precision 30.92
```

best 结果：

```text
A1-order+TWC best:           success 53.16, precision 63.35
A2-order-dyn+TWC best:       success 45.24, precision 57.43
A3-order-gate-safe best:     success 50.99, precision 60.17
A3-order-conf-res-gate best: success 62.04, precision 76.30
```

关键诊断：

```text
A1-order+TWC:
  twc_valid_ratio mean 0.750, tail1000 0.753
  loss_twc tail1000 0.0081
  vs A1-order final success -0.07, precision +3.24

A2-order-dyn+TWC:
  twc_valid_ratio mean 0.750, tail1000 0.750
  loss_twc tail1000 0.0077
  vs A2-order-dyn final success -22.73, precision -31.28

A3-order-gate-safe:
  obs_alpha_dyn_mean mean 0.127, tail1000 0.116
  vs A2-order-dyn final success -2.64, precision -8.45

A3-order-conf-res-gate:
  obs_alpha_dyn_raw_mean mean 0.493
  obs_alpha_dyn_clamped_mean mean 0.181
  obs_dyn_residual_norm mean 0.0315
  best-final gap: success 30.86, precision 45.38
```

结果解读：

- 当时只修复了 validity mask，没有修复 nonzero candidate 的共享坐标；因此 A1 的 precision-positive 和 A2 的后期崩坏都不能归因给 TWC。
- 后续必须先用 corrected sampler 复验 candidate 1/2/3、dataset length 和 `coordinate_anchor`，再讨论权重、warmup 或与 dynamics 的组合。
- `A3-order-gate-safe` 相比旧 P5 full 安全很多，但仍低于 A2-order-dyn。它说明强融合问题被缓解，却还没有证明 gate 能带来最终收益。
- `A3-order-conf-res-gate` 出现非常高的旧 best checkpoint，但 last checkpoint 崩坏。后续 2026-07-08 的 best-e14 复测只有 28.06 / 37.70，旧 best 暂时不能作为确认收益。

当时决策：

```text
当时暂以 A2-order-dyn 作为主线配置。
当时把 TWC 视为 A1 上的 precision-positive 候选；2026-07-11 坐标审计后该判断已撤回。
gate-safe / conf-res 暂时都不能作为最终收益模块；conf-res 先复测 best checkpoint。
```

注意：2026-07-08 五次复核后，`A2-order-dyn` 已从“当前主配置”降级为“最值得诊断的真实时间使用方式”。后续优先级转向 variable-rate 协议和 residual dynamics。

图表和完整表格：

```text
compare_results/twc_gate_ablation_curves.png
compare_results/twc_gate_ablation_success_curve.png
compare_results/twc_gate_ablation_precision_curve.png
compare_results/twc_gate_ablation_best_final_summary.png
compare_results/twc_gate_ablation_twc_diagnostics.png
compare_results/twc_gate_ablation_gate_diagnostics.png
compare_results/twc_gate_ablation_metrics_points.csv
compare_results/twc_gate_ablation_metrics_summary.csv
compare_results/twc_gate_ablation_diagnostics_summary.csv
compare_results/twc_gate_ablation_comparison.md
```

## 9. 第九轮：2026-07-08 五次稳定性复核

比较：

```text
A3-conf-res best-e14 retest
A2-order-dyn seed43
A2-order-dyn seed44
A2-order-dyn+TWC w0.01 seed42
A3-conf-res rerun seed42
```

关键结果：

```text
A3-conf-res best-e14 retest:        success 28.06, precision 37.70
A2-order-dyn seed43 final:          success 23.64, precision 23.77
A2-order-dyn seed44 final:          success 46.90, precision 52.62
A2-order-dyn+TWC w0.01 final:       success 22.88, precision 24.27
A3-conf-res rerun seed42 final:     success 32.11, precision 31.87
```

关键诊断：

```text
A2-order-dyn+TWC w0.01:
  loss_twc tail1000 mean 0.0083
  twc_valid_ratio tail1000 mean 0.7541
  twc_center_gap tail1000 mean 0.1748
  twc_angle_gap tail1000 mean 0.0103

A3-conf-res rerun:
  obs_alpha_dyn_mean tail1000 mean 0.4988
  obs_alpha_dyn_clamped_mean tail1000 mean 0.1810
  obs_dyn_residual_norm tail1000 mean 0.0314
```

说明：

- `A3-conf-res best-e14 retest` 没有复现旧汇总里的 62.04 / 76.30，高 best 暂时应视为未确认信号。
- `A2-order-dyn` seed43 / seed44 差异很大，说明旧 seed42 的 precision-positive 结果不能直接当成稳定结论。
- `A2-order-dyn+TWC w0.01` 当时只有 validity mask 正常；后续坐标审计证明它仍受 nonzero candidate 坐标污染，不能据此归因权重或 TWC 机制。
- `A3-conf-res rerun` 仍低，gate/conf-res 现在应先转向评测路径、alpha/residual 行为和困难分桶诊断。

归档文件：

```text
compare_results/reports/latest_5runs_comparison.md
compare_results/data/latest_5runs_metrics_points.csv
compare_results/data/latest_5runs_metrics_summary.csv
compare_results/data/latest_5runs_diagnostics_summary.csv
compare_results/data/latest_5runs_hparams_summary.csv
compare_results/figures/line_charts/latest_5runs_success_curve.svg
compare_results/figures/line_charts/latest_5runs_precision_curve.svg
compare_results/figures/bar_charts/latest_5runs_best_final_summary.svg
compare_results/figures/diagnostics/latest_5runs_diagnostics_tail_mean.svg
```

## 10. 2026-07-16：corrected-TWC、HTV 与 TrajTrack 参考

### 10.1 Corrected-TWC seed42

| family | baseline final | corrected-TWC final | final delta | late-mean delta |
| --- | --- | --- | --- | --- |
| A1 | 51.23 / 57.86 | 52.72 / 62.89 | +1.49 / +5.03 | +0.99 / +2.67 |
| A2 | 50.96 / 63.31 | 50.04 / 61.25 | -0.93 / -2.07 | -1.33 / -2.53 |

两组 corrected run 均完成 60 epoch、75720 optimizer steps，TWC anchor gap max 和 current XYZ gap max 都为 0。旧 baseline 没有 commit 记录，所以本节当时只能支持 A1 进入同提交 A/B/C；该控制现已于 2026-07-21 完成，最终结论以 10.9 节为准，不再补 seed43/44。

完整报告：`compare_results/reports/corrected_twc_seed42_comparison.md`。

### 10.2 HTV 六组 seed42

| protocol | A2-A1 Success final | A2-A1 Precision final | 解释 |
| --- | ---: | ---: | --- |
| gap1124 | -4.01 | -9.55 | epoch10 曾有高点，后期明显崩落 |
| burst-drop | -7.45 | -14.40 | 强不规则条件下稳定低于 A1 |
| random20 | +9.09 | +14.23 | 温和随机丢帧下 final/late mean 都为正 |

六组都是确定性 seed42 配对，但没有冻结 manifest，且只在 mini_val 上评估。它们适合筛选假设，不适合统计性结论。最重要的机制问题不是继续扩大 feature concat，而是检查：nonzero candidate 是否制造伪速度、目标是否离开 search crop、dynamics proposal 何时优于 observation proposal。

完整报告：`compare_results/reports/htv_6runs_comparison.md`。

### 10.3 TrajTrack evaluator 边界

aligned seed42 运行预算与 plain SeqTrack3D 基本一致，但 evaluator 并不等价：TrajTrack 当前 `pre_w_refine()` 读取当前帧 GT，先用 GT overlap 判断是否 refinement，再用 GT overlap 从 proposals 中选最大者。因此 64.94 / 79.07 是 GT-assisted 参考，不是公平在线结果。

下一步应固定 epoch60 checkpoint，分别评测：

1. `pre_wo_refine()`：GT-free local proposal baseline。
2. paper-aligned GT-free refinement：只用 local/global proposal IoU、预测置信度、点数等推理时可得量。
3. 当前 `pre_w_refine()`：只保留为 oracle upper-bound 诊断。

完整报告：`compare_results/reports/trajtrack_gt_assisted_vs_plain_seqtrack_reference.md`。

### 10.4 P0-A / P0-B 机制诊断

| diagnostic | base/current | alternative | 当前判断 |
| --- | ---: | ---: | --- |
| standard center outside | base 15.97% | CV recenter 0.12% | crop 前瓶颈明确存在 |
| standard mean target-point recall | base 85.41% | CV recenter 99.95% | 移动中心比扩大 crop 更合理 |
| standard crop points mean | base 285 | expanded 1622 / CV 290 | expanded 背景代价过大 |
| gap1124 mean target-point recall | base 76.78% | expanded 89.08% / CV 98.96% | 2x 扩区仍不足，CV oracle 强 |
| burst-drop mean target-point recall | base 77.72% | expanded 87.65% / CV 99.05% | 结论与 gap1124 一致 |
| strong-protocol crop points mean | base 312/331 | expanded 1650/1719 / CV 286/290 | CV 收益不是来自更多背景点 |
| observation error | P50 0.213 m | current cap 0.02 m | 2 cm 覆盖不了主要误差 |
| actual residual | P50 7.25e-8 m | alpha 2e-5 | 默认 gate 近乎关闭 |
| gate gradient | P50 4.00e-10 | 31/64 batch 为 0 | 当前初始化难以有效学习 |

standard P0-B 使用前一帧 GT 框，是对在线 tracker 乐观的 oracle。P0-B2 已补齐 A1 recursive predicted history，并否定 raw CV 恒开启接入；P0-A 的误差又混入 out-of-crop failure，所以仍不能在当前统计上直接调大 bound。

完整报告与派生表：

- `compare_results/reports/p0_ab_diagnostics_20260717.md`
- `compare_results/data/p0_ab_diagnostics_20260717_summary.csv`

### 10.5 P0-B2 recursive predicted-history 诊断

| diagnostic | previous-A1 | pred-history CV | 判断 |
| --- | ---: | ---: | --- |
| standard overall recall | 69.69% | 72.61% | +2.91 pp，安全但低于 +5 pp 门槛 |
| gap1124 overall recall | 63.73% | 66.38% | +2.65 pp，No-Go |
| burst-drop overall recall | 63.24% | 66.27% | +3.03 pp，No-Go |
| gap1124 `>4 m` recall | 1.68% | 10.13% | +8.45 pp，低于 +10 pp |
| burst-drop `>4 m` recall | 1.81% | 11.76% | +9.96 pp，接近但未达到门槛 |
| reliable history (`prev error <=4 m`) | 93.56%–94.76% | 97.34%–98.64% | 时间外推在漂移前有效 |
| drifted history (`prev error >4 m`) | 0.28%–0.93% | 0.80%–1.61% | 漂移后两锚点近乎同时失效 |

三协议 checkpoint SHA256 相同，reference endpoints exact match，missing/unexpected 均为 0。A1 previous prediction error P95 为 78.33/89.25/94.84 m，说明平均 recall 背后存在灾难性递归长尾。结论不是放弃真实时间，而是把 trajectory proposal 从唯一重定心 anchor 改为由测试时可靠性控制的第二搜索假设。完整报告与派生表：

- `compare_results/reports/p0b2_recursive_crop_reachability_20260717.md`
- `compare_results/data/p0b2_recursive_crop_reachability_20260717_summary.csv`

### 10.6 P0-B3 reliability 与 passive complementarity

| diagnostic | standard | gap1124 | burst-drop | 判断 |
| --- | ---: | ---: | ---: | --- |
| all-13 trigger AUROC | 0.857 | 0.787 | 0.785 | 通过原预注册 threshold |
| all-13 trigger AUPRC | 0.742 | 0.660 | 0.671 | 高于 prevalence 0.459/0.311/0.313 |
| observation-only trigger AUROC | 0.853 | 0.867 | 0.873 | 跨协议更强，主要 reliability 信号来自上一 observation |
| no-raw-dt trigger AUROC | 0.857 | 0.865 | 0.872 | raw current delta_t 是 OOD 退化来源 |
| raw-CV dual recall gain | +3.04 pp | +2.88 pp | +3.15 pp | 强协议均低于 +5 pp，anchor No-Go |
| selector AUROC | 0.729 | 0.605 | 0.433 | post-crop 选择规则不泛化 |

数据完整性、分组、标签边界和 headline 指标均通过复核。结构性缺失只出现在上一 observation empty fallback 行，并由显式 indicator 与 fold 内 imputation 处理。当前服务器运行记录为 dirty `f28f495`，所以可用于机制决策，但正式复现必须冻结 clean commit 与 diagnostic script hash。

完整验证与 feature ablation：

- `compare_results/reports/p0b3_reliability_validation_20260720.md`
- `compare_results/data/p0b3_reliability_feature_ablation_20260720.csv`

### 10.7 P0-B4 independent observation reliability

| protocol | N | prevalence | AUROC | AUPRC-prev | Brier / prevalence baseline | ECE | recall | FPR | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1623 | 0.111 | 0.794 | 0.414 | 0.0795 / 0.0986 | 0.073 | 0.711 | 0.220 | report-only |
| gap1124 | 829 | 0.089 | 0.680 | 0.282 | 0.0859 / 0.0813 | 0.093 | 0.568 | 0.258 | Fail AUROC/recall |
| burst_drop | 815 | 0.085 | 0.712 | 0.328 | 0.0780 / 0.0775 | 0.089 | 0.609 | 0.248 | Fail AUROC/recall |

四份输入 CSV hash、reference exact-match、checkpoint hash、fit/eval tracklet disjoint 和标签过滤均通过检查。本地验证器独立重跑得到相同 calibrator、指标和 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`，只有约 `1e-15` 浮点差。

同批 raw-CV passive crop 在 standard/gap/burst 的 union gain 为 `+0.06/0.00/0.00 pp`；强协议没有 trajectory-only endpoint。当前 calibrator 和 raw-CV candidate 均不进入 frozen-state/active anchor。

完整报告：`compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。

### 10.8 P0-C frozen protocol 与同 checkpoint time-control 判定

clean commit `343145d` 已在 nuScenes mini_val 上生成 role-specific gap1124 manifest 与 offline shuffled-dt mapping。val/test 均从 `106 tracklets / 2285 frames` 冻结为 `91 / 1257`，drop ratio 为 `0.449891`；test selection SHA256 为 `85e5603c...f9649f6f`。shuffled mapping 含 `1257 endpoints / 1166 transitions`，满足 `1257 - 1166 = 91 tracklets`。

真实 batch 中三路 `delta_t_real` 完全一致；true effective time 与 real exact match，fixed 为 `[0.5, 0.5, 0.5]`，shuffled 为冻结映射后的不同 gap，最终得到：

```text
P0-C true/fixed/shuffled batch invariance: PASS
```

同一个 standard-trained A2 seed42 60ep final checkpoint 随后完成三路冻结评测：

| mode | Success | Precision | true − mode Success | true − mode Precision |
| --- | ---: | ---: | ---: | ---: |
| true | 55.2247 | 66.8854 | 0.0000 | 0.0000 |
| fixed | 54.7872 | 66.3624 | +0.4375 | +0.5231 |
| shuffled | 55.3480 | 66.8298 | -0.1233 | +0.0557 |

三份 provenance 的 commit `343145d`、checkpoint `b508f958...24ac87ad`、source config、seed42、selection hash、`91 tracklets / 1257 frames` 一致，resolved config 只在时间控制与 log_dir 上不同；console 与 TensorBoard events exact match。true 没有同时优于两个对照，最小差值为 `-0.1233 Success / +0.0557 Precision`，未达到 `+0.5 / +1.0`。正式判定：

```text
NO_GO_P0C_A2_TRUE_DT_PROMOTION
```

这说明当前 frozen feature-concat A2 对时间输入有响应，但正确 physical-time 对应关系没有稳定收益；不扩展 burst/fixed-gap/multiseed。完整报告见 `compare_results/reports/p0c_frozen_protocol_validation_20260720.md`。

### 10.9 TWC A/B/C 同提交 seed42

2026-07-21 已完成同提交 A/B/C 的本地事件复算、checkpoint hash、配置差异和诊断完整性审计。B/C resolved config 除运行元数据外只差 `twc_weight: 0.0 -> 0.05`；三组均有 12 个同步评测点，final 为 step75720。结果表明 `C-B` 在 Final/Best/Late mean 上全部为正，但 `B-A` 的损害更大，使 `C-A` 仍全面为负。该结果结束了“旧 baseline 不同提交”的归因问题，同时触发 TWC 主方法 promotion No-Go。详见 `compare_results/reports/twc_abc_seed42_comparison_20260721.md`。

### 10.10 M0 P0-C-D1 gap1124 三路 full

2026-07-21 回传包已通过 SHA256 校验并解包到 `output/diagnostics/m0_manual_20260721_v2/`。true/fixed/shuffled 各有 `1257` 个 endpoint、`91` 个 tracklet、`102` 个字段；复合键无重复，三路 endpoint key/order、真实时间、GT、checkpoint、source config、selection 和 manifest exact match。true effective time 与 real time 完全相等，fixed 恒为 `0.5 s`，shuffled 的 effective gap 多重集合与 true real gap 完全相同但 endpoint 映射被离线冻结置换。三路 exporter/summarizer summary 与 CSV round-trip 重算完全一致。

| mode | Success | Precision | mean center error | fallback |
| --- | ---: | ---: | ---: | ---: |
| true | 55.2247 | 66.8775 | 2.2704 m | 94 |
| fixed | 54.7872 | 66.3544 | 2.4614 m | 106 |
| shuffled | 55.3481 | 66.8218 | 2.2496 m | 92 |

配对结果为 true−fixed `+0.4376 Success / +0.5231 Precision`，true−shuffled `-0.1233/+0.0557`。两组 Success/Precision 的逐 tracklet bootstrap 95% CI 均跨 0，且没有同时达到预注册的 `+0.5/+1.0`。true 相对两个控制各有 `1079/1257` 个 endpoint 的预测中心改变，证明模型会响应 time input；但正确物理时间对应关系没有比 shuffled 更可靠。

这里的 Precision 使用 CSV round-trip float 解析，可与 exporter/summarizer 精确复现；pandas 默认解析会在极少数阈值边界上带来约 `+0.008 pp`，解释了第 10.8 节旧 aggregate 表中 `66.8854/66.3624/66.8298` 与本次 `66.8775/66.3544/66.8218` 的细微差异，不改变任何 paired delta 或 verdict。

分桶没有发现隐藏的 long-gap promotion：在 `≥2 s` 的 100 个 transition 上，true−shuffled 只有 `0.000 Success / +0.525 Precision`；所有 real-gap 桶中 true 对 shuffled 的 Success 都没有正优势。GT 位移 `≥0.5 m` 的四个桶中，true 相对两个控制的 Success 差均为 0，说明当前机制没有恢复高运动 endpoint。

true−fixed 的 overall mean error 虽改善 `0.191 m`，但主要由 tracklet `0cfdfb5bbe8a41268271e24f2edefb9c` 驱动；该序列三路从首个预测帧起 IoU 都为 0，只是 fixed 后期漂得更远。移除它后 mean-error 改善缩小到 `0.0397 m`，而 Success/Precision 判定不变。因此不能把长尾减轻写成成功恢复或 promotion。

正式结论仍为：

```text
NO_GO_P0C_A2_TRUE_DT_PROMOTION
```

本次 D1 完成了旧 aggregate 判定的 endpoint/tracklet 失败定位，不解锁当前 feature-concat A2 的 burst/fixed-gap/multiseed，也不解锁 M1 正式训练或 M2。服务器 provenance 为 dirty，但 exact exporter/config/checkpoint/manifest/CSV hash 已保存，且 paired 效应复现此前 clean aggregate；足以完成冻结诊断，正式论文归档仍保留 clean-worktree caveat。完整报告见 `compare_results/reports/m0_p0c_d1_full_analysis_20260721.md`，可执行复核见 `compare_results/notebooks/m0_p0c_d1_full_analysis_20260721.ipynb`。

### 10.11 M0-3 gap1124 proposal oracle

2026-07-21 正式输出来自 clean commit `1357923...`、seed42、mini_train。`11,424` 行经 nonresampled/full-history/crop-reachable/dynamics-valid/candidate0 筛选后得到 `1,311 endpoints / 213 tracklets`。`d_obs/d_dyn/oracle` mean error 分别为 `1.349/0.309/0.232 m`，oracle gain mean/median 为 `1.118/0.214 m`，去除 gain 最大 5% 后 mean 仍为 `0.816 m`。

由于 oracle gain 按构造非负，又独立计算了非 oracle 对照：`d_dyn` 相对 `d_obs` 的 mean/median gain 为 `1.040/0.175 m`，`81.31%` endpoint 更优；以 tracklet 为 bootstrap 单位，mean gain `0.803 m`、95% CI `[0.633,0.988]`，`87.32%` tracklet 为正。long-gap `417 endpoints / 133 tracklets` 的 oracle gain tracklet bootstrap mean `0.717 m`、CI `[0.493,0.967]`。因此正式决定为 `GO_M2_PROPOSAL_INNOVATION`。

该 Go 只说明 crop-reachable offline proposal 有互补空间，不是 tracking 指标涨点。primary cohort 使用 GT history、candidate0 并排除 18.24% resampled rows；sparse `target_points<=5` 仅 3 个样本，不能作 sparse claim。正式 M2 只能新增 `d_dyn-stopgrad(d_obs)` 的 strict-zero bounded innovation mode，旧 full-displacement residual 仅作负对照；旧 `0.1×0.2=0.02` 有效线段上限不应直接继承，并须重新通过 seed42 true/fixed/shuffled 与 standard guardrail。

### 10.12 M0-4 candidate dynamics audit

正式输出共 `20,204` 行，full-history usable `13,934`，四个 candidate 数量为 `3484/3481/3482/3487`。candidate0 jitter 精确为 0；candidate1/2/3 的 velocity/acceleration jitter P50 为 `0.611 m/s`、`2.128 m/s²`，分别是预注册阈值的 `12.22×/21.28×`，确认逐历史帧独立 candidate offset 会制造强伪导数。

按相同 endpoint 与 candidate0 配对后有 `8,515` 个比较、`2,904` 个 endpoint key、`235` 个 tracklet。非零 candidates 的 proposal error delta mean/median 为 `+0.0104/+0.0033 m`，tracklet bootstrap 95% CI `[+0.0093,+0.0155] m`，81.70% tracklet 为正。负效应稳定，但绝对均值约 1 cm，不能解释旧 A2 的全部失败。

正式决定为 `FREEZE_M1_SHARED_SE2`：M1 对同一样本全部历史框使用一个 shared SE(2) 变换，Dynamics label 从 canonical/一致变换轨迹计算；第一版不实现 smooth drift，也不允许依据后续 tracking 涨跌反向改判。完整分析见 `compare_results/reports/m0_m03_m04_analysis_20260721.md`。

代码复核进一步确认：`getOffsetBB` 在每个框自身局部朝向中解释平移，因此不能用“对全部历史框重复同一 offset 数组”冒充共同刚体变换；M1 必须围绕共同 anchor 实现 world-SE(2)。M0-4 当时将状态推进为 `Engineering GO / Formal-training HOLD`；后续 E0–E5 结果见第 10.13 节。

### 10.13 M1/M2 E0–E5 服务器硬门禁

2026-07-22，commit `9a0b26d` 使用固定 A1 checkpoint SHA256 `a2fbff...a82` 在 GPU2/3 完成五组工程 gate。三个配置 hash 与本地 commit 完全一致；A1 的 `320` 个 checkpoint tensor 全部匹配，新模型缺失的 `14` 个 tensor 均属于按设计新建的 DynamicsEncoder/physical-time adapter，unexpected key 为 0。shared world-SE(2) dataset-free、真实 loader/TWC 与 strict-zero A1 model equivalence 均 PASS。

| run | batch / optimizer | 样本 | invalid / empty / resampled | applied / clamp | bound violation max | grad max（encoder / adapter） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 2` | `0` | `0.0445 / 1.1858` |
| standard warmup | `2 / 2` | 4 | `0 / 1 / 0` | `0 / 2` | `0` | `0.0392 / 0` |
| standard fallback | `53 / 0` | 106 | `8 / 16 / 2` | `86 / 10` | `5.96e-8` | `0.2145 / 0.3027` |
| gap1124 active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 3` | `5.96e-8` | `0.1267 / 3.8867` |
| burst-drop active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 3` | `5.96e-8` | `0.0515 / 2.6593` |

五个 JSONL 共 `61` 个 batch、`122` 个样本，所有 loss/gradient finite；本地独立重算的 step、sample、fallback、applied/clamp、optimizer 和 bound 统计与 summary 完全一致。warmup 内执行两次 optimizer update 后 adapter/innovation output 与 effective scale 仍精确为 0，而 DynamicsEncoder 两步梯度非零。fallback 的 invalid `8`、empty `16` applied max 都精确为 0；第 53 batch 首次出现 `2` 个 sampler-resampled 样本并保持 finite。resampled 是替换成另一条有效训练样本，不是模型 strict-zero 条件。

正式决定为：

```text
PASS_M1_M2_E0_E5_ENGINEERING_GATES
HOLD_FORMAL_TRAINING_PENDING_E6
```

这只证明公式、数据路径、回退和优化器机制可安全运行；不能证明 tracking 指标上涨，也不能用随机初始化 DynamicsEncoder 下的 clamp ratio 调整 alpha/半径。完整复核见 `compare_results/reports/m1_m2_e0_e5_validation_20260722.md`。

### 10.14 M2 R1/R2/R3 standard 结果

三组训练的 checkpoint、event、provenance 与 config 已完成统一复核，均可进入比较。冻结 epoch60 `last.ckpt` 口径下：

- R1 A1-init M2：`55.303/67.182`，相对历史 A1 `+4.074/+9.318`。
- R2 scratch M2：`53.318/62.503`，相对历史 A1 `+2.090/+4.640`。
- R3 scratch W0：`28.999/28.023`；R2−R3 为 `+24.319/+34.480`。
- 后 5 个评测点均值仍保持 R1 > R2 > A1 >> R3，正差不是最后一次评测独有。

正式解释是 `M2 STANDARD SIGNAL POSITIVE / METHOD ATTRIBUTION AND CAUSAL-TIME HOLD`。R1−A1 混有额外 60 epoch，R2−R3 则受到 shared-SE(2) W0 collapse 的强交互影响；当前不把任一差值写成 M2 相对 SeqTrack3D 的因果净收益。

## 11. 当前各实验共同说明了什么

可以支持的结论：

- P0-C 与 M2 R1 两轮独立控制都显示正确 `delta_t` alignment 没有超过 fixed/shuffled；当前 physical-time method claim 已 No-Go，不再通过追加 seed 或复杂时间模块补救。
- SeqTrack3D 主干对原始 order-time token 很敏感，直接替换为 real-time token 会破坏已学到的时间/顺序语义。
- feature-concat A2 仍是失败消融：它在强 gap/burst 下不稳且 P0-C promotion No-Go；但 M0-3 表明冻结 DynamicsEncoder proposal 在严格 offline cohort 中有明显信息，因此保留它作为 M2 proposal source，不复活旧 feature concat 或 hand-crafted Gate。
- 当前 `cand1` 结果不支持简单移除非 0 candidate；multi-candidate 训练暂时应保留。
- M1/M2 的 shared-SE(2)、canonical label、strict fallback、warmup、受界 innovation 和 optimizer 路径已通过 E0–E6；R1 formal 在 standard/gap1124 相对 matched A1 都有正信号，逐 tracklet bootstrap 的 Success/Precision CI 均为正。
- R1 同 checkpoint 时间控制确实改变预测和 innovation 半径，但 true 没有优于 fixed/shuffled；这排除了“时间分支完全没工作”，也否定了“正确物理时间造成涨点”的当前解释。
- R2 能从 scratch 达到 `53.318/62.503`，说明 full M2 不依赖 A1 初始化才可工作；R1 的更高结果说明 A1 初始化提供了有利起点，但还不能排除 extra continuation。
- R3 在 shared-SE(2) 下严重塌陷，说明 M1 数据定义与 M2 结构存在强交互；R2−R3 不能被当成标准 SeqTrack3D baseline 差值。
- 小权重 displacement 辅助监督不伤主线，并给 precision 带来温和正向信号，但不是主要收益来源。
- 旧 TWC 只有 validity mask 生效，坐标共享仍有缺陷；旧 A1 正向和 A2 负向信号均已撤回。
- corrected-TWC 的共享 offset、`coordinate_anchor` fail-fast 和 optimizer-step 对齐已实现；同提交 A/B/C 已证明 `C-B` 为正，但 C 仍显著低于 single-view A，主方法 promotion No-Go。
- P5 full 旧结果不能作为最终 gate 结论；gate-safe 比旧 P5 full 安全，但仍低于 A2-order-dyn。
- conf-res 旧 best checkpoint 未被最新 best-e14 复测确认；当前不能按旧 best 写正向收益。
- corrected-TWC 不再补训练 seed；只允许冻结 A/B/C final checkpoint 做一次 strong-cadence 与 evaluation-only path-variance 收尾，gate 仍不与 residual 同时启用。
- TrajTrack 当前高分含 GT oracle；它只能提示 trajectory proposal 的潜力，不能证明公平收益。
- standard/gap1124/burst-drop oracle 均证明高速位移下 search crop 本身会丢目标；recursive 诊断进一步说明 raw predicted CV 只能在历史可靠时改善，不能从已漂移状态独立恢复。
- P0-B3 开发集曾显示 previous-observation quality 可以预测可见目标的下一 crop miss，但 P0-B4 独立验证未达到预注册排序和运行点召回门槛；该信号只能保留为开发集诊断，不能升级为方法贡献。
- raw-CV 第二 crop 与 observation 失败高度重叠，mini_val 强协议的 trajectory-only endpoint 为 0；当前 calibrator、candidate 和 selector 均不进入 active tracker。

还不能说明的事情：

- 还不能说完整 CT-SeqTrack 已经稳定超过 SeqTrack3D。
- 还不能说 TWC 已稳定有效或能与 dynamics 组合；当前只确认 A1 paired control 内 `C-B` 的单 seed 净效应，端到端 `C-A` 仍明显为负。
- 还不能说 gate 有效；gate-safe final 不够好，conf-res best 复测未确认，但仍可做困难样本诊断。
- 64-batch residual 分桶显示 candidate0 的 observation error 中位数略低，但四个 candidate 都有大长尾；这不足以彻底解释 candidate noise，也不支持简单移除 nonzero candidate。
- active dual-anchor 已在预注册入口处停止，不应再列为待补性能结果；M0-3 与 R1 formal 已确认 proposal 有 aggregate 互补性，但 long-displacement crop recovery 与 burst robustness 仍未建立。
- R1 相对 matched A1 出现 tracking 正信号，但由于训练预算和 augmentation 不完全匹配，仍不能写成 M2 结构的因果净增益；同 checkpoint controls 已明确回答 true-dt 不优于 fixed/shuffled。
- 还不能说 burst-drop 会给 physical-time 更明显的正结果；gap1124 已提供高 `dt` 变异且 causal gate 失败，burst 只能作为通用 proposal/crop robustness 的条件性补充。
- 还不能说 displacement loss 是必要模块，因为当前只是小幅、不决定性的正向信号。
- 还不能说 physical `delta_t` 提高了 reliability prediction；raw `current_delta_t` 在当前 standard-only calibrator 中反而造成强协议过触发。
- 不能把 observation-only trigger 写成最终 uncertainty model；P0-B4 独立 mini_val 已 No-Go，且标签只覆盖 target visible 条件下的 crop miss。

## 12. 接下来应该做什么

当前优先顺序：

```text
1. P0-C frozen triplet 与 TWC A/B/C standard seed42 均已完整归档；两条方法 promotion 都为 No-Go，不扩展训练 seed。
2. P0-C-D1 的 long-gap、首次失控、连续失败和 empty fallback 定位已完成；并行复用同一 endpoint/per-tracklet logger，对冻结 A/B/C final checkpoint 做 standard/gap1124/burst-drop/unseen-fixed-gap 与 path-variance 收尾，不改模型或 checkpoint。M0-2 不阻塞写代码，但阻塞 M0 完成。
3. R1/R2/R3 以及 R1 standard/gap1124 八组 controls 已完成；冻结 `tracking positive / physical-time No-Go / attribution Hold`，不再用 best epoch或修改 alpha/R 重解释。
4. 固定 R1 权重完成 full/adapter-only/innovation-only/both-off 运行时 2×2，并审计 `d_obs/d_dyn/d_final`、candidate-frame/canonical target、双 loss 梯度与 recursive error process。
5. 补 A1-init W0 continuation，排除 R1 的额外 60 epoch；补 current-code scratch legacy-candidate W0，解释 R3 的 shared-SE(2) collapse。
6. burst-drop 只在通用 proposal 归因仍成立时补作 robustness/crop 证据；不补 physical-time seed，不启动 timestamp-conditioned M3/M4。
7. 若 matched attribution 也不支持结构净增益，则转 variable-rate benchmark/diagnosis；若支持，则围绕 time-agnostic bounded proposal correction 重新预注册，而不是恢复已失败的时间主张。
```

可选复核：

```text
A2-order-dyn-cand1-240ep
```

作用是让 `num_candidates=1` 的 cand1 与 `num_candidates=4` 的 A2-order-dyn 做 optimizer-step 对齐。只有这个版本也退化，才能更干净地说明移除非 0 candidate 本身有问题。

### 12.1 A1-order+TWC

目的：

```text
先在没有 dynamics 的 order-time 主干上检查 TWC 是否有效。
```

看什么：

- 旧 run 虽然 `twc_valid_ratio > 0` 且 final precision +3.24，但坐标共享缺陷使该增益无法归因。
- 先用 corrected sampler、candidate 1/2/3 和相同 optimizer steps 重跑原权重；不要先做权重网格。
- 同时直接报告同一 endpoint 下不同采样路径的预测方差是否下降。

### 12.2 A2-order-dyn+TWC

目的：

```text
检查真实时间 dynamics prior 和 TWC 是否互补。
```

看什么：

- 旧 active / w0.01 退化都受坐标污染，不能判断 TWC 与 dynamics 是否互补。
- corrected A1+TWC 没有形成稳定证据前，不重跑 A2+TWC，也不与 gate/residual 混合。

### 12.2b A2-residual-dyn

目的：

```text
检查真实时间 dynamics prior 是否能以更保守的 residual correction 形式稳定发挥作用。
```

当前实现：

- 主干保持 `A1-order`，不要把 raw real-time token 放回主干。
- `DynamicsEncoder` 继续预测 `velocity_pred / dynamics_displacement_pred`。
- 最终中心预测采用小幅 residual：`center = obs_center + scale * alpha * clamp(dyn_disp)`。
- 已支持 `scale / max_norm / max_alpha / warmup / long_gap_only / sparse_only`，gate 近零初始化且受 `dynamics_valid` 约束。
- standard 真实 batch 已证明默认 `scale=0.1, max_norm=1.0, max_alpha=0.2, warmup=5` 数值稳定但功能上近乎关闭：实际 residual P50 仅 `7.25e-8 m`。
- 不在混入 out-of-crop error 的统计上调大 bound；先解决 pre-crop reachability，再在 reachable subset 预注册 gate init/scale/bound。
- 与当前 `A2-order-dyn` feature-concat 版本做同 seed 对照。

判断标准：

- P0-B4 已停止 reliability-aware active dual-anchor；末端 residual 现在只允许在 crop-reachable mini_train subset 做一次预注册 kill-test，不能自动升级为正式方法。
- 如果普通 final 只持平，但 long-gap / sparse bin 稳定提升，可以作为更强论文证据。
- 如果 reachable subset 中 residual 仍长期近零，应停止该 gate/bound 设计，而不是继续扩大网络。

### 12.3 A3-order-gate-safe / conf-res

目的：

```text
在干净的 order-time 主干上重新测试保守 gate。
```

看什么：

- gate-safe 比旧 P5 full 稳，但 final 仍低于 A2-order-dyn。
- conf-res best-e14 复测没有复现旧 best；先核对旧汇总路径，再回到更保守的 alpha / residual 约束。
- 分桶分析仍有价值，但它现在用于解释 gate 行为，而不是证明 gate 已经有效。

## 13. 后续需要补的诊断

建议增加 dynamics 诊断日志：

```text
candidate_id
candidate0_loss_velocity
candidate_nonzero_loss_velocity
velocity_label_norm
velocity_pred_norm
dynamics_displacement_norm
dynamics_valid_ratio
```

作用：

- 直接判断非 0 candidate 是否让 dynamics 监督变脏。
- 判断 velocity prediction 是否爆炸、塌缩或量级不匹配。
- 给 cand1 / disp 的结果提供机制解释，而不是只看 success 和 precision。

## 14. 当前论文叙事建议

暂时不要写：

```text
CT-SeqTrack full model outperforms SeqTrack3D.
```

当前更稳的写法：

```text
We study whether 3D single-object trackers remain robust under within-track
irregular observation schedules. Matched true/fixed/shuffled-time controls
separate sensitivity to a time input from benefits of physically aligned time,
while endpoint-consistent history resampling isolates path robustness from
crop and coordinate changes.
```

中文主线：

```text
真实 timestamp 改变历史状态的物理含义，但当前证据只支持先把 variable-rate
问题和 matched time controls 做清楚。同提交 A/B/C 只确认 TWC 对受损 paired-view
路径的部分修复，不能作为超过 A1 的贡献，而且它不读取真实 delta_t；只有新的显式时间机制
在 true/fixed/shuffled 中形成因果正信号，论文才能恢复 timestamp-native 方法主张。
```

论文可行性、claim 审计与方法/benchmark 分叉见 `compare_results/reports/paper_viability_and_execution_20260720.md`。
## 2026-07-30 Δt-PFTC seed42 部分运行

第四模块的首个 formal-named artifact 并未完成 60 epoch。训练 events 止于
step29,091（共 29,092 step，约 epoch23.05），只验证到 epoch20，`last.ckpt`
为 epoch19。epoch20 得到 `49.056 Success / 63.870 Precision`，相对 B0
epoch60 的 `53.360/64.382` 仍低 `4.304/0.512`；由于 early validation 波动大且
缺少后 40 epoch，不能判定最终涨点。

机制审计发现三项决定性问题：canonical yaw 使用了与项目约定相反的
`R(+yaw)`；foreground feature std 从 epoch1 的 `0.0947` 收缩到 epoch20 的
`0.0210`，而 match count/distance 稳定，说明 raw SmoothL1 有平凡收缩风险；
单卡训练为 `3.689 s/step`，约是 B0 的 10.2 倍。weighted/raw PFTC loss 差异
中位数又只有 0.074%，尚无 physical-time 增量证据。

当前决策为 `NO-GO_CURRENT_IMPLEMENTATION / INCONCLUSIVE_IDEA`。旧 epoch19
checkpoint 不续训；修正几何、防坍缩和性能以后，先做 B0/PFTC-U/Δt-PFTC
各 5 epoch 机制 kill-test，通过后才重启正式 60-epoch 三臂。完整报告见
`compare_results/reports/pftc_b4_seed42_partial_diagnosis_20260730.md`。
