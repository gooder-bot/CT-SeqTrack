# M2 standard / gap1124 同 checkpoint 控制分析

**日期：2026-07-24｜结论：M2 tracking 信号成立，但 physical-time 因果主张 No-Go；方法归因仍 Hold。**

## 执行摘要

- R1 M2 相对历史 A1 在 standard 为 **+4.133 Success / +9.445 Precision**，在 gap1124 为 **+2.279/+4.143**；逐 tracklet bootstrap 的两项主指标 95% CI 均为正。
- 同一 R1 checkpoint 改成 fixed/shuffled 时间后，standard 差异接近零；gap1124 中 shuffled 反而比 true 高 **+0.318 Success / +0.209 Precision**。
- 因此现有结果支持“R1 训练得到更强 tracker”的描述，但不支持“正确物理时间映射造成涨点”。R1 还包含额外 60 epoch continuation、shared-SE(2) 和 M2 联合改变，不能把 M2−A1 直接写成模块净增益。
- 按已冻结计划，timestamp-conditioned M3/M4 不解锁。下一步先做同 checkpoint adapter/innovation 2×2、A1-init W0 continuation、legacy-candidate W0 和 proposal 语义/递归误差审计。

## 数据与完整性

- 来源：`D:/desktop/research/CT-SeqTrack/server_results/m2_standard_gap8_473738f_20260723_235400`。
- 包内 artifact manifest：**89/89 PASS**。
- 8 个 run 均来自 clean commit `473738f`；M2/A1 checkpoint SHA256 与冻结合同一致。
- standard 为 106 tracklets / 2285 endpoints；gap1124 为 91 / 1257。每个协议内 endpoint key、顺序、GT 与真实时间字段完全配对。
- 首轮 reference 校验遗漏了每条轨迹的 GT 初始化帧；恢复时只从 validator reference 移除 106/91 个初始帧，8 份结果 CSV 仍完整保留初始帧。全量 CSV 的跨 run endpoint identity 复核为 exact。

## 八组原始指标复算

| run | Success | Precision | mean error | empty |
| --- | ---: | ---: | ---: | ---: |
| m2_standard_true_seed42 | 55.283 | 67.206 | 2.282 | 217 |
| m2_standard_fixed_seed42 | 55.253 | 67.216 | 2.286 | 217 |
| m2_standard_shuffled_seed42 | 55.216 | 67.120 | 2.290 | 217 |
| a1_standard_true_seed42 | 51.150 | 57.760 | 2.912 | 229 |
| m2_gap1124_true_seed42 | 59.063 | 70.857 | 2.101 | 113 |
| m2_gap1124_fixed_seed42 | 59.191 | 70.843 | 2.125 | 111 |
| m2_gap1124_shuffled_seed42 | 59.381 | 71.066 | 2.073 | 111 |
| a1_gap1124_true_seed42 | 56.784 | 66.714 | 2.249 | 107 |

指标由原始 CSV 中 21 点 Success/Precision 曲线独立积分复算，与服务器 `m0_summary.json` 最大绝对误差小于 `1e-10`。

## 冻结门槛

| gate | observed | decision |
| --- | ---: | ---: |
| standard guardrail: M2−A1 ≥ −0.5/−1.0 | +4.133/+9.445 | PASS |
| gap1124 complementarity: M2−A1 ≥ +1/+2 | +2.279/+4.143 | PASS |
| standard true−fixed/shuffled ≥ +0.5/+1 | min +0.031/−0.010 | FAIL |
| gap1124 true−fixed/shuffled ≥ +0.5/+1 | min −0.318/−0.209 | FAIL |
| M3/M4 timestamp-dependent promotion | requires causal-time gates | LOCKED |

## 配对差异与 tracklet bootstrap

| comparison | ΔSuccess | ΔPrecision | error gain | S CI low | S CI high | P CI low | P CI high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_M2_minus_A1 | 4.133 | 9.445 | 0.630 | 1.920 | 5.454 | 4.486 | 10.634 |
| standard_true_minus_fixed | 0.031 | -0.010 | 0.004 | -0.061 | 0.250 | -0.089 | 0.400 |
| standard_true_minus_shuffled | 0.068 | 0.085 | 0.008 | -0.013 | 0.283 | -0.005 | 0.457 |
| gap1124_M2_minus_A1 | 2.279 | 4.143 | 0.148 | 0.687 | 3.940 | 1.452 | 6.022 |
| gap1124_true_minus_fixed | -0.127 | 0.014 | 0.024 | -0.390 | 0.293 | -0.218 | 0.537 |
| gap1124_true_minus_shuffled | -0.318 | -0.209 | -0.028 | -0.782 | 0.333 | -0.876 | 0.680 |

表中 CI 是以 tracklet 为抽样单位、20,000 次独立 bootstrap 得到的 tracklet-mean delta 区间；aggregate delta 按 endpoint 加权，因此两者中心值允许不同。

### 方法信号

- standard M2−A1 的 tracklet-mean 95% CI：Success `[1.920, 5.454]`，Precision `[4.486, 10.634]`。
- gap1124 M2−A1 的 tracklet-mean 95% CI：Success `[0.687, 3.940]`，Precision `[1.452, 6.022]`。
- gap1124 的 M2 empty fallback 为 113，A1 为 107：整体指标虽提高，但空搜索并未改善，增益不是由单纯减少 empty fallback 解释。
- gap1124 与 standard 使用不同 endpoint population；两者绝对分数不能直接解释为 gap 更容易或模型在 gap 下反而更强，只能在各自协议内做 matched comparison。

### 时间敏感但不具备正确对齐优势

| comparison | changed >1cm | mean shift m | p95 shift m |
| --- | ---: | ---: | ---: |
| standard_true_minus_fixed | 0.204 | 0.051 | 0.102 |
| standard_true_minus_shuffled | 0.211 | 0.054 | 0.073 |
| gap1124_true_minus_fixed | 0.316 | 0.098 | 0.261 |
| gap1124_true_minus_shuffled | 0.325 | 0.087 | 0.431 |

时间负对照确实改变了递归预测，说明分支不是完全失活；但正确时间没有稳定优于 fixed/shuffled。尤其 gap1124 的 shuffled 在两个主指标上均高于 true，这否定当前 R1 的 physical-time causal promotion。

### M2 运行时路径并未失活

| run | effective dt CV | applied rate | clamp rate | applied norm m | radius m | adapter norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_standard_true_seed42 | 0.050 | 0.938 | 0.329 | 0.365 | 0.749 | 0.885 |
| m2_standard_fixed_seed42 | 0.000 | 0.937 | 0.327 | 0.365 | 0.750 | 0.883 |
| m2_standard_shuffled_seed42 | 0.050 | 0.937 | 0.329 | 0.364 | 0.749 | 0.883 |
| m2_gap1124_true_seed42 | 0.622 | 0.908 | 0.273 | 0.387 | 0.962 | 0.874 |
| m2_gap1124_fixed_seed42 | 0.000 | 0.908 | 0.336 | 0.351 | 0.750 | 0.842 |
| m2_gap1124_shuffled_seed42 | 0.622 | 0.908 | 0.270 | 0.376 | 0.960 | 0.861 |

gap1124 true 将平均 innovation radius 从 fixed 的 `0.750 m` 提高到 `0.962 m`，平均 applied norm 从 `0.351 m` 提高到 `0.387 m`，且约 90.8% 非初始 endpoint 实际应用 innovation。模型确实响应 effective time，但这种响应没有带来正确时间对齐优势。

## gap1124 实际时间分桶

| comparison | real dt | n | ΔSuccess | ΔPrecision | error gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| gap1124_M2_minus_A1 | 0.75-1.0 | 165 | 3.030 | 5.576 | 0.139 |
| gap1124_M2_minus_A1 | 1.0-2.0 | 257 | 1.518 | 4.037 | 0.124 |
| gap1124_M2_minus_A1 | <0.75 | 644 | 2.585 | 4.216 | 0.182 |
| gap1124_M2_minus_A1 | >=2.0 | 100 | 3.100 | 5.350 | 0.141 |
| gap1124_true_minus_shuffled | 0.75-1.0 | 165 | -0.182 | 0.045 | -0.030 |
| gap1124_true_minus_shuffled | 1.0-2.0 | 257 | -0.603 | -0.399 | -0.032 |
| gap1124_true_minus_shuffled | <0.75 | 644 | -0.311 | -0.233 | -0.030 |
| gap1124_true_minus_shuffled | >=2.0 | 100 | -0.150 | -0.175 | -0.027 |

M2−A1 在四个 real-dt 桶都为正，说明正信号并不只集中在长间隔；true−shuffled 在四个 real-dt 桶的 Success 全为负，`≥2 s` 也没有隐藏的 physical-time promotion。分桶只用于定位，不用于重新选择阈值。

## gap1124 GT 位移分桶

| comparison | GT displacement m | n | ΔSuccess | ΔPrecision | error gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| gap1124_M2_minus_A1 | 0.5-1.0 | 14 | 0.714 | 0.357 | 0.103 |
| gap1124_M2_minus_A1 | 1.0-2.0 | 7 | 0.714 | 2.857 | 0.432 |
| gap1124_M2_minus_A1 | 2.0-4.0 | 17 | 4.706 | 5.147 | 0.514 |
| gap1124_M2_minus_A1 | <0.5 | 1191 | 2.326 | 4.278 | 0.094 |
| gap1124_M2_minus_A1 | >=4.0 | 28 | 0.000 | 0.000 | 2.180 |
| gap1124_true_minus_shuffled | 0.5-1.0 | 14 | 0.000 | 1.071 | 0.002 |
| gap1124_true_minus_shuffled | 1.0-2.0 | 7 | 0.000 | 0.000 | -0.084 |
| gap1124_true_minus_shuffled | 2.0-4.0 | 17 | 1.765 | 3.088 | 0.036 |
| gap1124_true_minus_shuffled | <0.5 | 1191 | -0.361 | -0.277 | -0.025 |
| gap1124_true_minus_shuffled | >=4.0 | 28 | 0.000 | 0.000 | -0.184 |

`≥4 m` 只有 28 个 endpoint，M2−A1 的 Success/Precision 增益均为 0；mean error 虽下降，但没有跨过 tracking 曲线阈值。这个结果与既有 crop-reachability 诊断一致：大位移目标常在网络 forward 前已离开固定 crop，末端 proposal correction 不能稳定恢复。

## 代码机制解释

fixed/shuffled 只替换 `DynamicsEncoder`、physical-time adapter 和 `R(Δt)` 消费的 effective time；order-time 主干、GT、候选和监督仍保持不变。当前预测同时经过两条显式时间路径：

1. `DynamicsEncoder(ref_boxs, delta_t_effective, current_delta_t_effective)` 产生 dynamics proposal，并通过 zero-init adapter 改写 observation feature；
2. `d_final = d_obs + α·clip(d_dyn−stopgrad(d_obs), R(Δt_effective))`，其中 `α=0.75`、`R(Δt)=min(0.5+0.5Δt, 2.0)`。

因此 true≈fixed/shuffled 不是“模型完全没读时间”，而是当前学到的时间条件没有转化为正确 alignment 的性能优势。结合 R3 shared-SE(2) W0 塌陷和 candidate-frame/canonical-target 语义风险，更合理的解释是：R1 的正信号主要来自 continuation、联合表征或通用 proposal correction，物理秒数不是已证实的增益来源。

## Validation Report

### Overall Assessment: Share with caveats

数据、配对身份、计算和 bootstrap 均已复核，可用于内部路线决策；但不能对外宣称 M2 的正确物理时间有效，也不能宣称 M2 已因果超过 SeqTrack3D。

### 必须保留的限制

- 当前只有 seed42、mini_val、standard 与 gap1124；burst、unseen schedule、full data 和第二数据集未完成。
- A1 是历史 checkpoint，R1 从 A1 再训练 60 epoch，M2−A1 混有额外训练预算。
- R3 不是可靠 matched baseline；shared-SE(2) W0 的异常塌陷尚未解释。
- 参考帧 workaround 不改变预测或最终 CSV，但正式 exporter 应在下一提交中修复初始帧 `observed_keys`。

## 决策与下一步

1. 将状态更新为 **`M2 TRACKING SIGNAL POSITIVE / PHYSICAL-TIME CAUSAL CLAIM NO-GO / METHOD ATTRIBUTION HOLD`**。
2. 不为 physical-time claim 追加 seed43/44，也不立即启动 timestamp-conditioned M3/M4。
3. 第一优先级完成 frozen R1 的 full / adapter-only / innovation-only / both-off；它决定正信号来自哪条运行时路径。
4. 补 A1-init W0 continuation，分离额外 60 epoch；补 current-code legacy-candidate W0，解释 shared-SE(2) collapse。
5. 审计 candidate-frame `box_label`、canonical `motion_label`、`d_obs/d_dyn/d_final` 与 recursive-error process。若确认语义错配，再开 M1.5 新分支。
6. burst-drop 只在通用 proposal 路线仍成立后补作 robustness/crop 恢复证据；它不能复活已失败的 gap1124 physical-time causal gate。

