# P0-B4 Observation Reliability 独立验证

更新时间：2026-07-20

## 1. 决策结论

P0-B4 在 `mini_train -> mini_val` 的独立、冻结验证中得到：

```text
NO_GO_OBSERVATION_RELIABILITY_VALIDATION
```

两个预注册强协议 `gap1124` 和 `burst_drop` 都没有同时通过全部门槛。失败项一致：

- AUROC 低于 `0.75`；
- standard fitting 上冻结的运行阈值在 mini_val 的 recall 低于 `0.70`。

因此，当前 `observation_v1` calibrator 不进入 reliability-updated Kalman/frozen-state anchor，不进入 active dual-anchor，也不能根据 mini_val 重新选择特征、L2 或阈值后把结果重报为 confirmatory。这个 No-Go 否定的是**当前冻结 calibrator 与其预注册下游路径**，不是证明所有 observation reliability 信号都不存在。

验证状态：**Share with caveats**。数据、计算和判定可以支持当前 No-Go；复现记录仍有 dirty worktree 和 exact server script 未随压缩包回传的 caveat。

## 2. 控制数据与完整性

本次控制数据来自：

```text
transfer/p0b4_full_results.tar.gz
SHA256 f4d06bb1116080850155595b6180dcc560d291323ed1d5526132dac21d3bca57
```

压缩包无覆盖解压到：

```text
server_results/p0b4_full_20260720/
```

完整性检查：

- standard/gap1124/burst-drop evaluation CSV 分别为 `1979 / 984 / 978` 行；文件 SHA256 与服务器 final summary 完全一致。
- 可见目标且标签有效的正式评估行为 `1623 / 829 / 815`；其余行为当前目标不可见或标签不适用，不是意外丢行。
- `(tracklet_key, frame_token)` 无重复；fit 与三个 eval 的 labeled tracklet 交集均为 0。
- 三协议 `reference_match.exact_match == true`，missing/unexpected endpoint 均为 0。
- 三协议使用同一 A1 checkpoint，SHA256 均为 `a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82`。
- fitting CSV 的 SHA256 为 `755588d...9939af1`，与本地已有 P0-B3 standard CSV 完全一致。
- 日志没有 traceback、NaN、Inf、endpoint mismatch 或 checkpoint mismatch。

服务器 summary 记录 commit `f28f495...` 且 dirty；`validate_observation_reliability.py` 和依赖脚本的服务器 SHA256 与当前本地文件不同。使用当前本地验证器对四份 CSV 独立重跑后，calibrator、全部 headline metrics 和 verdict 仍一致，差异只在约 `1e-15` 的浮点舍入。因此该 caveat 不改变本次 No-Go，但后续正式实验必须从 clean GitHub commit 运行，或者把 exact script 一并保存。

## 3. 冻结验证结果

预注册门槛为：AUROC `>=0.75`、`AUPRC-prevalence >=0.15`、ECE `<=0.10`、FPR `<=0.30`、operating recall `>=0.70`；两个强协议必须全部通过。

| protocol | N | positives | prevalence | AUROC | AUPRC-prev | Brier | ECE | activation | recall | precision | FPR | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1623 | 180 | 0.1109 | 0.7938 | 0.4144 | 0.0795 | 0.0732 | 0.2742 | 0.7111 | 0.2876 | 0.2197 | report-only |
| gap1124 | 829 | 74 | 0.0893 | 0.6804 | 0.2821 | 0.0859 | 0.0934 | 0.2859 | 0.5676 | 0.1772 | 0.2583 | Fail AUROC/recall |
| burst_drop | 815 | 69 | 0.0847 | 0.7120 | 0.3275 | 0.0780 | 0.0890 | 0.2785 | 0.6087 | 0.1850 | 0.2480 | Fail AUROC/recall |

相对门槛的差值：

- gap1124：AUROC `-0.0696`，recall `-0.1324`；
- burst_drop：AUROC `-0.0380`，recall `-0.0913`。

AUPRC margin、ECE 和 FPR 均通过，说明信号不是完全随机；但它既没有达到要求的排序稳定性，也没有在冻结运行点召回足够多的 crop miss。activation 约 28%，真实 prevalence 只有 8%–9%，precision 只有 17.7%–18.5%。这不适合作为拒绝状态更新的在线控制信号。

## 4. 分布漂移与校准解释

fitting 与 independent evaluation 的标签基率发生了明显变化：

| split/protocol | prevalence | empty fallback | mean predicted risk |
| --- | ---: | ---: | ---: |
| mini_train standard fit | 0.2827 | 0.1479 | 0.2827 |
| mini_val standard | 0.1109 | 0.0622 | 0.1841 |
| mini_val gap1124 | 0.0893 | 0.0543 | 0.1782 |
| mini_val burst_drop | 0.0847 | 0.0479 | 0.1736 |

mini_val 上上一 observation 的 search points 更多、empty fallback 更少、motion dynamic probability 更低，整体明显比 mini_train fitting 数据容易。冻结模型仍给出约 17%–18% 的平均风险，约为强协议真实 prevalence 的两倍。

在强协议上，模型 Brier 还略差于只预测本协议 prevalence 的常数基线：

- gap1124：`0.08587` vs `0.08130`；
- burst_drop：`0.07803` vs `0.07749`。

这进一步说明当前概率不能作为跨 split 的稳定 uncertainty 值使用。不能用 mini_val 重新校准后覆盖本次结论；若未来使用完整 nuScenes 重新研究 reliability，必须事先定义新的训练/验证协议，并把本批结果保留为失败的 confirmatory test。

## 5. 样本集中度与不确定性

mini_val 的正例高度集中在少数 tracklet：

| protocol | labeled tracklets | positive tracklets | positives | top-5 tracklet positive share |
| --- | ---: | ---: | ---: | ---: |
| standard | 96 | 15 | 180 | 66.1% |
| gap1124 | 91 | 10 | 74 | 74.3% |
| burst_drop | 91 | 9 | 69 | 78.3% |

这使 tracklet-level post-hoc bootstrap 区间较宽，说明 mini_val 不能支持“所有 observation reliability 方法都无效”这种普遍结论。但预注册判定依据点估计和固定门槛，不因事后区间而改变；当前 calibrator 仍必须 No-Go。

## 6. Passive raw-CV 互补性在 mini_val 上进一步消失

同批 P0-B4 passive logger 的 crop 互补性为：

| protocol | observation has-point | raw-CV has-point | oracle union | trajectory-only endpoints | union gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 88.91% | 88.48% | 88.97% | 1 | +0.06 pp |
| gap1124 | 91.07% | 90.59% | 91.07% | 0 | +0.00 pp |
| burst_drop | 91.53% | 90.92% | 91.53% | 0 | +0.00 pp |

因此两个强协议里，raw recent-history CV 第二 crop 没有找回任何 observation-only miss。P0-B3 mini_train 的 `+2.88/+3.15 pp` 已经低于门槛，mini_val 又降为 0。当前 raw-CV candidate 不仅 selector 不可靠，候选本身也没有独立互补空间。

## 7. 研究路线决定

立即停止：

- 不实现基于当前 `observation_v1` calibrator 的 Kalman/frozen-state anchor；
- 不实现 active dual-anchor、learned gate 或更大 trajectory encoder；
- 不在 mini_val 上调特征、L2、阈值或 crop scale后重报 confirmatory 结果；
- 不把 P0-B3 的 post-hoc observation-only 高 AUROC 写成已独立验证的可靠性贡献。

下一阶段转为两条较窄、可防御的工作：

1. **P0-C / benchmark pivot**：先完成 train/eval virtual-rate 分离、稳定 token manifest、held-out cadence 和 `true/fixed/shuffled-dt` 一致性检查。把项目主结果重心收敛为 variable-rate 3D SOT benchmark/diagnosis，而不是未验证的 reliability-controlled tracker。
2. **一次性机制收尾**：P0-A 只在 mini_train crop-reachable subset 统计 observation correction 分布，并先核对当前 residual 是“完整 dynamics displacement 再相加”还是“预测 observation error”。不得直接放大 `max_residual_norm`；若继续 residual，只允许一个预注册设置和 true/fixed/shuffled 对照。corrected-TWC 也只做同提交 `single-view / paired-view weight0 / corrected-TWC` 的 seed42 因果控制，通过后才补 seed。

这一路线符合 Stop/Pivot 规则：不继续叠加 Mamba、ODE/CDE、复杂 memory 或新的 learned uncertainty head。

## 8. 来源

- `server_results/p0b4_full_20260720/output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_summary.json`
- `server_results/p0b4_full_20260720/output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_calibrator.json`
- `server_results/p0b4_full_20260720/output/diagnostics/reliability_signals/{standard,gap1124,burst_drop}_p0b4_val/`
- `server_results/p0b4_full_20260720/output/diagnostics/crop_reachability/{standard,gap1124,burst_drop}_val_reference/`
- `server_results/p0b4_full_20260720/logs/diagnostics/`
- 本地独立复算：`server_results/p0b4_full_20260720/recomputed/`
