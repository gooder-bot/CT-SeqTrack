# P0-B3 Reliability 数据验证与特征消融

更新时间：2026-07-20

## 1. 结论

本地下载的三协议 P0-B3 数据通过完整性、唯一性、范围、标签一致性、endpoint 配对、checkpoint hash 和指标复算检查，可以用于当前机制筛选。预注册汇总判定仍为：

```text
RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO
```

但该名称必须收紧解释：**已成立的是 observation-quality reliability proxy，不是 timestamp-aware reliability 的因果收益。** 原始 recent-history CV anchor 的互补性不足，不能进入 active inference；显式 raw `current_delta_t` 在 standard-only calibrator 中还造成明显跨协议分布外退化。

当前总判定：

```text
observation reliability proxy: Conditional Go
raw predicted-history CV anchor: No-Go
current post-crop selector: No-Go
timestamp contribution to reliability: unproven
active dual-anchor tracking: not started
```

## 2. 数据与粒度

| protocol | endpoint rows | columns | tracklets | visible rows | endpoint duplicates | exact reference match |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 4246 | 120 | 260 | 2996 | 0 | yes |
| gap1124 | 2127 | 120 | 243 | 1503 | 0 | yes |
| burst-drop | 2098 | 120 | 243 | 1478 | 0 | yes |

三组使用同一 A1 checkpoint SHA256：

```text
a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82
```

原始 CSV SHA256 与汇总 JSON 完全一致。三协议均无重复 `(tracklet_key, frame_index, frame_token)`，关键 ID、时间、crop、candidate error 和 recall 字段无缺失；`delta_t` 全部为正，error 非负，recall 均落在 `[0,1]`。

本地复核时冻结的关键 SHA256：

```text
diagnose_reliability_signals.py   d7c8b27337518cf0d2fa79acb320b038cf2eb781e41fd67295f2621aed77d389
summarize_reliability_signals.py  6f121fb8487c0ea80f22a7c7b4cd79373564968c21113a45f99ebf2e23e7d1b0
standard endpoints                755588d0147bb065c57d90909902986f787d8951e7ccadc1d8525cb8f9939af1
gap1124 endpoints                 61e955c5f51e43efdad9e3141be612a468c5206b4798386391be49f1c6e9ae44
burst-drop endpoints              158ac342e3ea09106cf2dd60d113b2e0466c3ee5c4eae6daa1e5319df9af75b8
analysis summary                  e6bcb2f13ce7220c365c191f258229f11b3589dfb6e95ee1570dd89eea117c34
feature-ablation CSV              c63fe564f139359a34180fa64f3dd68f4f764cca952c8cbb18ee07186d120219
```

上一 observation forward 失败时，foreground 特征结构性缺失：standard/gap/burst 分别为 443/231/240 行，占 eligible visible endpoints 的 14.79%/15.37%/16.24%。这些行与 `prev_obs_empty_fallback=True` 一一对应，属于预期缺失，并由显式 empty indicator 与训练折内 median imputation 处理，不是下载损坏。

## 3. 方法核查

- `trigger` 只使用 `prev_obs_*`、当前可得的 `delta_t` 与 CV geometry；未使用 GT、current foreground、candidate error 或 drift label。
- `current_evidence` 和 `selector` 是 post-crop 诊断，不能冒充 pre-crop trigger。
- fold 由 `sha256(seed|tracklet_key)` 唯一决定；同一 tracklet 在三协议中进入相同 fold，没有 frame-level 随机切分泄漏。
- calibrator 和阈值只在 standard fold 内拟合，再原样评估 gap/burst。
- 当前标签只在 `current_target_visible=True` 的 endpoint 定义，所以结论是“可见目标是否离开 observation crop”，不是完整遮挡检测或通用 uncertainty estimation。

三项任务及 passive complementarity 的关键指标已从 CSV 独立复算，与已有 summary 的最大绝对误差小于 `6e-16`。

## 4. 预注册结果

### 4.1 Pre-crop trigger

| protocol | prevalence | AUROC | AUPRC | ECE | activation | miss recall | precision | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 0.2827 | 0.8572 | 0.7419 | 0.0241 | 0.3662 | 0.7639 | 0.5898 | 0.2094 |
| gap1124 | 0.3486 | 0.7871 | 0.6596 | 0.1341 | 0.6028 | 0.8473 | 0.4901 | 0.4719 |
| burst-drop | 0.3572 | 0.7847 | 0.6705 | 0.1553 | 0.5710 | 0.8333 | 0.5213 | 0.4253 |

两种强协议均达到预注册的 `AUROC >= 0.75` 与 `AUPRC - prevalence >= 0.15`。因此失败风险具有可预测性；但强协议 ECE、触发率和 FPR 明显升高，standard 阈值不能直接当成校准良好的部署概率。

### 4.2 Raw-CV crop complementarity

| protocol | obs recall | raw-CV recall | dual oracle recall | gain over obs | required |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 0.6976 | 0.7269 | 0.7280 | +0.0304 | - |
| gap1124 | 0.6368 | 0.6629 | 0.6657 | +0.0288 | +0.05 |
| burst-drop | 0.6319 | 0.6613 | 0.6634 | +0.0315 | +0.05 |

raw-CV 与 observation 的失败高度重叠；强协议分别仍有 491/483 个 visible endpoint 两者同时 miss。即使 oracle 逐 endpoint 选择较好 crop，增益也达不到 5 pp。因此不能把 raw recent-history CV 接入 active tracker，也没有必要继续调它的 selector。

### 4.3 Post-crop selector

| protocol | N | positive prevalence | AUROC | AUPRC | ECE | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 381 | 0.0945 | 0.7292 | 0.2627 | 0.0428 | 0.3130 |
| gap1124 | 92 | 0.3478 | 0.6052 | 0.4676 | 0.4013 | 0.7500 |
| burst-drop | 114 | 0.3947 | 0.4325 | 0.3380 | 0.4904 | 0.8261 |

selector 标签分布跨协议变化很大，burst-drop AUROC 已低于随机水平，不能用于主动选择 observation/raw-CV candidate。

## 5. 诊断性特征消融

下表为同一 grouped 5-fold 流程下的本地 post-hoc 诊断，不属于原预注册主判定。完整数值见 `compare_results/data/p0b3_reliability_feature_ablation_20260720.csv`。

| trigger features | gap AUROC/AUPRC | burst AUROC/AUPRC | gap/burst ECE | 解释 |
| --- | --- | --- | --- | --- |
| all 13 | 0.787/0.660 | 0.785/0.671 | 0.134/0.155 | 通过原 Go 门槛，但跨协议校准变差 |
| previous observation only | 0.867/0.778 | 0.873/0.789 | 0.061/0.057 | 更强、更稳定，说明主要信号来自上一观测质量 |
| time + CV geometry only | 0.553/0.457 | 0.557/0.490 | 0.273/0.292 | 单独接近随机，不足以预测失败 |
| all except raw `current_delta_t` | 0.865/0.787 | 0.872/0.804 | 0.061/0.060 | 去掉 raw dt 后恢复跨协议性能 |
| previous observation + raw `current_delta_t` | 0.792/0.652 | 0.790/0.657 | 0.126/0.142 | 复现主要退化，raw dt 是当前 OOD 问题来源 |

standard 的 `log(current_delta_t)` 标准差很小，standard-only 拟合后直接外推到 gap/burst 大时间间隔，容易把 cadence 变化当作失败概率并过度触发。它不是标签泄漏，但不支持“真实时间提高 reliability prediction”的因果表述。

此外，`prev_obs_forward_ran` 与 `prev_obs_empty_fallback` 完全互补，`pred_cv_available` 在 full-history eligible rows 中恒为真；后续正式 calibrator 应去掉这些冗余字段。

## 6. 下一步决策

1. 代码继续以 GitHub commit 版本化；服务器可手动同步，但必须保存本批 CSV/JSON、脚本与 checkpoint hash。当前运行来自 dirty `f28f495` 工作树，因此该批证据继续带 caveat；后续 P0-B4 以输出中自动记录的 git/dirty 和 SHA256 为准。
2. 将 reliability 与 timestamp trajectory 解耦。正式 trigger 先采用 `prev_obs_*` observation-only 版本，在 standard 上冻结模型和阈值，再用独立 split 验证；不要部署当前 all-13 calibrator。
3. 下一阶段只做一个独立 passive anchor kill-test：维护拒绝低可靠 observation 更新的 frozen-state / timestamp-aware Kalman 状态；它不能退化为最近两次预测的 raw CV。
4. passive anchor 必须同时比较 `true-dt / fixed-dt / shuffled-dt`，并与 raw CV、previous anchor、2x expanded、random second anchor 做相同 endpoint/crop/计算预算对照。物理时间因果性应在 active inference 前先筛掉。
5. 只有新 anchor 在 gap1124 和 burst-drop 的 dual-oracle target-point recall 均比 observation 提高至少 5 pp、standard 不下降超过 1 pp、点数/计算预算满足预注册约束时，才进入 active inference。
6. active 阶段固定 observation-only trigger；先报告首次失控、连续失败、empty fallback、Success/Precision 和 FPS。若 post-crop selector 仍不能跨协议稳定，不增加学习式 selector，而应停止双分支选择路线。
7. 新 anchor 或 true-dt 因果对照失败时，按计划 pivot 为 variable-rate 3D SOT benchmark/diagnosis；不增加 Mamba、ODE、occupancy memory 或大 trajectory encoder。

## 7. 可发布性判断

```text
Data integrity: Ready to use
Pre-registered mechanism verdict: Ready to share with caveats
Timestamp-aware reliability claim: Not supported
Raw-CV active anchor: Rejected
Paper-level method claim: Not ready
```

主要限制是：当前仍为 nuScenes-mini `mini_train` 的开发诊断，同一底层场景经不同 cadence 重采样；没有独立 validation/test、没有 active tracking 指标，也没有 clean-commit 复现。
