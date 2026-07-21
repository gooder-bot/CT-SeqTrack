# CT-SeqTrack 已完成记录

更新时间：2026-07-21

这份文件统一记录已经完成的工程验收、历史实验和可供回查的关键输出。当前和未来任务只维护在 `need_to_do.md`；研究定位和论文边界见 `refined_plan.md`；简洁实验结论见 `sum_results.md`。

注意：本文件是历史归档。下方旧日志里的“下一步”“待后续确认”只代表当时上下文，不代表当前任务；当前任务一律以 `need_to_do.md` 为准。

---

## 0. 完成总览

### 工程链路

- [x] P0：真实时间字段主链路闭合。
- [x] P1：真实时间 batch 字段、CPU forward、GPU loss、2-step train smoke test 通过。
- [x] P2：scalar-preserving `TimeEncoding` 已实现，`raw / mlp / fourier` smoke test 通过。
- [x] P3：`DynamicsEncoder` / Velocity Branch 已实现，forward / loss / 2-step train smoke test 通过。
- [x] P4：Time-resampling Consistency 已实现；2026-07-11 修复 nonzero candidate 的共享坐标系缺陷，2026-07-16 已完成 corrected A1/A2 seed42 训练，anchor/current XYZ gap max 均为 0。
- [x] P5：Observability Gate 已实现，forward / loss / 2-step train smoke test 通过。
- [x] nuScenes-mini-HTV / virtual-rate 数据层、检查脚本和 6 个 A1/A2 smoke 配置已实现。
- [x] P0-B standard/gap1124/burst-drop full-history crop oracle 已完成；强协议 base recall 为 76.78%/77.72%，GT-history CV 为 98.96%/99.05%。
- [x] P0-B2 三协议 recursive predicted-history 已完成；endpoints 完全匹配同一 checkpoint，always-on raw CV recenter 按预注册门槛判定 No-Go。
- [x] P0-B3 passive reliability 三协议 full 已完成并在本地独立复核：reference endpoints/checkpoint hash 完全一致，预注册结论为 `RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`；可靠性信号只能收窄为 observation-quality proxy，raw-CV anchor 与当前 selector 均为 No-Go。
- [x] P0-B4 10-tracklet smoke 与完整 mini_val 冻结验证已完成并本地复算：gap/burst AUROC `0.680/0.712`、固定阈值 recall `0.568/0.609`，正式结论为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`；reliability-controlled anchor 在实现前停止。
- [x] P0-A standard warmup/active 真实 batch 诊断已完成：默认 residual 数值稳定但实际修正约 `1e-7 m`，未通过非平凡幅度验收。
- [x] TWC 同提交 A/B/C seed42 已完成并本地复核：Final `B-A=-15.30/-24.18`、`C-B=+8.31/+11.74`、`C-A=-7.00/-12.44`；主方法 promotion No-Go，不补 seed43/44。
- [x] 2026-07-21 完成旧路线阶段决策：P0-B、P0-C 与 TWC 的核心筛选已足够关闭旧扩展路线，项目转入 M0；这是一项研究阶段决策，不代表 M1–M4 已实现或已获得性能增益。
- [x] 2026-07-21 完成 M0 P0-C-D1 gap1124 三路 full 冻结诊断：true/fixed/shuffled 各 `91` 个 tracklet、`1257` 个 endpoint，endpoint/order/checkpoint/config/selection/manifest exact match，时间干预按定义生效；true 相对 fixed 为 `+0.438/+0.523`，相对 shuffled 为 `-0.123/+0.056`，逐 tracklet bootstrap 不支持稳定 Success/Precision 正效应，再次确认 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`。旧 2-tracklet smoke 只保留为首帧口径修复的工程记录。
- [x] 当前六组新消融 YAML 已创建：

```text
cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_disp.yaml
cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_twc.yaml
cfgs/seqtrack3d_nuscenes_a3_order_gate_safe.yaml
cfgs/seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml
cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml
cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml
cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml
cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_random20.yaml
```

### 已完成实验

完整结果文件位于 `compare_results/`，简洁叙事见 `sum_results.md`。

- [x] `SeqTrack baseline vs CT-SeqTrack P5 full`：P5 full final 明显退化，说明不能把 real time、dynamics、gate 混在一起下结论。
- [x] `A1-raw vs A2 raw-dyn`：A1-raw 崩坏，A2 raw-dyn 明显恢复，说明 dynamics 分支有价值，但 raw real-time 主干路径有问题。
- [x] `A1-pseudo / A1-MLP / A1-Fourier`：A1-pseudo 接近 baseline，MLP/Fourier 没有救回 real-time A1，说明问题不是简单时间编码函数。
- [x] `A1-scaled / A2-scaled-dyn`：缩放 real time 仍未修复，说明主干对时间 token 语义敏感，不只是数值尺度问题。
- [x] `A1-order / A2-order-dyn`：恢复 order-time 主干后，A1-order 基本修复 A1 崩坏；A2-order-dyn final precision 高于 baseline，是当前最强正向信号。
- [x] `A2-order-dyn-cand1 / A2-order-dyn-disp`：cand1 在 60 epoch 协议下明显退化但 step 未对齐；disp 与 A2-order-dyn 基本持平，precision 小幅更高。
- [x] 旧 `A1-order+TWC / A2-order-dyn+TWC`：两组已跑完并整理，但 `twc_valid_ratio=0`，说明 TWC 项未激活；这些结果不能作为 TWC 有效或无效的最终结论。
- [x] validity-fixed active `A1-order+TWC / A2-order-dyn+TWC`：实验已完成，但 2026-07-11 发现 nonzero candidate 坐标污染；旧 A1 正向与 A2 负向归因均已撤回。
- [x] `A3-order-gate-safe / A3-order-conf-res-gate`：gate-safe 比旧 P5 full 安全但低于 A2；conf-res 旧 best 很高但 final 崩坏。
- [x] 2026-07-08 五次稳定性复核：A3 best-e14 retest、A2 seed43、A2 seed44、A2+TWC w0.01、A3 conf-res rerun 均已整理到 `compare_results/reports/latest_5runs_comparison.md`。
- [x] corrected A1/A2+TWC seed42：A1 相对配置级 baseline final `+1.49/+5.03`，A2 `-0.93/-2.07`；完整结果见 `compare_results/reports/corrected_twc_seed42_comparison.md`。
- [x] 同提交 TWC A/B/C seed42：完成 provenance/config/checkpoint/event 审计、CSV 汇总和 PNG/SVG 图表；确认 `C-B` 净正效应但 C 仍显著低于 A，详见 `compare_results/reports/twc_abc_seed42_comparison_20260721.md`。
- [x] gap1124 / burst-drop / random20 的 A1/A2 六组 HTV 筛选：A2 只在 random20 为正，在两个强不规则协议为负；见 `compare_results/reports/htv_6runs_comparison.md`。
- [x] TrajTrack aligned seed42 训练完整性与 evaluator 审计：确认当前高分路径使用 GT-assisted refinement，不能作为公平在线排名；见 `compare_results/reports/trajtrack_gt_assisted_vs_plain_seqtrack_reference.md`。
- [x] 逐页视觉核查 SeqTrack3D 与 TrajTrack 本地论文，并用 HVTrack、StreamTrack、MambaTrack3D、ChronoTrack 的官方论文页面收窄 related-work 边界。

### 当前结论

```text
真实时间方向没有被否定；
当前最稳论文边界仍是保留 SeqTrack3D 主干的 order-time 语义，
把真实 delta_t 的主张收窄为冻结协议下的因果消融，
末端 bounded residual 暂时只保留为一次性机制收尾。
但最新复核显示 A2-order-dyn 仍有明显 seed sensitivity，
普通 fixed-step 全局涨点把握不高；
递归 predicted-history 已证明 raw CV 恒开启不足；P0-B3 又证明 raw-CV 被动互补增益不足 5 pp，
而当前 post-crop selector 在强协议接近或差于随机；两条路径都不进入 active 闭环。
P0-B4 独立验证进一步显示 observation-only trigger 的强协议 AUROC 与 recall 均未过线，
同批 raw-CV 第二 crop 在强协议没有任何 trajectory-only endpoint，
因此 reliability-updated Kalman/frozen-state 与 active dual-anchor 在实现前停止；
M0 中的 P0-C-D1 已完成；下一步完成冻结 A/B/C checkpoint 的 strong-cadence/path-variance、reachable-subset proposal oracle 和 candidate 伪速度审计；M1 只并行准备代码骨架与零初始化等价性测试。
corrected-TWC 的同提交 `C-B` 有正信号，但 paired-view 退化更大，C 仍低于 A；TWC 主方法 promotion 已 No-Go；
HTV 六组显示旧 feature-concat dynamics 只在温和 random20 为正，强 gap/burst 为负；
TrajTrack 当前本地高分含 GT oracle，只能作为实现诊断；
gate / conf-res 目前也不能作为稳定主配置。
```

## 2026-07-21：M0 P0-C-D1 gap1124 三路 full

回传包 `output/diagnostics/m0_transfers/m0_manual_20260721_v2.tar.gz` 的 SHA256 为 `7806ccd3652092aa58aa3047932eed28d1492afed2aaf3e4910fa247b54c45a2`。解包后的 true/fixed/shuffled 各有 `1257` 个 endpoint、`91` 个 tracklet、`102` 个字段；复合键无重复，三路 endpoint key/order、真实时间、GT、checkpoint、source config、selection 和 manifest 全部一致。true effective time 与 real time exact match，fixed 恒为 `0.5 s`，shuffled 使用同一真实 gap 多重集合的冻结置换。

- true：`55.2247 Success / 66.8775 Precision`，mean center error `2.2704 m`，fallback `94`。
- fixed：`54.7872 / 66.3544`，mean error `2.4614 m`，fallback `106`。
- shuffled：`55.3481 / 66.8218`，mean error `2.2496 m`，fallback `92`。
- true−fixed：`+0.4376/+0.5231`；true−shuffled：`-0.1233/+0.0557`。均未通过预注册的 `+0.5 Success / +1.0 Precision`，且逐 tracklet Success/Precision bootstrap 95% CI 均跨 0。
- true 与两个控制各有 `1079/1257` 个 endpoint 的预测中心改变，说明模块读取了时间；但正确时间没有比打乱时间更可靠。长 gap 分桶也没有显示 true 对 shuffled 的 Success 正优势。
- true−fixed 的 `0.191 m` mean-error 优势主要由一条三路均已失控的长尾 tracklet 驱动；移除该条后只剩 `0.0397 m`，不能用 overall mean error 讲 promotion。

服务器运行记录为 dirty commit，但 exporter/config/checkpoint/manifest/CSV 精确 hash 已保存，且 paired 效应与此前 clean aggregate 一致。因此该输出足以完成 M0 冻结机制诊断和 No-Go 决策；正式论文归档仍保留 clean-worktree/source-bundle provenance caveat。完整分析见 `compare_results/reports/m0_p0c_d1_full_analysis_20260721.md`，可执行复核见 `compare_results/notebooks/m0_p0c_d1_full_analysis_20260721.ipynb`。

## 2026-07-21：TWC A/B/C seed42 同提交结果

结果目录 `output/paper_twc_abc_20260720_183711/` 的 A/B/C 三组均来自 commit `343145d`，tracked source clean；seed42、mini_train/mini_val selection、batch16、candidate4、60 epoch、1262 steps/epoch 和每5 epoch评测完全一致。B/C resolved config 除运行路径/tag 外只差 `twc_weight: 0.0 -> 0.05`。

- A single-view final：`50.01 Success / 58.20 Precision`。
- B paired-view weight0 final：`34.71 / 34.02`，`B-A=-15.30/-24.18`。
- C corrected-TWC final：`43.01 / 45.76`，`C-B=+8.31/+11.74`，但 `C-A=-7.00/-12.44`。
- Late mean 的 `C-B=+7.35/+10.64`、`C-A=-6.36/-9.24`，结论不依赖 final 单点。
- B/C 的 75720 步 valid ratio 序列完全一致，anchor/current-point gap max 全程为0；C 的末1000步 center/angle gap 比 B 低 `2.17%/6.13%`。
- A/B/C `last.ckpt` SHA256 分别为 `08b27a65...d7de1`、`24f2c20d...fa04c9`、`a26c59de...7b2ca`。

生成 `tools/summarize_twc_abc_seed42.py`、6份 CSV、3组 PNG/SVG 图表和完整报告。最终记录为 `C_MINUS_B_POSITIVE_ON_STANDARD_SEED42` 与 `NO_GO_TWC_MAIN_METHOD_PROMOTION`；不补 seed43/44，只允许冻结 checkpoint 的 strong-cadence/path-variance 输出型收尾。

## 2026-07-20：P0-B3 reliability 三协议回传与复核

新增 `tools/diagnose_reliability_signals.py`，保持 P0-B2 脚本与输出不变。正常 observation candidate 是唯一递归更新，raw real-dt predicted-history CV 只作为 passive 第二 crop forward；工具记录 foreground probability/count/entropy/margin、motion-state probability、crop points、empty fallback、CV speed/shift、anchor/candidate agreement、稳定 tracklet key、cfg/checkpoint/reference hash 和离线 GT 标签。

新增 `tools/summarize_reliability_signals.py`，不引入 scikit-learn 依赖。汇总器使用 `sha256(seed|tracklet_key)` 固定分折，只在 standard fold 上拟合 NumPy logistic calibrator 和运行阈值，再不改阈值评估 gap1124/burst-drop；分别输出 pre-crop trigger、current-crop evidence、post-crop selector、Brier/ECE 和 passive raw-CV crop complementarity。

2026-07-20 进一步新增 `tools/validate_observation_reliability.py` 与 `tools/run_p0b4_observation_validation.sh`。前者固定 `observation_v1` 的5个非冗余上一观测特征，fit/eval tracklet 有交集时 fail fast，evaluation 阶段严格复用 training median/mean/scale、logistic 权重与 threshold；后者串联 mini_val 三协议 reference endpoint、P0-B3 passive logger、exact-match/checkpoint hash 检查和 frozen validation，并将10-tracklet smoke 与 full confirmatory tag 分开。两个文件已通过本地 `py_compile`/`--self-test` 与 Bash `-n`，服务器 smoke 与 full 均已完成。

服务器已完成 standard/gap1124/burst-drop 三协议 full passive diagnostic，本地下载后完成以下复核：

- endpoints 为 `4246 / 2127 / 2098`，其中 current target visible 且历史完整的评估样本为 `2996 / 1503 / 1478`；tracklet 为 `260 / 243 / 243`。
- 三协议无重复 endpoint、关键字段缺失、非有限值、非正 `delta_t` 或标签/union/selector 逻辑矛盾；reference endpoint exact match，checkpoint SHA256 均为 `a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82`。
- 原始 CSV SHA256 与 summary 一致；用现有汇总器重新计算后，报告指标最大绝对差约 `5.3e-16`。
- 前景统计缺失率为 `14.79% / 15.37% / 16.24%`，与 `prev_obs_empty_fallback=True` 完全对应，是工具定义下的结构性缺失，不是数据损坏。

预注册的 13 特征 pre-crop trigger 在 standard/gap/burst 上 AUROC 为 `0.857/0.787/0.785`，AUPRC 为 `0.742/0.660/0.671`，通过 reliability 判据；但强协议 ECE 升至 `0.134/0.155`、FPR 升至 `0.472/0.425`。passive raw-CV 与 observation crop 的 target-point union recall 只提高 `3.04/2.88/3.15 pp`，低于强协议 `5 pp` 门槛，因此正式决定为 **`RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`**。

post-crop selector 在 standard/gap/burst 的 AUROC 为 `0.729/0.605/0.433`，强协议 FPR 为 `0.750/0.826`，不能用于 active proposal selection。进一步的诊断性特征消融发现：

- 只用 9 个 `prev_obs_*` 特征时，gap/burst AUROC 为 `0.867/0.873`，明显优于全 13 特征。
- 删除 raw `current_delta_t` 后，gap/burst AUROC 为 `0.865/0.872`、ECE 为 `0.061/0.060`、FPR 为 `0.171/0.159`。
- 只用 time/CV geometry 时 AUROC 仅 `0.529/0.553/0.557`；把 raw `current_delta_t` 加回 observation 特征会显著恶化强协议校准和误报。

因此 P0-B3 支持的是“开发集上上一 observation 质量可预测”，不是“物理时间已经提供可靠性因果信号”。raw-CV active anchor 与当前 selector 停止；其当时保留的 independent frozen-state 候选随后已被 P0-B4 No-Go 取消。详细复核见 `compare_results/reports/p0b3_reliability_validation_20260720.md`，特征消融见 `compare_results/data/p0b3_reliability_feature_ablation_20260720.csv`；运行和判定口径见 `tools/P0_AB_DIAGNOSTICS.md` 第 5 节。

## 2026-07-20：P0-B4 independent mini_val 冻结验证

- 压缩包 `transfer/p0b4_full_results.tar.gz` 的 SHA256 为 `f4d06bb1116080850155595b6180dcc560d291323ed1d5526132dac21d3bca57`，包含三协议 reference、原始 CSV/summary、最终 calibrator/report 和日志。
- evaluation CSV 为 `1979/984/978` 行，正式 visible+labeled 行为 `1623/829/815`；无重复 endpoint，fit/eval tracklet 无交集，reference exact match，三协议 checkpoint hash 一致。
- gap1124/burst-drop AUROC 为 `0.680/0.712`，运行点 recall 为 `0.568/0.609`，没有通过预注册的 `0.75/0.70`；AUPRC margin、ECE、FPR 通过也不能覆盖失败项。
- 强协议 Brier 略差于 prevalence 常数基线；mini_train fit prevalence `0.283` 降到 mini_val 的 `0.089/0.085`，表明明显 split/难度漂移。
- raw-CV passive second crop 在 gap/burst 的 trajectory-only endpoint 均为 0，union gain 均为 `0.00 pp`。
- 用当前本地验证器独立重跑得到相同 calibrator、指标和 verdict，只有约 `1e-15` 浮点差；服务器运行仍记录为 dirty `f28f495`，exact server script 未随包回传，因此正式复现保留 provenance caveat。

最终决定为 **`NO_GO_OBSERVATION_RELIABILITY_VALIDATION`**：不实现当前 calibrator 控制的 Kalman/frozen-state 或 active dual-anchor，不在 mini_val 上重调。完整报告见 `compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。

后续要做的事情不要写在本文件，统一放到 `need_to_do.md`。

---

## 2026-07-17：P0-B2 recursive predicted-history 回传

### 已完成输出

- `output/diagnostics/recursive_crop_reachability/standard_a1_recursive/`
- `output/diagnostics/recursive_crop_reachability/gap1124_a1_recursive/`
- `output/diagnostics/recursive_crop_reachability/burst_drop_a1_recursive/`
- `logs/diagnostics/p0b2_standard_a1_recursive.log`
- `logs/diagnostics/p0b2_gap1124_a1_recursive.log`
- `logs/diagnostics/p0b2_burst_drop_a1_recursive.log`
- `compare_results/reports/p0b2_recursive_crop_reachability_20260717.md`
- `compare_results/data/p0b2_recursive_crop_reachability_20260717_summary.csv`

### 完整性

- 三协议 endpoints 为 4246/2127/2098，均与 oracle reference exact match，missing/unexpected 为 0。
- 三组使用相同 checkpoint SHA256 `a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82`；日志无 traceback。

### 结果与决定

- previous-A1 recall 为 69.69%/63.73%/63.24%，pred-history CV 为 72.61%/66.38%/66.27%，只提高 2.91/2.65/3.03 pp。
- gap1124/burst-drop 的 `>4 m` 位移桶提高 8.45/9.96 pp；点数比为 0.98x/0.91x，standard 没有退化。
- 总体 +5 pp 与强协议 `>4 m` +10 pp 门槛未同时通过，故 **always-on raw predicted-history CV recenter No-Go**。
- 上一预测误差不超过 4 m 时，pred-CV recall 为 98.59%/97.34%/98.64%；超过 4 m 后只有 0.80%/1.21%/1.61%。CV 在漂移前有用，但不能从错误历史中恢复绝对位置。
- 下一方法假设改为 reliability-aware dual-anchor preventive search；`previous_prediction_error <= 4 m` 仅为离线 GT 分桶，不能用于在线 gate。

---

## 2026-07-17：P0-A / P0-B 第一批服务器诊断

### 已完成输出

- `output/diagnostics/crop_reachability/standard_smoke/`
- `output/diagnostics/crop_reachability/standard_train/`
- `output/diagnostics/crop_reachability/gap1124_train/`
- `output/diagnostics/crop_reachability/burst_drop_train/`
- `output/diagnostics/p0a_standard_warmup_summary.json`
- `output/diagnostics/p0a_standard_active_summary.json`
- `compare_results/reports/p0_ab_diagnostics_20260717.md`
- `compare_results/data/p0_ab_diagnostics_20260717_summary.csv`

### P0-B standard 结论

- 4246 个 full-history endpoint 中，base crop 即使使用 previous-GT anchor，center outside 仍为 15.97%，all-corners-inside 仅 69.12%。
- 对当前 GT 框内有点的 2996 个 endpoint，base crop 的 any-target-point rate 为 90.72%，mean target-point recall 为 85.41%。
- 位移 `>4 m` 的 960 个 endpoint 中，base center outside 为 70.63%，mean recall 只有 45.20%。
- 2x expanded 将 mean recall 提升到 99.57%，但平均点数增至 base 的 5.68 倍。
- GT-history constant-velocity recenter 将 mean recall 提升到 99.95%，平均点数仅为 base 的 1.02 倍。
- center-outside 发生在 75/260 条 tracklet；最多的 10 条只解释 43.66%，不是单条异常序列造成。

结论边界：CV 使用 GT history，只是 oracle reachability。它证明“移动搜索中心”值得做，不证明 GT-free 在线收益。

### P0-B gap1124 / burst-drop 结论

- gap1124/burst-drop 分别有 2127/2098 个 full-history endpoint，位移 P95 为 16.89/18.61 m，明显高于 standard 的 7.16 m。
- base center-outside 为 25.20%/23.83%，mean target-point recall 为 76.78%/77.72%。
- 2x expanded 的 recall 只有 89.08%/87.65%，平均点数增至 base 的 5.29/5.19 倍；固定扩大 crop 在强协议下既昂贵又不足。
- GT-history CV recenter 的 recall 为 98.96%/99.05%，平均点数为 base 的 0.92/0.88 倍。
- 位移 `>4 m` 时，base recall 只剩 23.10%/26.85%，expanded 为 62.09%/56.53%，CV oracle 仍为 96.40%/96.69%。
- base 越界覆盖 106/243 和 105/243 条 tracklet，top-10 失败序列只解释 29.10%/32.40%，不是少数异常 tracklet。

结论边界：这是 GT-history oracle，不能写成在线收益。后续 P0-B2 已补齐递归预测误差，并判定 raw predicted-history CV 恒开启 No-Go。

### P0-A standard 结论

- A1 checkpoint 有 320 个 observation key 匹配；新增 dynamics/gate 的 14 个 missing key 符合预期。
- warmup 2-batch 中 residual 与 gate gradient 严格为 0，loss/backward finite。
- active 64-batch 的 observation error P50/P75/P95 为 `0.213 / 0.577 / 3.838 m`。
- 默认理论上限为 0.02 m，但 alpha 约 `2e-5`，实际 residual P50 仅 `7.25e-8 m`。
- gate gradient P50 为 `4.00e-10`，31/64 batch 为 0；encoder gradient finite。
- `applied_ratio=1` 只来自 `norm > 1e-8` 的布尔阈值，不代表 correction 有实际作用。
- candidate0 的 observation P50 为 0.193 m，candidate1/2/3 为 0.217/0.215/0.216 m；四组都有数米级 P95 长尾。

默认 residual 通过数值 smoke，但未通过功能验收。当前不根据这些混入 out-of-crop failure 的误差直接放大 bound。

### 当时未完成部分（P0-B2 已在上节补齐）

- P0-A 尚未运行完整训练 split、gap1124/burst-drop 和真正的 2-step optimizer smoke。
- full-history batch 的 `dynamics_valid_ratio=1`，没有覆盖真实 `dynamics_valid=0` 样本。
- previous-A1-prediction 与 A1-prediction-history-CV 已于 P0-B2 完成；当前剩余缺口是 active dual-anchor 在线闭环，而不是被动 reachability。

---

## 2026-07-16：最新数据整理、论文核查与路线收敛

### Corrected-TWC seed42

- A1 baseline 51.23 / 57.86，corrected-TWC 52.72 / 62.89，final delta 为 +1.49 / +5.03。
- A2 baseline 50.96 / 63.31，corrected-TWC 50.04 / 61.25，final delta 为 -0.93 / -2.07。
- 两个 corrected run 均有 12 个评测点、epoch60 checkpoint、75720 optimizer steps；anchor gap max 与 current XYZ gap max 都为 0。
- baseline 是旧 run 且没有 commit 记录，所以当前只记录为配置级参考。A1 需要 seed43/44 和同提交 paired baseline；A2 不进入 TWC 主线。

### HTV 六组

- gap1124：A2-A1 final 为 -4.01 / -9.55；A2 epoch10 早期高点后明显回落。
- burst-drop：A2-A1 final 为 -7.45 / -14.40。
- random20：A2-A1 final 为 +9.09 / +14.23，late mean 同样为正。
- 六组均是 seed42、mini_val 开发证据，virtual-rate seed 固定但未冻结 manifest。
- 当前数据不支持旧 feature-concat dynamics 作为强 gap 主方法，支持继续测试 observation-first bounded residual、candidate 伪速度和 search-crop 上限。

### TrajTrack 公平性审计

- aligned run 完成 60 epoch，训练预算与 plain SeqTrack3D 基本对齐。
- 当前 `pre_w_refine()` 使用当前帧 GT overlap 决定是否 refinement，并按 GT overlap 从 trajectory proposals 中选最大者。
- 64.94 / 79.07 与 plain SeqTrack3D 50.99 / 59.96 的差值混合了模型和 evaluator oracle，不能写成方法增益。
- 下一步公平参考固定 epoch60 checkpoint，分别运行 `pre_wo_refine()`、GT-free paper-aligned refinement 和 oracle upper bound。

### 文献核查后的方向

- SeqTrack3D 的固定历史窗口消融已经暴露长历史误差累积，提示必须诊断 candidate 扰动对速度的污染。
- TrajTrack 证明低维 bbox trajectory proposal 值得研究，但 CT 只借鉴 GT-free local/global proposal agreement。
- HVTrack/MambaTrack3D 已覆盖 fixed-interval HTV 和 SSM-HTV；CT 必须强调 tracklet 内不规则物理 `delta_t`、一个模型跨 unseen cadence。
- ChronoTrack 已覆盖泛化的 temporal consistency 叙事；TWC 必须限定为不同采样路径到同一物理 endpoint 的一致性。
- 下一方法方向收敛为 observation-first timestamp-conditioned dual-proposal residual，不转向完整 TrajFormer、Mamba 或 ODE/CDE。

---

## 2026-07-11：TWC 共享坐标修复与 bounded residual 实现

### TWC 根因与修复

旧 paired sampler 分别调用 A/B 两路预处理。candidate 1/2/3 会各自随机扰动最近历史框，而这个框同时决定当前帧 search crop 和全部输出框的局部坐标系。旧检查比较的是归一化后的 `ref_boxs[:, 0]`；每一路 anchor 相对自身变换后都接近零，所以不同坐标系也会误判为共享成功。

已完成：

- 新增 `utils/twc_utils.py`，以绝对 `frame_id` 为键为 paired sample 只采样一次 candidate perturbation。
- A/B 两路按各自 `prev_frame_ids` 从同一 offset map 取值；共同历史帧，尤其 `t-1` anchor，使用完全相同的 offset、crop 和局部坐标系。
- A/B 共同绝对帧还共享 point regularization seed，避免相同 raw crop 被独立随机抽成不同的 1024 点；检查工具会直接核对共同历史帧与当前帧 XYZ。
- sampler 在坐标归一化前输出 `coordinate_anchor`，并输出 `candidate_id / candidate_offsets` 供回归检查。
- `compute_twc_loss()` 改为检查未归一化的 `coordinate_anchor` 和当前帧 sampled XYZ；字段缺失或 gap 超阈值默认直接报错。
- `twc_candidate_zero_only=True` 不再把 `num_candidates` 改成 1，因此不会把每 epoch 样本数和 optimizer steps 缩成四分之一。
- `check_twc_batch.py` 新增 candidate 1/2/3、dataset length、共享 frame offset、sampling seed / XYZ、anchor、search crop 点数等硬断言。
- `check_forward_batch.py / check_train_steps.py / check_time_batch.py` 不再在 TWC 模式下偷偷强制 candidate0。

结论边界：旧 TWC run 的两路 supervised loss 仍然有效，但 nonzero candidate 的跨 view consistency loss 混入了坐标/crop 差异。旧 A1 precision-positive 和 A2 崩坏都不能继续归因给 TWC，必须修复后重跑。

### Observation-first bounded residual

已实现 `dynamics_motion_mode: residual_limited`：

```text
obs_motion = motion_mlp(point_feature)
dyn_disp = clamp_norm(dynamics_displacement_pred, max_residual_norm)
alpha_dyn = max_alpha * sigmoid(small_gate(stats)) * dynamics_valid
final_center = obs_center + residual_scale * alpha_dyn * dyn_disp
```

实现约束：

- residual 模式不再把 `z_dyn` 拼接进 observation motion head。
- gate 末层零权重、近零 alpha 初始化，训练开始时接近原 observation 解。
- 支持 norm clamp、warmup、long-gap-only、sparse-only 和 `dynamics_valid` mask。
- residual 只修改中心，角度仍由 observation branch 给出。
- residual 模式禁止和旧 `ObservabilityGate` 同时开启，避免两套 gate 语义混杂。
- 记录 alpha、raw/clamped norm、clamp/applied ratio、effective scale 和 `obs_dyn_center_gap`。

新增 standard、`gap1124`、`burst_drop`、`random20` 四个 residual 配置。当前只完成工程实现，不记录任何性能正结论。

### 本地验收

```text
python -m py_compile ...                         PASS
python tools/check_twc_shared_coordinates.py    PASS
python tools/check_residual_dynamics.py          PASS
SEQTRACK3D residual method-level smoke           PASS
TWC anchor/current-XYZ positive + negative guard PASS
git diff --check                                PASS
```

本机 Python 缺少 `easydict / nuscenes-devkit`，因此真实 nuScenes paired batch、完整 model forward/loss 和 2-step backward 留到服务器环境执行，具体命令与门槛见 `need_to_do.md`。

---

## 2026-07-08：nuScenes-mini-HTV 数据层与第一批运行启动

### 已完成工程

已在当前 `CT-SeqTrack` 工程中完成 nuScenes-mini virtual-rate / HTV 数据协议支持：

```text
datasets/nuscenes_lidar_mf.py
datasets/__init__.py
tools/check_virtual_rate_batch.py
tools/build_virtual_rate_manifest.py
```

完成内容：

- `NuScenesMFDataset` 支持在 tracklet 构造后生成虚拟变帧率序列。
- 已支持 `gap_pattern`、`burst_drop`、`random_drop`、`stride` 等模式。
- preload cache 名称已加入 virtual-rate tag，避免误读 fixed-step 旧缓存。
- `datasets/__init__.py` 已透传 `virtual_rate_*` 配置。
- `tools/check_virtual_rate_batch.py` 可打印 tracklet keep indices、timestamp gaps、batch 里的 `delta_t/current_delta_t/valid_mask`。
- `tools/build_virtual_rate_manifest.py` 已用于后续 formal manifest 工作。

### 已创建配置

```text
cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml
cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml
cfgs/seqtrack3d_nuscenes_a1_order_vr_random20.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_gap1124.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_burst_drop.yaml
cfgs/seqtrack3d_nuscenes_a2_order_dyn_vr_random20.yaml
```

### 服务器检查结果

`gap1124` metadata 检查已通过：

```text
virtual-rate mode=gap_pattern
tracklets=243/274
frames=2856/5051
drop=0.435
keep examples: [0,1,2,4,8,...]
timestamp gaps include about 0.5 / 1.0 / 2.0 seconds
gap_cv about 0.58-0.61
```

`burst_drop` full-history batch 检查已通过：

```text
virtual-rate mode=burst_drop
train frames=2827/5051, drop=0.440
val frames=1251/2285, drop=0.453
prev_frame_ids [2 1 0]
valid_mask [1 1 1]
delta_t [1.4545531, 0.55020094, 0.49989605]
current_delta_t 1.4545531
```

`random20` metadata 检查已通过：

```text
virtual-rate mode=random_drop
frames=4074/5051
drop=0.193
metadata gap_cv about 0.37-0.56
```

说明：`random20` 的某个 full-history batch 可能刚好落在普通相邻 gap 上，这是随机丢帧的正常现象，不代表协议失败。

### 第一批 6 组后台运行已启动

本批实验已按 `seed=42 / batch_size=16 / epoch=60 / workers=4 / preloading / check_val_every_n_epoch=5` 在服务器后台启动。此处只记录“已启动”和当时配置，不代表已经得到 final 指标；最终结果仍需跑完后再整理到本文件和 `compare_results/`。

| 协议 | 模型 | GPU | tag | log |
| --- | --- | ---: | --- | --- |
| gap1124 | A1-order | 0 | `htv_gap1124_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_gap1124_a1_order_seed42_w4_60ep_bs16.log` |
| gap1124 | A2-order-dyn | 0 | `htv_gap1124_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_gap1124_a2_order_dyn_seed42_w4_60ep_bs16.log` |
| burst_drop | A1-order | 1 | `htv_burst_drop_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_burst_drop_a1_order_seed42_w4_60ep_bs16.log` |
| burst_drop | A2-order-dyn | 1 | `htv_burst_drop_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_burst_drop_a2_order_dyn_seed42_w4_60ep_bs16.log` |
| random20 | A1-order | 1 | `htv_random20_a1_order_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_random20_a1_order_seed42_w4_60ep_bs16.log` |
| random20 | A2-order-dyn | 1 | `htv_random20_a2_order_dyn_seed42_w4_60ep_bs16` | `logs/vr_htv/htv_random20_a2_order_dyn_seed42_w4_60ep_bs16.log` |

当时观察到 `burst_drop A2-order-dyn` 进程 PID 为 `1714325`，日志进入 train / val preload 阶段：

```text
Global seed set to 42
virtual-rate mode=burst_drop tracklets=243/274 frames=2827/5051 drop=0.440 tag=vr_burst_k323_s233
preloading data into memory
reading from annos
saving loaded data to /home/lishengjie/data/nuscenes-mini/preload_nuscenes_Car_mini_train_v1.0-mini_10_-1_vr_burst_k323_s233.dat
virtual-rate mode=burst_drop tracklets=91/106 frames=1251/2285 drop=0.453 tag=vr_burst_k323_s233
preloading data into memory
reading from annos
```

### 文档清理

已按新的文档分工清理：

- `need_to_do.md`：只保留当前正在跑、待整理、后续要做的任务。
- `done.md`：保留已经完成的工程、实验、检查输出和当时状态。

---

## 2026-07-08：论文叙事与执行路线重定向

### 已更新文档

```text
README.md
refined_plan.md
sum_results.md
need_to_do.md
```

### 新的当前判断

- 普通 fixed-step benchmark 上追求整体 final 稳定涨点的把握不高。
- 更稳的论文主战场是 variable-rate / high-temporal-variation / long-gap / sparse 子集。
- `A2-order-dyn` 仍是最有价值的真实时间线索，但 feature-concat dynamics 已暴露 seed sensitivity。
- 当时计划优先实现保守 residual dynamics；该实现已于 2026-07-11 完成，服务器验收仍待办。
- 当时把 TWC 视为 A1-order 上的 precision-positive 候选；2026-07-11 坐标审计后该判断已撤回。

### 文档口径

```text
Fixed-step 3D SOT hides the physical meaning of irregular frame intervals.
CT-SeqTrack should first expose this issue with variable-rate evaluation,
then use a conservative timestamp-conditioned residual dynamics prior to
improve tracking under long gaps and sparse observations.
```

---

## 2026-07-08：关联实验图表整理

### 已生成文件

```text
compare_results/reports/related_comparisons.md
compare_results/data/related_comparisons_metrics_summary.csv
compare_results/data/related_comparisons_metrics_points.csv
compare_results/figures/bar_charts/*related group charts
compare_results/figures/delta_charts/*related group charts
compare_results/figures/line_charts/*related group charts
tools/summarize_related_comparisons.py
```

### 覆盖分组

- Main A1/A2/P5 Progression (60ep)
- A1 Time Encoding / Main-Branch Variants (60ep)
- A2 Dynamics Variants (60ep)
- TWC-Related Runs (60ep)
- A3 / Gate Variants (60ep)
- A2 Seed Stability (60ep)
- Long Training Stability (180ep)

每个分组都包含 SeqTrack baseline 或对应 180ep baseline，并重新计算 final/best 相对 baseline 的 delta。每组生成 final score 柱状图、final delta 柱状图、success 曲线和 precision 曲线。

---

## 2026-07-08：latest 5 runs + baseline 图表整理

### 已生成文件

```text
compare_results/reports/latest_5runs_with_baseline_comparison.md
compare_results/data/latest_5runs_with_baseline_metrics_summary.csv
compare_results/data/latest_5runs_with_baseline_metrics_points.csv
compare_results/figures/bar_charts/latest_5runs_with_baseline_final_scores.svg
compare_results/figures/delta_charts/latest_5runs_with_baseline_final_delta_vs_baseline.svg
compare_results/figures/line_charts/latest_5runs_with_baseline_success_curve.svg
compare_results/figures/line_charts/latest_5runs_with_baseline_precision_curve.svg
tools/summarize_latest_5runs_with_baseline.py
```

### 内容

- 将 2026-07-08 五次最新实验与 60ep `SeqTrack baseline` 合并。
- baseline 取自 `twc_gate_ablation_metrics_*` 中的 `SeqTrack baseline`，避免混入 180ep baseline。
- 生成 final score 柱状图、final delta vs baseline 柱状图、success 曲线和 precision 曲线。
- 输出表格中为每个 run 重新计算 `final_delta_vs_baseline` 和 `best_delta_vs_baseline`。

---

## 2026-07-08：五次稳定性复核整理

### 覆盖实验

```text
A3-conf-res best-e14 retest
A2-order-dyn seed43
A2-order-dyn seed44
A2-order-dyn+TWC w0.01 seed42
A3-conf-res rerun seed42
```

### 关键结果

```text
A3-conf-res best-e14 retest:        success 28.06, precision 37.70
A2-order-dyn seed43 final:          success 23.64, precision 23.77
A2-order-dyn seed44 final:          success 46.90, precision 52.62
A2-order-dyn+TWC w0.01 final:       success 22.88, precision 24.27
A3-conf-res rerun seed42 final:     success 32.11, precision 31.87
```

### 关键诊断

```text
A2-order-dyn+TWC w0.01:
  twc_valid_ratio tail1000 mean 0.7541
  loss_twc tail1000 mean 0.0083

A3-conf-res rerun:
  obs_alpha_dyn_mean tail1000 mean 0.4988
  obs_alpha_dyn_clamped_mean tail1000 mean 0.1810
  obs_dyn_residual_norm tail1000 mean 0.0314
```

判断：

- `A3-conf-res best-e14 retest` 没有复现旧汇总里的 62.04 / 76.30，高 best 暂时不能作为确认收益。
- `A2-order-dyn` seed43 / seed44 差异很大，旧 seed42 的 precision-positive 信号需要多 seed 统计支撑。
- 当时认为 `A2-order-dyn+TWC w0.01` 的退化不只是权重问题；2026-07-11 坐标审计后，该 TWC 归因已失效。
- `A3-conf-res rerun` 仍低，后续应先做评测路径核对和 gate 行为分桶。

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

---

## 2026-06-03：active TWC / gate-safe / conf-res 结果整理

### active TWC（历史记录；2026-07-11 后归因失效）

关键结果：

```text
A1-order final:          success 51.23, precision 57.86
A1-order+TWC final:      success 51.16, precision 61.10
A2-order-dyn final:      success 50.96, precision 63.31
A2-order-dyn+TWC final:  success 28.23, precision 32.04
```

关键诊断：

```text
A1-order+TWC twc_valid_ratio mean 0.750, tail1000 0.753
A2-order-dyn+TWC twc_valid_ratio mean 0.750, tail1000 0.750
```

判断：

- 当时只确认 validity mask 已激活；后续证明 nonzero candidate 的 A/B 坐标仍不共享。
- A1 的 +3.24 precision 和 A2 的退化均保留为历史观测，但不能归因给 TWC 或它与 dynamics 的兼容性。

### gate-safe / conf-res

关键结果：

```text
A2-order-dyn final:               success 50.96, precision 63.31
A3-order-gate-safe final:         success 48.32, precision 54.87
A3-order-conf-res-gate final:     success 31.17, precision 30.92
A3-order-conf-res-gate best:      success 62.04, precision 76.30
```

判断：

- `A3-order-gate-safe` 比旧 P5 full 的 `31.19 / 31.89` 安全很多，但仍低于 A2-order-dyn。
- `A3-order-conf-res-gate` 的旧 best checkpoint 信号很强，但 last checkpoint 崩坏；后续 2026-07-08 best-e14 复测未复现旧 best。
- 当前主配置仍应保持 `A2-order-dyn`，不要把 active TWC 或 gate 直接写成 full-model 收益。

归档文件：

```text
compare_results/twc_gate_ablation_comparison.md
compare_results/twc_gate_ablation_metrics_summary.csv
compare_results/twc_gate_ablation_metrics_points.csv
compare_results/twc_gate_ablation_diagnostics_summary.csv
compare_results/twc_gate_ablation_curves.png
compare_results/twc_gate_ablation_success_curve.png
compare_results/twc_gate_ablation_precision_curve.png
compare_results/twc_gate_ablation_best_final_summary.png
compare_results/twc_gate_ablation_twc_diagnostics.png
compare_results/twc_gate_ablation_gate_diagnostics.png
```

---

## 2026-06-02：cand1 / disp / order+TWC 正式结果整理

### cand1 / displacement 诊断

关键结果：

```text
SeqTrack baseline final:    success 50.99, precision 59.96
A2-order-dyn final:         success 50.96, precision 63.31
A2-order-dyn-cand1 final:   success 26.68, precision 24.50
A2-order-dyn-disp final:    success 50.54, precision 63.85
```

判断：

- `A2-order-dyn-cand1` 明显退化，但 `num_candidates=1` 让 60 epoch 只有约 18899 step，而 cand4 的 A2-order-dyn 是 75719 step；当前只能说明 60 epoch cand1 协议不稳，不能彻底判死 candidate noise 假设。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本持平，final precision 高 0.53；小权重 displacement loss 不伤主线，但不是决定性收益来源。

归档文件：

```text
compare_results/cand1_disp_dynamics_comparison.md
compare_results/cand1_disp_dynamics_metrics_summary.csv
compare_results/cand1_disp_dynamics_metrics_points.csv
compare_results/cand1_disp_dynamics_curves.png
compare_results/cand1_disp_dynamics_best_final_summary.png
```

### order+TWC 诊断

关键结果：

```text
A1-order final:          success 51.23, precision 57.86
A1-order+TWC final:      success 45.61, precision 50.77
A2-order-dyn final:      success 50.96, precision 63.31
A2-order-dyn+TWC final:  success 38.27, precision 38.85
```

关键诊断：

```text
两组 order+TWC 的 loss_twc / twc_valid_ratio / twc_center_gap / twc_angle_gap
全程为 0。
```

判断：

- 当前两组 order+TWC 不是 active-TWC 训练结果，而是 paired-view / cand1 / reduced-step 训练结果。
- 下降幅度不能解释成 TWC 与 order-time 主干或 dynamics prior 不兼容。
- 下一步应先修复 TWC validity / logging，使 `twc_valid_ratio` 非 0 后，先重跑 `A1-order+TWC`，再重跑 `A2-order-dyn+TWC`。

归档文件：

```text
compare_results/twc_order_ablation_comparison.md
compare_results/twc_order_ablation_metrics_summary.csv
compare_results/twc_order_ablation_metrics_points.csv
compare_results/twc_order_ablation_twc_diagnostics_summary.csv
compare_results/twc_order_ablation_curves.png
compare_results/twc_order_ablation_step_aligned_curves.png
compare_results/twc_order_ablation_delta_summary.png
compare_results/twc_order_ablation_twc_diagnostics.png
```

---

## 2026-05-26：P0-P2 验收状态

### 总结

- `P0` 真实时间主链路的代码修改已完成。
- `P1-1` 真实 batch 时间字段验收已完成。
- `P1-2` CPU forward smoke test 已完成，输出 tensor 全部 finite。
- `P1-2b` GPU loss smoke test 已完成，forward 输出和所有 loss 均 finite。
- `P1-3` 真实时间小训练步已通过，`loss_total` 和梯度均 finite，没有 NaN。
- `P2-1` scalar-preserving `TimeEncoding` 已实现。
- `P2-2` `raw / mlp / fourier` forward smoke test 已通过。
- `P2-3` `raw / mlp / fourier` GPU loss smoke test 已通过。

### 已完成文件

```text
CT-SeqTrack/datasets/misc_utils.py
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/models/base_model.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/models/time_encoding.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

### 时间字段链路

训练侧 `motion_processing_mf()` 已输出：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
velocity_label
```

测试侧 `MotionBaseModelMF.build_input_dict()` 已输出：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
```

模型侧已改为：

```python
delta_T = input_dict["delta_T"]
corner_stamps = create_corner_timestamps_from_deltas(delta_T, 8)
corner_stamps = self.time_encoder(corner_stamps)
box_seq_corners = torch.cat((box_seq_corners, corner_stamps), dim=-1)
```

点云时间也走同一个 `TimeEncoding`：

```python
points = self.encode_point_time(input_dict["points"])
```

### P1-1：真实 batch 时间字段

真实时间关键输出：

```text
points shape: (2, 4096, 5)
valid_mask: [1 1 1]
timestamps: [-0.499305 -1.049506 -1.549402  0.      ]
delta_T:    [-0.499305 -1.049506 -1.549402]
delta_t:    [0.499305  0.55020094 0.49989605]
current_delta_t: 0.499305
```

判断：

- 历史帧 `points[..., 3]` 为负数，当前帧为 `0`。
- `delta_T` 为负数。
- `delta_t` 为正数。
- nuScenes keyframe 约 2Hz，`delta_t` 在 `0.5s` 附近正确。
- 出现 `0.5502s` 说明确实读取了真实 timestamp，不是硬编码 `0.5`。

固定伪时间对照：

```text
valid_mask: [1 1 1]
timestamps: [-0.1 -0.2 -0.3  0. ]
delta_T:    [-0.1 -0.2 -0.3]
delta_t:    [0.1 0.1 0.1]
current_delta_t: 0.1
```

判断：

```text
use_real_time=True/False 已经能在 batch 级正确切换。
```

### P1-2：CPU forward smoke test

关键输出：

```text
device: cpu
valid_mask: [1 1 1]
delta_T: [-0.499305 -1.049506 -1.549402]
delta_t: [0.499305 0.55020094 0.49989605]

pred_bc: shape=(1, 4096, 9), finite=True
motion_cls: shape=(1, 2), finite=True
estimation_boxes: shape=(1, 4), finite=True
seg_logits: shape=(1, 2, 4096), finite=True
motion_pred: shape=(1, 4), finite=True
aux_estimation_boxes: shape=(1, 4), finite=True
ref_boxs: shape=(1, 3, 4), finite=True
valid_mask: shape=(1, 3), finite=True
updated_ref_boxs: shape=(1, 3, 4), finite=True
```

判断：

```text
delta_T -> create_corner_timestamps_from_deltas() -> box corner timestamp -> Transformer forward
主链路已通过 shape 和 NaN/Inf 检查。
```

### P1-2b：GPU loss smoke test

关键输出：

```text
device: cuda
valid_mask: [1 1 1]
delta_T: [-0.499305 -1.049506 -1.549402]
delta_t: [0.499305 0.55020094 0.49989605]

loss_motion_cls: 0.686676, finite=True
loss_center: 0.000131, finite=True
loss_angle: 0.000000, finite=True
loss_total: 3.694282, finite=True
loss_seg: 0.665731, finite=True
loss_center_aux: 0.523946, finite=True
loss_center_motion: 0.000000, finite=True
loss_angle_aux: 0.035597, finite=True
loss_angle_motion: 0.000000, finite=True
loss_center_ref: 0.447077, finite=True
loss_angle_ref: 0.031530, finite=True
loss_bc: 2.033965, finite=True
```

判断：

```text
GPU forward + compute_loss() 已通过 finite 检查。
真实时间字段已经能进入模型并参与完整 loss 计算。
```

### P1-3：真实时间小训练步

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=9.924500 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=2 loss_total=12.464478 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p1_3_real_time_gpu0_loss.jsonl
last checkpoint: output/p1_3_real_time_gpu0_ckpt/last.pt
```

判断：

```text
真实时间字段不仅能进入 forward 和 loss，而且已经能参与最小 optimizer 更新。
```

注意：

- 当前只跑了 `max_steps=2`，足够作为 smoke test，不等价于正式小 epoch。
- `check_train_steps.py` 保存普通 PyTorch `state_dict` checkpoint，不是 Lightning `.ckpt`。

### P2：TimeEncoding

当前实现：

- 新增 `CT-SeqTrack/models/time_encoding.py`。
- `seqtrack3d.py` 中 point time 和 box corner time 共用同一个 `TimeEncoding`。
- `seqtrack3d_nuscenes.yaml` 和 `seqtrack3d_waymo.yaml` 已加入 `time_encoding / time_scale / time_clip / time_fourier_bands / time_hidden_dim`。
- 默认 `time_encoding: raw`，保持真实时间 scalar 行为。
- `mlp` 和 `fourier` 仍保持一个时间通道，不改变 PointNet/Transformer 输入维度。
- `raw / mlp / fourier` 的 forward smoke test 和 GPU loss smoke test 均 finite。
- 当前帧 `t=0` 输出保持为 `0`。

---

## 常用验收命令

服务器路径：

```text
CT-SeqTrack: /home/lishengjie/study/lcyu/CT-SeqTrack
nuScenes-mini: /home/lishengjie/data/nuscenes-mini
config: /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
```

真实 batch 时间字段：

```bash
python tools/check_time_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

固定伪时间对照：

```bash
python tools/check_time_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history \
  --pseudo-time
```

CPU forward：

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/check_forward_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --no-loss
```

小训练步：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0
```

共享服务器上的低占用长版命令见：

```text
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

注意：`compute_loss()` 内部仍有 `.cuda()`，CPU 下不能直接算 loss。CPU forward 只验证模型 forward 主链路。

---

## 2026-05-26：P3 Dynamics / Velocity Branch 实现

### 代码改动

新增：

```text
CT-SeqTrack/models/dynamics.py
```

修改：

```text
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
```

### 实现内容

- 新增 `DynamicsEncoder`，输入 `ref_boxs / delta_t / valid_mask`。
- 按历史框序列构造真实时间差分动力学特征：

```text
d_i     = c_i - c_{i-1}
v_i     = d_i / max(delta_t_i, eps)
omega_i = wrap(theta_i - theta_{i-1}) / max(delta_t_i, eps)
speed_i = ||v_i||
gap_i   = delta_t_i
valid_i = valid_mask_i * valid_mask_{i-1}
```

- 使用 per-step MLP + masked mean/max pooling 得到 `z_dyn`。
- 输出 `velocity_pred` 作为轻量速度监督分支。
- 在 `seqtrack3d.py` 中通过 `use_dynamics_encoder` 开关控制是否启用。
- 默认配置保持关闭，不影响 P0-P2 baseline。
- 启用后 coarse motion branch 使用 `torch.cat([point_feature, z_dyn], dim=1)`。
- `compute_loss()` 在存在 `velocity_label` 时加入：

```text
loss_velocity = SmoothL1(velocity_pred, velocity_label)
loss_total += velocity_weight * loss_velocity
```

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/models/dynamics.py CT-SeqTrack/models/seqtrack3d.py
```

已通过直接加载 `dynamics.py` 的纯张量 smoke test：

```text
z shape: (1, 128), finite=True
velocity_pred shape: (1, 3), finite=True
dynamics_valid shape: (1, 1), value=[[1.0]]
invalid-history case: z sum=0.0, velocity sum=0.0, dynamics_valid=[[0.0]]
```

未在本地跑完整 `check_forward_batch.py`，因为本机缺少部分训练依赖和数据集。完整 forward/loss 与训练步检查已在服务器完成。

```text
check_forward_batch.py
check_train_steps.py --max-steps 2
```

### 服务器 P3 forward + loss smoke test

已在服务器使用 P3 配置运行：

```bash
python tools/check_forward_batch.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p3_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history
```

关键输出：

```text
using batch_idx=12
timestamps shape=(1, 4): [-0.499305 -1.049506 -1.549402  0.      ]
delta_T shape=(1, 3): [-0.499305 -1.049506 -1.549402]
delta_t shape=(1, 3): [0.499305   0.55020094 0.49989605]
current_delta_t shape=(1,): 0.49930500984191895
valid_mask shape=(1, 3): [1 1 1]
device: cuda

pred_bc: shape=(1, 4096, 9), finite=True
velocity_pred: shape=(1, 3), finite=True
dynamics_valid: shape=(1, 1), finite=True
motion_cls: shape=(1, 2), finite=True
estimation_boxes: shape=(1, 4), finite=True
seg_logits: shape=(1, 2, 4096), finite=True
motion_pred: shape=(1, 4), finite=True
aux_estimation_boxes: shape=(1, 4), finite=True
ref_boxs: shape=(1, 3, 4), finite=True
valid_mask: shape=(1, 3), finite=True
updated_ref_boxs: shape=(1, 3, 4), finite=True

loss_motion_cls: 0.772253, finite=True
loss_center: 0.001527, finite=True
loss_angle: 0.006683, finite=True
loss_total: 10.814757, finite=True
loss_seg: 0.737634, finite=True
loss_center_aux: 1.532263, finite=True
loss_center_motion: 0.000000, finite=True
loss_angle_aux: 0.481590, finite=True
loss_angle_motion: 0.000000, finite=True
loss_center_ref: 1.518901, finite=True
loss_angle_ref: 0.385201, finite=True
loss_velocity: 0.002232, finite=True
loss_bc: 2.024360, finite=True
```

判断：

```text
P3 dynamics branch 已能在真实 batch 上进入 forward 和 compute_loss。
velocity_pred / dynamics_valid / loss_velocity / loss_total 均 finite。
```

注意：

- PointNet2 的 `SyntaxWarning: "is" with a literal` 仍是旧代码警告，不影响本次结果。
- 当前已完成 forward + loss smoke test，输出 `velocity_pred / dynamics_valid / loss_velocity` 均正常。

### 服务器 P3 train-step smoke test

已在服务器继续使用同一份 P3 配置运行 2-step 训练检查：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
timeout 20m nice -n 19 \
python tools/check_train_steps.py \
  --cfg /home/lishengjie/study/lcyu/CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p3_dyn.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --log-file output/p3_dyn_real_time_gpu0_loss.jsonl \
  --checkpoint-dir output/p3_dyn_real_time_gpu0_ckpt \
  --tag p3_dyn_real_time_gpu0
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=22.146883 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=4.546315 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p3_dyn_real_time_gpu0_loss.jsonl
last checkpoint: output/p3_dyn_real_time_gpu0_ckpt/last.pt
```

判断：

```text
P3 dynamics branch 不仅能完成 forward 和 compute_loss，也已经能完成 backward、梯度裁剪、optimizer step、loss log 写出和 checkpoint 保存。
loss_total 与 grad_norm 均 finite，P3 工程验收完成，可以进入 P4。
```

### 当前 GitHub 同步范围

本轮同步建议包含以下文件，构成 P0-P3 的可复现工程快照：

```text
.gitignore
need_to_do.md
refined_plan.md
done.md
CT-SeqTrack/models/time_encoding.py
CT-SeqTrack/models/dynamics.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_train_steps.py
CT-SeqTrack/tools/P1_3_SAFE_TRAIN_STEPS.md
```

同步后建议从 P4 开始新一轮改动，不再继续在同一个提交里混入 TWC 代码。

---

## 2026-05-27：P4 Time-resampling Consistency 代码实现

### 代码改动

新增：

```text
CT-SeqTrack/tools/check_twc_batch.py
```

修改：

```text
CT-SeqTrack/datasets/misc_utils.py
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
```

### 实现内容

- `get_history_frame_ids_and_masks()` 增加 `offsets` 参数，默认保持 `[1, 2, ..., hist_num]` 行为。
- `MotionTrackingSamplerMF` 在 `use_twc=True` 时返回 `view_a / view_b` paired batch。
- 第一版 paired view 使用同当前帧、同最近历史 anchor、不同旧历史路径：

```text
view_a: [t-1, t-2, t-3] -> t
view_b: [t-1, t-3, t-5] -> t
```

- TWC 默认关闭，不影响 P0-P3 baseline。
- `SEQTRACK3D.forward()` 和 `compute_loss()` 支持 nested paired batch。
- 新增 `compute_twc_loss()`，只约束最终 `aux_estimation_boxes`。
- paired supervised loss 使用 `0.5 * (L_a + L_b)`，避免监督项权重翻倍。
- `check_train_steps.py` 和 `check_forward_batch.py` 已支持递归移动 nested dict。
- `check_time_batch.py`、`check_forward_batch.py`、`check_train_steps.py` 增加 `--twc` 临时开关，可在不修改 YAML 的情况下启用 paired-view 检查。

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/datasets/misc_utils.py CT-SeqTrack/datasets/sampler.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_train_steps.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_time_batch.py CT-SeqTrack/tools/check_twc_batch.py
python -m py_compile CT-SeqTrack/datasets/sampler.py
```

已通过 offsets 纯函数检查：

```text
(7, 3, None)      -> ([6, 5, 4], [1, 1, 1])
(7, 3, [1,3,5])  -> ([6, 4, 2], [1, 1, 1])
(2, 3, [1,3,5])  -> ([1, 0, 0], [1, 0, 0])
bad offsets       -> correctly raises ValueError
```

### 服务器 P4 paired batch 检查

已在服务器运行：

```bash
python tools/check_twc_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

关键输出：

```text
using batch_idx=3
view_a prev_frame_ids: [5 4 3]
view_a history_offsets: [1 2 3]
view_a timestamps: [-0.49572396 -0.95052814 -1.4509721   0.        ]
view_a delta_T:    [-0.49572396 -0.95052814 -1.4509721]
view_a delta_t:    [0.49572396 0.45480418 0.50044394]
view_a current_delta_t: 0.4957239627838135
view_a current_timestamp: 1532402930.648325
view_a valid_mask: [1 1 1]

view_b prev_frame_ids: [5 3 1]
view_b history_offsets: [1 3 5]
view_b timestamps: [-0.49572396 -1.4509721  -2.500478    0.        ]
view_b delta_T:    [-0.49572396 -1.4509721  -2.500478]
view_b delta_t:    [0.49572396 0.9552481  1.049506]
view_b current_delta_t: 0.4957239627838135
view_b current_timestamp: 1532402930.648325
view_b valid_mask: [1 1 1]

shape_mismatches: none
same_current_timestamp: [True, True]
same_anchor_ref_box: [True, True]
different_delta_T: [True, True]
full_history_a: [True, True]
full_history_b: [True, True]
twc_valid: [True, True]
```

判断：

```text
paired view 数据构造正确。view_a / view_b 共享同一当前帧和最近历史 anchor，
只改变更早历史路径；nested batch 能被 DataLoader 正常 collate。
```

### 服务器 P4 forward + loss smoke test

已在服务器运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --twc
```

关键输出：

```text
using batch_idx=5
device: cuda

output view_a:
pred_bc / motion_cls / estimation_boxes / seg_logits / motion_pred /
aux_estimation_boxes / ref_boxs / valid_mask / updated_ref_boxs: finite=True

output view_b:
pred_bc / motion_cls / estimation_boxes / seg_logits / motion_pred /
aux_estimation_boxes / ref_boxs / valid_mask / updated_ref_boxs: finite=True

loss_total: 4.892723, finite=True
loss_total_sup: 4.892723, finite=True
loss_total_a: 4.889785, finite=True

loss_twc: 0.000004, finite=True
twc_valid_ratio: 1.000000, finite=True
twc_center_gap: 0.002937, finite=True
twc_angle_gap: 0.002928, finite=True
```

判断：

```text
paired forward 与 paired loss 均通过 finite 检查。
twc_valid_ratio=1.0 说明当前 batch 中所有样本均满足同当前时刻、同 anchor、
不同历史路径和完整历史条件。loss_twc 约 4e-6，远小于 loss_total，
当前权重下不会主导训练；center / angle gap 均很小，说明两条历史采样路径
在未训练初始状态下已能产生接近的最终框，TWC 项的量级是安全的。
```

### 服务器 P4 train-step smoke test

已在服务器运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --twc
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
use_twc: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=10.901030 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=9.719684 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/check_train_steps_loss.jsonl
last checkpoint: output/check_train_steps_ckpt/last.pt
```

判断：

```text
P4 TWC 不仅能完成 paired batch、forward 和 compute_loss，也已经能完成
backward、梯度裁剪、optimizer step、loss log 写出和 checkpoint 保存。
loss_total 与 grad_norm 均 finite，P4 工程 smoke test 已通过。
```

### 待后续确认

```text
1. 再跑一次 use_twc=False 的默认 forward/train-step，对照确认默认路径仍与 P0-P3 兼容。
2. 检查 output/check_train_steps_loss.jsonl 中是否写入 loss_twc / twc_valid_ratio /
   twc_center_gap / twc_angle_gap。
3. 进入小规模训练和消融前，建议为 P4 单独保存一份配置，例如
   cfgs/seqtrack3d_nuscenes_p4_twc.yaml。
```

### 2026-05-27：P4 剩余收口项取消为当前阻塞项

决定：

```text
P4 已有 paired batch / forward / loss / 2-step train smoke test 记录，足够支撑进入 P5。
默认路径回归、JSONL 字段复核和 P4 专用配置暂不作为当前阻塞任务。
这些项目后续如进入正式消融，再和实验配置一起补齐。
```

下一步：

```text
进入 P5 Observability Gate。
P5 第一版保持轻量：只在 coarse motion branch 中融合 observation feature 与 P3 dynamics prior。
不引入复杂 memory、不引入多模态、不改 TWC 和 Transformer refine。
```

---

## 2026-05-27：P5 观测可靠性统计量实现

### 代码改动

修改：

```text
CT-SeqTrack/datasets/sampler.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/models/base_model.py
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/tools/check_time_batch.py
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_twc_batch.py
```

### 实现内容

- 训练侧 `motion_processing_mf()` 在当前搜索区域裁剪后、`regularize_pc()` 前记录真实点数：

```text
num_points_in_search = this_frame_pc.nbr_points()
```

- 测试侧 `MotionBaseModelMF.build_input_dict()` 使用同一口径写入 `num_points_in_search`。
- `SEQTRACK3D.build_observability_stats()` 已构造第一版 P5 观测可靠性统计量：

```text
obs_stats = [
  log1p(num_points_in_search),
  log1p(soft_fg_count_current),
  mean_fg_score_current,
  valid_history_ratio,
  current_delta_t / time_scale
]
```

- `soft_fg_count_current / mean_fg_score_current` 只来自当前帧 chunk 的 `seg_logits`，不混入历史点云。
- 默认 `obs_stats_detach_seg=True`，避免后续 gate 通过统计量反向操纵 segmentation confidence。
- 检查脚本已增加 `num_points_in_search` 打印。

### 本地检查

已通过：

```text
python -m compileall CT-SeqTrack/datasets/sampler.py CT-SeqTrack/models/base_model.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_time_batch.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_twc_batch.py
```

未在本地运行完整模型 import 单元测试，因为本地环境缺少 `easydict` 等训练依赖；后续服务器 smoke test 可通过 `check_forward_batch.py` 查看 `obs_stats / obs_*` 输出是否 finite。

---

## 2026-05-27：P5 Observability Gate 主体实现

### 代码改动

新增：

```text
CT-SeqTrack/models/observability.py
CT-SeqTrack/tools/check_observability_gate.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml
```

修改：

```text
CT-SeqTrack/models/seqtrack3d.py
CT-SeqTrack/cfgs/seqtrack3d_nuscenes.yaml
CT-SeqTrack/cfgs/seqtrack3d_waymo.yaml
CT-SeqTrack/tools/check_forward_batch.py
CT-SeqTrack/tools/check_train_steps.py
need_to_do.md
```

### 实现内容

- 新增 `ObservabilityGate`：

```text
point_feature: B,256
z_dyn: B,dynamics_hidden_dim
obs_stats: B,5
dynamics_valid: B,1

z_dyn_proj = Linear(z_dyn) -> B,256
alpha = softmax(MLP(obs_stats)) -> [alpha_obs, alpha_dyn]
fused_feature = alpha_obs * point_feature + alpha_dyn * z_dyn_proj
```

- `gate_mlp` 最后一层权重初始化为 0，bias 初始化为 `[obs_gate_init_obs_bias, 0]`，训练初期偏向 observation。
- `dynamics_valid < obs_gate_min_dyn_valid` 时强制 `alpha_dyn=0, alpha_obs=1`。
- `SEQTRACK3D` 中新增 `use_observability_gate` 开关；打开 P5 时必须同时打开 `use_dynamics_encoder`。
- P5 打开时 motion feature 保持 256 维并复用原始 `motion_mlp`；P3 dynamics-only 路径仍使用 `torch.cat([point_feature, z_dyn])`。
- `compute_loss()` 已记录：

```text
obs_alpha_obs_mean
obs_alpha_dyn_mean
obs_alpha_dyn_min
obs_alpha_dyn_max
obs_gate_entropy
obs_num_points_search_mean
obs_soft_fg_count_mean
obs_mean_fg_score
obs_valid_history_ratio
obs_current_delta_t_ratio
```

- `obs_gate_entropy_weight` 已接入，默认 `0.0`，当前不改变 loss。
- `check_forward_batch.py` 和 `check_train_steps.py` 新增 `--obs-gate`，可临时打开 `use_dynamics_encoder=True` 和 `use_observability_gate=True`。
- 新增 P5 专用 nuScenes 配置：

```text
cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml
```

其中 `use_dynamics_encoder=True`、`use_observability_gate=True`、`use_twc=False`。

### 本地检查

已通过：

```text
python CT-SeqTrack/tools/check_observability_gate.py
python -m compileall CT-SeqTrack/models/observability.py CT-SeqTrack/models/seqtrack3d.py CT-SeqTrack/tools/check_observability_gate.py CT-SeqTrack/tools/check_forward_batch.py CT-SeqTrack/tools/check_train_steps.py
```

纯张量 smoke test 关键输出：

```text
fused shape: (2, 256), finite=True
alpha: [[0.7310586  0.26894143]
        [1.         0.        ]]
alpha_sum_ok: True
invalid_dyn_ok: True
```

服务器 P5 forward + loss smoke test 已通过。运行命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history \
  --obs-gate
```

关键输出：

```text
using batch_idx=12
timestamps shape=(1, 4): [-0.499305 -1.049506 -1.549402  0.      ]
delta_T shape=(1, 3): [-0.499305 -1.049506 -1.549402]
delta_t shape=(1, 3): [0.499305   0.55020094 0.49989605]
current_delta_t shape=(1,): 0.49930500984191895
num_points_in_search shape=(1,): 3.0
valid_mask shape=(1, 3): [1 1 1]
device: cuda

velocity_pred: shape=(1, 3), finite=True
dynamics_valid: shape=(1, 1), finite=True
obs_alpha: shape=(1, 2), finite=True
obs_alpha_obs: shape=(1,), finite=True
obs_alpha_dyn: shape=(1,), finite=True
obs_gate_entropy: shape=(1,), finite=True
obs_stats: shape=(1, 5), finite=True
obs_num_points_search / obs_soft_fg_count / obs_mean_fg_score /
obs_valid_history_ratio / obs_current_delta_t_ratio: finite=True

loss_total: 4.333134, finite=True
loss_velocity: 0.001596, finite=True
obs_num_points_search_mean: 3.000000, finite=True
obs_soft_fg_count_mean: 497.636841, finite=True
obs_mean_fg_score: 0.485973, finite=True
obs_valid_history_ratio: 1.000000, finite=True
obs_current_delta_t_ratio: 0.998610, finite=True
obs_alpha_obs_mean: 0.731059, finite=True
obs_alpha_dyn_mean: 0.268941, finite=True
obs_alpha_dyn_min: 0.268941, finite=True
obs_alpha_dyn_max: 0.268941, finite=True
obs_gate_entropy: 0.582203, finite=True
```

判断：

```text
P5 forward + loss 主链路已通过。当前样本 num_points_in_search=3，属于极稀疏搜索区域；
obs_stats、obs_alpha、obs_gate_entropy 和所有 tracking loss 均 finite。
alpha_obs≈0.731 / alpha_dyn≈0.269 与 obs_gate_init_obs_bias=1.0 的初始化一致，
说明 gate 初始化和 dynamics_valid 有效路径正常；这不是训练后 gate 已学到策略的结论。
```

### 服务器 P5 train-step smoke test

已通过。运行命令：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0 \
  --obs-gate \
  --log-file output/p5_obs_gate_loss.jsonl \
  --checkpoint-dir output/p5_obs_gate_ckpt \
  --tag p5_obs_gate
```

关键输出：

```text
device: cuda:0
max_steps: 2
batch_size: 1, workers: 0
use_real_time: True
use_twc: False
use_observability_gate: True
cuda memory fraction limit: 0.2
frozen BatchNorm modules: 28
step=1/2 batch_idx=0 loss_total=14.486424 grad_norm=1.000000 lr=0.00010000
step=2/2 batch_idx=1 loss_total=5.297028 grad_norm=1.000000 lr=0.00010000
finished train-step check
loss log: output/p5_obs_gate_loss.jsonl
last checkpoint: output/p5_obs_gate_ckpt/last.pt
```

判断：

```text
P5 Observability Gate 已能完成 forward、compute_loss、backward、梯度裁剪、
optimizer step、loss log 写出和 checkpoint 保存。P5 工程 smoke test 已通过。
默认关闭路径回归检查已取消，不再作为后续计划任务；下一步重点转向困难子集评估、
正式消融，以及观察训练后 alpha 是否随 sparse / gap / fg score 分桶发生合理变化。
```
---

## 2026-07-20：P0-C 冻结协议工程与服务器真实 batch 验收完成

- `datasets/__init__.py` 已支持 train 与 val/test/eval cadence 字段分离，并保留旧无前缀配置的兼容回退。
- nuScenes virtual-rate manifest 升级为 stable-token v2：以 version/split/scene/instance 建键，记录 protocol、endpoint、commit 与多层 SHA256，并对错 split/role/protocol/tracklet/hash fail fast。
- 新增离线 split-wide shuffled-dt manifest；batch 与递归评测同时保留 real/effective time，且模型只有 `DynamicsEncoder` 读取 effective time。
- 新增 `tools/check_p0c_time_controls.py`、`tools/build_dynamics_time_manifest.py`、P0-C gap1124 配置和 `protocols/README.md`；`main.py` 会写 `run_provenance.json`。
- 本地已通过相关 Python 文件 `py_compile`、split-aware config/hash 自测和 synthetic effective-time 自测。
- 服务器 clean commit `343145d` 已生成 val/test gap1124 cadence manifest 与 test shuffled-dt manifest；val/test 均为 `91/106 tracklets`、`1257/2285 frames`，test endpoint selection SHA256 为 `85e5603c...f9649f6f`。
- shuffled mapping 为 `1257 endpoints / 1166 transitions`，满足 `1257 - 1166 = 91 tracklets`；真实 nuScenes batch 的 frame/crop/candidate/label/real-time 不变量检查最终输出 `P0-C true/fixed/shuffled batch invariance: PASS`。
- 本节完成的是协议工程和输入公平性验收；当时三路性能仍待执行，随后已在下一节完成并判定 No-Go。验证报告见 `compare_results/reports/p0c_frozen_protocol_validation_20260720.md`。

## 2026-07-20：P0-C 同 checkpoint 三路 time-control 性能完成并 No-Go

- 使用 standard-trained A2-order-dyn seed42 60ep final/last checkpoint，SHA256 为 `b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad`。
- true/fixed/shuffled 三份 provenance 共享 commit `343145d`、source config `69b801f7...7d658`、test manifest/selection、seed42 和 `91 tracklets / 1257 frames`；resolved config 只在时间控制字段与 log_dir 上不同。
- 指标分别为 true `55.2247 / 66.8854`、fixed `54.7872 / 66.3624`、shuffled `55.3480 / 66.8298`。true 相对 fixed 为 `+0.4375 / +0.5231`，相对 shuffled 为 `-0.1233 / +0.0557`。
- 未达到 true 同时超过两个对照且 `Success >= +0.5 / Precision >= +1.0` 的预注册门槛，正式判定 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；不扩展 burst/fixed-gap/multiseed。
- tarball、三个原始 manifest file/content hash、selection/permutation hash、console 与 TensorBoard events 已在本地独立复核；完整报告见 `compare_results/reports/p0c_frozen_protocol_validation_20260720.md`。

## 2026-07-20：P0 后论文可行性与 code-to-claim 审计完成

- 沿 `sampler -> time fields -> DynamicsEncoder -> residual/TWC loss -> protocol/provenance` 重新核对代码与实验结论。
- 确认 A1 corrected-TWC 使用 `main_time_source=order` 且关闭 DynamicsEncoder；其正信号只能支持 history-resampling consistency，不能直接支持 physical timestamp 收益。
- 确认当前 bounded residual 将两个由完整 displacement 标签监督的 proposal 直接相加，存在重复运动的定义歧义；后续改为先做 crop-reachable `d_obs -> d_dyn` oracle blend，oracle 通过才允许修正公式和训练。
- 确认 stable virtual-rate/effective-time/provenance 当前只完整接入 nuScenes；Waymo 与统一 endpoint/per-tracklet tracking logger 仍是 benchmark 缺口。
- 将论文优先级改为 P0-C-D1 -> 同提交 TWC A/B/C seed42 -> 通过后多 seed/完整数据/第二数据集；TWC 与 residual 均失败则转多模型 variable-rate benchmark/diagnosis。
- 完整报告见 `compare_results/reports/paper_viability_and_execution_20260720.md`，并已同步更新 `README.md`、`refined_plan.md`、`need_to_do.md`、`sum_results.md` 与根目录早期思路文档的状态说明。
