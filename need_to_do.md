# CT-SeqTrack 当前执行清单

更新时间：2026-07-21

本文只维护会影响论文结论的未完成工作，并按重要性排序。已完成内容见 `done.md`，结果口径见 `sum_results.md`，研究定位见 `refined_plan.md`。新候选方法的完整公式、代码落点和分阶段验收见 `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md`。

## 当前阶段状态

> **决策：前序核心筛选实验已经结束，项目自 2026-07-21 起正式进入 M 阶段。当前活动阶段是 M0；允许并行进行 M1 的代码骨架与零初始化等价性准备，但 M1 正式训练以及 M2–M4 尚未解锁。**

| 阶段 | 状态 | 当前允许的工作 |
| --- | --- | --- |
| 旧路线筛选：P0-B / P0-C / TWC | **已关闭** | 只保留冻结输出和论文失败边界，不再补训练 seed 或复活旧 Gate/TWC 组合 |
| M0：冻结输出、oracle 与 candidate 审计 | **进行中** | endpoint logger、P0-C-D1、A/B/C strong-cadence/path variance、residual oracle、candidate 伪速度审计 |
| M1：物理一致 augmentation + zero-init dual clock | **工程准备可开始** | 可搭接口、配置、单元测试和 A1 数值等价性；完成 M0 关键诊断前不提交正式训练结论 |
| M2：proposal innovation | **锁定** | 仅在 crop-reachable `d_obs -> d_dyn` oracle 通过后实现和训练 |
| M3：asymmetric path distillation | **锁定** | 仅在 M1/M2 的 `true-dt` 同时优于 fixed/shuffled 且不破坏 A1 后启动 |
| M4：filter / trajectory tube | **锁定** | 仅在 proposal 互补性、predicted-history tube oracle 和 uncertainty calibration 均通过后启动 |

“进入 M 阶段”不等于“新方法已经成立”。在 M1/M2 通过预注册时间负对照之前，文稿中仍只能称其为 candidate method。

## 0. 当前主线与结论边界

当前主线收敛为：

> 面向不规则采样和变帧率 3D 单目标跟踪，先建立冻结、可复现的 variable-rate / held-out-cadence 评测协议，量化 search reachability、递归漂移和时间接入方式的失败边界；只有预注册的 `true/fixed/shuffled-dt` 控制出现因果正信号，才保留轻量时间模块。

目前只能确认：时间戳、virtual-rate、TWC、feature dynamics 和 bounded residual 已有不同程度的工程实现或实验信号；还不能声称真实 `delta_t` 稳定提分，也不能声称模型已经具备跨采样率泛化能力。

现有结果的使用边界：

- 同提交 A1 TWC A/B/C seed42 已完成：`B-A=-15.30/-24.18`、`C-B=+8.31/+11.74`、`C-A=-7.00/-12.44`。TWC 相对 paired control 有净效应，但只恢复 paired-view 损失约一半，主方法 promotion No-Go。
- A1+TWC 的 `main_time_source=order` 且 `use_dynamics_encoder=false`；现有 `C-B` 只能支持 resampling consistency 的局部机制，不是 `delta_t` 收益。
- corrected A2+TWC seed42 为 `-0.93 / -2.07`，暂不继续组合 A2+TWC。
- feature-concat A2 在 random20 为正，在 gap1124 和 burst-drop 为负，不能作为主创新结论。
- residual A2 已完成 standard 的真实 batch warmup/active forward-loss-backward 诊断，但默认实际修正仅约 `1e-7 m`、gate 梯度极小，尚未完成 2-step optimizer、完整 split、强 gap 或跟踪性能验证。
- P0-B2 三协议递归诊断已完成：predicted-history CV 相对 previous-A1 只提高 2.65–3.03 pp，未达到预注册门槛；它在可靠历史桶达到 97.34%–98.64% recall，但在上一预测误差超过 4 m 后两者都近乎失效。always-on raw CV recenter 已 No-Go。
- P0-B3 三协议已完成并复核：13 特征 trigger 通过预注册判据，但 passive raw-CV union gain 仅 2.88–3.15 pp，当前 selector 在强协议 AUROC 为 0.605/0.433；正式决定为 `RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`。消融表明预测力主要来自 `prev_obs_*`，raw `current_delta_t` 是跨协议失准的主要来源，因此只能称 observation reliability Conditional-Go，不能称 timestamp-aware reliability 已成立。
- P0-B4 已在独立 mini_val 上完成冻结验证：`observation_v1` 在 gap/burst 的 AUROC 为 `0.680/0.712`、运行点 recall 为 `0.568/0.609`，均未通过 `0.75/0.70` 门槛；正式决定为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`。同批 raw-CV 第二 crop 在两个强协议的 trajectory-only endpoint 都为 0，因此 reliability-controlled anchor 与 active dual-anchor 路线停止，不在 mini_val 上重调。
- TrajTrack 的 `64.94 / 79.07` 来自 GT-assisted evaluator，只能作为 oracle 诊断。

## 1. 四个核心决策问题（执行顺序见第 2 节）

P0-B 已在独立验证入口处 No-Go，P0-C A2 true-dt 与 P1-D TWC A/B/C 也都未通过主方法 promotion。旧路线不再启动新训练；当前正式执行 M0，并允许并行搭建 M1 的代码骨架、配置和零初始化等价性测试。M1 的正式训练与 M2–M4 必须按第 2、3 节逐级解锁，不启动复杂 trajectory/gate 扩展。

### P0-A：bounded residual 可能小得几乎不起作用

**问题**：当前默认最大修正量仅为 `0.1 × 0.2 × 1.0 = 0.02 m`，而 gate 近零初始化。若真实 observation error 明显大于 2 cm，或 alpha/梯度长期接近零，该分支即使存在也几乎不改变预测。

**2026-07-17 状态**：standard active 64-batch 的 observation error P50/P75/P95 为 `0.213 / 0.577 / 3.838 m`；alpha 固定约 `2e-5`，实际 residual P50 仅 `7.25e-8 m`，gate grad P50 仅 `4.00e-10` 且 31/64 batch 为 0。默认配置数值稳定，但没有通过“非平凡修正幅度”验收。完整证据见 `compare_results/reports/p0_ab_diagnostics_20260717.md`。

- [x] 在 standard 真实 full-history batch 上完成 warmup 2-batch 与 active 64-batch forward/loss/backward。
- [ ] 先核对目标定义：`motion_pred` 与 `dynamics_displacement_pred` 都是完整 displacement，当前直接相加可能重复计算运动。
- [x] 确认 warmup 内 residual/gate gradient 为 0；active 的 alpha、raw/clamped/applied residual 与 gradient 全部 finite。
- [ ] 在 crop-reachable subset 保存 `d_obs / d_dyn / d_gt`，计算线段 `d_obs -> d_dyn` 的 oracle 最优 blend 与 long-gap/sparse 分桶。
- [x] 确认 residual observation head 输入仍为 256 维，没有额外拼接 `z_dyn`。
- [ ] 只有 oracle blend 通过预注册收益门槛，才把公式改为 `d_obs + alpha * clamp(d_dyn - stopgrad(d_obs))`，再补 invalid-history、gap/burst forward/backward 和 2-step optimizer smoke。
- [x] 记录 gate bias、gate gradient norm、alpha 分位数、applied ratio、clamp ratio 和 applied norm；注意当前 `applied_ratio=1` 只表示 norm 大于 `1e-8`，不表示修正有实际作用。
- [ ] oracle 通过后才依据训练 split reachable subset 一次性预注册 gate init/scale/bound，并运行一个 seed42 `true/fixed/shuffled`；不得根据 test/mini_val 涨跌反复调参。

**验收**：当前默认配置未通过。oracle blend 无稳定空间即终止 residual；只有 oracle 与修正后的 seed42 time-control 均通过，才允许正式训练和多 seed。

### P0-B：长 gap 的失败可能发生在 search crop 之前

**问题**：当前 search crop 围绕最近历史框生成。长间隔或 burst-drop 下，目标可能在进入网络前已经离开 crop；此时无论最终 residual 多准确，2 cm 级后处理都无法追回目标。

**2026-07-17 状态**：oracle 与 recursive predicted-history 均已完成。GT-history CV 在三协议约为 99% recall，但 predicted-history CV 相对 previous-A1 仅提高 2.91/2.65/3.03 pp，低于总体 +5 pp 门槛；强协议 `>4 m` 位移桶只提高 8.45/9.96 pp，也未同时达到 +10 pp。上一预测误差不超过 4 m 时 pred-CV recall 为 98.59%/97.34%/98.64%，超过 4 m 后只有 0.80%/1.21%/1.61%。完整证据见 `compare_results/reports/p0b2_recursive_crop_reachability_20260717.md`。

**2026-07-20 P0-B3 状态**：三协议 full passive diagnostic 已完成并在本地通过 endpoint/hash/schema/逻辑一致性与指标复算。全特征 trigger AUROC 为 `0.857/0.787/0.785`，但强协议校准和误报恶化；raw-CV union gain 为 `3.04/2.88/3.15 pp`，低于 5 pp；selector AUROC 为 `0.729/0.605/0.433`。`prev_obs_only` 在 gap/burst 反而达到 `0.867/0.873`，删除 raw `current_delta_t` 后达到 `0.865/0.872`，说明当前信号是 observation-quality proxy，而不是已验证的 timestamp mechanism。

**2026-07-20 P0-B4 状态**：mini_train standard 上一次性拟合5特征 `observation_v1`，在 disjoint mini_val 三协议冻结评估。gap/burst AUROC 为 `0.680/0.712`，固定阈值 recall 为 `0.568/0.609`，两者均未通过预注册门槛；Brier 还略差于各强协议 prevalence 常数基线。最终判定为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`。完整复核见 `compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。

- [x] 在 standard 模型 forward 前统计 target-in-base-crop recall、center-outside 和目标点保留率。
- [x] 补齐 gap1124、burst-drop 的 summary/CSV，并按 `current_delta_t`、真实位移和目标点数分桶。
- [x] 在 standard 比较 base、2x expanded、GT-history constant-velocity recentered crop 的 oracle recall 与背景点开销。
- [x] 新增独立 `diagnose_recursive_crop_reachability.py`，被动比较四种 anchor，并记录 endpoint/hash/连续失败信息。
- [x] 对三协议递归误差、连续失败、empty fallback 和可靠/失控分桶完成检查，确认预测历史存在灾难性长尾漂移。
- [x] 在服务器运行 predicted-history 诊断：四种 anchor 使用完全一致 endpoints、同一 checkpoint，missing/unexpected 均为 0。
- [x] 按预注册门槛判定 always-on raw predicted-history CV recenter 为 No-Go，不接成唯一 search anchor。
- [x] 新增独立 P0-B3 passive dual-forward logger，记录 foreground/crop points/empty fallback/CV shift/speed/proposal agreement、稳定 tracklet key、hash 和严格 GT-only 离线标签。
- [x] 新增纯 NumPy 分组汇总器，把 pre-crop trigger、current-crop evidence 和 post-crop selector 分开评估，并输出 AUROC、AUPRC、Brier、ECE、固定阈值运行点及 Go/No-Go。
- [x] 新增独立 `validate_observation_reliability.py`：固定5个非冗余 `observation_v1` 特征，只在 standard fitting CSV 上拟合预处理、logistic 权重和阈值；强制 fit/eval tracklet 不重叠，并在独立协议上冻结评估。新增 `run_p0b4_observation_validation.sh` 串联 mini_val 三协议 reference、passive logger、完整性检查和最终验证。
- [x] 在服务器完成 self-test、model-load smoke、10-tracklet smoke 和 standard/gap1124/burst-drop 三协议 full P0-B3；reference endpoints 与同一 checkpoint exact match。
- [x] 按预注册门槛判定 reliability proxy 并检查 passive raw-CV dual crop union gain：trigger 通过，但 raw-CV anchor 不通过；current foreground 未冒充 pre-crop trigger。
- [x] 判定当前 post-crop selector No-Go，不进入 active proposal selection。
- [x] 诊断输出已自动记录 git 状态、脚本/config/CSV/checkpoint SHA256。P0-B3 与 P0-B4 服务器 summary 都记录为 `f28f495...` dirty；P0-B4 exact server script 未随结果包回传，但本地当前算法复算指标一致到约 `1e-15`。该批可用于 No-Go，后续正式运行必须改为 clean GitHub commit。
- [x] 完成 P0-B4 10-tracklet smoke 与完整 mini_val；独立冻结验证得到 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`，不得在 mini_val 上重调后重报。
- [x] 根据预注册入口条件取消 reliability-updated Kalman/frozen-state passive anchor；这表示路线停止，不表示该模块已经实现。
- [x] 取消 reliability-controlled anchor 的 `true/fixed/shuffled-dt`、active selection 和 learned gate；这些控制只在 P0-C 的通用时间协议中保留，不再用于复活当前 calibrator。
- [ ] expanded/recentered crop 必须保持相同训练步和模型容量，不能把更大搜索区收益写成真实时间收益。

**验收**：P0-B4 已触发 No-Go。当前 frozen observation reliability、raw-CV candidate、post-crop selector 和 active dual-anchor 全部停止；不增加新 reliability 特征或更大 trajectory encoder。P0-B 只保留为论文中的机制诊断与失败边界。

### P0-C：当前 HTV 实验还不是“未见 cadence 泛化”

**问题**：train/val 目前复用同一组 `virtual_rate_*`，已有六组结果更接近“分别在各协议上训练和评测”。这不能支持“一个模型跨采样率泛化”的论文主张；现有 manifest 的 split 内序号建键也不适合正式冻结协议。

- [x] 工程上拆分 `train_virtual_rate_*` 与 `val/test/eval_virtual_rate_*`，允许 standard-train、variable-rate-test；旧配置仍回退到无前缀字段。
- [x] 增加 `virtual_rate_manifest_train / val / test`；v2 manifest 使用 dataset version + split + scene token + instance/tracklet token 稳定建键。
- [x] v2 manifest 记录 protocol、seed、endpoint 数、代码 commit、selection/content/file SHA256；schema、split、role、protocol、tracklet set、长度或 hash 不匹配时 fail fast。
- [x] 在训练采样和递归评测的同一字段契约中实现 `dynamics_time_mode: true | fixed | shuffled`。
- [x] `fixed/shuffled` 只改变 dynamics effective time；旧 `delta_t/current_delta_t` 明确保留为 real-time alias，主干 order-time、frames、crop、candidate、标签和 real-time velocity supervision 不变。
- [x] batch 同时保留 `delta_t_real/effective`；shuffled 使用离线冻结、split 内一一 permutation、累计 effective timestamp 和 mapping hash。
- [x] 增加 `tools/check_p0c_time_controls.py`：纯函数 self-test 与真实 batch 三路不变量检查已实现。
- [x] 明确禁用未接入模型的 `dynamics_use_acceleration=true`，避免产生伪消融。
- [x] `main.py` 每个 run 写出 `run_provenance.json`，保存 commit、dirty status、原始/解析后 cfg hash、manifest/mapping hash、seed、checkpoint hash 和 checkpoint 规则。
- [x] 在服务器 clean commit `343145d` 上生成 val/test cadence manifest 和 test shuffled-time manifest；gap1124 保留 `91/106` tracklets、`1257/2285` frames，test selection SHA256 为 `85e5603c...f9649f6f`，shuffled mapping 为 `1257 endpoints / 1166 transitions`，真实 nuScenes batch invariance 输出 PASS。
- [x] 用 standard-trained seed42 A2 60ep `last.ckpt`（SHA256 `b508f958...24ac87ad`）完成 gap1124 `true/fixed/shuffled-dt` 三次评测；三份 provenance 的 commit/config/checkpoint/selection/91 tracklets/1257 frames 一致。true 相对 fixed 为 `+0.438/+0.523`，相对 shuffled 为 `-0.123/+0.056`，未达到 `+0.5 Success / +1 Precision`。
- [ ] P0-C-D1 只补一次输出型复跑：保存 per-tracklet/endpoint 指标、gap bin、首次失控、连续失败与 empty fallback，计算三路 paired delta；不改 checkpoint、模型、manifest、seed 或阈值。现有 aggregate outputs 无法完成该诊断。
- [x] 根据预注册规则判定 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；不扩展 burst-drop、未见 fixed-gap或多 seed，也不把 A2 gap1124 表现归因于正确 physical-time alignment。

**验收**：协议、公平输入、原始 manifest/hashes、三路 aggregate 性能与 provenance 均已验收；A2 true-dt promotion 为 No-Go。完整报告见 `compare_results/reports/p0c_frozen_protocol_validation_20260720.md`。P0-C 只剩可选但预注册的一次 D1 失败定位，不再是模型扩展入口。

### P1-D：TWC A/B/C 已完成，主方法 promotion No-Go

**问题**：当前 TWC 目标同时包含两条历史视图的 supervised loss，因此单 seed 提升可能来自 paired-view 数据增强，而不一定来自 consistency loss。

```text
A. single-view A1
B. paired views + twc_weight=0
C. paired views + corrected-TWC
```

- [x] A/B/C 来自同一提交 `343145d`，使用相同 seed42、candidate4、optimizer steps 和 checkpoint 规则；12 个评测点与 final step75720 一致。
- [x] 已完成效应分解：Final `B-A=-15.30/-24.18`，`C-B=+8.31/+11.74`，`C-A=-7.00/-12.44`。
- [ ] evaluation-only 测试同一 endpoint 的多条合法历史路径，报告 center/angle gap 和 prediction variance。
- [x] seed42 的 `C-B` 为正，但 C 仍显著低于 A，触发 standard guardrail；不补 seed43/44。
- [x] TWC 不作为当前主贡献；只保留“部分修复 paired-view 退化”的机制结果。若输出型收尾只降方差不涨指标，最多写作稳定性 regularizer。
- [x] 不启动 A2/residual+TWC 组合。

**验收**：augmentation 与 consistency 净效应已经分开，`C-B` 已证明；但 C 没有恢复到 A，故 `NO_GO_TWC_MAIN_METHOD_PROMOTION`。完整报告见 `compare_results/reports/twc_abc_seed42_comparison_20260721.md`。剩余 evaluation-only path variance 只作冻结 checkpoint 收尾，不再作为多 seed 入口。

## 2. M 阶段启动与立即执行清单

P0-B、P0-C 和 TWC 主方法 promotion 均已 No-Go，旧路线的核心筛选阶段已经完成。现在从 M0 开始：先用冻结输出和离线 oracle 关闭证据缺口，同时可以准备 M1 工程；这不授权直接启动新的大模型训练。

实际执行顺序：

1. P0-B4 与 P0-C 已绑定 commit `343145d`；服务器三路结果、原始 manifest、console/events/provenance 已拉回并通过本地 hash/指标复算。
2. P0-C aggregate 判定为 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；不补 burst、未见 fixed-gap或多 seed。
3. 同提交 TWC A/B/C standard seed42 已完成并判定 `NO_GO_TWC_MAIN_METHOD_PROMOTION`；不补 seed43/44。
4. **M0-1/2**：复用一个 endpoint/per-tracklet logger，完成 P0-C-D1；并对冻结 A/B/C final checkpoint 评测 standard、gap1124、burst-drop 和 unseen fixed gap，报告 paired delta、path variance、首次失控、连续失败和 empty fallback，不改变预测路径。
5. **M0-3**：P0-A 只做 crop-reachable `d_obs -> d_dyn` oracle blend，保存 `d_obs/d_dyn/d_gt`，并报告 long-gap/sparse bins；oracle 不通过即停止 M2。
6. **M0-4**：离线审计 candidate0/1/2/3 的 velocity、acceleration 和 dynamics proposal error，量化逐历史框独立 jitter 制造的伪速度；该项可与第 4、5 项并行。
7. **M1 工程准备**：在不启动正式训练的前提下，搭建 physical-consistent augmentation、zero-init dual-clock adapter、配置开关和测试；关闭 adapter 时、以及其零初始化时，输出必须在数值容忍度内恢复同提交 A1。
8. **M1/M2 训练解锁**：candidate 审计完成后才冻结一种物理一致 augmentation；M2 还必须等待 residual oracle 通过。第一轮只跑预注册 seed42 mini true/fixed/shuffled，不扫大网格。
9. **M3 解锁**：只有 dual-clock/innovation 的 true-dt 同时超过 fixed/shuffled、满足 standard guardrail 且相对新的 single-path control 为正，才实现 asymmetric canonical-teacher -> irregular-student path distillation。
10. **M4 解锁**：只有 state prior、predicted-history tube oracle 和 uncertainty calibration 均通过，才实现 continuous-discrete filter / trajectory tube；旧 hand-crafted Gate 不复活。
11. 若 M0 oracle/candidate 审计和后续 time-control 都无可推广正信号，正式收敛为多模型、多数据集的 variable-rate 3D SOT benchmark/diagnosis，不增加复杂时序模块。

### M0 完成定义

- [ ] 整理当前文档、脚本和配置并建立可回查的 clean code/config commit；大体积结果不要求入库，但必须保存路径与 SHA256 索引。新正式运行不得沿用 dirty provenance。
- [ ] P0-C-D1 三路 per-tracklet/endpoint 输出完成并通过 endpoint/hash/checkpoint 一致性检查。
- [ ] 冻结 A/B/C 四协议输出和 evaluation-only multi-path variance 完成。
- [ ] crop-reachable proposal oracle 完成，明确 `GO_M2_PROPOSAL_INNOVATION` 或 `NO_GO_M2_PROPOSAL_INNOVATION`。
- [ ] candidate 伪速度审计完成，并在 shared SE(2) 与 smooth drift 中只冻结一个 M1 augmentation 方案。

以上五项完成后，M0 才能标记完成。trajectory-tube oracle 是 M4 的单独前置条件，不阻塞 M1/M2，但必须在 M4 实现前完成。

| 方法 | 目的 |
| --- | --- |
| A1-order | observation baseline |
| A2 feature-concat true-dt | 旧时间接入方式参考 |
| observation-reliability-updated Kalman/frozen-state | P0-B4 入口 No-Go，停止实现 |
| A2 residual true-dt | reachable-subset refinement 消融 |
| A2 residual fixed-dt | 同容量时间负对照 |
| A2 residual shuffled-dt | 物理时间对应关系负对照 |
| constant-velocity/Kalman | 无学习、GT-free 低复杂度轨迹基线 |
| dual-clock adapter true/fixed/shuffled | 保留 order clock，验证 zero-init physical-time 增量是否有因果收益 |
| proposal innovation | 在 `d_obs` 与 `d_dyn` proposal 之间有界插值，禁止完整位移相加 |
| asymmetric path distillation | canonical EMA teacher 监督 irregular true-time student；第一轮 `beta=0` |
| continuous-discrete filter / trajectory tube | 后置高上限方案；只有先验、crop oracle 与 calibration 通过后执行 |

统一要求：

- [ ] 所有学习方法使用同一 commit、candidate4、manifest、batch 规则、optimizer steps 和预先规定的 checkpoint 口径。
- [ ] 主结果优先使用预先固定的 final epoch；如使用 best，只能由独立 validation metric 选择。
- [ ] 报告逐 tracklet paired delta，不从多个 epoch 中事后挑最高 test 结果。
- [ ] 第一轮不扫大网格；若 true-dt 没有同时优于 A1、fixed-dt 和 shuffled-dt，先定位机制，不补 seed43/44。
- [ ] 只有 seed42 出现因果正信号，才补 seed43/44，并报告 mean±std 与 tracklet-level paired bootstrap CI。

### 第一轮 Go 条件

- TWC 的 `C-B` 已为正，但 `C-A` 在 standard 明显退化，主方法 gate 已失败；剩余同 endpoint prediction variance 与强协议只用于机制收尾。
- 强 gap 上若有收益，仍需逐 tracklet paired delta 证明不只来自单个 tracklet；不能用它事后取消 standard No-Go。
- 若进入 residual，oracle blend 必须先有空间，随后 true-dt 同时优于 fixed-dt 和 shuffled-dt，并具有非平凡 applied ratio。
- 新 dual-clock/innovation 的 mini promotion 还要求：standard 相对 A1 不低于 `-0.5 Success / -1.0 Precision`，且 gap1124 或 burst-drop 至少一项达到 `+1 Success / +2 Precision`。
- asymmetric path distillation 必须相对新的 single-path dual-clock control 为正，并且最终不低于 A1；不能只相对受损的 paired control 为正。

若不满足，按第 4 节诊断，不立即增加网络复杂度。

## 3. M1–M4 分阶段解锁

### M1：物理一致 augmentation 与 zero-init dual clock（工程准备已解锁）

- [ ] 读取并冻结 M0 的 candidate0/1/2/3 审计结论；不得在 M1 中用训练结果反向改写伪速度判据。
- [ ] 在 shared SE(2) 与 smooth drift 中预注册一个正式 augmentation；Dynamics label 从 canonical/一致扰动轨迹计算。
- [ ] 实现 zero-init physical-time adapter；初始化后与 A1 在同 batch 上的输出误差必须低于数值容忍度。
- [ ] 保留 SeqTrack3D order embedding；禁止再次用 raw seconds 直接替换主干 order token。
- [ ] 正式训练前冻结 clean commit、唯一 augmentation 定义和 seed42 mini 配置；不得边看 test 结果边改变扰动过程。

### M2：Proposal innovation（尚未解锁）

- [ ] 只在 crop-reachable oracle blend 通过后实现：`innovation=clip(d_dyn-stopgrad(d_obs), R(delta_t))`。
- [ ] `d_final=d_obs+alpha*innovation`；删除旧正式路径中的完整 `d_obs + alpha*d_dyn` 语义。
- [ ] `alpha=0` 必须严格恢复 A1；记录 innovation norm、applied ratio、clamp ratio、alpha 分布和梯度。
- [ ] mini seed42 只跑一个预注册配置，并用同一 checkpoint 做 true/fixed/shuffled；不扫 scale/gate 网格。

### M3：Endpoint-consistent asymmetric path distillation（尚未解锁）

- [ ] canonical dense path 使用 EMA teacher；irregular true-time path 使用 student。
- [ ] 第一轮固定 `beta=0`：`L=L_sup_A+lambda_path*w_A*D(stopgrad(p_A),p_B)`。
- [ ] `w_A` 只能来自 teacher 的推理时可得 confidence/uncertainty，不能读取当前 GT。
- [ ] fixed/shuffled 只用于因果评估，不进入 path consistency 训练。
- [ ] 不复用当前 corrected-TWC checkpoint 继续训练；新设计必须从同提交 A1 初始化并重新做 A/B/control。

### M4：Continuous-discrete filter 与 trajectory tube（尚未解锁，后置）

- [ ] 进入条件：M2 dynamics proposal 有互补性、true-dt 通过两负对照、predicted-history tube oracle 有明显 crop recall 空间。
- [ ] 先实现固定 process/observation covariance 的 GT-free filter baseline，再决定是否学习 `Q(delta_t)` 和 `R_obs`。
- [ ] trajectory tube 沿传播方向增长、横向受限，并与 baseline crop 共享固定 point budget/FLOPs 口径。
- [ ] learned covariance 必须先通过 NLL、coverage、ECE/calibration，再允许进入 tracking 闭环。
- [ ] 旧 P5 hand-crafted observability Gate 不复活、不在 mini_val 上重调，也不列论文贡献。

### P1：未见 cadence 泛化

- [ ] Standard-only train：一个 checkpoint 直接测试 standard、gap1124、burst-drop 和未见 fixed gap。
- [ ] Mixed-cadence train：训练时故意留出至少一种 gap pattern 和一种 drop probability。
- [ ] Unseen-schedule test：不重训、不改 threshold，直接测试 held-out schedule。
- [ ] 分开报告 in-domain、seen-cadence 和 unseen-cadence；只有最后一项成立，才写“跨采样率泛化”。

### P1：完整数据与统计

- [ ] mini 通过后迁移到完整 nuScenes trainval，先完成 Car，再决定是否扩展类别。
- [ ] 至少补一个第二数据集或官方 HTV 协议；Waymo 需要先补齐等价 virtual-rate manifest 支持。
- [ ] 三个 seed 报逐 seed、paired mean±std、final/late mean 和 tracklet-level bootstrap；不能把序列帧当独立样本 bootstrap。
- [ ] 公平基线至少包含同提交 SeqTrack3D/A1、feature A2、CV/Kalman 和 TrajTrack GT-free。
- [ ] 最终再报告参数量、FLOPs、FPS、显存和新增分支开销。

### P1：TrajTrack GT-free 公平对照

- [ ] 固定现有 epoch60 checkpoint，评测 `pre_wo_refine()` 与只依赖 local/global proposal agreement 的 GT-free hard switch。
- [ ] `pre_w_refine()` 只作为 GT-assisted oracle，不能进入主表排名。
- [ ] proposal 选择函数不得接收当前 GT、`this_bb` 或 GT-derived mask。
- [ ] 增加 GT 独立性测试：只改变当前 GT、保持输入和 proposals 不变，GT-free 输出必须不变。
- [ ] 固定 checkpoint、tracklet 顺序、evaluator、threshold，并报告 Success/Precision 与 FPS。

## 4. 失败时的额外诊断

P0-A/P0-B/P0-C 已覆盖 residual、crop 和协议检查；这里只保留会影响下一步模型设计的额外诊断，并按实验失败类型触发。

### 若 dynamics 在强 gap 退化：检查 candidate 伪速度

- [ ] 比较 candidate0 与 candidate1/2/3 的 velocity/dynamics proposal error。
- [ ] 用共享刚体扰动或时间相关轨迹扰动，对照当前逐历史框独立扰动。
- [ ] 从 A1 递归预测误差拟合更真实的 candidate 噪声，再判断是否需要 reliability weight。

### 若 true/fixed/shuffled 差异异常：检查时间与样本质量

- [ ] 统计 timestamp 缺失、fallback、重复、非单调、单位异常和 `delta_t` 分布。
- [ ] 检查协议过滤后的 tracklet 数、endpoint 数、类别和轨迹长度，排除样本难度变化。
- [ ] 检查 real/effective time 日志与 hash，确认没有读取错字段或 permutation 泄漏。

## 5. 暂缓模型扩展

- [ ] 不上 Mamba、复杂 Transformer、ODE/SDE/CDE 或多传感器异步融合。
- [ ] 不同时开发 gate、uncertainty head、TWC 和大 trajectory encoder。
- [ ] 旧 reliability Gate 已由 P0-B4 No-Go；不把它作为新方法入口，也不在 mini_val 上重调。
- [ ] 只有 proposal innovation、time controls 与 tube oracle 先通过，才实现 covariance-based filter；若 calibration 失败则退回固定小 `alpha`。
- [ ] trajectory proposal 只作为预防性第二搜索假设；raw always-on recenter 与已漂移后的单锚点恢复均停止。

## 6. 复现与论文交付底线

- [ ] 从实际训练服务器导出锁定环境，修正 `requirement.txt` 中注释 torch、`tdqm` 拼写、松散版本和停用依赖问题。
- [ ] 最小测试覆盖：manifest split、time switch、residual bound/gradient、GT-free evaluator 和 TWC shared coordinates。
- [ ] 每个论文数字绑定 cfg、manifest、checkpoint、CSV/report 和 commit；README 不作为最终数据源。
- [ ] 主表使用 GT-free、公平训练预算和预先固定的 checkpoint 规则。
- [ ] 不使用“完整 continuous-time tracker”“首次 HTV”“首次使用历史 trajectory”或“full model 稳定超过 SeqTrack3D”等超出证据的表述。

## 7. Stop / Pivot

P0-B4 已触发 reliability-controlled dual-anchor 的 Stop 条件；当前 Pivot 从 P0-C 冻结协议与 benchmark/diagnosis 开始。

满足任一项则停止继续堆时间模块：

- 连续两轮配对实验中，`true-dt - fixed/shuffled-dt < 0.5 Success / 1 Precision`。
- residual 在合理校准后仍长期梯度、alpha 或 applied ratio 接近零。
- 收益只来自扩大 crop、更多参数、更多训练步、checkpoint 选择或按 protocol 分别重训。
- unseen cadence 不成立，或 full data 与 mini 的方向相反。
- corrected-TWC 虽有 `C-B`，但无法恢复到 single-view A；该 Stop 条件已由 standard A/B/C 触发，不再补训练 seed。
- 新 asymmetric distillation 若仍只优于 paired control、不能回到 single-path A1，则停止该分支，不通过调 `beta/lambda` 大网格继续追分。
- dual-clock adapter 若在同一 checkpoint 下不能形成 `true > fixed` 且 `true > shuffled`，停止 physical-time method claim；保留协议资产。

Pivot：把工作收敛为 variable-rate 3D SOT benchmark/diagnosis，或使用可解释的 constant-velocity/Kalman trajectory fallback；不再通过增加复杂时序模块追分。
