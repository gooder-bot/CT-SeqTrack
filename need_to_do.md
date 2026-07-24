# CT-SeqTrack 当前执行清单

更新时间：2026-07-24

本文只维护会影响论文结论的未完成工作，并按重要性排序。已完成内容见 `done.md`，结果口径见 `sum_results.md`，研究定位见 `refined_plan.md`。新候选方法的完整公式、代码落点和分阶段验收见 `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md`。

## 当前阶段状态

> **决策：commit `473738f` 的 R1 final 已完成 standard/gap1124 八组 same-checkpoint controls。M2−A1 为 `+4.133/+9.445` 与 `+2.279/+4.143`（Success/Precision），逐 tracklet bootstrap 的两项 95% CI 均为正；但 true−fixed/shuffled 在 standard 仅 `+0.031/−0.010`、`+0.068/+0.085`，在 gap1124 为 `−0.127/+0.014`、`−0.318/−0.209`。正式状态为 `M2 TRACKING SIGNAL POSITIVE / PHYSICAL-TIME CAUSAL CLAIM NO-GO / METHOD ATTRIBUTION HOLD`。当前不补 physical-time seed，也不以 burst 推翻 gap1124 的因果失败；先做 adapter/innovation 2×2、candidate/target/递归误差审计、A1-init W0 continuation 与 current-code legacy W0。timestamp-conditioned M3/M4 不解锁。**

| 阶段 | 状态 | 当前允许的工作 |
| --- | --- | --- |
| 旧路线筛选：P0-B / P0-C / TWC | **已关闭** | 只保留冻结输出和论文失败边界，不再补训练 seed 或复活旧 Gate/TWC 组合 |
| M0：冻结输出、oracle 与 candidate 审计 | **进行中，3/4 诊断完成** | P0-C-D1、M0-3、M0-4 已完成；只剩 M0-2 A/B/C strong-cadence/path variance 与文档/provenance 收口 |
| M1：物理一致 augmentation + zero-init dual clock | **formal 路径冻结，性能语义待审计** | shared SE(2)、canonical label、zero-init adapter 已用于 R1/R2；R3 collapse 要求补 legacy-candidate W0，并核查训练候选误差与递归预测误差是否同分布 |
| M2：proposal innovation | **tracking 正信号；physical-time No-Go；归因 Hold** | standard/gap1124 controls 已完成；当前只做 evaluation-only 机制审计与两个 matched attribution baseline |
| M3：asymmetric path distillation | **timestamp-conditioned 路线锁定** | 当前 causal-time gate 已失败；若以后改做 time-agnostic endpoint/path robustness，必须作为新假设重新预注册 |
| M4：filter / trajectory tube | **当前时间条件路线锁定** | 不沿当前 R1 时间主张推进；只有独立 state-prior、predicted-history tube oracle 和 calibration 新证据齐备后才重新评估 |

“进入 M 阶段”不等于“新方法已经成立”。本轮预注册时间负对照已经失败：文稿可以报告 M2 tracker 正信号，但不能把涨点归因于正确 physical time，也不能把当前 R1 写成已验证的 timestamp-aware method。

## 0. 当前主线与结论边界

当前主线收敛为：

> 面向不规则采样和变帧率 3D 单目标跟踪，先建立冻结、可复现的 variable-rate / held-out-cadence 评测协议，量化 search reachability、递归漂移和时间接入方式的失败边界；只有预注册的 `true/fixed/shuffled-dt` 控制出现因果正信号，才保留轻量时间模块。

目前可以确认：R1 相对 matched A1 在 standard 与 gap1124 都有正 tracking 信号；但同 checkpoint 的 correct `delta_t` 没有优于 fixed/shuffled，physical-time 因果主张已 No-Go。模型是否具备跨采样率泛化、M2 结构本身贡献多少，仍需分别通过 held-out robustness 与 matched attribution 回答。完整八组复核见 `compare_results/reports/m2_standard_gap8_analysis_20260724.md`。

### 0.1 `delta_t` 可辨识性与 HTV 定位

本地冻结 endpoint 统计显示：standard 的 `delta_t=0.4974±0.0228 s`，CV 只有 `4.59%`，且 `86.55%` 位于 `0.5±0.01 s`；gap1124/burst-drop 的 CV 分别为 `58.94%/62.63%`。因此 standard 的论文角色固定为性能 guardrail，不能单独用于证明物理时间机制；主证据必须来自同一 tracklet 内不规则 cadence、held-out schedule 和同 checkpoint `true/fixed/shuffled`。

受控丢帧可作为 virtual-rate / irregular-observation stress test，但 HVTrack 已在 ECCV 2024 构造 KITTI-HV，不能再使用“首次 HTV/首次 skip-frame”表述。CT 的可辨识边界固定为：within-track irregularity、matched endpoint 时间干预、one-checkpoint 跨 cadence 泛化和 failure diagnosis。完整决策与实验矩阵见 `compare_results/reports/htv_identifiability_and_execution_plan_20260722.md`。

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
- M0-3 gap1124 proposal oracle 已通过：primary cohort 为 `1311 endpoints / 213 tracklets`，oracle gain mean/median 为 `1.118/0.214 m`；更重要的是 `d_dyn` 本身在 `81.31%` endpoint 上优于 `d_obs`，tracklet bootstrap mean `0.803 m`、95% CI `[0.633, 0.988]`，正式决定为 `GO_M2_PROPOSAL_INNOVATION`。
- M0-4 candidate 审计已通过：非零 candidates 的伪速度/伪加速度 P50 为 `0.611 m/s`、`2.128 m/s²`，分别是阈值的 `12.22×/21.28×`；matched proposal error penalty mean `+0.0104 m`，tracklet bootstrap CI `[+0.0093,+0.0155] m`。M1 正式冻结 shared SE(2)，不选 smooth drift。

### 0.2 2026-07-23 代码级复核：当前最强的涨点假设

R3 的训练 loss 正常收敛但递归评测严重塌陷，优先指向训练目标、候选误差过程或递归推理分布不一致，而不是普通的容量不足。逐行复核当前 sampler、forward、loss 和 recursive evaluator 后，记录以下待验证假设：

1. **shared-SE(2) 保留了物理轨迹，却可能把训练历史做得过于干净。** 训练时所有历史 GT 框共享一个刚体误差，anchor-normalized `ref_boxs` 与 canonical GT 轨迹数值一致；递归评测时 `ref_boxs` 来自逐帧 `results_bbs`，误差会随时间漂移、突变并累积。M0-4 证明 independent white jitter 会制造伪导数，但尚未证明恒定 shared-SE(2) 与真实递归误差过程匹配。
2. **canonical physical motion 与 candidate-frame box correction 不是同一个目标。** 设最近 GT 框坐标下的真实位移为 `d_phys`，sample-level candidate 局部平移为 `u=[dx,dy]`、yaw 为 `phi`，则当前 GT 相对扰动 anchor 的中心目标满足 `d_box=R(-phi)(d_phys-u)`。candidate0 中二者相同；candidate1/2/3 中不同。当前 offset 分布的 XY RMS 约为 `0.245 m`，因此不能只按 shape 相同就把两种 displacement 视为同一 proposal。
3. **同一个 post-innovation coarse proposal 同时受到两种语义的监督。** `loss_center_motion` 将其拉向 candidate-invariant `motion_label`，`loss_center` 又通过 `estimation_boxes` 将其拉向 candidate-dependent `box_label`；`dynamics_displacement_label` 则继续监督 canonical `d_phys`。现有 E2 只证明 physical label 不被 candidate 污染，没有证明最终 innovation 两端处在同一 proposal 语义下。
4. **M2 可能既是有效先验，也在补偿 shared-SE(2) 的训练—推理缺口。** 因此 R2−R3 的巨大差值不能直接作为 M2 净增益；必须先做 label/gradient/error-process 审计与 candidate-path 二因素对照。
5. **5-epoch warmup 后的硬切换可能破坏 A1 起点。** 当前 adapter/innovation scale 从 0 直接切到 1；R1 从 epoch5 `44.222/59.165` 降至 epoch10 `39.146/37.837`，R2 也在启用后明显震荡。该现象只登记为下一版优化假设；在 R1 formal controls 与 matched attribution 完成前不修改冻结 warmup。

边界必须分开：

- 当前 R1 的 shared-SE(2)、`alpha=0.75`、`R(dt)`、warmup=5 和 final checkpoint 继续作为已冻结实验合同，不因本次复核重写结果。
- `FREEZE_M1_SHARED_SE2` 仍表示“第一版物理导数审计选择了共同刚体变换”，不等于“shared-SE(2) 已被证明是递归 tracking 的性能最优误差增强”。
- 只有第 2 节的 matched audit 证实上述问题后，才允许在新的 clean commit 中预注册 M1.5；不得在当前 R1 上事后修补并继续沿用同一实验标签。

## 1. 四个核心决策问题（执行顺序见第 2 节）

P0-B 已在独立验证入口处 No-Go，P0-C A2 true-dt 与 P1-D TWC A/B/C 也都未通过主方法 promotion。旧路线不再启动新训练；当前执行 M0 收口，并正式启动 M1/M2 的代码、配置、单测和 smoke。M1/M2 的正式训练以及 M3–M4 必须按第 2、3 节逐级解锁，不启动复杂 trajectory/gate 扩展。

### P0-A：bounded residual 可能小得几乎不起作用

**问题**：当前默认最大修正量仅为 `0.1 × 0.2 × 1.0 = 0.02 m`，而 gate 近零初始化。若真实 observation error 明显大于 2 cm，或 alpha/梯度长期接近零，该分支即使存在也几乎不改变预测。

**2026-07-17 状态**：standard active 64-batch 的 observation error P50/P75/P95 为 `0.213 / 0.577 / 3.838 m`；alpha 固定约 `2e-5`，实际 residual P50 仅 `7.25e-8 m`，gate grad P50 仅 `4.00e-10` 且 31/64 batch 为 0。默认配置数值稳定，但没有通过“非平凡修正幅度”验收。完整证据见 `compare_results/reports/p0_ab_diagnostics_20260717.md`。

- [x] 在 standard 真实 full-history batch 上完成 warmup 2-batch 与 active 64-batch forward/loss/backward。
- [x] 核对目标定义：`motion_pred` 与 `dynamics_displacement_pred` 都是完整 displacement，当前直接相加会造成定义歧义；正式候选改为 proposal innovation。
- [x] 确认 warmup 内 residual/gate gradient 为 0；active 的 alpha、raw/clamped/applied residual 与 gradient 全部 finite。
- [x] 在 crop-reachable subset 保存 `d_obs / d_dyn / d_gt` 并完成 oracle、long-gap/sparse 分桶；决定为 `GO_M2_PROPOSAL_INNOVATION`。sparse 仅 3 个样本，不作结论。
- [x] 确认 residual observation head 输入仍为 256 维，没有额外拼接 `z_dyn`。
- [x] oracle 已通过，公式已改为 `d_obs + alpha * clamp(d_dyn - stopgrad(d_obs))`，dataset-free fallback/2-step 已通过。
- [x] proposal innovation 已完成 invalid-history、gap/burst、empty-search 与 sampler-resampled 真实 batch forward/backward；E0–E6、R1/R2/R3 训练与 standard 结果完整性均已通过。
- [x] 记录 gate bias、gate gradient norm、alpha 分位数、applied ratio、clamp ratio 和 applied norm；注意当前 `applied_ratio=1` 只表示 norm 大于 `1e-8`，不表示修正有实际作用。
- [x] oracle 后已只依据 mini_train reachable subset 一次性冻结 `alpha=0.75`、`R(dt)` 与 warmup；R1 true/fixed/shuffled 同 checkpoint 评测已完成，physical-time gate FAIL，不得根据结果反复调参。

**验收**：旧默认 residual 配置仍未通过，但 M0-3 已证明 proposal 互补性并解锁 M2 公式重构。只有修正后的 seed42 time-control 与 standard guardrail 均通过，才允许补多 seed 和方法 promotion。

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
- [x] P0-C-D1 三路 full 输出型复跑与失败定位完成：true/fixed/shuffled 各 `91` 个 tracklet、`1257` 个 endpoint，endpoint/order/checkpoint/config/selection/manifest exact match，real/effective time 干预通过检查；保存了 per-tracklet/endpoint、gap/位移分桶、首次失控、连续失败、fallback、bootstrap 与 leave-one-tracklet-out 结果。
  - true−fixed 为 `+0.4376 Success / +0.5231 Precision`，true−shuffled 为 `-0.1233/+0.0557`；Success/Precision 的逐 tracklet bootstrap 95% CI 均跨 0，再次确认 promotion No-Go。
  - true 与两个控制各有 `1079/1257` 个 endpoint 的中心预测改变，说明模型会响应时间；但 true 对 shuffled 没有稳定正确性优势。`≥2 s` 桶 true−shuffled 也只有 `0.000/+0.525`。
  - true−fixed 的 mean-error 改善主要来自一条三路均已失控的长尾序列；删除该条后 `-0.191 m` 缩小到 `-0.0397 m`，不能据此晋级。
  - 运行时仓库为 dirty，但 exact exporter/config/checkpoint/manifest/CSV hash 已保存，且 paired 效应复现此前 clean aggregate；足以完成诊断，正式论文归档保留 clean provenance caveat。旧 2-tracklet smoke 只保留作首帧口径修复记录。
- [x] 根据预注册规则判定 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；不扩展 burst-drop、未见 fixed-gap或多 seed，也不把 A2 gap1124 表现归因于正确 physical-time alignment。

**验收**：协议、公平输入、原始 manifest/hashes、三路 aggregate 与 endpoint-level paired diagnosis 均已验收；A2 true-dt promotion 为 No-Go，P0-C 不再有待补实验，也不是模型扩展入口。协议报告见 `compare_results/reports/p0c_frozen_protocol_validation_20260720.md`，D1 完整报告见 `compare_results/reports/m0_p0c_d1_full_analysis_20260721.md`。

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

P0-B、P0-C、TWC 与当前 M2 physical-time promotion 均已 No-Go。M1/M2 现在进入“tracking 正信号归因”阶段；当前授权只覆盖 R1 final 的 evaluation-only 路径分解、语义/递归误差审计和两个缺失的 matched baseline，不自动扩展到新 seed、mixed-cadence、timestamp-conditioned M3 或 M4。

当前启动边界：

- **现在允许**：执行不改权重的 adapter/innovation 0/1 机制消融与 candidate/target/error-process 审计；设计并运行 A1-init W0 continuation、current-code legacy-candidate W0。
- **并行允许**：M0-2 只读评测冻结 A/B/C checkpoint；它不阻塞 M1/M2 写代码，但完成前不能把 M0 标记为完成。
- **暂不允许**：除两个明确 attribution baseline 外启动其他训练；evaluation-only 机制消融不得变成 mini_val scale 搜索；不扫 gate/alpha/R/warmup，不为 physical-time claim 补 seed43/44，不开始 timestamp-conditioned M3/M4。

实际执行顺序：

1. P0-B4 与 P0-C 已绑定 commit `343145d`；服务器三路结果、原始 manifest、console/events/provenance 已拉回并通过本地 hash/指标复算。
2. P0-C aggregate 判定为 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；不补 burst、未见 fixed-gap或多 seed。
3. 同提交 TWC A/B/C standard seed42 已完成并判定 `NO_GO_TWC_MAIN_METHOD_PROMOTION`；不补 seed43/44。
4. **M0-1 已完成**：P0-C-D1 三路 full endpoint/per-tracklet 诊断再次确认 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`，不再追加该 A2 的 cadence 或 seed。
5. **M0-2**：用同一 logger 对冻结 A/B/C final checkpoint 评测 standard、gap1124、burst-drop 和 unseen fixed gap，报告 paired delta、path variance、首次失控、连续失败和 empty fallback，不改变预测路径。
6. **M0-3 已完成**：gap1124 crop-reachable proposal oracle 得到 `GO_M2_PROPOSAL_INNOVATION`；`d_dyn` 本身相对 `d_obs` 的 tracklet bootstrap mean gain 为 `0.803 m`，95% CI `[0.633,0.988]`。
7. **M0-4 已完成**：candidate jitter 明确制造伪速度/伪加速度，matched proposal penalty 的 tracklet CI 不跨 0；M1 唯一 augmentation 冻结为 shared SE(2)，第一版不做 smooth drift。
8. **M1 E0–E5 已完成**：sample-level world-SE(2)、canonical label、zero-init adapter、真实 loader/TWC、A1 strict-zero 等价和 warmup 内两步 optimizer 后 exact-zero 均在 commit `9a0b26d` 通过。
9. **M2 E0–E5 已完成**：独立 proposal-innovation、invalid/empty exact-zero、sampler-resampled coverage、standard/gap/burst finite、active 2-step 与 correction bound 均通过；旧 `residual_limited` 保持原义。
10. **R1/R2/R3 已完成**：三组均为 clean `473738f`、退出码 0、epoch59/global step75720、12 个评测点和 75720 条 loss；R1 的 35 项 manifest 全匹配。
11. **standard 结果已冻结**：R1 `55.303/67.182`、R2 `53.318/62.503`、R3 `28.999/28.023`；只使用 epoch60 `last.ckpt`，不以 best epoch 改写结论。
12. **解释边界已冻结**：R1−A1 有 extra 60-epoch confound；R2−R3 受 shared-SE(2) W0 collapse 影响；二者都不是 M2 相对 SeqTrack3D 的完整因果效应。
13. **R1 formal controls 已完成**：standard/gap1124 八组输出通过 `89/89` artifact hash、endpoint identity、原始 CSV 指标复算和逐 tracklet bootstrap；standard guardrail 与 gap complementarity PASS，两个 causal-time gate FAIL。
14. **physical-time 决策已冻结**：不补 seed43/44，不用 burst-drop 复活当前因果主张；burst 仅在通用 proposal 归因成立后作为额外 robustness/crop 证据。
15. **同 checkpoint 机制消融**：固定 R1 权重，执行 full、adapter-only、innovation-only、both-off 四路 forward；它只回答运行时路径贡献，不替代 matched retraining attribution。
16. **endpoint 与语义审计**：导出 `d_obs/d_dyn/d_final/box_label/motion_label`、innovation norm/clamp、candidate、foreground、gap 和失控状态；检查 target mismatch、双 loss 梯度夹角和训练/递归历史误差分布。
17. **归因补全**：训练 A1-init W0 continuation，排除 extra training；训练 current-code legacy-candidate W0，定位 R3 collapse。
18. **条件补全二因素表**：只有 current-code legacy W0 恢复到有效 baseline，才允许预注册 scratch legacy-candidate M2；它用于判断 M2 在非塌陷 candidate path 上是否仍有净增益，不在看到 mini_val 后扫配置。
19. **M1.5 条件解锁**：只有语义/梯度/误差过程审计或二因素对照支持具体机制，才实现 physical-motion/candidate-correction 分头、相关误差轨迹和渐进 warmup；这是一条新实验线，不回写 R1。
20. **M3/M4 当前决定**：timestamp-conditioned 路线不解锁。若保留 M3，只能改写为 time-agnostic endpoint/path robustness 并重新预注册；M4 必须等待独立 tube oracle/calibration 证据。
21. **论文 Pivot**：当前优先保留 variable-rate benchmark/diagnosis 与通用 proposal correction；不再通过增加复杂时间模块追分。

### 2.1 P0：冻结 checkpoint 的正式评测与机制分解

预注册的 R1 formal controls 已完成：

- [x] standard 与 gap1124 各完成 `true/fixed/shuffled`，三路使用同一 R1 epoch60 `last.ckpt`、同 endpoints、同 cadence manifest 和同 checkpoint rule。
- [x] 同 endpoints 导出 matched A1；保存 endpoint、per-tracklet、summary、provenance、manifest 与 SHA256，包内 `89/89` hash 校验通过。
- [x] 完成 tracklet-level paired bootstrap、real-`delta_t`/位移分桶、预测位移差和 fallback 检查；M2−A1 两协议 CI 均为正，所有 time-control CI 均跨 0。
- [x] 冻结门槛判定：standard guardrail PASS，gap1124 complementarity PASS；standard 与 gap1124 的 `true > fixed/shuffled` 均 FAIL。
- [ ] burst-drop 仅在通用 proposal 路线通过归因后再决定是否补作 robustness；它不是 physical-time gate 的补考。

在不改权重、不改变 architecture/state_dict 的前提下，再做 evaluation-only 2×2：

| 诊断名 | `physical_time_adapter_scale` | `dynamics_innovation_scale` | 解释边界 |
| --- | ---: | ---: | --- |
| full | 1 | 1 | 冻结 R1 原路径 |
| adapter-only | 1 | 0 | 去掉最终 innovation |
| innovation-only | 0 | 1 | 去掉 feature adapter |
| both-off | 0 | 0 | 仅观察共同训练后的主干/辅助监督遗留效应 |

- [ ] 四路只允许使用预声明的 0/1 开关，不扫连续 scale，不根据 mini_val 选最佳值。
- [ ] `both-off` 仍包含被 M2 共同训练过的主干权重，因此它不是 W0 retraining，不能进入“结构净增益”主表。
- [ ] 若 both-off 保留大部分 R1 收益，优先研究 continuation/representation effect；若 innovation-only 保留主要收益，优先修 proposal 语义；若 adapter-only 为主，优先验证真实时间因果性而不是扩大 dynamics head。

### 2.2 P0：candidate、target、gradient 与递归误差过程审计

该审计应先于任何新复杂模块，输出独立 JSON/CSV 与可复查脚本：

- [ ] 对同一 full-history endpoint 固定 point seed，分别构造 candidate0/1/2/3，验证 shared-SE(2) 下 `ref_boxs≈canonical_ref_boxs`，同时统计 `||box_label[:3]-motion_label[0,:3]||` 与 `||box_label[:3]-dynamics_displacement_label||`。
- [ ] 按 candidate 分组报告 target mismatch 的 mean/P50/P95/max、translation/yaw 分量和 crop/foreground 条件；candidate0 应接近 0，非零 candidate 必须与 sampled transform 的解析公式一致。
- [ ] 分别对 `loss_center`、`loss_center_motion` 求 `motion_pred` 或 `motion_mlp` 参数梯度，报告 cosine、norm ratio 与负夹角比例；重点比较 candidate0 和 candidate1/2/3。
- [ ] 从 frozen A1、R1、R2 的递归输出导出历史预测误差序列，与 training independent/shared-SE(2) 的 error、error velocity、error acceleration、yaw drift、连续失控长度比较；不得只比较单帧 offset 方差。
- [ ] 在训练/评测 logger 中记录 `d_obs/d_dyn/d_final` 对 `box_label` 与 `motion_label` 的两套误差、raw/clamped/applied innovation norm、radius、clamp/applied ratio、adapter norm、candidate、foreground、gap 和 first-failure 状态。
- [ ] 明确区分“canonical physical label 不变量通过”与“最终 proposal 语义一致”；前者不能替代后者。

审计判定：

| 观察 | 解释 | 下一步 |
| --- | --- | --- |
| nonzero candidate target mismatch 大且双 loss 梯度经常相反 | physical motion 与 candidate correction 混用 | 解锁 M1.5 分头目标设计 |
| shared 训练误差过程远窄于递归误差，legacy W0 恢复 | shared-SE(2) train/inference mismatch | 解锁相关误差轨迹增强 |
| 两类 mismatch 均弱，legacy W0 也塌陷 | 问题更可能来自当前代码/训练配置 | 停止新结构，做 current-code A1 exact replication |
| R1 true 不超过 fixed/shuffled | M2 可能只是一般运动先验/容量 | 停止 physical-time method claim，保留 benchmark 与通用 proposal 诊断 |

### 2.3 P1：两个必需 baseline 与条件二因素对照

当前必须新增且只有两个训练：

1. **A1-init shared-SE(2) W0 continuation**：与 R1 相同 A1 init、60 epoch、75720 step、seed、数据、checkpoint rule，关闭 DynamicsEncoder、adapter、innovation 与其 auxiliary loss。
2. **current-code scratch legacy-candidate W0**：与 R3 相同 commit、seed、step、batch、数据和 checkpoint rule，只把 candidate path 恢复为 legacy independent。

完整二因素表为：

| Candidate path | W0 | M2 |
| --- | --- | --- |
| legacy candidate | current-code scratch W0（必需） | scratch legacy M2（条件触发） |
| shared-SE(2) | R3（已有） | R2（已有） |

分支规则：

- [ ] 若 A1-init W0 接近 R1，R1−A1 主要是 continuation/shared-data effect，M2 净贡献不足；先停止扩 M2。
- [ ] 若 A1-init W0 明显低于 R1，保留 M2 净贡献候选，再结合 same-checkpoint 2×2 判断 adapter/innovation 主路径。
- [ ] 若 current-code legacy W0 恢复到历史 A1 附近而 R3 继续塌陷，确认 candidate path 是关键交互；随后才运行一次冻结配置的 scratch legacy M2，完成二因素表。
- [ ] 若 current-code legacy W0 也塌陷，先做 current-code A1 exact replication 和 loss/label audit，不启动 legacy M2、M3 或 M4。
- [ ] 如果 legacy M2 相对 legacy W0 仍为正，优先把“canonical supervision + realistic error augmentation + bounded innovation”作为涨点主线；如果只在 shared 路径为正，则必须把 M2 写成对 shared 训练缺口的耦合补偿，不能泛化成独立模块收益。

### 2.4 P2：M1.5 性能修复（条件解锁，不属于当前 R1）

只有 2.2/2.3 给出明确证据后，才在新 clean commit 预注册以下最小改造：

1. **目标分头**：`d_phys` 只表示最近 canonical GT frame 下的 physical displacement/velocity；`d_box` 只表示当前 GT 相对最新 candidate/predicted anchor 的 proposal。最终 coarse box 和 Transformer 只消费 candidate-frame proposal，不再让一个 post-innovation 输出同时承担两种目标。
2. **同语义 innovation**：只有 `d_obs_box` 与 `d_prior_box` 已位于同一 anchor frame、同一预测语义时，才计算 `d_obs_box+alpha*clip(d_prior_box-stopgrad(d_obs_box))`；canonical `d_phys` 不再未经转换直接参与 candidate correction。
3. **相关误差轨迹增强**：从 mini_train frozen A1 recursive errors 只拟合一次 bounded AR(1)/correlated drift 的平移、yaw、漂移率和长尾分位数；保留 shared bias，但允许随时间平滑变化。它不得重新把 candidate derivative 当 physical GT label。
4. **渐进启动**：先冻结 A1 主干训练 DynamicsEncoder auxiliary head，再用预声明的 epoch 区间把 adapter/innovation 从 0 线性升到 1；A1 主干使用更低 LR。不得根据当前 mini_val 曲线搜索 ramp 长度。
5. **最小消融**：旧 frozen R1、matched W0、目标分头 only、相关误差 only、分头+相关误差、再加 staged warmup；不同时引入 M3/M4/Gate。

M1.5 的目标是修复训练—递归推理的一致性并争取稳定涨点，不改变 M0-4 对 independent white jitter 伪导数的原始结论，也不把新分支冒充成原 R1 的预注册结果。

### M0 完成定义

- [ ] 整理当前文档、脚本和配置并建立可回查的 clean code/config commit；大体积结果不要求入库，但必须保存路径与 SHA256 索引。新正式运行不得沿用 dirty provenance。
- [x] P0-C-D1 三路 per-tracklet/endpoint 输出完成并通过 endpoint/hash/checkpoint 一致性检查；full paired effect、bootstrap、分桶和长尾敏感性分析已归档，结论为 No-Go。
- [ ] 冻结 A/B/C 四协议输出和 evaluation-only multi-path variance 完成。
- [x] crop-reachable proposal oracle 完成，决定为 `GO_M2_PROPOSAL_INNOVATION`；包含 dynamics-only、trimmed、tracklet bootstrap 与 long-gap 稳健性检查。
- [x] candidate 伪速度审计完成，M1 唯一 augmentation 冻结为 shared SE(2)，第一版排除 smooth drift。

以上五项完成后，M0 才能标记完成。M0-2 与 M1/M2 工程可以并行，但 M0-2 尚未完成时不得把 M0 写成已完成。trajectory-tube oracle 是 M4 的单独前置条件，不阻塞 M1/M2，但必须在 M4 实现前完成。

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
- [x] 第一轮未扫大网格；true-dt 没有同时优于 fixed-dt 和 shuffled-dt，当前转入机制定位，不补 seed43/44。
- [x] seed42 未出现因果正信号，因此不触发 seed43/44；现有 tracklet-level paired bootstrap 已作为 No-Go 证据归档。

### 第一轮 Go 条件

- TWC 的 `C-B` 已为正，但 `C-A` 在 standard 明显退化，主方法 gate 已失败；剩余同 endpoint prediction variance 与强协议只用于机制收尾。
- 强 gap 上若有收益，仍需逐 tracklet paired delta 证明不只来自单个 tracklet；不能用它事后取消 standard No-Go。
- 若进入 residual，oracle blend 必须先有空间，随后 true-dt 同时优于 fixed-dt 和 shuffled-dt，并具有非平凡 applied ratio。
- 新 dual-clock/innovation 的 mini promotion 原要求：standard guardrail、gap1124 complementarity 和 `true-dt` 相对 fixed/shuffled 时间门槛同时通过。前两项已通过，时间门槛已失败，因此 timestamp-conditioned promotion 正式 No-Go。
- asymmetric path distillation 必须相对新的 single-path dual-clock control 为正，并且最终不低于 A1；不能只相对受损的 paired control 为正。

若不满足，按第 4 节诊断，不立即增加网络复杂度。

## 3. M1–M4 分阶段解锁

### M1：物理一致 augmentation 与 zero-init dual clock（R1 formal 路径已冻结）

- [x] 读取并冻结 M0 的 candidate0/1/2/3 审计结论；不得在 M1 中用训练结果反向改写伪速度判据。
- [x] 正式 augmentation 预注册为 shared SE(2)，第一版不做 smooth drift；Dynamics label 从 canonical/一致扰动轨迹计算。
- [x] 第一代码切片：已新建 `utils/candidate_utils.py` 与 `tools/check_candidate_shared_se2.py`，dataset-free 世界坐标刚体变换测试通过；sampler 已接入，`utils/twc_utils.py` 的 absolute-frame 共享语义未改。
- [x] 新增 `candidate_trajectory_mode: independent | shared_se2`，默认 `independent`，旧配置和 A1 数据路径不变。
- [x] 实现围绕最近历史 anchor 的共同世界坐标 SE(2)：历史中心共用一次旋转/平移，yaw 同加 `dtheta`，没有复用每框局部 offset 冒充刚体变换。
- [x] Dynamics displacement/velocity label 由未扰动 canonical GT trajectory 和真实 `delta_t` 显式计算，不再从 candidate `ref_boxs` 构造监督。
- [x] 保存 augmentation mode、sample-level local/world transform、canonical refs 与等价 local-offset 审计字段；candidate0 有显式 identity 快路。
- [x] 纯函数共同刚体变换、anchor-normalized trajectory、canonical label、degrees/radians 与双路径共享已通过；2026-07-22 服务器真实 loader 的 full-history candidate1/2/3 与 TWC 回归通过。
- [x] zero-init physical-time adapter 已实现且 dataset-free exact identity 通过；2026-07-22 服务器同 A1 权重、同 batch 的 motion/output/loss strict-zero 等价通过（A1 `320/334` 张量匹配，14 个新 dynamics/adapter 张量按设计新建）。
- [x] 保留 SeqTrack3D order embedding；新工程配置固定 `main_time_source: order`，physical time 只进入 DynamicsEncoder、zero-init adapter 与 `R(delta_t)`。
- [x] formal 配置与唯一 augmentation 已静态冻结；commit `473738f` 的 server manifests/preflight 与 R1 训练均已完成，不得根据 standard/mini_val 结果事后改变原实验扰动过程。
- [ ] 按 2.2 完成 candidate-frame `box_label`、canonical `motion/dynamics label` 与最终 proposal 的语义/梯度审计。
- [ ] 比较 training shared-SE(2) 与 frozen recursive predictions 的历史 error process；shared-SE(2) 的 formal freeze 不视为性能最优性证明。

### M2：Proposal innovation（tracking 正信号；physical-time No-Go；归因待定）

- [x] 新增独立显式 `proposal_innovation` 模式：`innovation=clip(d_dyn-stopgrad(d_obs), R(delta_t))`；旧 `residual` alias 和语义未改。
- [x] 实现 `d_final=d_obs+alpha*innovation`；旧完整 `d_obs + scale*alpha*d_dyn` 仅保留为历史负对照。
- [x] 新公式只有一个 `[0,1]` effective alpha；既有 mini_train oracle 已一次性确认并冻结 `alpha=0.75`、`R(dt)=min(0.5+0.5dt,2.0)`，没有读取 mini_val/test 或搜索网格。
- [x] zero-scale/disabled、`alpha=0`、`dynamics_valid=0`、empty-search 的 strict fallback 与真实模型 A1 等价均通过；warmup 内执行两次 optimizer update 后 adapter/innovation output 与 effective scale 仍精确为 0，DynamicsEncoder 梯度非零。
- [x] 已新增 raw/clamped/applied innovation、半径、alpha、applied/clamp ratio、invalid fallback、adapter/encoder gradient 诊断；旧 residual 字段保留。
- [x] commit `9a0b26d` 的服务器硬门禁通过：五组共 `61` 个 batch/`122` 个样本全 finite；fallback 为 `8 invalid / 16 empty / 2 resampled`，invalid/empty applied max 精确为 0；standard/gap/burst active 均完成 2-step，encoder/adapter 梯度非零，bound violation max `5.96e-8`。
- [x] R1/R2/R3 完成预注册 60 epoch/75720 step 训练，完整性与 standard final 已审计；R2/R3 只回答当前 shared-SE(2) scratch 交互，不把 R3 当历史 A1。
- [x] 用 R1 同一 final checkpoint 完成 standard/gap1124 true/fixed/shuffled 与 matched A1；八组 endpoint/integrity/metrics/bootstrap 已独立复核。physical-time causal gate FAIL，详见 `compare_results/reports/m2_standard_gap8_analysis_20260724.md`。
- [ ] 用 R1 同一 checkpoint 完成 full/adapter-only/innovation-only/both-off 四路 evaluation-only 机制消融；不把 both-off 当 matched W0。
- [ ] 将 proposal-innovation 与 adapter 的 endpoint/epoch 诊断量写入正式 logger；当前 output 字段存在不等于训练曲线已记录。
- [ ] 补 A1-init W0 continuation 与 current-code legacy-candidate W0，分别排除 extra continuation 和 shared-SE(2) baseline collapse。

### E0–E6：从工程 GO 升级到正式训练 GO

- [x] **E0 默认回归**：新功能默认关闭；旧 config、自检和 A1 batch/output/loss 不回归。
- [x] **E1 几何不变量**：shared SE(2) 是真实共同刚体变换，不是重复局部 offset。
- [x] **E2 physical-label 不变量**：candidate 改变不污染 canonical GT velocity/acceleration；该 gate 不负责证明 candidate-frame box proposal 与 canonical physical motion 语义相同。
- [x] **E3 公式不变量**：zero/invalid/empty/warmup 严格回到 A1；正式路径不再叠加两个完整位移。
- [x] **E4 数值安全**：三协议与所有 fallback batch 的 loss/gradient finite；sampler-resampled 已显式覆盖。
- [x] **E5 可训练性**：至少 2-step optimizer，innovation 分支启用后有非零有限梯度和非平凡但受界的修正。
- [x] **E6 可复现性**：唯一 true-dt seed42 配置、共享 warmup=5、A1 `--init_checkpoint`、candidate/point sampling、1262×60=75720 steps、last-only checkpoint、same-checkpoint controls、归档脚本、commit `473738f` 的 server cadence/shuffled manifests 与 preflight provenance 均已完成。

2026-07-22 硬门禁复验来自 clean commit `9a0b26d` 和固定 A1 SHA256 `a2fbff...a82`。五组 summary 均 `requirements_passed=true`，本地从 JSONL 独立复算计数、finite、梯度与 bound 完全一致，正式判定 `PASS_M1_M2_E0_E5_ENGINEERING_GATES`。随后 `m2_e6_parameter_freeze_20260722.{json,md}` 完成单规则参数确认，commit `473738f` 完成 server manifests/preflight。2026-07-23 R1/R2/R3 完成并得到 standard 正信号；2026-07-24 standard/gap1124 formal controls 进一步确认 tracking 正信号，但否定正确物理时间的因果 promotion。当前进入纯 attribution HOLD，完整分析见 `compare_results/reports/m2_three_run_analysis_20260723.md` 与 `compare_results/reports/m2_standard_gap8_analysis_20260724.md`。

### M1.5：Proposal 语义与递归误差过程修复（条件候选）

**当前状态：未解锁、未实现。** 它只在 2.2/2.3 证实 target conflict、shared-to-recursive distribution gap 或 candidate-path interaction 后启动，且必须使用新的实验标签和 clean commit。

- [ ] 将 canonical `d_phys` 与 candidate-frame `d_box` 分头建模和监督。
- [ ] 最终 innovation 的两端必须处于相同 anchor/frame/target semantics；增加解析几何与 candidate0/1/2/3 单测。
- [ ] 相关误差轨迹只从 mini_train frozen recursive errors 预注册，不从 mini_val/test 调分布。
- [ ] 使用 staged freeze/unfreeze、分组 LR 和线性 ramp，避免 epoch5 从 0 到 1 的硬切换；schedule 在新训练前冻结。
- [ ] 第一轮只比较分头、相关误差、渐进启动三个正交因素，不进入 path distillation/filter/tube。

### M3：Endpoint-consistent asymmetric path distillation（尚未解锁）

**当前决定：timestamp-conditioned M3 不启动。** R1 的 `true > fixed/shuffled` 前置条件已在 standard 与 gap1124 同时失败。若以后研究 endpoint/path consistency，只能把它作为不依赖真实秒数的 time-agnostic robustness 新假设重新预注册，不能沿用当前 physical-time promotion 口径。

- [ ] canonical dense path 使用 EMA teacher；irregular true-time path 使用 student。
- [ ] 第一轮固定 `beta=0`：`L=L_sup_A+lambda_path*w_A*D(stopgrad(p_A),p_B)`。
- [ ] `w_A` 只能来自 teacher 的推理时可得 confidence/uncertainty，不能读取当前 GT。
- [ ] fixed/shuffled 只用于因果评估，不进入 path consistency 训练。
- [ ] 不复用当前 corrected-TWC checkpoint 继续训练；新设计必须从同提交 A1 初始化并重新做 A/B/control。

### M4：Continuous-discrete filter 与 trajectory tube（尚未解锁，后置）

**当前决定：保持锁定。** 当前仓库没有 persistent `mu/P`、covariance propagation、measurement update 或 trajectory-tube crop；旧 sampler 中禁用的 `KalmanFiltering` 片段不属于 M4。M2 causal-time gate 已失败，因此不沿当前 timestamp-conditioned 路线进入 M4；只有通用 state prior 的 matched attribution、predicted-history tube oracle 与 uncertainty calibration 形成新的独立证据链后，才重新评估。

进入条件：

- [x] M2 dynamics proposal 在递归 tracking 中有 aggregate 互补性且 standard guardrail 通过；但同 checkpoint `true > fixed` 与 `true > shuffled` 已失败，所以当前 M4 promotion 条件不成立。
- [ ] 使用 frozen M2 predicted history 完成固定点预算 tube oracle；GT-history oracle 不能替代该项。
- [ ] predicted tube 相对 baseline crop 有非平凡 `tube-only reachable`、crop recall 或 first-failure/consecutive-failure 改善；预先冻结阈值，不能看 test 后定义。
- [ ] 固定 covariance 或 learned covariance 在独立 split 通过 NLL、coverage、ECE/reliability；未校准 uncertainty 不进入在线融合。

`M4-0` predicted-history tube oracle：

- [ ] 从 frozen M2 endpoint logger 导出 prior center、velocity、`delta_t`、valid/fallback，不改变 checkpoint 和预测路径。
- [ ] 构造 `baseline crop union bounded trajectory tube`；不得用 predicted center 替换唯一 anchor。
- [ ] 固定总 point budget、candidate、endpoint、point seed；记录 baseline/tube/union target points、background ratio、crop recall、tube-only reachable 和 empty fallback。
- [ ] 分 standard、gap1124、burst-drop、unseen cadence、long-gap、`>4 m` displacement 与 previous-error buckets；若无独立 complementarity，M4 在此停止。

`M4-1` fixed-covariance GT-free filter：

- [ ] 新增 per-tracklet `mu_t_plus/P_t_plus/last_timestamp/valid-reset state`；状态为 `[x,y,z,vx,vy,vz,yaw,yaw_rate]`，box size 第一版不建模。
- [ ] 实现解析 constant-velocity/constant-turn `f/F(dt)` 与固定 PSD `Q(dt)/R_obs`；不实现 ODE/CDE/Mamba。
- [ ] measurement 只使用推理时可得 observation center/yaw；yaw innovation wrap，矩阵更新使用 solve/Cholesky 和 Joseph form。
- [ ] 新 tracklet、非法/非单调 `dt`、invalid history、非 finite/非 PSD covariance 必须 reset 或 strict baseline fallback；禁止用当前 GT 触发。
- [ ] 首先比较 A1、M2 fixed-alpha、CV/Kalman fixed-Q/R；fixed filter 无正收益则不学习 covariance。

`M4-2` filter + tube：

- [ ] 只有 M4-1 为正才将 prior mean/covariance 接入 search support。
- [ ] tube 沿传播/velocity 方向增长、横向由 `P_xy` 限制，低速退化到 posterior yaw，长宽有上限。
- [ ] 分别报告 tube-only、filter-only、filter+tube，并保持 point budget、candidate、FLOPs 和 checkpoint 口径公平。

`M4-3` learned covariance：

- [ ] 只有 fixed filter 与 tube 均为正才学习 `Q_theta(delta_t, history)` 或 `R_obs(feature/stats)`。
- [ ] 通过 Cholesky/softplus 保证 PSD；输入不能包含当前 GT、GT overlap 或真实预测误差。
- [ ] 先报告 Gaussian NLL、50/90/95% coverage、置信区域大小、ECE/reliability、分 `delta_t` calibration、eigenvalue/finite/reset 统计；必要时补 NEES。
- [ ] calibration 失败则退回 M2 固定小 alpha/固定 covariance，不增加 learned Gate。

正式消融与边界：

- [ ] 至少包含 A1、M2 fixed-alpha、fixed CV/Kalman、tube-only、filter-only、fixed-Q/R full、learned-R、learned-Q/R 与 `true/fixed/shuffled`。
- [ ] 记录 Success/Precision、crop recall、first failure、连续失败、fallback、参数量、FLOPs、FPS、显存和实际 crop point count。
- [ ] M4 与 M2 不对同一 observation/dynamics pair 连续修正两次；M4 gain 替换 fixed alpha，M2 只保留 baseline/fallback。
- [ ] 旧 P5 hand-crafted observability Gate 不复活、不在 mini_val 上重调，也不列论文贡献。
- [ ] 只有 persistent state、covariance propagation/update 与 tube 均实际实现并通过上述 gate，才允许在论文中使用“continuous-discrete state filtering”。

### P1：未见 cadence 泛化

- [ ] Standard-only train：一个 checkpoint 直接测试 standard、fixed skip、gap1124、random20/40、burst-drop 和未见 schedule。
- [ ] Fixed skip 至少包含 `K=2/3/4`（约 1.0/1.5/2.0 s）；它用于与既有固定 interval HTV 对齐，不作为 CT 独立创新。
- [ ] Mixed-cadence train：训练时故意留出至少一种 gap pattern 和一种 drop probability；训练 manifest 在运行前冻结。
- [ ] Unseen-schedule test：不重训、不改 threshold，直接测试 held-out pattern/probability；per-cadence specialization 只能作上界。
- [ ] train regime 内保持相同 optimizer steps；丢帧不得同时减少训练预算。
- [ ] 分开报告 in-domain、seen-cadence 和 unseen-cadence；只有最后一项成立，才写“跨采样率泛化”。
- [ ] 丢关键帧只称 virtual-rate / irregular-observation stress test；除非补真实系统统计或 raw sweep 标签，不写成真实 LiDAR packet loss。

### P1：完整数据与统计

- [ ] mini 通过后迁移到完整 nuScenes trainval，先完成 Car，再决定是否扩展类别。
- [ ] 至少补一个第二数据集或官方 HTV 协议；Waymo 需要先补齐等价 virtual-rate manifest 支持。
- [ ] 三个 seed 报逐 seed、paired mean±std、final/late mean 和 tracklet-level bootstrap；不能把序列帧当独立样本 bootstrap。
- [ ] 公平基线至少包含同提交 SeqTrack3D/A1/W0、HVTrack、一个运动/轨迹方法、CV/Kalman 和 TrajTrack GT-free；MambaTrack3D 仅在可核验复现时加入。
- [ ] 最终再报告参数量、FLOPs、FPS、显存和新增分支开销。

### P1：TrajTrack GT-free 公平对照

- [ ] 固定现有 epoch60 checkpoint，评测 `pre_wo_refine()` 与只依赖 local/global proposal agreement 的 GT-free hard switch。
- [ ] `pre_w_refine()` 只作为 GT-assisted oracle，不能进入主表排名。
- [ ] proposal 选择函数不得接收当前 GT、`this_bb` 或 GT-derived mask。
- [ ] 增加 GT 独立性测试：只改变当前 GT、保持输入和 proposals 不变，GT-free 输出必须不变。
- [ ] 固定 checkpoint、tracklet 顺序、evaluator、threshold，并报告 Success/Precision 与 FPS。

## 4. 失败时的额外诊断

P0-A/P0-B/P0-C 已覆盖 residual、crop 和协议检查；这里只保留会影响下一步模型设计的额外诊断，并按实验失败类型触发。

### 若 R3/shared 路径塌陷：检查 candidate 语义与误差过程

- [ ] 按 candidate0/1/2/3 复算 `ref_boxs`、`box_label`、`motion_label`、`dynamics_displacement_label` 与解析 transform，不再只检查 physical-label invariance。
- [ ] 计算 `loss_center` 与 `loss_center_motion` 对 coarse motion head 的梯度 cosine/norm，定位 candidate-dependent 冲突。
- [ ] 用 A1/R1/R2 frozen recursive predictions 统计相关误差、error velocity/acceleration、yaw drift 与连续失败，再与 independent/shared training augmentation 对照。
- [ ] 若需要相关误差增强，只在 mini_train 拟合并作为新 M1.5 预注册实验；不复活旧 reliability Gate。

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

- 当前 R1 在 standard/gap1124 的 `true-dt - fixed/shuffled-dt` 均未达到 `0.5 Success / 1 Precision`；physical-time method claim 的 Stop 条件已触发。
- residual 在合理校准后仍长期梯度、alpha 或 applied ratio 接近零。
- 收益只来自扩大 crop、更多参数、更多训练步、checkpoint 选择或按 protocol 分别重训。
- unseen cadence 不成立，或 full data 与 mini 的方向相反。
- corrected-TWC 虽有 `C-B`，但无法恢复到 single-view A；该 Stop 条件已由 standard A/B/C 触发，不再补训练 seed。
- 新 asymmetric distillation 若仍只优于 paired control、不能回到 single-path A1，则停止该分支，不通过调 `beta/lambda` 大网格继续追分。
- dual-clock adapter 若在同一 checkpoint 下不能形成 `true > fixed` 且 `true > shuffled`，停止 physical-time method claim；保留协议资产。
- R1 both-off 若保留大部分收益且 A1-init W0 也接近 R1，停止把运行时 adapter/innovation 写成主要涨点来源；将结果归为 continuation/representation effect。
- current-code legacy W0 若恢复而 shared W0 持续塌陷，停止把 shared-SE(2) 当作默认性能增强；保留它作为物理一致 control，并只通过新预注册 M1.5 测试相关误差轨迹。
- current-code legacy W0 若同样塌陷，停止 candidate/M2 扩展，先完成 current-code A1 exact replication 与 loss/label 修复。

Pivot：把工作收敛为 variable-rate 3D SOT benchmark/diagnosis，或使用可解释的 constant-velocity/Kalman trajectory fallback；不再通过增加复杂时序模块追分。
