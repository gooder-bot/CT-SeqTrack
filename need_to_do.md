# CT-SeqTrack 当前执行清单

更新时间：2026-07-20

本文只维护会影响论文结论的未完成工作，并按重要性排序。已完成内容见 `done.md`，结果口径见 `sum_results.md`，研究定位见 `refined_plan.md`。

## 0. 当前主线与结论边界

当前主线收敛为：

> 面向不规则采样和变帧率 3D 单目标跟踪，先建立冻结、可复现的 variable-rate / held-out-cadence 评测协议，量化 search reachability、递归漂移和时间接入方式的失败边界；只有预注册的 `true/fixed/shuffled-dt` 控制出现因果正信号，才保留轻量时间模块。

目前只能确认：时间戳、virtual-rate、TWC、feature dynamics 和 bounded residual 已有不同程度的工程实现或实验信号；还不能声称真实 `delta_t` 稳定提分，也不能声称模型已经具备跨采样率泛化能力。

现有结果的使用边界：

- corrected A1+TWC seed42 为 `+1.49 Success / +5.03 Precision`，但 baseline 来自旧提交且只有一个 seed。
- corrected A2+TWC seed42 为 `-0.93 / -2.07`，暂不继续组合 A2+TWC。
- feature-concat A2 在 random20 为正，在 gap1124 和 burst-drop 为负，不能作为主创新结论。
- residual A2 已完成 standard 的真实 batch warmup/active forward-loss-backward 诊断，但默认实际修正仅约 `1e-7 m`、gate 梯度极小，尚未完成 2-step optimizer、完整 split、强 gap 或跟踪性能验证。
- P0-B2 三协议递归诊断已完成：predicted-history CV 相对 previous-A1 只提高 2.65–3.03 pp，未达到预注册门槛；它在可靠历史桶达到 97.34%–98.64% recall，但在上一预测误差超过 4 m 后两者都近乎失效。always-on raw CV recenter 已 No-Go。
- P0-B3 三协议已完成并复核：13 特征 trigger 通过预注册判据，但 passive raw-CV union gain 仅 2.88–3.15 pp，当前 selector 在强协议 AUROC 为 0.605/0.433；正式决定为 `RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO`。消融表明预测力主要来自 `prev_obs_*`，raw `current_delta_t` 是跨协议失准的主要来源，因此只能称 observation reliability Conditional-Go，不能称 timestamp-aware reliability 已成立。
- P0-B4 已在独立 mini_val 上完成冻结验证：`observation_v1` 在 gap/burst 的 AUROC 为 `0.680/0.712`、运行点 recall 为 `0.568/0.609`，均未通过 `0.75/0.70` 门槛；正式决定为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`。同批 raw-CV 第二 crop 在两个强协议的 trajectory-only endpoint 都为 0，因此 reliability-controlled anchor 与 active dual-anchor 路线停止，不在 mini_val 上重调。
- TrajTrack 的 `64.94 / 79.07` 来自 GT-assisted evaluator，只能作为 oracle 诊断。

## 1. 四个最高优先级问题

P0-B 已在独立验证入口处 No-Go，不能再作为当前方法主线。接下来先完成 P0-C 的冻结协议，把工作收敛为 variable-rate benchmark/diagnosis；P0-A 与 P1-D 各只保留一次窄机制验收，不启动主线大规模训练或复杂 trajectory/gate 扩展。

### P0-A：bounded residual 可能小得几乎不起作用

**问题**：当前默认最大修正量仅为 `0.1 × 0.2 × 1.0 = 0.02 m`，而 gate 近零初始化。若真实 observation error 明显大于 2 cm，或 alpha/梯度长期接近零，该分支即使存在也几乎不改变预测。

**2026-07-17 状态**：standard active 64-batch 的 observation error P50/P75/P95 为 `0.213 / 0.577 / 3.838 m`；alpha 固定约 `2e-5`，实际 residual P50 仅 `7.25e-8 m`，gate grad P50 仅 `4.00e-10` 且 31/64 batch 为 0。默认配置数值稳定，但没有通过“非平凡修正幅度”验收。完整证据见 `compare_results/reports/p0_ab_diagnostics_20260717.md`。

- [x] 在 standard 真实 full-history batch 上完成 warmup 2-batch 与 active 64-batch forward/loss/backward。
- [ ] 补 gap1124、burst-drop 的 forward/loss/backward，并在三协议完成真正的 2-step optimizer smoke。
- [x] 确认 warmup 内 residual/gate gradient 为 0；active 的 alpha、raw/clamped/applied residual 与 gradient 全部 finite。
- [ ] 在真实 invalid-history batch 上确认 `dynamics_valid=0` residual 严格为 0；standard full-history 运行的 valid ratio 为 1，没有覆盖该条件。
- [x] 确认 residual observation head 输入仍为 256 维，没有额外拼接 `z_dyn`。
- [ ] 遍历完整训练 split，并只在 crop-reachable subset 统计 `||GT motion - observation proposal||` P50/P75/P95；当前 1024 样本混入 out-of-crop error，不能直接用于调 bound。
- [x] 记录 gate bias、gate gradient norm、alpha 分位数、applied ratio、clamp ratio 和 applied norm；注意当前 `applied_ratio=1` 只表示 norm 大于 `1e-8`，不表示修正有实际作用。
- [ ] 预裁剪可达性解决后，再依据训练 split reachable subset 一次性预注册 gate init/scale/bound；不得根据 test/mini_val 涨跌反复调参。

**验收**：当前默认配置未通过。后续版本必须在有效样本上同时具有非平凡 gate gradient、可见修正幅度和预注册上界，且仍保持稳定、可归因。

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
- [ ] 在服务器 clean commit 上生成 val/test cadence manifest 和 test shuffled-time manifest，跑真实 nuScenes batch invariance；本地因没有 nuScenes 开发包只能完成 `py_compile`、config/hash 与 effective-time 纯函数测试。
- [ ] 用同一个冻结 A2 checkpoint 完成 gap1124 `true/fixed/shuffled-dt` 三次评测并核对三份 `run_provenance.json`；之后才扩展 burst-drop 和未见 fixed-gap。

**验收**：一个 standard-only 或 mixed-cadence checkpoint 可在不重训、不改 threshold 的条件下测试 held-out schedule；所有方法共享相同 endpoints。工程入口已完成，服务器 manifest/batch smoke 和第一组三路冻结评测尚未完成；命令见 `protocols/README.md`。

### P1-D：TWC 缺少 `paired-view + twc_weight=0` 控制组

**问题**：当前 TWC 目标同时包含两条历史视图的 supervised loss，因此单 seed 提升可能来自 paired-view 数据增强，而不一定来自 consistency loss。

```text
A. single-view A1
B. paired views + twc_weight=0
C. paired views + corrected-TWC
```

- [ ] A/B/C 必须来自同一提交，使用相同 seed、candidate4、optimizer steps 和 checkpoint 规则。
- [ ] 用 `B-A` 衡量 paired-view augmentation，用 `C-B` 衡量 TWC 净贡献。
- [ ] evaluation-only 测试同一 endpoint 的多条合法历史路径，报告 center/angle gap 和 prediction variance。
- [ ] 先跑 seed42；只有 `C-B` 为正且路径方差下降，才补 seed43/44。
- [ ] 若 B 已解释全部收益，TWC 不作为主贡献；若只降方差不涨指标，只写作稳定性 regularizer。
- [ ] 不再启动 A2/residual+TWC 组合，直到 A1 上的净贡献得到三 seed 支持。

**验收**：能够把 paired augmentation 和 consistency loss 的收益分开，并在同提交配对实验中证明 `C-B`。

## 2. 立即执行的最小实验

P0-B 已 No-Go，当前不启动新的主线训练。先完成 P0-C 的协议工程，再决定是否执行一个 seed42 的窄机制控制。

实际执行顺序：

1. 将当前脚本、文档和 P0-B4 verdict 提交到 clean GitHub commit；后续服务器运行必须使关键脚本来自该 commit，避免再次只留下 dirty hash。
2. 将本轮 P0-C 工程提交到 clean GitHub commit；在服务器按 `protocols/README.md` 生成 role-specific cadence/time manifests，并先让真实 batch invariance 输出 PASS。
3. 用同一个冻结 A2 checkpoint 做 gap1124 `true/fixed/shuffled-dt` held-out cadence 评测；不重训、不改 threshold，核对 endpoints/manifest/checkpoint hash 后再做 reachability/递归失败报告。
4. 只做一次 P0-A 收尾：在 mini_train crop-reachable subset 统计 `GT motion - observation proposal`，先核对 residual 目标/公式，再一次性预注册 init/scale/bound；不直接调大 `max_residual_norm`，不扫网格。
5. 若仍需要方法贡献，优先做同提交的 `single-view A1 / paired-view weight0 / corrected-TWC` seed42 控制；只有 `C-B` 为正且路径方差下降才补 seed43/44。
6. 若 P0-C、residual 或 TWC 的因果控制仍无正信号，正式 Pivot 为 variable-rate 3D SOT benchmark/diagnosis，不再增加时序模块。

| 方法 | 目的 |
| --- | --- |
| A1-order | observation baseline |
| A2 feature-concat true-dt | 旧时间接入方式参考 |
| observation-reliability-updated Kalman/frozen-state | P0-B4 入口 No-Go，停止实现 |
| A2 residual true-dt | reachable-subset refinement 消融 |
| A2 residual fixed-dt | 同容量时间负对照 |
| A2 residual shuffled-dt | 物理时间对应关系负对照 |
| constant-velocity/Kalman | 无学习、GT-free 低复杂度轨迹基线 |

统一要求：

- [ ] 所有学习方法使用同一 commit、candidate4、manifest、batch 规则、optimizer steps 和预先规定的 checkpoint 口径。
- [ ] 主结果优先使用预先固定的 final epoch；如使用 best，只能由独立 validation metric 选择。
- [ ] 报告逐 tracklet paired delta，不从多个 epoch 中事后挑最高 test 结果。
- [ ] 第一轮不扫大网格；若 true-dt 没有同时优于 A1、fixed-dt 和 shuffled-dt，先定位机制，不补 seed43/44。
- [ ] 只有 seed42 出现因果正信号，才补 seed43/44，并报告 mean±std 与 tracklet-level paired bootstrap CI。

### 第一轮 Go 条件

- residual true-dt 同时优于 A1、fixed-dt 和 shuffled-dt。
- standard 上不出现明显退化，强 gap 上的收益不只来自单个 tracklet。
- residual 具有非平凡 applied ratio，收益不能由 crop、参数量、训练步或 checkpoint 选择解释。

若不满足，按第 4 节诊断，不立即增加网络复杂度。

## 3. 因果正信号后再做的论文实验

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
- [ ] 只有测试时 reliability proxy 与无训练 dual-anchor 先通过，才实现学习式 gate 或轻量 dual proposal。
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
- corrected-TWC 的 `C-B` 不复现，或只证明 paired-view augmentation 有效。

Pivot：把工作收敛为 variable-rate 3D SOT benchmark/diagnosis，或使用可解释的 constant-velocity/Kalman trajectory fallback；不再通过增加复杂时序模块追分。
