# CT-SeqTrack 当前执行清单

更新时间：2026-07-17

本文只维护会影响论文结论的未完成工作，并按重要性排序。已完成内容见 `done.md`，结果口径见 `sum_results.md`，研究定位见 `refined_plan.md`。

## 0. 当前主线与结论边界

论文主线暂定为：

> 面向不规则采样和变帧率 3D 单目标跟踪，用测试时可靠性信号协调 observation anchor 与真实时间驱动的 trajectory anchor，在首次漂移前维持搜索可达性，再由 SeqTrack3D observation 主干完成 refinement，并验证物理时间在未见采样节奏下是否具有因果收益。

目前只能确认：时间戳、virtual-rate、TWC、feature dynamics 和 bounded residual 已有不同程度的工程实现或实验信号；还不能声称真实 `delta_t` 稳定提分，也不能声称模型已经具备跨采样率泛化能力。

现有结果的使用边界：

- corrected A1+TWC seed42 为 `+1.49 Success / +5.03 Precision`，但 baseline 来自旧提交且只有一个 seed。
- corrected A2+TWC seed42 为 `-0.93 / -2.07`，暂不继续组合 A2+TWC。
- feature-concat A2 在 random20 为正，在 gap1124 和 burst-drop 为负，不能作为主创新结论。
- residual A2 已完成 standard 的真实 batch warmup/active forward-loss-backward 诊断，但默认实际修正仅约 `1e-7 m`、gate 梯度极小，尚未完成 2-step optimizer、完整 split、强 gap 或跟踪性能验证。
- P0-B2 三协议递归诊断已完成：predicted-history CV 相对 previous-A1 只提高 2.65–3.03 pp，未达到预注册门槛；它在可靠历史桶达到 97.34%–98.64% recall，但在上一预测误差超过 4 m 后两者都近乎失效。always-on raw CV recenter 已 No-Go。
- TrajTrack 的 `64.94 / 79.07` 来自 GT-assisted evaluator，只能作为 oracle 诊断。

## 1. 四个最高优先级问题

前三项依次决定末端 residual 是否还有价值、pre-crop trajectory guidance 是否成立，以及物理时间收益能否被因果归因；第四项决定 TWC 能否保留为论文贡献。先解决前三项，再启动主线大规模训练；TWC 控制组可后置。

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

- [x] 在 standard 模型 forward 前统计 target-in-base-crop recall、center-outside 和目标点保留率。
- [x] 补齐 gap1124、burst-drop 的 summary/CSV，并按 `current_delta_t`、真实位移和目标点数分桶。
- [x] 在 standard 比较 base、2x expanded、GT-history constant-velocity recentered crop 的 oracle recall 与背景点开销。
- [x] 新增独立 `diagnose_recursive_crop_reachability.py`，被动比较四种 anchor，并记录 endpoint/hash/连续失败信息。
- [x] 对三协议递归误差、连续失败、empty fallback 和可靠/失控分桶完成检查，确认预测历史存在灾难性长尾漂移。
- [x] 在服务器运行 predicted-history 诊断：四种 anchor 使用完全一致 endpoints、同一 checkpoint，missing/unexpected 均为 0。
- [x] 按预注册门槛判定 always-on raw predicted-history CV recenter 为 No-Go，不接成唯一 search anchor。
- [ ] 扩展递归日志，记录测试时 confidence/foreground/crop points/empty fallback/CV shift/speed/proposal agreement，并评估故障预测 AUROC、AUPRC 与 calibration。
- [ ] 仅当可靠性代理有效时，实现无训练 active dual-anchor；用同一 A1 checkpoint 检查首次失控、连续失败和在线跟踪指标。
- [ ] expanded/recentered crop 必须保持相同训练步和模型容量，不能把更大搜索区收益写成真实时间收益。

**验收**：P0-B2 被动诊断已完成并判定 raw always-on CV No-Go。下一阶段不再问“单一 CV anchor 能否恢复已漂移目标”，而是验证测试时信号能否在 GT-free 条件下识别风险，以及 dual-anchor 是否能预防首次失控、缩短连续失败。最终 bounded residual 仍不作为当前第一修正位置。

### P0-C：当前 HTV 实验还不是“未见 cadence 泛化”

**问题**：train/val 目前复用同一组 `virtual_rate_*`，已有六组结果更接近“分别在各协议上训练和评测”。这不能支持“一个模型跨采样率泛化”的论文主张；现有 manifest 的 split 内序号建键也不适合正式冻结协议。

- [ ] 拆分 `train_virtual_rate_*` 和 `eval_virtual_rate_*`，允许 standard-train、variable-rate-test。
- [ ] 增加 `virtual_rate_manifest_train / val / test`；使用 dataset version + split + scene/instance/tracklet token 稳定建键。
- [ ] manifest 记录 protocol、seed、endpoint 数、代码 commit 和 SHA256；不匹配时 fail fast。
- [ ] 在同一代码路径实现 `dynamics_time_mode: true | fixed | shuffled`。
- [ ] `fixed/shuffled` 只改变 dynamics effective time，不改变 main order-time、frames、crop、candidate、标签或 optimizer steps。
- [ ] batch 同时保留 `delta_t_real/effective`；shuffled 使用离线冻结、split 内 permutation 和 mapping hash。
- [ ] 增加回归测试，证明 true/fixed/shuffled 除 dynamics effective time 外完全一致。
- [ ] 删除或明确禁用未接入模型的 `dynamics_use_acceleration`。
- [ ] 每个 run 保存 commit、dirty status、cfg hash、manifest hash、seed 和 checkpoint 规则。

**验收**：一个 standard-only 或 mixed-cadence checkpoint 可在不重训、不改 threshold 的条件下测试 held-out schedule；所有方法共享相同 endpoints。

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

完成 P0-A、P0-B 和 P0-C 的工程验收后再启动训练。第一轮只用 seed42 做机制筛选，优先 `gap1124` 和 `burst-drop`，standard 只检查正常节奏是否明显退化。

实际执行顺序：

1. 扩展 recursive diagnostic，收集只依赖测试时信息的 reliability signals，并用离线 GT 标注建立 drift/next-crop-failure 评估表。
2. 预注册信号筛选口径，报告 AUROC、AUPRC、calibration 和按协议稳定性；禁止把 `previous_prediction_error <= 4 m` 本身用于在线 gate。
3. 只有可靠性代理有效时，固定 A1 checkpoint 实现无训练 dual-anchor active inference，优先验证首次失控时间和连续失败长度。
4. active dual-anchor 通过后，完成 P0-C 的稳定 manifest 与 `true/fixed/shuffled` causal time switch；同时补 P0-A reachable-subset 统计。
5. 只有 active 机制和时间因果性均为正，才进入训练、seed43/44、未见 cadence 和完整数据。

| 方法 | 目的 |
| --- | --- |
| A1-order | observation baseline |
| A2 feature-concat true-dt | 旧时间接入方式参考 |
| reliability-aware dual-anchor true-dt | 当前主假设；先做无训练主动推理 |
| reliability-aware dual-anchor fixed/shuffled-dt | 物理时间负对照 |
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

满足任一项则停止继续堆时间模块：

- 连续两轮配对实验中，`true-dt - fixed/shuffled-dt < 0.5 Success / 1 Precision`。
- residual 在合理校准后仍长期梯度、alpha 或 applied ratio 接近零。
- 收益只来自扩大 crop、更多参数、更多训练步、checkpoint 选择或按 protocol 分别重训。
- unseen cadence 不成立，或 full data 与 mini 的方向相反。
- corrected-TWC 的 `C-B` 不复现，或只证明 paired-view augmentation 有效。

Pivot：把工作收敛为 variable-rate 3D SOT benchmark/diagnosis，或使用可解释的 constant-velocity/Kalman trajectory fallback；不再通过增加复杂时序模块追分。
