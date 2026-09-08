# v27 B2/B3 实验与代码通路审计

日期：2026-09-07。范围：本地新增 `20260905-173118-27_full_minus_b3-*`、`20260905-173121-27_full-*`。所有原始 `output/` 只读；本次没有修改训练模型和配置。本报告为主报告的模块专项证据，不替代 SeqTrack 对照公平性审计。

## 结论

**不能根据当前分数判定 B2/B3 无效，也不能说 v27 全部完成。发现了可执行复现的 P0：旧 presence≥0.5 门仍被当作正式候选的结构有效性，阻断 B1+B2 输出、污染动作漏斗和校准导出；Full 安装有效策略后还可能因这两个字段不一致而在评测时抛异常。**

Full 的全部训练期 dev 结果都没有安装校准；第 60 轮 321 帧明确记录 `missing_calibration_artifact`，0 次动作。因此它的 S/P 是自己的 observation 输出，不能作为 B3 selective 成效或失败的依据。训练完成也不意味着校准和官方 mini_val 评测完成。

## 1. 已确认的字段错误、影响及复现

当前执行链：

1. `models/ct_v2/evidence_memory.py:845` 计算结构有效性 `availability = extension_count>0 & finite(raw_xy)`，这一步正确。
2. 同文件 `:880`—`:882` 又产生旧字段 `candidate_valid = availability & (extension_presence_probability>=0.5)`。
3. `models/seqtrack3d.py:3002` 读取旧字段为 `candidate_available`，`:3049` 将其写为 `ct_policy_candidate_valid`，`:3168` 的 `bounded_always` 实际按它执行。由此违反“有合法 extension 和有限候选即可执行、取消固定 presence 门”的 v27 规格。
4. `utils/v27_evaluation.py:58` 读取 `ct_policy_candidate_valid` 当作 `structural_available`；`:60`—`:64` 把低 presence 候选的 bounded box 替换成 observation。CSV 不仅丢失有效性，连本应评估的 bounded 动作收益也丢失了。
5. `utils/v27_eval_reporting.py:33`—`:43` 的 raw/bounded 漏斗又只统计这个错误有效集合，所以 raw 候选潜在收益也被低报。
6. Full 的实际 B3 路由器却正确接收 `ct_b2_available`：`models/seqtrack3d.py:3057`；`models/ct_v2/action_v27.py:142`—`:153` 不设 presence 门。安装策略后，只要它接受一个 presence<0.5 的结构合法候选，`utils/v27_evaluation.py:66`—`:67` 就因错误字段而报 `v27 action applied without a finite structural candidate`。
7. 校准 runner 通过该公共 evaluator 导出并闭环：`tools/ct_action_v27_runtime.py:125`。阈值候选与策略集合又依赖 `structural_available`：`utils/action_calibration_v27.py:216`、`:224`。所以不能绕过 evaluator 仅删除报错；需要统一整条数据合同。

已执行 [reproduce_presence_contract.py](reproduce_presence_contract.py)，它使用真实 B2、真实 B3 和真实公共 evaluator，只以合成 B0 观测替代 nuScenes 输入。输出 [presence_contract_reproduction.json](presence_contract_reproduction.json)：

```text
unique extension points = 4
presence = 0.1
ct_b2_available = 1
ct_search_candidate_valid = 0
B1+B2 bounded_always current host applies = false
calibrated B3 always applies = 1
public evaluator -> RuntimeError: v27 action applied without a finite structural candidate
```

现有 `tests/test_ct_v27_evaluation.py:48` 的 FakeHost 直接把 `ct_policy_candidate_valid` 设为真实可用性，哪怕 presence=0 也设 True；因此该测试满足理想合同，却没有覆盖真实 B2→host 字段转换，解释了为什么之前合同测试通过而此错误仍存在。

**修复范围**：v27 下用结构合法性统一 `candidate_available`、`ct_policy_candidate_valid`、正式 bounded_always、B3、H1/H3、导出和校准；presence 保持为学习特征与独立监督。旧版本分支保持原样。增加真实 B2 低 presence→host→B3 calibrated→公共 evaluator 的集成合同测试，再验证真实轨迹 always/never。不可仅降低 presence 阈值、删除异常、或只改 summary。

## 2. 第 60 轮：门挡住了什么

两组均为 dev 的 1 场景、12 条轨迹、321 帧（其中 309 个非首帧），不是官方 mini_val。下表从完整 CSV 重算，真实结构候选用 `ct_search_extension_selected_count>0` 识别；这些实际前向均有有限原始候选和指标。

| 指标 | B1+B2 | Full |
|---|---:|---:|
| 真正 extension 非空帧 | 177 | 140 |
| selected 确有目标点帧 | 60 | 55 |
| 被报告为 structural 的帧 | 2 | 12 |
| 被旧门挡住的结构合法帧 | 175 | 128 |
| 被挡住的含目标点帧 | 59/60 | 46/55 |
| 实际执行动作 | 2 | 0（缺校准） |
| presence 中位数 | 0.2902 | 0.3438 |
| raw 位移超出动作半径 | 173/177 | 140/140 |

Full 中 98 个结构合法帧的 q>0，其中 88 个被旧 presence 字段标成无效。说明上述校准后异常不仅是人为构造的极端边界；实际模型已有可能触发它的候选。最终拟合策略是否接受这些帧需真实重评确认。

原 summary 的 raw 全帧净 U 仅为 +0.0312/+0.4517 个百分点。按所有结构合法帧重新累加 CSV 的 raw 同状态收益后，为 **+3.7150/+5.1480 个百分点**。这只是逐帧、同一已发生状态下的 unbounded counterfactual；它不是 bounded 正式效果，也不是替换递归状态后的闭环提升。

主 endpoint CSV 单独缺失低 presence 帧的真实 bounded 动作，但另存的 `candidate_diagnostics/epoch_60.csv` 保留了无条件 bounded 候选和由 `models/base_model.py:1660`—`:1673` 直接计算的 S/P gain。将全部 309 个 `(tracklet_id,frame_id)` 与主表一一对齐后，observation_distance 完全一致；门内 2/12 帧的 utility_gain 也完全一致；候选 sidecar 的结构可用性正确为 177/140。

据此恢复完整真实 bounded H1：

| 同一已发生状态下的动作统计，全 321 帧分母 | B1+B2 | Full |
|---|---:|---:|
| 全部合法 bounded 候选 ΔS（百分点） | +0.10903 | -0.01558 |
| 全部合法 bounded 候选 ΔP（百分点） | -0.26480 | ≈0 |
| 全部合法 bounded 候选 ΔU（百分点） | **-0.07788** | **-0.00779** |
| bounded helpful/harmful 帧数 | 26 / 12 | 8 / 6 |
| 只选当前 H1 正收益动作的 oracle ΔU（百分点） | +0.33489 | +0.12072 |

Full 在固定 `q>0` 的 98 个候选上 one-step ΔU 为 +0.03894 个百分点（7 helpful、2 harmful）。这仅是已知分数的诊断，不是校准结果。上述 oracle 也只是固定状态下 H1 的选择上限，**不是任何闭环策略的性能上限**。

因此，修复旧门是正确运行和解释结果的前提，但现有证据不支持“移除门就会涨分”。raw 潜力几乎没有转化成 bounded 当前收益，比单纯“动作没执行”更深的瓶颈是小幅更新与重度漂移恢复需求不匹配。

## 3. 新增点证据已出现价值，主要瓶颈在获取和执行

第 60 轮 ID 漏斗：

| 阶段 | B1+B2 目标点数 | Full 目标点数 |
|---|---:|---:|
| 全局可见目标 | 13098 | 13098 |
| B0 raw crop | 1834 | 1192 |
| support raw | 2378 | 2208 |
| 排除整个 B0 raw crop 后 novel | 759 | 1210 |
| 768 prepool | 759 | 1210 |
| 256 selected | 720 | 1167 |
| top-mode consensus | 499 | 908 |

768→256 保留目标点 **94.86% / 96.45%**，因此当前证据不支持优先加大选点预算。novel 点相对全局可见目标仅 **5.79% / 9.24%**；这其中同时包含“已被 B0 crop 覆盖，无需新增”的合理排除与“支持区域没有找到目标”的真实错失，不能将 novel/global 直接解释为 B1 的独立召回率。需要结合 B0 raw 和 `global_novel_target_count` 条件分母进一步拆分获取问题。

在 selected 确实含目标点的帧中，raw 候选已经明显改善局部定位：

| 条件：selected 含目标 | B1+B2 | Full |
|---|---:|---:|
| 帧数 | 60 | 55 |
| observation→raw 平均中心误差（m） | 6.47→3.70 | 6.04→3.13 |
| observation→raw 中位中心误差（m） | 6.11→1.67 | 5.66→1.39 |
| raw helpful / harmful 帧 | 47 / 2 | 46 / 2 |
| top-mode 仍包含目标的帧 | 47/60 | 45/55 |
| 这些帧 top-mode 目标点比例 | 31.1% | 43.4% |

条件筛选使用 GT，仅作诊断，不能成为推理门或主结果。selected 无目标帧仍会产生背景共识：B1+B2 117 帧、Full 85 帧，其 raw 全帧净 U 分别 -1.480/-0.167 个百分点。这说明“新增证据有价值，但目标身份区分和动作选择仍必要”，不能直接将 unbounded raw 当正式输出。

动作半径约为 nuScenes 0.5s 下的 0.75m，而有目标帧的 observation 误差中位已约 5.7—6.1m，且 97.7%/100% 原始候选被裁剪。恢复出的完整 bounded H1 也证实：当前动作对已严重漂移帧几乎没有即时 S/P 收益。**先修门恢复既定 v27 行为、补真实闭环，再优先设计动作半径的质量条件化或多步恢复；不要依据 raw 分数直接放开所有大动作。**

## 4. B2/B3 的学习信号与实际耦合

正确落实的部分包括：B0/历史特征 detach 后进入 memory（`seqtrack3d.py:2805`、`evidence_memory.py:101`）；memory 局部 x=length/y=width（`:143`）；当前、前景记忆、上下文分别汇聚、局部几何后 relation 选点（`:740`—`:787`）；post-attention targetness 单独控制 vote weight（`:811`—`:820`）；只对目标点监督 vote、只对 selected target-bearing 行监督 raw（`seqtrack3d.py:5317`—`:5343`）；Full B3 与 H3 正确使用结构有效性（`utils/v27_training.py:18`—`:22`、`:46`、`:141`）。

因此本次发现的旧门主要损坏**正式 B1+B2 输出、动作导出/校准和 Full 安装策略后的评测一致性**，不是 B3 训练时又用 presence 门过滤。v27 训练明确 observation 递归（`seqtrack3d.py:3185`），故修复该门本身不要求改变已学参数或重新训练损失。

训练期后 3 轮信息（`ct_epoch_calibration_*` 是机制训练轨迹的整 epoch 统计，不是离线阈值校准）：

| 指标 | B1+B2 | Full |
|---|---:|---:|
| selected-presence AUROC | 0.7340 | 0.7204 |
| selected-presence AUPRC | 0.5027 | 0.5299 |
| selected-presence ECE | 0.0193 | 0.0215 |
| 含目标行 presence 平均值 | 0.3484 | 0.3862 |
| 无目标行 presence 平均值 | 0.2360 | 0.2588 |
| 有效结构行 bounded ΔU 均值 | -0.000300 | -0.003490 |
| B3 helpful AUROC | — | 0.6795 |
| B3 harmful AUROC | — | 0.8098 |
| B3 helpful/harmful AUPRC | — | 0.4639 / 0.4712 |
| B3 q 对即时 ΔU 的 MSE | — | 0.00960 |

presence 概率普遍低于 0.5，与真实类比例和有限可分性有关，不能单凭低分断言 head 崩溃；它的 AUPRC 高于约 27%—31% 的正例占比。更不应把这个辅助概率门作为是否存在真实点的结构条件。relation 使用类别加权 BCE，分数用于排序，其 ECE 偏高不能直接等同于选点失败。

H3 第 58/59/60 轮有效 shadow 行分别为 **54/39/46**，每轮 mechanism 真实 endpoint 为 **4003**，故 late-3 实际行曝光率 **139/12009=1.1575%**。每轮都有 251 条机制 batch 日志，H3 记录值仅 0 或 1/16，与每次只选一 slot 一致。第 60 轮另一个口径“含 H3 的 batch 比例”为 46/251=18.33%，二者不能混用。every-two-batches-one-slot 的理论行上限约 3.125%，再乘约 38% 结构可用率后约 1.2%；当前数据不支持调度失效的判断。未逐项导出末帧/失败原因汇总，后续应补全而非凭空假定 shadow 失败。

## 5. 可复用的 checkpoint 与修改优先级

1. **立即修公共结构候选合同，先不改参数形状、损失、半径和训练轨迹。** 当前 B1+B2/Full 的 58/59/60 checkpoint 可以用于修复后重新推理、导出动作、每个 Full checkpoint 重新校准。这是同一已训练模型的推理修复，不是冻结后从其他实验 checkpoint 再训练。严格完整加载仍应使用各自 run 的 resolved_config；不能混用后端或配置。
2. **不要复用此前生成的错误 endpoint/artifact。** calibration artifact 绑定代码内容（`action_calibration_v27.py:268`、`:305`），修复后必须重新导出和拟合。原始训练结果保留，修复后的推理输出单列版本。不得将旧主 endpoint CSV 的 blocked bounded gain=0 当真实零收益；本次从候选 sidecar 恢复的 H1 仅用于审计，不能替代真实完整闭环。
3. **在当前 checkpoint 上先补可解释的动作实验**：同一状态的 raw/bounded 所有合法候选、never 和修复后 bounded_always 闭环、Full calibration→locked dev→official mini_val。保持既定 0.75m 半径，明确修复本身的影响，再决定算法升级。
4. **若修门后 bounded 仍显著落后于 raw潜力**：设计尺寸/误差证据条件化的恢复半径或渐进恢复动作；B3 的输入、H1、H3 和校准必须统一使用新的实际动作，需从 epoch0 重新训练 B3 所属正式实验，不能拿旧 utility head 直接宣称新动作已校准。
5. **若 selected 有目标而 consensus 仍失败**：优先检查目标/背景混合模式、模式排序与记忆污染。现在 memory 的 foreground role 只是过去预测框内点（`evidence_memory.py:167`），跟踪漂移后可能把背景标作目标记忆。可借鉴稳定首帧身份记忆、历史质量软权重、目标与上下文的身份分离，但需要真实 memory 诊断/empty-memory 灵敏度验证；当前结果还不能证明具体污染因果。
6. **若获取阶段已丢失大部分可新增目标**：优先修 B0/时间/坐标或调整获取范围，不继续堆叠 B2 attention。保持 768→256 预算，现有高目标保留率并不支持此处优先扩容。

建议补充日志：真实 scene token/tracklet key（当前 dev CSV 全部 scene_id=unknown、tracklet_id=0..11）；结构有效性与 presence 分字段；observation/raw/bounded/final 的 XY、半径、clip fraction；所有结构候选的 H1/H3 有效性和 shadow failure 分类。这样下一轮可做逐轨迹配对与复算，避免再次把输出协议错误误认为模块失败。

## 可复算附件

- [b2_b3_evidence.json](b2_b3_evidence.json)：全部 dev 轮次条件统计、训练专属标量和精确 H3 分子分母。
- [b2_b3_dev_epochs.csv](b2_b3_dev_epochs.csv)：12 次 dev 诊断的候选、目标、旧门、raw 收益轨迹。
- [extract_b2_b3.py](extract_b2_b3.py)：只读原始结果的复算脚本。普通 loss 标量仍为 batch 均值，不能误当点/行总体加权均值；整 epoch 统计与精确 H3 已另行标注。
- [reproduce_presence_contract.py](reproduce_presence_contract.py)：无数据集的真实 B2/B3/evaluator 复现。
