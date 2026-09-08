# CT-SeqTrack v27：五臂 mini 实验、数据链路与改进方案

审计日期：2026-09-07；代码版本：`8b8b8d9`。原始实验目录只读。本轮交付数据表、图、根因审计和可执行问题复现；下文列出的模型修复与新设计尚未实施。

> 后续B0专项复核发现更早的训练/特征问题：五臂四view的BC损失双计、预测历史prior语义回归、继承的帧/通道混排与历史mask遗漏。修复优先级应先处理B0；完整补充见[B0根因与时空融合方案](B0_ROOT_CAUSES_AND_FUSION.md)。本报告的原始指标与下游presence问题仍成立，但不能把下游动作合同当成B0低分的解释。

## 1. 判断

**五个实验确实正常训练完了，但不能据此宣布 v27 已完成验证、已经涨分，或直接进入完整 nuScenes 正式实验。** 当前主要问题已经从“能否跑完”转为“实际执行的动作是否符合设计、比较是否匹配、取得的证据能否转化为有效位置更新”。

本次发现一个明确的 P0 代码遗漏：旧 `presence >= 0.5` 门仍被当成候选合法性使用，阻断 B1+B2 的 `bounded_always`，并污染主逐帧导出与校准候选集合。真实 B2、B3 和公共评估器的 CPU 复现进一步证明：Full 装入允许这类动作的策略后，可能抛出 `v27 action applied without a finite structural candidate`。因此，之前“取消 presence 硬门已经全部落实”的判断需要更正。

**修门也不等于马上涨分。** 通过两份逐帧文件交叉复核，恢复出的真实有界动作单步净收益仍接近零；当前约 0.75m 的半径把有价值的 raw correction 大幅截断。这是后续设计必须处理的第二个核心问题。

## 2. 分数、基线和可比范围

### 2.1 五臂第 60 轮的实际结果

以下全部来自 **dev：scene-0061，12 条轨迹、321 帧，包含 12 个首帧和 309 个预测帧**。S/P 单位为百分数；U=(S+P)/2。不是官方 mini_val 结果。

| 运行 | S | P | U | 相对独立 B0 的 ΔU | 实际动作帧/309 |
|---|---:|---:|---:|---:|---:|
| B0 | 17.952 | 15.553 | 16.752 | — | 0 |
| B1-CfC | 19.198 | 17.383 | 18.290 | +1.538 | 0 |
| B1-GRU | 20.896 | 18.645 | 19.770 | +3.018 | 0 |
| B1+B2 | 16.612 | 14.167 | 15.389 | −1.363 | 2 |
| Full | 17.593 | 15.202 | 16.398 | −0.354 | 0 |

这里的 ΔU 仅描述最终运行差异，**不是模块因果增益**。B0 权重已经跨臂分叉；CfC/GRU 两臂最终又只输出 observation，不能将上表解读成“GRU 插件直接涨了 3 分”。

Full 的 321 条 endpoint 均记录 `missing_calibration_artifact`，全部输出 observation。因此它目前不是完成校准后的 Full selective 成绩。五个目录只包含每五轮的 dev 验证，没有找到对应官方 mini_val 评测或 Full 校准 artifact。

第 58/59/60 轮 checkpoint 都在，但第 58、59 轮没有对应评测。**不能用 50/55/60 三次验证冒充 late-3=58/59/60。**

已对同一组 12 条轨迹做配对 bootstrap：CfC、GRU、B1+B2、Full 相对 B0 的 ΔU 95%区间分别为 `[-0.478,4.682]`、`[-0.248,7.268]`、`[-6.944,2.313]`、`[-1.291,0.786]`，均跨 0。这是固定单场景内、按轨迹重采样的描述性区间；没有跨场景或跨 seed 稳定性的含义，也不能消除 B0 不同造成的混杂。

### 2.2 本地原 SeqTrack 基线

重新读取了冻结参考仓库 `../seqtrack/output/` 的 TensorBoard，未只引用旧报告：

| 历史 SeqTrack 运行 | 第60轮 S | P | U | 训练更新数 | 评测 |
|---|---:|---:|---:|---:|---|
| 20260528，workers=12 | 50.986 | 59.962 | 55.474 | 75,720 | 官方 mini_val |
| 20260801，workers=4 | 31.684 | 31.337 | 31.510 | 75,720 | 官方 mini_val |
| 20260825 重跑，workers=4 | 27.997 | 26.484 | 27.240 | 75,720 | 官方 mini_val |

三次历史结果自身波动很大，不能任意挑最高或最低作为有利基线。它们使用原 8 场景 mini_train、原独立候选目标、历史评价代码；v27 是 6 场景训练、63,420 次 B0 更新、四视图加权、首帧尺寸和公共数据修复。历史代码 `../seqtrack/models/base_model.py:231` 还读取当前 GT 的 `wlh` 作为输入尺寸，v27 使用首帧尺寸；这一实现差别也需明确记录，不能把纠正数据使用方式算成算法涨分。

**不能把 v27 的 17–21 与历史 SeqTrack 的 28–51 直接相减。也不能只把历史 SeqTrack checkpoint 在 scene-0061 上重评来构造公平 dev 对照，因为它曾使用该场景训练。** 应补跑 `cfgs/27_seqtrack_reference.yaml`，匹配 6 场景、seed42、60轮、bs16、workers4、每5轮dev和预算，再统一评官方 mini_val。该 reference 已有配置，本次尚无对应运行证据。

## 3. 训练和 checkpoint：确实完成了什么

- 五组均有 `max_epochs=60 reached`；日志未发现 Traceback、RuntimeError 或 OOM。
- 15 个 late-3 文件全部可反序列化，内部 epoch 为 57/58/59，边界标记完整。最终 `global_step=63420`。
- 启用插件各有 15,060 次模块更新；每轮 mechanism 为 251 步、4,003 个预测 endpoint，覆盖 226 条有非首帧的训练轨迹。另两条只有首帧，不存在机制预测更新。
- 最终模型和 Adam 浮点状态均有限，所有启用参数有 Adam 状态和实际更新；没有冻结、跨运行 checkpoint 初始化或未训练的启用参数。
- mini 实际划分为 6 train / 1 calibration (`scene-0796`) / 1 dev (`scene-0061`)；官方 mini_val 是 `scene-0103, scene-0916`。不能只看 YAML 的 `val_split: mini_val`，实际 dataset 还按 `protocol_role='dev'` 选择场景。

这些证据证明本次训练成功，不等于尚未执行的校准、正式评测及任意配置恢复也一定成功。校准路径已经发现确定的逻辑冲突，见下文。

### B0 为何仍不对齐

五臂 B0 初始化 hash 相同；首 100 个完整 observation batch 逐字节相同；首次四个 view 的 loss 相同；但第一次 Adam 更新后的参数和 Adam hash 已不同。插件第一次执行在 global step 4，首次分叉发生得更早。最终全部 28 个 BN 的计数均为 63,420，全局 Python/NumPy/Torch CPU/CUDA RNG 状态相同。

这将首个差异收窄到了 **observation forward 之后的反向/优化器数值路径**。CUDA 非确定性归约或 Adam 路径是优先假设；还没有实际逐层梯度快照，不能声称已找到具体算子。现有证据不支持将首次差异归因于 CfC 参数量、输入不同、插件梯度串入或多更新 B0 BN。最终 B0 参数相对 L2 差异约 20.9%–21.5%，递归轨迹已实质不同。

下一次定位只需先在同一 GPU 顺序跑两次 B0、一次 GRU 的 100-step，从同 seed scratch 开始，保存逐层 forward/梯度/BN/Adam 和参数差异，再按同臂重复与跨臂差异区分原因。不要先再投入五次60轮。无需冻结 B0 或分阶段训练插件。

另一个日志陷阱：`loss_loss_b0_transaction` 同一个 step 可先写 observation、再写 mechanism 上的监测值；后者没有训练 B0。不能按“每step最后一条”求 B0 loss。本报告只用独占 observation 的 `loss_loss_b0_view0` 汇总。

## 4. 实际数据链路与代码根因

```text
observation 数据流 → B0四视图损失 → B0参数更新

mechanism 数据流：
过去预测框/质量/有效时间 → B1物理位移与获取margin
  → 实际AcquisitionRecord → 当前全云的endpoint/tube/corridor
  → 排除整个B0 raw crop的原始ID → novel pool → 768
  → B2局部几何+base/foreground/context memory → 256 → vote/raw候选
  → bounded action → B3当前/未来收益监督

训练递归：host提交observation；H3使用独立状态。
正式B1+B2：应提交结构合法的bounded action。
正式Full：应提交校准策略选出的bounded action。
当前错误：旧presence门污染候选有效性，推理/导出与训练采用不同合同。
```

### 4.1 P0：残留 presence 门与结构有效性冲突

| 位置 | 当前行为 | 后果 |
|---|---|---|
| `models/ct_v2/evidence_memory.py:880–882` | `candidate_valid = availability & evidence_present` | 将 learned presence 阈值混入结构有效性 |
| `models/seqtrack3d.py:3002,3049` | 读取旧字段写 `ct_policy_candidate_valid` | 把残留门传给 host 和导出 |
| `models/seqtrack3d.py:3168` | bounded_always 根据上述字段执行 | B1+B2 并未真正执行计划中的结构合法即更新 |
| `utils/v27_evaluation.py:58–67` | 用旧字段决定候选存在，冲突时抛异常 | 漏记动作标签；Full 校准/评测可能中断 |
| `utils/v27_eval_reporting.py:32` | 只对误标的 structural 集合统计 raw/bounded | 原 summary 严重缩小候选分母 |

训练 B3 的 `utils/v27_training.py:18` 优先使用 `ct_b2_available`，B3 forward 也接收结构可用性，因而这里是**训练与部署/报告合同不一致**，不是 B3 从未学习。

| 第60轮真实数量 | B1+B2 | Full |
|---|---:|---:|
| 有非空 selected evidence 的预测帧 | 177 | 140 |
| selected 内有目标点的帧 | 60 | 55 |
| 主表标成 structural_available 的帧 | 2 | 12 |
| 被旧门漏掉的结构候选帧 | 175 | 128 |
| 真正提交动作的帧 | 2 | 0 |

最小复现使用真实 B2、真实 B3 和公共 evaluator，只有 B0/点云输入为合成夹具：4 个唯一 extension 点、presence=0.1 时 `ct_b2_available=1`、`ct_search_candidate_valid=0`；B3 装 `always` 后执行动作，evaluator 必现上述 RuntimeError。见 [复现脚本](reproduce_presence_contract.py) 和 [执行结果](presence_contract_reproduction.json)。这是可触发的代码错误，不是“将来或许有风险”的泛泛推测。

修复应统一所有读写端：v27 的结构候选仅由非空唯一有效证据、有限候选和有限动作几何决定；presence 保留为连续输入/辅助监督。不要通过取消异常、强行改表或把阈值设成0绕开冲突。补上跨 B2→host→B3→evaluator→calibration 的同候选集测试。

### 4.2 B1：获取已生效，但无法从错误历史中凭空恢复位置

四个 B1 臂均有 297/309 帧使用真实 learned 获取；12 帧是每条轨迹首次 query，没有有效 transition。本次没有旧版“margin 丢失导致全量 CV fallback”的现象。

| 第60轮获取指标 | CfC | GRU | B1+B2 | Full |
|---|---:|---:|---:|---:|
| 全局 novel 目标点 | 11,737 | 10,095 | 11,264 | 11,906 |
| 实际取得 novel 目标点 | 330 | 624 | 759 | 1,210 |
| 实际 novel recall | 2.81% | 6.18% | 6.74% | 10.16% |
| novel pool→768 目标点保留率 | 100% | 100% | 100% | 100% |

Full 在 observation 误差>10m 的帧中，有 9,365 个全局 novel 目标点，仅取得16个；B1 相对 CV 的平均调整约0.50m。当前主要损失发生在 support 之前，不在768采样预算。

margin 分支确实更新：late-3 有效 batch 的平行 margin 大约3.27–3.86m，垂直 margin 约1.0003m，贴近下界。需要补 `no_novel / reachable_novel / outside_maximum_support` 三组标签计数和 margin 分布，才能判断是合理监督还是大量无novel样本压过稀有正例。不能仅凭均值认定bug或直接调大 sigma；sigma 与获取范围本来就是不同目标。

另有已确认的指标语义问题：训练物理目标是 `Rᵀ(GT_t−GT_(t−1))`，当前 dev NLL/coverage 却拿 `Rᵀ(GT_t−prediction_(t−1))` 评价。二者相差历史定位误差。应分别记录 physical-delta RMSE/NLL/coverage 与 endpoint/search error，不能以当前日志判定 CfC 或 GRU 的统计预测优劣。对应 `utils/candidate_utils.py:82`、`models/base_model.py:1139,1358`。

### 4.3 B2：保留了目标点，身份污染仍在，但更大的障碍在动作端

| 第60轮，目标点跨endpoint求和 | B1+B2 | Full |
|---|---:|---:|
| 768 pool 中目标点 | 759 | 1,210 |
| 256 selected 中目标点 | 720 | 1,167 |
| 选点目标保留率 | 94.86% | 96.45% |
| top-mode 中目标点 | 499 | 908 |
| selected含目标帧：observation→raw平均距离 | 6.47→3.70m | 6.04→3.13m |
| selected含目标帧：observation→raw中位距离 | 6.11→1.67m | 5.66→1.39m |

这些局部数值支持“B2可以利用部分新增证据”，不支持“B2已经闭环涨分”。在 selected 含目标帧中，top-mode 的目标比例仍约31.1%/43.4%，说明几何共识仍可能混入背景或错误实例。全体帧的 top-mode 目标比例更低，不能混用这两个分母。

因此当前不应优先增加选点数、堆叠更深 attention 或换更大 backbone。先修动作合同，再改善记忆身份线索、模式质量与获取位置；保留现有局部几何、relation/targetness分工和长车vote设计作为可消融组件。

### 4.4 B3与动作半径：raw收益几乎没有转化为实际有界收益

主 endpoint 表漏掉的 H1 可以由 `candidate_diagnostics/epoch_60.csv` 恢复：其 v27 `success_gain/precision_gain/utility_gain` 在 `models/base_model.py:1660` 使用完整有界动作计算，没有套旧presence门。已逐行匹配309个endpoint，并与门内记录交叉验证。以下分母均为321帧，首帧/无动作收益为0。

| 同一当前状态下的单步反事实 | B1+B2 | Full |
|---|---:|---:|
| 采用 raw candidate 的净 ΔU | +3.715 分 | +5.148 分 |
| 采用真实 bounded candidate 的净 ΔU | −0.0779 分 | −0.0078 分 |
| bounded helpful/harmful帧 | 26 / 12 | 8 / 6 |
| 完美挑选H1正收益动作的 ΔU | +0.3349 分 | +0.1207 分 |
| 发生截断的结构候选帧 | 173/177 | 140/140 |

这不是闭环效果或未来收益上界。特别是第一行不能写成“修复后预计涨3–5分”：这些候选基于原轨迹，真正应用后会改变后续输入、memory与获取位置。

当前 `r=min(0.5+0.5Δt,2)`，nuScenes常见Δt≈0.5s时只有约0.75m。多米误差即使沿正确方向纠正0.75m，也可能仍在 Precision 的2m阈值外、IoU仍为0；离散S/P贡献不变，B3就看到大量零收益。这解释了为什么 raw localization 改善明显、bounded收益却近零。**只把 `radius_max` 从2调大不会改变普通0.5s帧的0.75m半径。**

B3不是完全没学：训练 late-3 的 help/harm AUROC约0.680/0.810，但只代表训练分布。有效 H3 行为54/39/46，共139/12,009=1.157%；每2个机制batch最多1slot再乘结构可用性和末尾限制，本就会很稀疏，不能误判调度没有执行。增加H3数量之前，应先确认动作本身具有可辨识收益。

## 5. 低分的根因分层

1. **明确代码错误**：旧presence门及候选有效性字段冲突；B1物理预测指标与endpoint指标混用；同名loss混写易误读。
2. **已观察到的机制瓶颈**：严重漂移后support取不到目标；约0.75m动作半径几乎消掉raw候选的S/P收益；因此B3选择空间本身很弱。
3. **需要进一步定位的训练问题**：B0第一次更新后的数值分叉；尚需服务器同臂重复和梯度级证据确认具体运算。
4. **数据与学习分布问题**：单dev场景很小；B0 observation loss从约7.3降至0.30，但dev没有相应改善。符合训练分布与递归推理分布不匹配/过拟合的症状，尚不能仅凭曲线分离两者。

B0第60轮309个预测帧中，230帧全云有目标点，但231帧base crop内无目标，117帧base完全空，260帧最终距离>2m。早期失效随后让B1围绕错误历史搜索。首query就失败的tracklet3/8/10全云目标点为0；tracklet7的8个目标点全部已在base。它们分别需要处理不可观测状态与B0已有稀疏证据的错误回归，不能全部归因于搜索范围不足。

## 6. 修改顺序与研究方向

### 第一优先：修合同，利用已有checkpoint补清楚事实

- 统一v27结构有效性和有界动作构造器，覆盖B1+B2、Full、endpoint导出和校准；保留legacy行为仅供旧版本读取，v27不再消费它。
- 修B1两类指标名称/目标；给observation与mechanism日志不同前缀；为endpoint补真实scene/token，而非`unknown`和仅位置序号。
- 使用现有58/59/60 checkpoint，先在dev做各自模型的 observation-only 对照与修正后的 bounded-always。Full每个checkpoint单独在calibration场景拟合、闭环选策略，dev仅诊断。生成新artifact，不覆盖原数据。
- 同步完成上述同GPU的100-step B0重复定位。修复推理/报告不会要求将现有模型作跨实验初始化；改变训练目标/网络后的正式实验仍全部scratch。

### 第二优先：调整可执行动作，而不只调整筛选器

先在已有checkpoint上做明确标记的推理敏感性：比较0.75m、1.5m、3m和raw诊断的当前/闭环收益、背景误更新与恢复时机。半径变化需改变base/per-second项或独立动作预算；同时重建该半径的实际H1/H3和校准，不能沿用只学过0.75m动作的B3分数作正式结论。

若较大半径确实恢复了有用候选，建议下版采用**少量不同幅度的候选动作+动作条件收益头**：例如保留observation以及同一vote方向的三个有界幅度，让B3比较具体动作的S/P收益，而不是只对一个常常无效的小动作作接受/拒绝。共享证据特征，输入实际位移/半径/clip ratio，每个动作单独H1，H3轮换采样动作。这样可以把已有raw候选的价值转成可学习的动作选择。幅度集合是待验证设计，不是本次数据证明的最佳超参数。

这会改变B3任务及训练标签，需新schema、从epoch0联合训练、同一优化过程更新所有启用模块；不冻结、不从旧checkpoint接着训练新结构。

### 第三优先：改善早期递归状态和身份记忆

- B0增加稀疏点、纯背景、预测历史误差的诊断切片，比较训练与递归时的质量/运动分布。优先用因果的预测质量、历史一致性及训练增强降低背景下任意大跳；不可用当前GT可见性作推理门，也不要重新把N=1/2清零。
- 如果给B0增加新的可靠性回归/状态约束，应明确这是算法改动，不能伪称原SeqTrack或普通数据修复；保留独立reference及相应消融。
- B1补三类margin监督分层；针对中等漂移调整获取位置与历史质量，而非无限扩大背景域。首query可研究按首帧yaw/尺寸的stationary最小margin获取，但应单独标source，它解决不了当前全局无目标的帧。
- B2优先保留可靠目标身份：可研究首帧身份锚点加质量选择的近期memory，避免仅由最近错误框不断覆盖目标线索；前景/上下文分路和同类干扰区分比增加memory长度更值得验证。先证明污染存在与改动收益，再扩结构。
- CfC、GRU继续保留，默认GRU只是工程选择。用同一份预测历史replay两种B1并核对输入digest，比较物理位移、实际novel recall、背景竞争与耗时；这种固定checkpoint比较仍属于敏感性，完整后端对照需匹配训练路径。

## 7. 文献如何借鉴到当前问题

- **SeqTrack3D（2024）**明确使用点云与框序列，并讨论随机GT偏移难以模拟长历史测试误差的问题。对本项目的启发是检查训练扰动与真实递归漂移的匹配程度，而不是先增加历史长度。[原文](https://arxiv.org/html/2402.16249v1)
- **CXTrack（CVPR 2023）**在传播几何特征的同时逐层传播/细化targetness，讨论直接门控或对身份线索施加dropout的局限。可借鉴身份线索独立传播和辅助中心监督，改进当前“模式几何一致但身份不纯”的问题；不是照搬整个网络。[原文](https://arxiv.org/html/2211.08542v1)
- **MBPTrack（ICCV 2023）**利用历史targetness memory和首帧box prior；其memory规模实验也显示更多历史并不总更好。本项目可重点研究可靠身份锚点、记忆质量与几何/身份分路，不宜只把36个token盲目加大。[原文](https://arxiv.org/html/2303.05071v1)
- **HVTrack（ECCV 2024）**使用相对位姿记忆、局部/扩展特征尺度和上下文抑噪。注意其“base-expansion”主要是特征感受野分支，并不等于本项目按原始ID获取novel点云；可借鉴抗同类干扰的方法，但本项目需要单独证明实际增量证据价值。[原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01145.pdf)
- **CfC（Nature Machine Intelligence 2022）**提供显式时间依赖的连续时间后端依据，并不能推出它在这里必然胜过GRU。本次真正短板在递归定位、获取与动作收益链，换循环单元不能代替修链路。[原文](https://doi.org/10.1038/s42256-022-00556-7)

以上是针对已观察到瓶颈的设计借鉴，不是声称这些论文已经验证了CT-SeqTrack的新方案。没有参考TrajTrack。

## 8. 下一步实验清单

| 顺序 | 工作 | 完成后能回答的问题 |
|---|---|---|
| 1 | 修presence/结构合同，复现用例转为集成测试 | 实际输出、H1/H3、导出和校准是否针对同一个动作 |
| 2 | 已有checkpoint做本模型observation vs bounded/selective；Full逐checkpoint校准 | 插件在相同B0下到底改变了多少闭环S/P |
| 3 | 同GPU两次B0+一次GRU的100-step诊断 | 跨臂分叉是否来自同臂也存在的数值非确定性 |
| 4 | 补匹配的SeqTrack reference；补五臂58/59/60官方mini_val评测 | 可比baseline和正式late-3是否成立 |
| 5 | 根据动作半径/记忆/稀疏分层证据选择最少算法改动，全部scratch验证 | 哪项改动带来可复现收益，而非只增加复杂度 |
| 6 | mini链路和收益确认后跑完整nuScenes | 350 train_track训练、150官方val上的五类别论文证据 |

完整数据继续遵守350训练/150评测；17/18阈值拟合/诊断子集属于训练集合，官方val不参与阈值拟合或最佳epoch选择。当前mini结果不包含Bus/Trailer等长目标或完整数据上的性能证据。

本轮没有可靠的部署FPS对照：四插件任务同卡并行，诊断还包含GT标签/几何计数、同步计时等开销。已有runtime可用于定位工程耗时，不能与论文FPS直接比较。

## 9. 可复查产物

- [总览图 PNG](audit_overview.png) / [PDF](audit_overview.pdf)，配套[作图来源与口径](figure_provenance.json)。
- [第60轮主表](dev_epoch60_summary.csv)、[每五轮验证曲线](dev_validation_curves.csv)、[B0纯observation loss](b0_view0_training_loss.csv)、[逐轨迹表](dev_epoch60_tracklets.csv)。
- [历史SeqTrack原始基线汇总](historical_seqtrack_baselines.csv)、[描述性配对区间](dev_descriptive_paired_deltas.csv)。
- [训练/checkpoint/B0公平性审计](checkpoint_fairness.md)、[B1审计](b1_audit.md)、[B2/B3审计](b2_b3_audit.md)。
- 主数据采集脚本：`artifacts/ct_checks/analyze_20260907_v27_mini.py`；子审计脚本和JSON均在本目录。原始五臂各321条主endpoint、309条候选诊断保留在原output目录。

本报告把确认错误、观察到的性能瓶颈、需要验证的假设和未执行的后续工作分开。真实训练成功是已经取得的工程进展；论文收益仍需修正动作链后，以匹配的完整闭环评测证明。
