# v37：保护强主路的扩展候选与最终框核验

**2026-10-21 B2 梯度修正已实施为独立配方。** `v37_centroid_recipe` 仅切断identity×reliability到重心的直接权重梯度，分子/分母共用detached mass；votes、live center、精修和显式两头监督保留。共享接口默认关闭，以下原v37定义仍用于旧五组。新三组身份、残余梯度边界和操作见 [B2三组说明](CTSEQTRACK_V37_B2_CENTROID_LAUNCH.md)，尚无正式结果。

**2026-10-21 结果补充。** 下文为 10/09 已实施结构的定义，原五组现已完成；Full FP32 final60/late3=49.9136/59.5274、50.0715/59.6327，低于强 B1 且未过原门。输入/证据权限隔离已经实现，但没有证明强主路训练不受辅助梯度影响，也没有证明恢复链有效。当前问题、实验进展与文献提案见 [研究审查](CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md)。该审查没有修改本页结构、loss、配置或已训练源码；不能把建议写成 v37 已实现能力。

2026-10-09。实现方案与五组预登记，不是正式涨分结果。登记文件为 `cfgs/ct_seqtrack/37_registration.json`，操作见 [启动说明](CTSEQTRACK_V37_LAUNCH.md)。

## 研究问题与共同起点

v36 五组已经完成且未达原总体/移动门。v37 回到强 v35 B0/B1/W：四分支 1/4/4/8、十轮窗口课程、112 reserve、首测合法精确初框、33 维历史描述、coarse 条件、原物理时间 B1 与同源 accepted 状态。q0 局部证据恢复 LWH 归一化半径 1、每角最多 16 个唯一点。v36 的测量事件时间链、连续长分支课程和独立测量提交不进入本版。

研究假设是：在保护 q0 和 B1 的前提下，extension-only 身份检索与顺序去重能够产生额外可定位候选；在最终框处独立估计 S/P 效用，并将排序与记忆写入质量分开，可减少“有好候选但选不对”和错误写入。它是否提高真实闭环分数、是否减少持续失跟，须由固定五组检验。

## B0/B1 与状态边界

`models/ct_v37/model.py` 的 B0/B1 直接使用旧观察、decoder、prior 和逐项 loss；新增配置适配发生在独立 `as_v35_config` 中，不改旧默认值或配置 SHA。新 schema 为 `ct_seqtrack.structural_evidence.v37`，有独立 runtime、配置、源码和执行精度身份。

q0/history 的共享头及旧全局来源保持；q0 定位只读当前 B0 局部证据，不读 extension 或 memory K/V。B2/Full 的新 mode 头在独立 RNG 作用域初始化，避免改变共同主路和 B1 初始化。共同参数初始内容摘要进入 run manifest；恢复和比较分别核对配置、源码、run、精度、排程及完整 epoch。

所有启用参数正常学习；detach 是梯度/递归耦合边界。每 batch 一次 forward、backward、Adam，随后由 host 提交 detached accepted 状态。GT 只用于合法首帧初始化、监督、提交后的被动台账。

## B2：当前几何产生 vote，历史记忆提供身份

获取器、768 extension 池、identity 选取 256 个真实点及 36 个 raw memory 槽沿用既有容量。36 槽为首帧 12 FG 加最近两次各 8 FG+4 BG。历史点按保存框归一化重新编码，只用于身份判断；不将历史 canonical 坐标运输到当前，不做刚性对应估计，不增加 canonical-coordinate 辅助头。

vote 几何只由当前 extension 点、当前几何特征和当前观察产生。固定 selected IDs 后，改变 memory 不应改变几何 vote；identity 排名与 mode 加权可能随 memory 改变，这仍是需要审计的污染通道。identity 与 reliability 共同决定聚合权重，GT 仅监督 identity、foreground vote 和 reliability。

两个 mode 顺序形成：1 m 聚类，0.75 m 排除，mode0 的真实成员不能再次成为 mode1 成员。最终精修后两中心相距不超过 0.75 m 时按固定顺序去除 mode1。记录 seed 与 final 的 GT 误差及“seed 好、精修坏”比例，区分获取、投票、精修和选择失败。

mode 的局部读取为当前 B0 加自身成员，使用 LWH 归一化半径 1、每角最多 16 点；新增 residual pose head 与 q0/history 的原 pose head 分开。共享 decoder 参数仍可收到 mode 定位梯度，因此本版保护读取/参数职责，不声称 Full 中 q0 训练轨迹逐位等于 B1。

## B3：最终框排序与唯一输出提交

候选固定 `[q0, prior, mode0, mode1]`。prior 使用原 v35 `prior.box`，始终作为传播候选；四帧全空时 q0 等于 prior，并标记 fallback，在 Full 竞争中无效。B1+B2 仍输出 q0，即使是合法的全空 fallback。

独立 verifier 在最终框的 8 角读取当前点，LWH 半径 1、每角 16 个唯一点。其定位框、特征、prior 与统计输入全部 detach；核验参数正常学习。35 维统计包含相对 prior/coarse 的几何量、来源、原支持统计及 weak/supported/历史年龄。评分读取权限：q0 和 prior 可读当前 B0+selected256；mode 只读当前 B0+自身成员。评分不读取记忆坐标，不允许 GT 进入输入。

verifier 输出 S、P、Qmode 三 logits。排序效用为 `(sigmoid(S)+sigmoid(P))/2`，在有效候选中 argmax，同分按候选次序优先 q0。S/P 标签直接来自注册 benchmark 指标的单帧贡献；pairwise ranking 使用标签差的方向、margin 和权重，不另调门槛。此评分是监督效用预测，不宣称已实现外部统计校准。

提交质量与排序分开：q0 使用旧 quality head；mode 使用 Qmode；prior、fallback、无效 mode 的提交质量为 0。Qmode 初始概率 0.1，采用旧 q0 几何质量标签公式。只有被选中的唯一输出能提交；没有第二个独立“测量候选”。

提交证据权限比 q0 评分更窄：q0 只允许当前 B0，mode 允许当前 B0+自身 selected 成员。必须在自己的最终框内有至少 3 个 unique predicted FG，且自身提交质量不低于 0.5，才更新受支持状态及 memory。prior/fallback 只传播位置，不伪造观测时间或写 memory。memory 中允许保存授权的低概率 BG；高概率框外点不得冒充 FG。

## 完整训练图

B1+B2 与 Full 都训练完整身份、vote、mode 与 verifier，只改变输出选择/提交政策；不把 shadow 臂偷换成不训练 mode 或 scorer 的旧 B1+B2。原 B0/history/B1 逐项 loss 保留，新增总 loss 为：

`L = L_v35_B1 + 0.2 L_identity + L_vote + 0.1 L_reliability + L_modes + 0.125 L_S + 0.125 L_P + 0.5 L_Qmode + 0.1 L_rank`。

`L_modes = 2 mean(0.5 seed_center_loss + 0.5 final_center_loss) + 10 mean(1-cos(yaw_error))`，仅有效、至少一个真 FG 且成员纯度≥0.5 的 mode 监督定位。identity/reliability 使用前景背景平衡；vote、mode、S/P/Q 和 rank 按各自有效端点/候选归约。被动 `loss_statistics` 同时记录分子、分母和实际 batch 贡献，避免把不同归约伪装成相同权重。无额外结构坐标辅助任务。

## 五组与固定结论口径

| 标签 | 训练图 | 输出/提交 | 精度 |
|---|---|---|---|
| b0 | 强 v35 B0 | 原 q0 | optimized_fp32 |
| b1 | 强 v35 B0+B1 | 原 q0 | optimized_fp32 |
| b1_b2 | 完整 B2/mode/verifier | q0 shadow，提交限 B0 | optimized_fp32 |
| full | 完整图 | 最终框效用选择，唯一输出提交 | optimized_fp32 |
| full_bf16 | 与 Full 同构 | 与 Full 相同 | 局部 optimized_bf16 |

所有组 mini Car、seed42、scratch60、batch16、LR=1.5e-4、20/50 衰减、无 warmup；每组 71,700 Adam 更新，共 300 epoch/358,500 更新。正式 BF16 仅此 v37 Full；Trainer、参数、Adam、loss/几何/状态保持 FP32，BF16 只覆盖已标明的特征算子。不追加完整训练、LR、seed 或模块组合。

主模型预先固定为 Full FP32。比较原始 58/59/60 帧，以 final60 与 late-3 的 S/P 对原固定 R/C 设门；额外报告新 B1 相对旧强 B0/B1、每级候选链、Full 相对新 B1/B1+B2 和 BF16 相对 Full。任何较高 best epoch、对照或 BF16 都不能替换主模型。共同帧切片、>5m/>10m、未恢复失跟和 memory 错写作为风险复核；严重漂移风险不能被平均分掩盖，也不改写原数值门。

候选记录保留所有最终框、prior 量、获取覆盖、selected 原点、vote、mode 成员、seed/final 几何、评分与自身支持权限、唯一输出和提交证据。比较器独立核对选择、质量、原点去重、框内支持和 memory 写入，并从原帧重算指标；同状态 oracle 只是一步上界，不是闭环性能。

## 文献启发与证据边界

- [UniKPT, CVPR 2026 原论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Xiao_Generalizable_Structure-Aware_Keypoint_Correspondence_for_Category-Unified_3D_Single_Object_Tracking_CVPR_2026_paper.pdf)：结构对应和置信度启发身份/几何职责区分；本版不复刻它的对应或 canonical 运输，也未使用它作为新增监督的依据。
- [ChronoTrack 原论文](https://arxiv.org/html/2604.13789v1)：长期前景身份与短期背景差异启发身份记忆审计；其时序训练和匹配机制不同，不能据此断言本项目记忆有效。
- [BEVTrack, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0250.pdf)：监督定位不确定性与真实递推评测需分开；其 residual likelihood 训练不是本版可直接使用的提交校准器。
- [DAM4SAM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf)：可靠性与干扰区分共同决定记忆价值；任务是 2D 视频/SAM2，与稀疏 3D 框跟踪存在差异，不把 wrong-box bank 当负例模板库。

以上为机制启发，不构成“首创”“SOTA”或涨分证据。memory 污染仍可能通过身份选择侵入候选，mode 仍可能只学到近 q0 的冗余框，S/P 排序与实际恢复收益也可能不一致；正式结果失败时应据候选链和漂移台账收窄结论。
