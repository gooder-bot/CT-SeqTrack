# CT-SeqTrack 当前问题、研究依据与优化顺序（2026-10-21）

## 1. 结论与证据范围

**优先守住强 B1 的学习与递推，再改恢复候选的身份/几何和输出替换资格；效率方面先消除重复评测工作。** 当前证据不支持继续把分解投影、缩 batch、扩大点池或增加循环层数作为主要投入方向。另一个不能忽略的问题是低运动目标的累计漂移：本地合法首框保持基线总分高于当前学习模型，但 B1 在运动帧上确有明显价值，不能简单改成全程不动。

本次是源码、既有实验与论文/代码的研究审查和文档更新，未修改生产模型/配置、未加载 checkpoint 推理、未训练、未操作服务器。以下建议均未实施、未登记新正式预算。没有读取或参考 TrajTrack。

- 当前工作区起点：`09a7809698d512f53d999574b00df2e0773fa93f`。旧五组结果属于 10/09 登记源码；解释原运行时使用 [0662c21 源码快照](../artifacts/ct_checks/20261016_v37_five_run_review/direction/source_0662c21/)，不能用 10/14 的新执行开关解释旧结果。
- 原五组正式结果：[10/16 总报告](../artifacts/ct_checks/20261016_v37_five_run_review/REPORT.html)、[方向审计](../artifacts/ct_checks/20261016_v37_five_run_review/direction/REPORT.md)、[结构审计](../artifacts/ct_checks/20261016_v37_five_run_review/architecture/REPORT.md)、[原帧独立重算](../artifacts/ct_checks/20261016_v37_five_run_review/formal/independent_scores.json)。
- 更晚的本地证据：[10/21 新 B1 执行实验](../artifacts/ct_checks/20261021_v37_execution_decision_review/execution/README.md)、[机制复盘](../artifacts/ct_checks/20261021_v37_execution_decision_review/mechanisms/MECHANISM_REVIEW.txt)、[工程复盘](../artifacts/ct_checks/20261021_v37_execution_decision_review/efficiency/ENGINEERING_REVIEW.txt)、[首框保持审计](../artifacts/ct_checks/20261021_learning_value_audit/summary.json)。本次将这些已存在、尚未进入主文档的结果同步进来，不冒充全部为本次新实验。
- 本次新增被动量化：[评测开销与训练深度](../artifacts/ct_checks/20261021_research_audit/performance/audit.json)及其 [复现脚本](../artifacts/ct_checks/20261021_research_audit/performance/audit.py)。仅解析记录和检查源码；没有 GPU 性能测量。
- 本次新增 [mode 局部读取复核](../artifacts/ct_checks/20261021_research_audit/b2/local_visibility_summary.json)及其 [脚本](../artifacts/ct_checks/20261021_research_audit/b2/audit_local_visibility.py)，重建 4096 个 formed mode 的邻域计数并与原台账核对。
- 证据分层：**事实**是源码行为或已留存数据；**解释**是与事实相容但尚未因果验证的机制；**提案**是待检验改法。单 seed mini、同轨迹相邻三轮不能构成三次独立重复。下文的优先级不是可量化的涨分概率。

## 2. 实验进度必须更新

### 2.1 原 v37 五组已完成

每组 scratch60 / 71,700 更新，总计 300 epoch / 358,500 更新。统一 2285 帧，包括 106 初始化帧、2179 预测帧。S/P 均为百分制；late-3 固定为 58/59/60 均值。

| 配方 | final60 S/P | late-3 S/P |
|---|---:|---:|
| B0 | 52.3162 / 62.0230 | 53.7629 / 64.6080 |
| B1 | 53.7133 / 66.2746 | 52.8541 / 65.3082 |
| B1+B2，始终输出 q0 的 shadow | 51.4376 / 62.4376 | 50.3158 / 60.6831 |
| Full FP32，预登记主模型 | 49.9136 / 59.5274 | 50.0715 / 59.6327 |
| Full 局部 BF16 | 49.9672 / 59.5667 | 49.4121 / 58.4333 |

Full 相比 B1，final60 下降 **3.7998 / 6.7473 pp**，late-3 下降 **2.7826 / 5.6754 pp**，未通过原总体/移动门槛。它高于旧 v35 Full 和 v36 主模型，说明部分退化得到修复，但不能据此宣称完整链优于强基线。FP32 主结果不由 BF16 替代。

### 2.2 10/16 启动的新 B1 对照已有结果

| 新执行配方 | final60 S/P | late-3 S/P | 本地状态 |
|---|---:|---:|---|
| unique_mask + 原 concat，batch16 | 53.7133 / 66.2746 | 52.8541 / 65.3082 | 完成，原联合门通过 |
| unique_mask + 分解投影，batch16 | 50.6663 / 61.5088 | 51.0835 / 62.2319 | 完成，未通过 |
| 分解投影 batch16，仅第 20 轮衰减 | 52.5952 / 64.2374 | 49.0430 / 59.6349 | 完成，未通过 |
| 分解投影 batch4，20/50 衰减 | 未有正式 final60 | 未有 late-3 | 本地完整审计/验证到 41 轮，日志进入 42 轮 |

concat 对旧强 B1：60 份训练诊断、共同 12 次验证、末三轮共同原帧字段逐值相同。hold_lr 对 factorized：前 50 轮诊断和验证相同，第 51 轮首次分叉。二者是很有价值的控制证据，但仍限当前 seed。B1 实际生效的公共去重只有 `unique_mask`，不能把它写成 Full 五项去重都已正式验证。

batch4 第 41 轮为 43.8807/49.2505；60 轮需 286,620 更新，约为 batch16 的 3.9975 倍。当前没有显示值得继续以小 batch 搜索来改进强 B1 的积极信号；是否结束既有任务由用户决定，本次没有停止任务，也不把未完成运行标成正式失败。

四个新运行 manifest 实际均为 `CUDA_VISIBLE_DEVICES=1`，启动重叠；这与启动页的分卡计划不同。并发负载使墙钟不能作为分解加速依据；此外旧 B1 每 5 轮验证、新四组每轮验证，新四组彼此的验证频率相同。本地快照并不表示服务器当前实时状态。

**更正旧审计中的显存表述**：三个原 `results.json` 有 `peak_gpu_allocated_mib`：concat 7081.8872 MiB，factorized/hold_lr 均 6792.7847 MiB，相差 289.1025 MiB（约 4.08%）。入口没有 reset_peak_memory_stats，这是本进程截至结果写出时的 allocator allocated 高水位，可能包含 fit 前初始化，以及训练/validation/late3；缺少受控配对、reserved 和整卡占用数据，不能证明稳定的净显存收益，但也不能再说完全没有峰值记录。旧 artifacts 原样保留，在本页纠正。

**研究工作配方建议保留原 concat batch16 + 原 20/50 衰减。** 分解与 hold_lr 的配置和证据继续保留用于复现；不据本审查修改旧运行或默认再扩到 Full。

## 3. 创新方向：哪些已经有内容，哪些还只是主张

项目的研究价值在于把 **物理时间下的预测、当前扩展证据、长期身份、输出选择和可信更新** 放进同一个可审计闭环。v37 已明确了若干重要边界：B2 不直接写主路定位证据；固定 selected IDs 时，历史 memory 不直接改变当前几何 vote；核验输入 detach；最终只有一个输出提交；传播框不能冒充当前测量写 memory。这些是实际实现的设计内容。

但现有证据尚不能支持“连续时间优于离散时间”“memory 改善恢复”“B3 已校准”“Full 涨分”或“SOTA”。尤其当前 B3 是监督效用预测，不是旧 v24 外部 calibration artifact 机制；旧文档中 B3 名称不能代替当前数学定义。对未来论文更稳妥的问题表述是：**怎样让恢复分支在有可靠当前证据时帮助跟踪，同时减少对原本正确主路和递推状态的损害？** 必须既有保持正常跟踪的结果，也有真实恢复收益，单纯拒绝所有 mode 只能守底。

## 4. 剩余问题与具体优化位置

### 4.1 低运动目标：累计更新损伤可能比恢复次数更影响总分

[首框保持审计](../artifacts/ct_checks/20261021_learning_value_audit/scores.csv)只在合法初始化时取一次世界坐标框，之后保持 XYZ/yaw/size；后续 GT 仅用于计分和分组。当前数据适配已做世界坐标/ego pose 处理，不能把这个基线误称为固定传感器坐标框，也不能外推为其他数据协议的结论。

| final60 切片 | 帧数 | 首框保持 S/P | B1 S/P | Full S/P |
|---|---:|---:|---:|---:|
| 总体，含初始化 | 2285 | 60.5503 / 72.9945 | 53.7133 / 66.2746 | 49.9136 / 59.5274 |
| 相邻 GT XY 位移 ≥0.15m 的运动端点 | 171 | 22.8801 / 25.9211 | 36.9444 / 44.9854 | 22.3246 / 25.4094 |
| 低运动端点 <0.15m | 2008 | 61.6758 / 75.5777 | 52.6980 / 66.3073 | 49.6190 / 60.2963 |

B1 在运动端点明显胜过保持基线，表明当前 B1 系统在这些端点有实用价值，不单独证明物理时间模块的因果贡献；大量低运动端点的负差把总体拉低。低运动不是严格静止，171 个运动端点也不同于正式门槛的 31 条曾移动轨迹全程 657 帧。不能用 GT 运动切片在推理时选策略，不能改正式门槛，也不能把该策略比较称作“训练没有学到东西”：后者需要同一模型 epoch0 的受控比较。

**值得新增的诊断**：从 B1 自己的输出检查小位移时的中心/yaw 抖动、累计漂移、正确观测被替换、观测缺失后何时开始偏移；同时看运动开始/停止边界。先比较无 GT 输入的保持/传播/观测决策能否辨认这些帧。可借鉴 M2-Track 的运动状态预测（第 5 节），但不直接加“预测静止则永久不更新”：这会错过启动运动，并把初始定位误差永久保留。更有价值的设计是学 **有当前证据的增量修正及其收益**，保留重新启动和纠错能力。

### 4.2 B0/共享训练：输入隔离没有保证强主路参数不受影响

shadow 始终输出 q0，却已经低于 B1。因此 Full 的全部下降不能归为 B3 选错。`ct_v36/evidence.py::identify/geometry_votes` 读取 live B0 特征；mode 仍经过共享 decoder、局部读和 prior context。`ct_v37/verification.py` 输入已 detach，不能说 verifier loss 直接反传破坏 B0。

正常可见的 coarse-good→q0-lost 在 Full 末三轮均为 0，v36 的固定 1m 主路局部邻域问题不能照搬到 v37。与此同时，36 个合法首测 raw0 帧上，Full q0 的 Success 相对 B1 在 58/59/60 分别低 8.61/6.25/9.03 pp；弱观测表示和辅助训练耦合仍值得查。相同初始化框和点数不等于逐点输入张量已核验相同。

**较优先提案**：恢复支路读取主路 detached 特征，mode 使用自己的小型 refiner；主路参数仍由原 B1 loss 正常学习，恢复参数也正常学习，不冻结旧权重。只 detach mode 输入却继续训练共享 decoder 不能完成保护。还须隔离新增 RNG 消耗、dropout/BN 调用和状态政策。先做同输入、同状态的 shadow 合同及分损失梯度审计，再判断是否值得新 scratch；负梯度余弦本身也不是最终退分的因果证明。

### 4.3 B2：当前优先瓶颈在取得目标点之后

Full e60 的 bad-q0 恢复链：503 个 q0 exact IoU<0.1 端点中，raw 含 GT 418；最大合法扩展域可达 198；实际搜索域/768 池均为 173；selected256 含 GT 为 171。随后仅 62 个机会形成至少一个正 mode，6 帧有 IoU≥0.5 的最终 mode，0 个好 mode 被选中。正 mode 定义含成员纯度≥0.5 且至少一个 GT 点；“6 个好最终框”不全是正 mode，不能把各数字无条件当作嵌套漏斗。

73 个 selected GT≥8 的机会仍只有 2 帧好 final，故不能全部归因于极端一两点稀疏。恢复前景点 vote 均误差约 1.427m，训练 e60 约 0.483m；67 个正 mode 的 seed→final 平均中心误差 1.387→1.397m，精修平均没有带来有效纠偏。训练/恢复分布不同，这个比较不是同样本泛化差的严格估计。

当前已经有 `RawLocalGeometry` 的最多 16 点局部交互、两 mode 顺序聚类、LWH 邻域局部读取；“加一个局部模块”不构成针对性新方案。更具体的风险是：

1. 当前点 encoder 吃 anchor-relative XYZ/LWH；memory 点却按保存框中心/yaw 规范化，二者共用 point encoder，身份比较参照不对称。
2. vote 的 raw 分量仍含大 anchor 偏移；local 几何并不自动消除 raw 的坐标敏感性。漂移几十米时可能偏离训练覆盖。
3. 几何 vote 还读当前 B0 特征，恰好在 B0 crop 丢目标时依赖可能无目标的上下文；这是待检验假设，不能仅据代码认定污染发生。
4. identity×reliability 决定 mode 权重。固定 IDs 的 vote 虽与 memory 无关，memory 仍可通过选点、加权和 support 间接改变几何结果。

**本次新增排除性证据**：从原点池重建 Full e60 的 4096 个 formed mode 局部读取，逐角 neighborhood/selected 计数与台账为 0 mismatches。上述 171 机会帧的 67 个正且合法 mode **全部读到真实前景点**，unique FG 中位 8、均值 15.22；没有因 top16 把前景全删光的例子。然而 33/67 的读入 FG 全部概率<0.5。该概率是局部读的输入特征，不是 hard mask，所以不能说 refiner 完全看不到它们。它说明应优先查内容/概率使用和监督有效性，而不是扩大邻域。seed yaw 误差中位 0.021rad、P90 0.840rad，也不支持把全部失败归为继承了错误 yaw。[复核结果](../artifacts/ct_checks/20261021_research_audit/b2/local_visibility_summary.json)

**优先候选**：恢复支路用点中心相对几何预测点到中心 offset，将 anchor 位移/年龄作为单独风险上下文；先固定 IDs/成员做坐标重参数敏感性检查。若身份参照问题得到支持，再考虑在候选形成后对成员规范化比较 identity。不能用 GT 居中作为部署输入，也不应一次改变坐标编码、候选形成次序、identity 权重和精修器而失去归因。

### 4.4 B2 身份：全局校准、帧内区分与记忆来源要分开

Full e60 恢复 selected256 中，2254 个真实前景点只有 94 个 identity≥0.5，约 4.17%；合并 AUC≈0.4485，而 169 个含正负点端点的帧内 AUC 中位≈0.6579。合并与帧内排序不同，不能只给一个 pooled AUC 就宣称所有帧完全反判。现有日志不含全 768 池 logits，不能把 selected 条件结果推广成完整池检索性能。

171 个机会中 110 个没有 recent memory；12 个有干净 recent memory 的机会表现较好，但它们 q0 距离也更小，记忆来源和难度混杂，不能宣称 recent memory 的因果提升。现有 36 槽容量已经包括首帧 12 FG 与最近两次各 8 FG+4 BG。

**较有依据的 PF-RPN/ChronoTrack 借鉴**：在已有 source/age/FG-BG metadata 和点级 memory attention 基础上，增加 initial-FG、recent-FG、recent-BG 的分源摘要/分路相似度，检验不同来源的竞争是否有害；保留 selected 预算和真实点 ID。当前 memory 并非被压成一个均值，不能把已有来源/年龄信息说成缺失。先在相同样本上比较来源置换/仅 initial/有无 recent 的敏感性。温度或单调校准只改变尺度，不能修复同一帧内错误排序；在完整池日志缺失时，不应优先重写 top-k 点池。

额外源码事实：`b0_summary` 对四帧所有 valid unique 点直接平均，没有区分前景/背景或 current/history；可能受背景密度变化影响，但 B0 特征含上下文，不能直接定为污染。可在固定 cache/memory 下比较 current-only、history-only 和软 FG 加权摘要对帧内 AUC、绝对 logit 的影响。当前数据路径 `point_arrays` 裁为 XYZ，B2 五维点输入的末两维为零，不能把它们解释成已利用反射率/时间特征。identity 与 reliability 都含 FG 监督，乘积也不是两个独立概率的联合置信。

### 4.5 训练覆盖：短窗口的联合状态覆盖仍待核查

Full e60 共 19,108 端点曝光：窗口 depth0 为 8208（42.96%），depth4–7 为 2130（11.15%），三历史均为预测框为 4812（25.18%）。q0 中心误差均值从 d0 约 0.330m（n=8208）到 d4 1.602m（n=581）、d7 2.496m（n=481）；不同深度样本难度不同，不能解释为递推深度的单因素因果效应。[本次审计](../artifacts/ct_checks/20261021_research_audit/performance/audit.json)

完整 60 轮 Full 几何正 mode 活跃曝光为 152,452/1,146,480=13.30%，vote 活跃 15.17%；不能称恢复支路完全没学。`frame_endpoint_mean` 仅对有监督端点平均，所以不能把 mode loss 再按 13.3% 的倒数放大。深分支 d4–7 的正 mode 活跃率 10.75%、raw 有目标但 crop 漏目标率 9.39%，支持继续检查恢复条件覆盖。[分深度/分阶段统计](../artifacts/ct_checks/20261021_v37_execution_decision_review/mechanisms/)

`ct_v31/data.py::_new_state` 的 initial memory 始终来自原轨迹 `first_frame`，其真实年龄会保留；**不能说短窗训练从未见过老 initial memory**。但 recent memory 为空、trusted/velocity 重建、previous_quality=1、supported 时间重置在窗口种子处，使“老 initial + 大 anchor 误差 + 长缺测 + 不可靠 recent/历史”的联合分布仍可能不足。

借鉴 DAgger 的关键是覆盖模型自己诱导的状态，而不是再把 GT 历史喂多一些。当前窗口内已经有 on-policy 递推，不能称完全 teacher forcing。先在训练集按 anchor 误差×点量×memory 来源/年龄×seed 质量分层；确有缺口后才考虑辅助恢复状态采样或连续一致扰动，保持 B1 主课程。额外 crop/forward 有成本，不能称免费；验证集失败帧不能回流训练。v36 整轨迹课程已有负结果，不直接整体恢复。

### 4.6 B3：先明确谁有权改变 accepted，再定义谁能写 memory

`ct_v37/model.py::select_candidates` 按 S/P 效用 argmax 选框；`ct_v31/data.py` 在 Q/支持门判断前已经写 accepted。因此 Q<0.5 不写 memory，并不意味着该框不会污染下一帧 crop/history。

Full e60 选 mode 29 次，其中 27 个纯背景且失跟；26 次 Q<0.5 仍已改变 accepted。须排除无效 fallback alias 后统计：末三轮有 **4/3/4 次合法 q0 已 IoU≥0.5，却选了 lost mode**；这些事件后可观察五帧全部仍 lost，是严重风险关联，但不能当作保留 q0 的反事实成绩。[修正后的事件表](../artifacts/ct_checks/20261021_v37_execution_decision_review/mechanisms/selected_mode_events.csv)

反面证据同样重要：6 个好 mode 都被明显低估，5/6 的 predicted-FG 支持不足 3 点；直接把现有 Q≥0.5、support≥3 搬到输出门，会同时拒绝大部分真实恢复机会。**可定位的稀疏输出**和**可写入强记忆/可信速度的测量**应有不同资格，不能用同一个门一刀切。

较优先设计是默认使用 B1 输出，让恢复候选同时证明自己的当前身份/几何证据与相对默认输出的替换收益。学习 `ΔU=U(candidate)-U(default)` 可作为训练目标，但应有 reject/无收益情形；不是把现有绝对 S/P 改名。先做固定权重、同事件前状态的少量政策分叉，滚完后续轨迹，量化“少错接管”与“漏掉恢复”的取舍。当前同状态 mode oracle 额外效用仅约 0.972 pp，不能承诺只换选择器便能追回全部 Full−B1 差距。

### 4.7 提交质量 Q：监督对象和实际提交内容不一致

当前 `Q_target=(0.5+0.5*IoU)/(1+d/half_diagonal)`，其中尺度为框水平对角线的一半，v37 调用时 purity=1。它不是 IoU≥0.5 的正确概率，也不是实际授权支持集合的纯度。q0 排序可看 B0+selected256，q0 提交只可看 B0，两种证据权限不同。

Full e60 的 798 次 memory 写入中，125 次存入 FG 纯度不达标；几何或纯度任一错误共 335 次。q0 写 795 次，其中 123 次前景不纯；mode 仅写 3 次、2 次不纯。故主要污染源是 q0，不能只针对 mode 加门。坏 q0 中还有 386 次同时 Q≥0.5 且 predicted-FG support≥3，说明当前门不等于真实可靠支持。

**优先提案**：提交头针对“候选自己实际有权提交的集合”学习几何正确性和支持身份可靠性；输出替换收益另学。训练 GT 可生成标签，不能成为推理特征或选择条件。先检查风险—覆盖关系、不同来源的可靠性和稀疏点条件，不能把 sigmoid 或新 BCE 头直接称已校准。GFL/QFL 可提供软质量监督与困难错误加权的实现参考，但 CT 已有连续质量/负候选监督，问题不在于缺一个叫 quality 的头。

### 4.8 B1：运动预测小误差不等于当前世界位置可靠

prior 被选中时 accepted=prior，innovation 恒 0，旧 transition 门通过；provenance 没有区分传播框和真实测量支持。Full e60 后续有 888 端点含传播边，249 个所有有效边均为传播。B1 fallback 也有同类逻辑，不能写成 Full 独有实现错误。

当前 prior sigma 主要监督 motion 残差，未显式监督或递推累计 anchor 误差。Full 支持年龄≥5s 的有效 prior 中，anchor 误差均约 8.124m，motion 误差 0.350m，sigma 0.505m。总位置误差的向量由 anchor 和 motion 两部分相加，不能把各自范数直接相加当恒等式。单步 sigma 小不能用作总位置置信度。

优先把 accepted 的来源、距最后可靠观测年龄、连续传播次数显式记录并用于独立风险判断；分别表达 motion 尺度与 anchor 风险。暂不屏蔽全部传播边、不把 CfC 整体替换成 v36 事件模型：这可能破坏已有效的缺测 prior。Recursive KalmanNet 可借鉴协方差预测/更新分工，不宜把其整套滤波器塞进已有均值/状态语义。

prior 在近似效用平局时还可能压过可更新的 q0；但已审计的 251 个正确强 q0 机会中，250 个 prior 也正确。选择 prior 是否因丢失观测更新而伤害未来，必须完整重放；不能从计数直接推出长期损伤，也不应给所有 prior 加固定惩罚。

### 4.9 次优先缺口：重复模式监督与效用地板

Full e60 训练正 mode 在最终去重前 2915、去重后 2676，少 239（8.20%）。但 9 个部署末轮文件未发现去重删掉正 mode 或好 final；其优先级低于身份/几何。可考察 formed/pre-dedup 正 mode 几何监督与 post-dedup 部署筛选分开，并按同目标归一权重；不能给背景 mode 强拉 GT，也不能把已去重 mode 的点级梯度说成完全消失。

234/2179 个 Full e60 预测端点的所有有效候选落在相同 S/P 地板，pairwise 真实差为 0；“rank 活跃端点”并不等于存在有效非零排序信号。可考虑只对恢复正例加连续距离/seed 改善辅助信号，但该项本身不能为全负 floor 候选建立完整排序，大误差且无身份证据的候选也不应被鼓励盲目移动。先改善候选供给，再决定该辅助项是否必要。

## 5. 文献与官方代码：按问题选择部件

以下是阅读论文/作者仓库后提出的迁移判断，不将其他任务/数据集的涨分移植成 CT 的预期涨分。未取得完整训练代码的工作明确标注；不使用二手博客代替技术证据。

| 工作与来源 | 可借鉴部分 | 对应问题与迁移限制 |
|---|---|---|
| [M2-Track，CVPR 2022](https://arxiv.org/abs/2203.01730)，[作者代码 M2TRACK](https://github.com/Ghostish/Open3DSOT/blob/main/models/m2track.py) | 实际代码有 motion_state_mlp，预测 motion mask 后对运动增量作门控；另有前框修正和第二阶段精修 | 与低运动漂移直接对应。已读摘要及 forward/loss 代码；本文不依赖其论文测速数值。CT 应预测合法在线运动/修正价值，不使用 GT 切片作门，不照搬阈值或抑制所有小位移 |
| [DAgger，AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)，[原文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | 训练覆盖学习器诱导的状态分布 | 支持核查递推分布差；CT 已有窗口内递推。只借状态覆盖原则，不照搬其迭代预算/模型挑选方式，不声称其理论保证直接适用于 CT |
| [Generalized Focal Loss，2020](https://arxiv.org/abs/2006.04388)，[MMDetection v2.28.2 QFL 实现](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/models/losses/gfocal_loss.py) | 负例质量为 0、正例用连续质量标签、误差相关加权 | 对应提交目标对齐和高分错误。已读方法与 QFL 实现；原文所链 implus/GFocal 本次访问 404，未冒称读到该仓代码。CT 已有软监督，需改目标语义而非只换 loss 名称；QFL 不自动提供概率校准 |
| [PF-RPN，2026 预印本](https://arxiv.org/abs/2603.17554)，[作者代码](https://github.com/tangqh03/PF-RPN) | 来源/尺度自适应摘要、少量原型条件化；多路候选互补选择的思路 | 更适合 B2 initial/recent/BG 身份摘要。其 2D 预训练 Grounding-DINO 前提、冻结部分和 900 queries 不适用 CT scratch-only；当前 173→171 的目标点保留说明先改采点未必最划算 |
| [LoopVL，2026 预印本](https://arxiv.org/abs/2609.38426)，[作者仓库](https://github.com/Tier-Flow/LoopVL) | 在反复更新中重新注入固定视觉锚点，可启发少步共享 mode 精修 | 只作为后续二步 refiner 思路；共享权重省参数不等于少计算，不能称推理加速。当前 seed→final 无明显改进，反复应用不可靠更新可能放大错误；公开推理代码不等于完整训练配方 |
| [ChronoTrack，2026 预印本](https://arxiv.org/html/2604.13789v1)，[作者代码固定版本](https://github.com/ujaejoon/ChronoTrack/tree/788a4a188e69ea363c61e973db061f651bed5f37) | 身份特征的时间一致性与 memory cycle consistency；近期 BG、长期 FG 分工 | 已读方法/消融及 task、memory_encoder、transformer；只加长期 token 的消融反而退分，说明应先学稳定身份而非加容量。训练 GT canonical 对齐只能用于监督；其保留 8 帧训练图与 CT 每步 detach 合同不同。论文 FPS 也不支持移植后加速 |
| [PTTR，CVPR 2022](https://arxiv.org/html/2112.02857v2)，[作者代码固定版本](https://github.com/Jasonkks/PTTR/tree/ade969329a825e1d9813d226983ff0f279f56d4c) | matching 中 Q/K 投影后 L2 归一化、双归一与 offset；先粗后细的局部证据使用 | 已读方法/消融和 multihead_attention/transformer；可先查 memory attention 投影后 Q/K norm 是否造成来源/年龄偏置，再试 cosine 变体，不能忽略最终 identity 已有 LN。CT 已有局部读取，不能把 PRM 当缺失模块；其 Waymo 排除目标点<10 的帧，不可直接推断极稀疏恢复效果 |
| [GeoTransformer，CVPR 2022](https://arxiv.org/html/2202.06688v2)，[作者代码固定版本](https://github.com/qinzheng93/GeoTransformer/tree/e7a135af4c318ff3b8d7f6c963df094d7e4ea540) | 分别在点云内用距离/角度表达结构、归一化匹配 | 已读正文、消融和 invariance 附录及 geometric transformer/matching 代码；启发身份内容的重锚鲁棒性，但 offset 仍需方向/等变信息。不能整套搬 KPConv、刚性配准/LGR 到 12 点首帧 memory；其对 RANSAC 的加速不代表 CT 提速 |
| [Recursive KalmanNet，EUSIPCO 2025](https://arxiv.org/html/2506.11639v1)，[作者代码](https://github.com/ixblue/RecursiveKalmanNet) | 保留状态协方差递推，更新使用 Joseph 形式与可学习半正定项 | 已核 `Algo/RecursiveKalmanNet.py::update_cov/step`。支持 anchor 风险记账，不支持把 motion sigma 当总不确定性；其简单测量噪声实验不覆盖身份错配。q0/prior 共享历史，不应当独立高斯随意融合 |
| [BEVTrack，IJCAI 2025](https://www.ijcai.org/proceedings/2025/0250.pdf)，[作者代码](https://github.com/xmm-prio/BEVTrack) | 分布自适应几何回归/RLE | 已核 simple_head、rle_loss、inference；推理只返回 coors，sigma 未用于选择/提交，不能为 CT 的 Q 或状态置信背书。B1 已有 beta-NLL，不是缺少任意概率损失；当前优先级低 |
| [DAM4SAM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.pdf)，[扩展原文](https://arxiv.org/html/2509.13864v1)，[作者代码](https://github.com/jovanavidenovic/DAM4SAM) | 近期与长期记忆分工；优先保留可靠且能区分干扰的样本 | 已核 track 中质量、面积稳定、干扰信息和间隔条件。适合 CT 的定额 memory replacement；先修 q0 错写，不能把低 Q 失败预测存作“负记忆”，不能照搬像素阈值 |
| [Non-Exchangeable Conformal Risk Control，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Conference.pdf)，[作者代码](https://github.com/deep-spin/non-exchangeable-crc) | 非交换条件下的风险评价与加权校准视角 | 已读理论条件与 split_CRC_LS 实现；CT 门改变后续状态分布，闭环风险未必随阈值单调，不能直接承诺 distribution-free 风险界。只借训练内 held-out 轨迹分组的风险—覆盖评价，不能随机拆帧制造独立性 |

**实际优先顺序**：低运动/错误更新参考 M2-Track 与质量目标对齐；身份稳定性参考 ChronoTrack、PF-RPN、PTTR；坐标鲁棒性参考 GeoTransformer；传播风险参考 Recursive KalmanNet；LoopVL 二步精修、RLE 和长期记忆替换后置。各工作只借最小相关部件，不组成一次全套更换。“前沿”本身不是优先级。

## 6. 提速：优先减少重复工作，不再用参数量推断运行成本

Full 约 3.956M 参数，FP32 参数约 15.09 MiB；这不包含 activation、梯度、Adam 和数据。静态 dense MAC 代理显示 observation 占约 80.5%，但没有计入排序、CPU 几何、D2H、序列化或数据等待，不能据此排除小模块的真实耗时。

本次新增审计确认：Full e60 `frames.jsonl` 约 230.170 MiB，2179 个预测行平均约 110,748 字节；抽样候选字典中 support_pool 和 extension 点级记录占大部分。候选记录的几何调用静态计数 21,198 次，按完全相同 box 签名可复用 6537 次（30.84%）；**这是调用去重比例，不是可承诺的提速比例**。

| 优先级 | 位置 | 最小优化方向 | 验证边界 |
|---|---|---|---|
| 高 | runtime.add_batch → candidate_records | 父 runtime 的逐字段 D2H 与后续 packed transfer 共享已物化结果；按完全相同输入复用框几何 | 保留所有原字段、分母、候选/提交权限与输出顺序，验证数值/日志一致；先 profiler 定位真实占比 |
| 高 | validation summary 与完整记录生命周期 | 检查只写 summary 时仍构建/保留完整候选字典的重复开销，做紧凑累积或流式结构化记录 | 必须保留正式要求的完整证据及诊断含义，不能关日志换“模型加速”；评测与纯模型耗时分别报告 |
| 中 | B2 attention | 当前 need_weights=True 后只取输出；评估不返回无用权重 | torch2.0.1 可能改变 SDPA/数值/确定性路径，不视为逐位无损；需配方与合同检查 |
| 中 | 局部邻域/selection | 复用已算距离/稳定排序/选点结果，避免重复 materialization | 不改变 raw ID 首有效槽、稳定同分规则、邻域上限或监督集合 |
| 证据不足 | 分解投影、BF16、batch4、更多循环 | 保留历史配方，不默认扩展 | 分解当前分数下降；BF16 末三轮未优于主模型；小 batch 更新数暴增；循环增加计算 |

真实速度报告应分 CPU 准备、H2D、模型 forward、状态提交、诊断 D2H/几何、写盘；用相同设备/负载/排程与日志，固定预热和交错重复，报告吞吐、中位/P90、allocated/reserved。静态 MAC 减少、并发墙钟、每步更快、整个训练更快是不同结论。当前没有本次新增 A40 测速结果。

## 7. 下一步如何收缩为可验证的研究

以下是建议顺序，不是自动追加训练计划。旧 output、配置身份和 checkpoint 源码保持不变。

1. **先完成已有证据的决策闭环。** 工作基线采用强 concat B1 的明确配方；batch4 保留未完成标记，不为分解再扫 batch/LR。报告首框保持及运动/低运动切片，保留原 R/C 门槛。
2. **低成本区分原因。** 用原登记源码与既有权重重放少量预先列出的事件：固定输入比较主路；固定权重分叉 mode 接管/近似平局政策；记录首框保持优势来自哪些失误。所有 rollout 必须看后续轨迹，不能把单帧 oracle 当正式结果。环境/数据不可用时只记录缺口，不虚构重放已完成。
3. **若只做一个下一版结构方向：强 B1 + 独立恢复支路 + 有证据的增益替换。** 先保护主路训练，再选一项恢复几何/身份修正；提交头按自己的支持集合学习。不要同一版同时换 B1 动力学、课程、采点、loss、门控、精度和 batch。
4. **训练覆盖针对恢复条件补齐。** 必须证明大 anchor 误差/仅 initial memory/长 unsupported 状态的缺口，训练数据内取样，保持相同正式端点与预算口径，明示额外计算。不能用当前验证轨迹调到恰好有效的阈值。
5. **按失败条件决定是否继续。** shadow 仍伤主路就查合同/梯度/状态，不堆层；好 mode 仍是个位数就先修供给；好候选多了却选不中再加强 selector；拒绝全部 mode 只算守底；低运动改善但运动启动显著退化不视为目标达成。
6. **任何正式结论仍用 scratch、固定 final60/late-3。** 后续多 seed/full 数据仅在研究方向明确后另行登记，不能把当前单 seed 描述性结果当泛化证明。

“涨分概率较高”只能作相对判断：**修正无证据接管与提交目标，比照搬整个 PF-RPN/LoopVL 更贴近已观察到的错误；恢复几何/身份和状态覆盖，比扩大点数更有依据；重复评测去重，比继续改变浮点计算顺序更有希望获得低风险工程收益。** 其中任何算法收益仍需受控闭环实验确认。

## 8. 本次交付验证

更新 README、AGENTS、need_to_do、最新问题、实验协议及三份 v37 设计/操作页，新增本研究审查；旧 artifacts 保持原样，本次计算另存独立目录。八个已完成运行的得分表直接对照原 `results.json`，局部读取与开销统计核对机器结果，新增文档本地链接和 `git diff --check` 通过。三项独立复核分别检查 B2、B1/B3、实验/效率口径，已修正来源信息已有、joint-state 仅为假设、峰值未 reset 等易误解表述。

复核脚本：[validate_docs.py](../artifacts/ct_checks/20261021_research_audit/validate_docs.py)；验证记录及输入 SHA256：[documentation_validation.json](../artifacts/ct_checks/20261021_research_audit/documentation_validation.json)。生产代码与配置没有改动，因此本次不重跑模型 pytest/compileall；也没有新的模型 forward、训练或 GPU profiling 结果。
