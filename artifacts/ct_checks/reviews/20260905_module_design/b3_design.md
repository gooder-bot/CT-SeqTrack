# B3 设计建议：先把动作收益学对，再提高可接受动作覆盖

审阅日期：2026-09-05。仅审阅，不修改模型、配置或输出；不参考 trajtrack。

## 结论

B3 建议保留为轻量的动作收益判断器，当前不值得扩大网络或换成复杂强化学习策略。优先级是：**统一动作和标签 → 修复 H3 样本截断 → 对齐评价指标与校准目标 → 冻结阈值后完整闭环验证**。只有 B2 能产生一定数量的有益候选后，B3 的涨分空间才存在。

主方案保持唯一递归状态写入者、B2 detach、严格 observation fallback 和原有残差上限。先验证现有单候选的判别能力；半步/整步动作选择作为第二阶段可选项。

## 已核对的代码逻辑

- `models/ct_v2/evidence_memory.py:740–995`：B3 将 B0/B1/B2 特征 detach，输入 evidence、presence、coarse/refined 一致性、prior/evidence 与 observation 位移、B1 sigma、dt/gap/age、B0 统计以及点/投票共识。64 维小 MLP 输出 helpful/harmful 和两个 expected-gain 头。
- 部署分数实际只有 `sigmoid(help)*(1-sigmoid(harm))`（:939）；expected center/IoU gain 仅作辅助训练，并未进入决策。该乘积是一个可以校准的排序分数，不能未经验证叫做“动作有益的概率”。
- 动作是将 raw 候选相对 B0 的平面残差裁剪到 `min(.5+.5*dt,2)` 米；通过 structure、presence、calibration 和 score 后替换 x/y；z/yaw 仍来自 observation（:941–961）。
- `models/seqtrack3d.py:5184–5266`：H1 helpful 要求 center gain > .05m 且 IoU gain 非负；H3 有标签时还需 H3 平均 center gain > .15m 且 IoU gain 非负。harmful 包含中心恶化、IoU 降低及 extension 没有目标。
- H3 在 :7985–8083 克隆同一个历史：一支执行 B0，一支执行有界修正，然后未来两帧均执行 observation；因此它估计的是**一次干预后继续 observation 的短期收益**，不是完整 selective 策略的长期价值。
- H3 标签还受 `ct_search_candidate_valid` 截断（:8005）。该变量在 evidence_memory.py:672–674 等于结构有效且 presence >= .5；初始 presence=.1，低分段难得到 H3 标签。后续校准可能接受低于 .5 的分段，存在监督覆盖错位。
- H3 平均收益本身已含 H1，训练再取 `.5*(H1+H3)`（:5230–5236），H=3 时等价于当前帧 2/3、后两帧各 1/6 权重。这不必然错误，但应明确设计目标，不能误写为等权三帧收益。
- `utils/action_calibration.py:170–245` 在风险约束下选 `mean_center_gain + mean_iou_gain` 最大的阈值；coverage 只作并列比较。这里混合米与 IoU，并且优化已接受动作的平均收益，可能错过更高的全帧净收益。

## 优先修改 1：训练、校准和部署必须评价同一个实际动作

H1 用真实有向 IoU 和与正式评估一致的中心误差计算；修复 wlh 轴序后，不再让近似 IoU 的符号决定 helpful/harmful。这些是 detach 标签，不需要可微几何。H1、H3、calibration 共用标签函数。

必须区分 `raw proposal`、`bounded proposal`、`accepted proposal`。风险模型标注与训练的是 bounded proposal；校准也必须评价它。增加一个 **bounded-always** 对照，否则 Full−B3 的 unbounded raw 与 Full 的 bounded+reject 同时变化，无法把收益归因于拒绝机制。

给风险特征显式加入 `bounded displacement / box diagonal`、`raw displacement / radius` 和裁剪比例。当前输入的是 raw disagreement，但执行的是 clipped action；dt 可让 MLP 间接推断 radius，显式动作特征更直接，代价只是几个输入维度。

## 优先修改 2：H3 按结构有效样本抽样，保留真实状态分布

移除 H3 监督生成中的“当前 learned presence >= .5”条件，改为结构有效、有限、有实际残差的候选，采用独立于当前预测分数的确定性轮转采样。维持既有 shadow 总预算，先只改变覆盖对象；记录 presence 分箱、候选有益/有害标签、dt、漂移和稀疏度各组的 H3 覆盖。

需要额外研究难例时，可按固定分层抽样并记录 inclusion probability，用权重恢复 canonical population 的训练期望；不能把人工空间恢复 candidate 当 B3 的正式在线分布。B3 存在稀少正例，不意味着可直接用无权重 1:1 采样后将 sigmoid 当自然分布概率。

明确写出短期效用，例如 `G_H = sum_k w_k [u(b^action_{t+k})-u(b^obs_{t+k})]`。选择固定 w 并注册；先比较 H1 与等权 H=3。保留 H3 valid mask，不能把“没有 H3 标签”当作 H3 无风险。当前完整门控条件在有/无 H3 两组不同，应报告各组 helpful base rate，避免学到采样选择差异。

成本：单个 H3 标签目前需要未来两帧×两支 = 4 个 observation forward。更合理的抽样可在不增加这一总量的情况下改善标签覆盖。

## 优先修改 3：用全帧净收益选择阈值，不用单次动作收益

`utils/metrics.py:78–122` 的 Success 是 IoU 阈值曲线的 AUC，Precision 是 0–2m 中心误差阈值曲线的 AUC。可以直接计算逐帧离散 AUC 标签；连续近似为：

`u(b) = IoU(b,g) + lambda * max(1 - error(b,g)/2m, 0)`。

这里两项均无量纲，lambda 必须先注册；如主攻 Success，则 Success 为主目标，Precision 作为不退化约束。不能在测试集上挑 lambda。H1/H3 labels 是 detach，可直接使用离散正式 metric，无需平滑近似。

保持原来的 harmful upper bound、最少 tracklet/action 数和最小覆盖约束。在风险可行阈值中选择：

`J(tau) = (1/N_all_frames) sum_i A_tau(i) * Delta_u(i)`。

也就是 `coverage × E[Delta_u | accepted]`，而不是只最大化后者。例：1% 帧每次收益 .4 的总体收益 .004，10% 帧每次收益 .1 的总体收益 .01；现在的校准选择可能偏向前者。该修改不需要放宽风险阈值，也不需要模型重跑，只需要正确且完整的动作 endpoint。

初始阶段保留 help/harm score 作为主版本，比较 expected-utility 排序版本。后者可将现有两个 gain 头改为与上述度量对齐的 gain（或一个标量效用头），仍由外部阈值校准约束风险；重点检验高收益候选排序是否改善。不要只因已经有 gain head 就将未经验证的线性组合直接上线。

## 优先修改 4：完整闭环验证，而非只重筛 observation 轨迹

当前 H3 适合当局部动作监督，保留其低成本用途。calibration/dev 必须在 checkpoint 固定后：calibration 选阈值 → 阈值冻结 → dev 完整 selective 递归跑每条轨迹。接受动作会改变下一帧 crop、历史框、B1、memory，不能用 observation 轨迹上的筛选结果代替实际部署增益。

分区至少按场景/完整轨迹隔离；同场景不同物体有相关性，区间估计优先以 scene 为独立重采样单位。重新反复设计模型或阈值会消耗 dev 的独立性，最终必须有 untouched test。

如果未来要做 on-policy B3 训练，可另开明确注册的新实验，而不是在当前 scratch formal 训练中途突然部署未校准动作。现阶段完整闭环测试比引入在线策略训练更优先。

## 第二阶段可选：半步/整步动作菜单

如果 oracle 分析发现“raw 方向有益但整步过冲”很常见，考虑 `alpha in {0,.5,1}`，其中非零动作都是同一个有界残差的倍数。共享小型 action-conditioned scorer，明确输入 alpha 和实际动作特征。B0 仍是零动作且唯一主干/状态写入者不变。

先用 H1 对半步、整步做离线 oracle，确认菜单有真实上界再训练。推理只多一次小 MLP，不增加 point encoder；H3 未来分支由 observation+1 action 变成 observation+2 actions，shadow forward 从 4 次升到 6 次，必须等总训练算力比较。

这比直接放大半径更适合轻中度漂移。对几十米丢失，现有局部 bounded correction 根本不是重捕获器，不能宣称解决长距离失锁。

## 文献依据与适配边界

1. [Learn then Test](https://arxiv.org/html/2110.01052v5)：模型训练后以有限样本有效检验和多重检验控制选择可行阈值；其正式设定使用 i.i.d. calibration 单元。项目当前 tracklet bootstrap + 网格选择不是 LTT 的完整理论实现。可借鉴风险可行集内追求有效覆盖，以及训练/校准分离；不能移植理论保证到相关视频帧。若真做统计保证，需要以独立场景/轨迹为 unit，并明确固定策略对应的轨迹级风险。
2. [SelectiveNet](https://arxiv.org/abs/1901.09192)：拒绝机制应通过 risk–coverage 曲线评估。对 CT 的适配是“接受动作的有害率/净收益—全帧覆盖率”曲线，并与简单置信门比较；原论文的端到端主干耦合不是当前必须复制的部分。
3. [PrDiMP / Probabilistic Regression for Visual Tracking](https://openaccess.thecvf.com/content_CVPR_2020/html/Danelljan_Probabilistic_Regression_for_Visual_Tracking_CVPR_2020_paper.html)：强调普通 tracking score 缺乏清晰概率解释。对 CT 的启发是区分 evidence presence、定位不确定性和“比 B0 更好”的动作收益；不能把存在目标或共识紧致当成修正有益。
4. [KeepTrack](https://openaccess.thecvf.com/content/ICCV2021/html/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.html)：同类干扰物可能外观十分相似，作者通过候选跨帧关联保留身份信息。对 CT 的合理借鉴是给 B2→B3 传入目标身份支持、竞争候选差异等证据；高 vote consistency 本身也可能是邻车的稳定共识。先改 B2 的身份辨识，B3 不宜另造大型多目标关联器。

这里提出的 CT 修改是根据源码与论文启发的研究假设，尚无本项目涨分证据；未声称任何模块保证提升。

## 最小消融和 B3 去留条件

在可用 B2 候选上，以同一个冻结 checkpoint 做诊断：observation、bounded-always、presence-only、现有 help/harm、效用排序五种决策。正式跨模块结论仍需匹配 scratch runs；同 checkpoint 决策对照只回答 selector 问题。

至少报告全帧 Success/Precision、全帧动作覆盖、有害率、净效用、track/scene bootstrap、额外时延；增加匹配覆盖下的 presence-only/simple-consensus 对照和 oracle selector 上界。分别比较 H1 与 H3，保持 shadow 预算相同。不要让 oracle GT 决策成为正式推理结果。

B3 值得作为论文模块保留的条件：B2 oracle 已有可见收益空间；实际 selector 在独立闭环上超过 bounded-always 和简单门；风险/覆盖曲线在相同覆盖下改善；增益不是靠覆盖为零、删掉失败 endpoint 或测试集调阈值取得。如果 B3 长期动作覆盖接近零，或不优于 presence+consensus，暂降为部署选项/附录，把主要开发投入 B1 获取和 B2 身份证据。
