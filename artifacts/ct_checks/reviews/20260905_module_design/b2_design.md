# B2 模块设计建议（2026-09-05）

本文件是设计审阅，不是已经实现或验证的涨分结论。阅读当前源码、已有审阅材料及 CXTrack、MBPTrack、StreamTrack、HVTrack 的原文方法与消融；没有参考 trajtrack。只新增本文件，没有修改正式配置、模型或 output。

## 结论

保留 B2 的 extension-only 职责。先完成 B1 获取入口、原始点身份、FPS、memory 轴序等正确性修复，再做一个有控制的主方向：**让新增点先获得局部几何，再显式比较目标和上下文证据，最后形成带绝对置信度的定位候选**。先不要增加整个 PointNet++/Transformer backbone、memory 长度或点数预算。

这一方向改善的是“新测量是否属于当前目标、它们支持哪里”，其研究价值需要与“只扩大搜索/只增加点数/直接复制 attention”区分。已有诊断提示获取召回远比 selection retention 更差，故 B2 结构改造必须排在获取接口修复后，不能替代 B1 修复。

## 当前真实数据与梯度通路

1. `utils/ct_search.py:1275` 的 endpoint/tube/corridor novel union 用确定性的 voxel round-robin 得到 768 prepool。local/corridor 名义预算 512/256，余量借用；不是模型在完整原始场景上直接学习采样。
2. `models/seqtrack3d.py:2720` 读取 B0 对齐点特征：current 1024×64；历史 3 帧各选 8 个预测框内 token、4 个框外近邻 token，共 36。历史 token 附局部 xyz、物理 age、相对 yaw、框内/上下文 role 和帧身份。
3. `models/ct_v2/evidence_memory.py:548` 对每个扩展点做 5 维逐点 MLP，叠加 B1 归一化纵横位置、dt/gap/valid 和 source embedding。**这里没有扩展点之间的局部几何聚合**。
4. `evidence_memory.py:568` 的 relation 用 base mean/max、memory mean/max 压成同一个全局 context，再对每个 prepool 点预测 logit。relation 在 point-to-memory cross-attention 之前，因此采样阶段没有访问逐 token 的目标匹配。
5. relation top128 + XY FPS96 + stateless exploration32 得到 256。随后 extension-only query 对 detached current base + detached memory 做一次 MHA；残差 gate 初始为 0。
6. targetness 和 point vote 由 enriched extension features 预测。坐标只能来自 `extension_xy + 4*tanh(offset)`；没有 extension 时精确回到 observation。z/yaw 沿用 observation。
7. votes 的权重是 `sigmoid(targetness)*sigmoid(relation)`，**不是在所有点上直接 softmax**。K=3 的 Huber consensus 按相对质量、inlier 比例和 covariance 排序。它能选择紧密模式，但紧密模式不等于正确身份。
8. B0 feature、历史 box/token、B1 center/sigma 均 detach；B2 的输出进入 B3 时再次 detach。B2 loss 训练 B2 参数，不回写 B0/B1，不负责写递归状态。

## 需要区分的几个事实

### 标签已经是目标实例，不宜说“只会分前景背景”

`seqtrack3d.py:5038–5185` 的 relation 和 targetness 标签来自当前被跟踪 GT 框内点；其他车在框外也为负。这是目标实例二分类的监督，不是“所有车辆皆正”的语义分割。问题是**表征和训练负例是否足以学出这个判别**：单点 XYZ、B1 位置和全局 pooled context，可能主要学到位置与物体性；原文 CXTrack 同样指出，几何相似注意力会关注相似物，需要运动和目标线索区别干扰物。

### presence 与 voting 的集合现在不同

`seqtrack3d.py:5165` 把 768 prepool 含目标定义为 extension presence 正例；`5167` 的 raw/vote loss 只在 256 selected 中有目标时成立。但 presence head 实际读的是 selected enriched mean/max。若 selection 丢掉全部目标，presence 的输入与标签便不完全匹配。

最低成本的修正是：把 B3 读取的 presence 定义为 `selected_has_target`，用于回答“当前参与投票的新测量里是否有目标”；prepool presence 只保留为获取诊断，若另设预测 head 必须读 prepool 特征。没有必要为记录一个统计量再增加一个 head。

还应记录 `selected_has_target` 下 raw candidate 的误差，而不要把 presence 定义成“比 B0 好”；是否改善 B0 是 B3 的职责。

### 绝对置信度被 consensus 归一化部分丢失

读取 `_consensus_vote` 并 CPU 调用原函数：四个紧密 votes 位于 `(3,0),(3.1,0),(3,0.1),(3.1,0.1)`。

| 每点 relation/targetness | center | consistency | effective mass |
|---|---|---|---|
| 都是 0.9 | (3.05,0.05) | 0.995012 | 3.24 |
| 都是 0.01 | (3.05,0.05) | 0.995012 | 0.0004 |

所以当前 consistency 是几何共识统计量，不是候选可靠概率。`ct_vote_effective_mass` 已输出，但 B3 consensus 输入没有这一项。B3 另有 presence、targetness mean/max，不能断言它完全看不到置信度；更精确的建议是把 **top-mode 的有效质量、均值 targetness、绝对支持点数、target-vs-context margin** 连同全局量输入 evidence presence/B3，并保持 B3 输入 detach。避免临时加一个未校准硬阈值后宣称解决。

## 主方向：小型局部几何与目标条件证据

### 最小实现

在 768 prepool 的 `extension_encoder` 后增加一个轻量局部聚合块；先试固定 k=8 或 16、64d、1 层。邻域仅从合法 extension 点构建，mask padding，保留原始点索引。输入相对坐标与特征差，例如：

`g_i = h_i + max_{j in N(i)} MLP([h_i, h_j-h_i, p_j-p_i])`。

这是一个起始工程候选，k 和半径并非文献保证的最优值。若担心 768 局部图成本，先测 profiler；在 256 选点后加块能改善 voting，但不能改善已经作出的 relation 选点，二者应明确区分。

relation context 保留现有 base mean/max，另外将历史 **预测框内 role** 与 **框外 context role** 分别聚合成 `m_target,m_context`，不要先混成一个 mean/max。对扩展局部 feature 显式输入 `sim(g_i,m_target)`、`sim(g_i,m_context)` 及它们的 margin；B1 几何保留为单独输入，不用 motion likelihood 硬乘死所有偏离 prior 的点。框内 token 是预测身份线索，不是 GT 身份证明。

relation 仍用独立 sigmoid，采样仍保留当前 128/96/32 探索保底。selected point-to-memory attention 保留；输出仅从 extension votes 来。这样首先改变最薄弱的表征与条件信息，不急于变成一套全新的 detector。

### loss 与 coupling

- relation：prepool target-instance balanced BCE，保持已有负例；targetness：selected balanced BCE。暂不同时更换 focal loss、采样比例、网络深度。
- vote：仍仅对 selected target 点做 SmoothL1；raw：仍仅对 target-bearing selected rows 做 SmoothL1。空/纯背景输入不强行获得 GT 中心梯度。
- presence：selected target-bearing BCE；原始 prepool presence 作为分层统计。
- 可选 identity ranking：只在有合法目标点和困难负点的行增加 `softplus(margin - s_pos + s_neg)`，其中 s 是目标-vs-context 身份分数；先做单独消融，不能同时引入后把涨分全归给局部块。
- 所有 B0/B1 输入在边界 detach；memory 选择用预测历史；B2 adapter/encoder/head 训练。GT 当前目标标签只参与 loss 和离线诊断，不能进入 forward 的 identity prototype。

### 最少但可解释的对照

修复后的原始 B2 → 仅局部几何 → 仅分离 target/context 条件 → 两者同时。固定 B1 backend、support、768/256预算、seed、B0 trajectory；模块仍按正式 protocol 从 epoch0 联训。机制比较可额外在同一冻结轨迹/点池上做配对评估，但该只读诊断不替代闭环总分。

报告：relation top-k enrichment、selected target retention、纯背景 presence FPR、target-bearing 时 vote/center error、raw harm、同类邻居/稀疏/大 dt 子组、整体 Success/Precision 和新增延迟。若目标点没被取到，这些表示能力改动没有恢复依据，应先解决 acquisition。

## 条件方向一：同类困难负例训练

**触发条件**：修复获取后，目标确实在 selected 中，但 relation/targetness 或 consensus 频繁被邻车/路边相似物吸引。

先使用机制流真实 scene 中的非目标同类实例作困难负例，依赖训练集 annotations 生成标签，输入只保留点、预测历史与合法时间。记录非目标实例 ID 仅用于 loss/诊断；禁止把 GT identity 泄漏给推理。先比较困难负例采样/BCE，再决定是否需要 contrastive。

若真实负例太少，可参考 StreamTrack 的同类负轨迹插入，但这是**新的 mechanism augmentation 协议**：要让连续帧的负轨迹位置、速度、遮挡一致，重算 novel 点身份和 source，独立 RNG，禁止污染 B0 observation 流和校准/dev/test；不能直接把 StreamTrack 的额外 GT query/EMA decoder 全搬进现有插件。新协议需预先登记并 scratch 训练。

收益假设是更低的同类混淆和 raw harm；仅有 mini 少量数据上的整体涨分不能证明 instance discrimination。

## 条件方向二：memory 的可靠性与时间对齐

**触发条件**：同一合法点池下 `memory=real` 相比 `empty` 无效或更差，且错框后的 memory inside/context role 明显污染；或长 gap/转向时 memory 影响恶化。

先在当前 36 slots 和三帧范围内做改造：保持角色分支，给 token 增加 detached、推理可得的预测框支持度/质量及 age 信息，采用软权重，不把低质量 history 全部硬清空。现有 age/yaw metadata 已有，不应把“第一次加入时序 memory”写成创新。

若错误确来自连续几帧都漂移，可以研究固定首帧 identity anchor + 最近可信历史。但当前 memory 在 forward 里由最近三帧重建，固定首帧缓存会改变在线状态/slot预算/训练输入，是**新研究协议**，需由 B0 host 拥有缓存写入并严格因果，不能让 B2 私自成为递归 state writer。先在相同总 slots 下比较，避免多给内存造成归因混淆。

MBPTrack 的启发是“几何与目标线索分路传播”，不是 memory 越长越好。HVTrack 的相对观测姿态与多尺度上下文是近邻设计；若借鉴观测角，必须区分本项目已有的 box yaw 差和 LiDAR 视角相对变化，它们不等价。

## 文献依据与创新边界

- [CXTrack 原文](https://arxiv.org/html/2211.08542v1)：目标提示逐层传播，局部 transformer 聚合，center embedding 辅助抑制同类干扰；其消融和失败分析表明纯几何注意力仍会关注相似物。对本项目的启发是让 identity 条件明确，并使用局部结构；直接复制整个 X-RPN 不构成新的贡献。
- [MBPTrack 原文](https://arxiv.org/html/2303.05071v1)：把 geometric feature 与 targetness mask feature 分开传播并共享 attention maps；不同 memory 长度的消融不是单调变好。当前三帧/36 token 本身不是待无限扩大的瓶颈。
- [StreamTrack 原文](https://arxiv.org/html/2303.07605v2)：跨帧 global/local hybrid attention；训练时插入同类负轨迹，并使用 query 级辅助 InfoNCE。这里建议的是按当前 extension-only 架构适配困难负例监督，不能把该文完整结构的结果当成本项目会涨分的证据。
- [HVTrack 原文](https://arxiv.org/html/2408.02049v1)：BEA 是 feature receptive field 的 base/expansion 两个尺度；CPA 按相关性聚合/压缩低重要性上下文。它与本项目新增原始测量的 extension 含义不同。可借鉴局部上下文判别，必须正面比较扩展搜索和记忆近邻方法。

可主张的候选论文逻辑是：**在固定额外测量预算下，让时间先验决定搜索支持、让身份条件解释新增点、让校准行动控制这些证据何时修正 observation**。这是待实验验证的联合设计方向，不是已经成立的创新或涨分结论。
