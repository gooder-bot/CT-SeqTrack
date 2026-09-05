**论文定位草案｜证据尚未完成**

工作题名：*Observation-Anchored Evidence Recovery for Irregular-Time 3D Single-Object Tracking*。它描述研究对象，不暗示已有稳定收益。

**一句话研究假设。** 在 SeqTrack3D 的固定观测路径之外，使用真实时间间隔确定有界的额外测量区域，只从新增点产生修正证据，并在独立数据上学习何时接受修正，可以在控制误修正的同时改善时间变化与局部失跟场景。这是待检验假设；目前尚不能补写“实验表明有效”。

**问题与边界。** 输入是第一帧真值框、当前及历史点云、因果预测框和时间戳；输出为当前 3D 框。B0 本身包含历史点云、框序列和时间表示，“observation”表示其架构责任，绝不表示无历史或绝对可靠。当前 B2/B3 主要修正 XY，保留 B0 的高度与姿态；有界局部修正不能自动称为长距离重检测。

**技术难点。** 预测历史可能累计漂移；扩张区域可能新增的都是背景；极少点的几何一致性容易被误当身份可靠；接受一次修正会改变未来 crop、history 与风险分布。因此需要分开验证获取、识别、定位与动作四个环节。

**核心设计判断。** 值得保留的设计是新增测量证据的归因和使用条件。B1 的职责是提供获取先验；B0 raw crop 构成固定参照；B2 的 vote 只能来自严格集合差内的新点；B3 以观测输出为参照执行有界动作，失败时回退。v26 实际获取边界来自独立 quantile margin head，不能写成由统计 sigma 直接控制 crop。主后端是 GRU，CfC 暂为后端诊断。

| 候选贡献 | 相对已有工作的必要差异 | 必须补上的证据 | 当前强度 |
|---|---|---|---|
| 时间条件的有界新增测量获取 | 相同 B0、相同空间/点预算下，实际多获取了当前目标的点 | 修复 prepass；fixed/CV/learned；true/fixed/shuffled；间隔/稀疏度分层 | 假设，入口缺陷尚存 |
| 来源可归因的证据恢复 | memory/base 只提供上下文，新增点才可投票；无新增证据精确回退 | no-extension、equal-budget sampling、目标点保留、候选 gain、身份错误率 | 合同已实现，收益未确立 |
| 可拒绝的有界测量修正 | 对已有 observation 的实际增益而非单一置信度排序 | 独立校准、真实几何标签、覆盖/伤害曲线、冻结后的整轨迹结果 | 实现存在，尚未完成校准 |

模块不能各自仅凭名字构成论文贡献。真实时间、扩张、memory 和 attention 已分别出现在 SeqTrack3D、HVTrack、MBPTrack、StreamTrack 等工作中。HVTrack 的 Base/Expansion 特征感受野与本项目 raw-point 集合差有明确区别，需要在方法和等预算对照中展示。[SeqTrack3D](https://arxiv.org/html/2402.16249v1)、[HVTrack](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01145.pdf)、[MBPTrack](https://arxiv.org/html/2303.05071v1)、[StreamTrack](https://ojs.aaai.org/index.php/AAAI/article/download/28196/28389)。

**目前可以陈述。** 已建立模块化研究实现、显式数据/梯度/状态责任、确定性采样、来源诊断和校准失效回退；已记录混合/负结果；已有针对若干关键缺陷的可复现证据。

**目前必须谨慎。** 五臂数值有明显 baseline 分歧；只有一个 seed 和 mini Car；B1 prepass 无效但后续 B1 forward 能产生日志；Full−B3 部分训练且没有部署 raw 候选；Full 未校准。B0 已见过 mechanism-heldout 轨迹的 observation 数据，不能笼统写“整模型未见的校准集”。

**目前不能陈述。** 稳定涨分、SOTA、CfC 优于 GRU、真实时间或 memory 的因果收益、分布无关有限样本风险保证、已解决长期失跟。bootstrap 经验区间不等同 Learn then Test 的理论保证。[LTT](https://arxiv.org/html/2110.01052v5)。

**最可能的审稿意见。** 增益到底来自新增证据还是 baseline 随机轨迹差异；为什么不是单纯扩大 crop；相同预算下 learned motion 是否优于 CV；B3 只在 observation 轨迹上校准是否适用于改变递归状态后的闭环；为什么比较论文采用不同的 nuScenes 子集和训练域。这些应由实验回答，不通过强化措辞解决。

**写作顺序。** 先修复并获得可信的 matched B0 与 raw evidence 曲线，再写方法的可证实部分和实验结果；结果支持后再定主贡献数量及最终题名。若时间对照不成立，应删弱物理时间主张；若 raw 候选有收益而 B3 持续无覆盖，应将 B3 列为尚未验证的可选部分。科学范围应服从结果。
