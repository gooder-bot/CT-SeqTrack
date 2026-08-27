# Systematic Literature Review: 3D LiDAR SOT 的搜索覆盖、递归漂移与点保留

**Date**: 2026-08-27  
**Papers surveyed**: 15  
**Scope**: 3D LiDAR single-object tracking 中的搜索中心、搜索范围、递归历史、稀疏点采样与投票；以 2019–2024 年代表性工作为主  
**Citation format**: APA 7th edition

## Executive Summary

相关文献形成了一个很稳定的因果顺序：搜索几何先决定目标点是否进入输入区域，点级关系建模、采样和投票只能改善区域内已有目标点的保留与定位。P2B、BAT、M²-Track、DMT 等方法大多沿用“以上一帧预测框为中心、向各方向扩大约 2 m”的局部搜索协议；它们的下游模块都不能恢复裁剪阶段已经遗漏的目标。PTTR 与 SyncTrack 证明 relation-aware/attention-guided sampling 可以显著改善稀疏目标点的保留，但论文同时把错误递归模板和目标不在搜索区列为边界。SeqTrack3D、M3SOT、MBPTrack 与 StreamTrack 说明有限历史信息能缓解稀疏和遮挡，却也共同显示：错误历史会污染后续预测，更多历史并不自动解决长期漂移。

对 CT-SeqTrack 的直接结论是：当前 B2 应先修复 support 的中心与覆盖，再讨论 relation-aware sampling。若目标已经落到递归锚点约 14–15 m 之外，围绕同一错误中心把 2 m/1 m margin 略微放大，或加入相近的 learned/CV 双中心，通常不足以构成重捕获机制。

## Methodology

本综述于 2026-08-27 尝试使用 arXiv API 查询 `3D object tracking`、类别 `cs.CV`、按 relevance 取前 20 篇；API 返回 HTTP 429，因此未重复请求。随后从 CVF Open Access、ECVA、AAAI 与 arXiv 论文页面选取 15 篇与本项目最相关的代表性工作。三组语言模型代理分别抽取研究问题、方法、搜索区域、采样/保留策略、主要发现和局限，主代理结合 CT-SeqTrack 的本地代码与 acquisition funnel 统一综合。

**Limitations of this review**: 这不是 PRISMA 风格的穷尽式数据库综述；筛选强调与当前 B2 决策的相关性。部分论文没有在正文中完整披露搜索 crop 的所有实现参数，因此个别实现细节还参考了论文公开代码或作者公开配置。论文报告的总体跟踪增益不能直接外推到 CT-SeqTrack 的 extension-only contract。

## Themes

### Theme 1: 搜索覆盖是采样、注意力和投票的上游硬边界

SC3D 把候选生成和相似度判断显式分开，并在当前 GT 附近做近似穷举，这展示的是“几何已覆盖”条件下的判别上限，而不是真实递归重捕获能力。P2B、BAT 与 MLVSNet 将搜索改为以上一帧预测为中心的局部 crop，再在区域内部匹配、投票或多层保留。P2B 的搜索中心消融尤其说明，使用递归预测中心相较 GT 中心会产生远大于多数下游模块的性能差距。

因此，`support_target=0` 与 `pool_target=0` 时，不应把失败归因于采样器或投票头。下游网络没有可恢复的正点。

### Theme 2: Relation-aware sampling 解决的是“池中有点、采样丢点”

PTTR 的 Relation-Aware Sampling 用模板—搜索关系保留更相关的搜索点，同时保留一半随机采样以降低错误模板的自强化。SyncTrack 的 APST 也利用模板到搜索 token 的注意力响应逐层保留搜索点。两者都证明稀疏小目标对采样策略敏感，但它们都在给定搜索区域内部工作。

这与 CT-SeqTrack 的漏斗门槛完全对应：只有当 `pool_target_count>0` 且 `sampled_target_count=0` 出现到足以影响总体召回时，relation-aware hybrid sampling 才是当前瓶颈的正确修复。

### Theme 3: 运动先验改善局部中心，但不能替代当前帧证据

M²-Track 与 DMT 都把运动作为粗中心或相对位姿先验，再依靠当前帧点进行分割、投票或精修。DMT 的实验表明单独运动预测明显弱于加入当前点云证据后的完整模型，也显示局部精修器只对有限中心扰动有效。它们仍使用上一预测附近的固定局部 crop，因此在递归锚点已经远离目标时同样会失效。

对 B2 而言，B1 应负责把 acquisition support 放到“目标可能出现”的位置，B2 evidence 再负责纠正；但 B1 learned 与 CV 若共享同一漂移锚点，两者彼此接近并不代表接近真实目标。

### Theme 4: 历史信息需要可靠性约束，数量增加并不等于重捕获

SeqTrack3D、M3SOT、MBPTrack 与 StreamTrack 都报告了有限多帧信息的收益，同时都观察到历史过长、历史预测不准或模板质量下降会导致性能饱和或退化。MBPTrack 的记忆与保守复用策略提示可以保存“最后可靠状态”，但其输入 support 仍必须含有可关联证据。

因此，若 bounded local shell 无法覆盖 CT-SeqTrack 的长尾漂移，更合理的下一步是少量、可诊断、因果的备用锚点或历史 corridor，而不是无条件堆叠更多历史帧或把整帧点云直接交给复杂网络。

### Theme 5: 上下文有用，但大范围背景必须受控

CXTrack 强调目标周围上下文会被传统裁剪忽略；MBPTrack 通过记忆与 box prior 传播目标线索；CorpNet、MLVSNet 和 STNet 在网络内部保留多尺度或关系上下文。这些工作支持 CT-SeqTrack 的“stable base + bounded shell”设计：保留 B0 的稳定观察核心，只增加有上限的外部证据区。然而它们不支持无约束扩大到十余米后依赖 attention 自动排除所有背景。

## Convergences and Disagreements

**Convergences**: 所有调查方法都把搜索覆盖视为下游定位的前提；关系采样、目标性评分、多层投票和注意力只能作用于已进入输入的点。多帧与运动先验普遍有帮助，但错误递归状态会传播。局部 search offset 约 2 m 是常见工程设置，却不是长期重捕获保证。

**Disagreements**: 方法在“应更依赖外观还是运动”“应保留几帧历史”“应随机、FPS 还是关系采样”上存在差异；这些差异主要发生在局部搜索仍有效的条件下。没有一篇调查论文证明，在目标完全不在原始 support 时，仅改变点采样或投票就能恢复目标。

## Gaps and Open Questions

- 现有 3D SOT 很少把完整 acquisition funnel 分解为全帧可观测性、原始 crop 覆盖、extension-only 去重和固定预算采样。
- 多数方法采用固定局部 crop，但较少单独报告递归中心误差分布与 crop recall 的关系。
- “最后可靠锚点”或有限重捕获 support 如何在不引入大背景和错误更新的前提下工作，仍缺少统一协议。
- 关系采样对错误模板可能产生自强化；混合随机/空间覆盖是常见稳健手段，但没有替代上游几何覆盖。
- mini 数据集上的少数长 tracklet 可能主导长尾漂移，需要 tracklet-level 报告和更大数据验证。

## Per-Paper Annotations

### Giancola et al. (2019)

**Research question**: 形状补全能否正则化稀疏点云的 Siamese 表示？  
**Methodology**: 对模型形状与候选点云编码、以余弦相似度选择候选，并联合形状补全损失。  
**Key findings**:
- 搜索策略与候选判别应分开评估。
- GT 邻域穷举给出判别上限，但不代表真实递归性能。
- 错误在线模板更新会累积漂移。  
**Limitations**: 依赖大量候选和 GT 中心附近的评估搜索，不具备真实区域外重捕获。

### Qi et al. (2020)

**Research question**: 能否以点到中心投票替代大量 3D 候选搜索？  
**Methodology**: 模板匹配、seed targetness、Hough voting 与候选验证。  
**Key findings**:
- 前一预测中心相较 GT 中心产生巨大性能差距。
- 投票能改善区域内定位。
- 模板过稀或递归进入杂乱背景后难以恢复。  
**Limitations**: 固定 2 m 局部搜索会遗漏快速运动或漂移目标。

### Wang et al. (2021)

**Research question**: 多层投票能否减少 PointNet++ 下采样造成的信息丢失？  
**Methodology**: 目标引导注意力、多层 Hough voting 与投票簇交互。  
**Key findings**:
- 多层种子比只用最深层更好。
- 目标引导注意力能压制区域内背景。
- 极端稀疏仍是失败边界。  
**Limitations**: 不能恢复原始搜索 crop 外的目标。

### Zheng et al. (2021)

**Research question**: BoxCloud 结构先验能否增强稀疏外观匹配？  
**Methodology**: 点到框中心/角点距离编码与 box-aware feature fusion。  
**Key findings**:
- 结构信息改善区域内目标区分。
- 首帧目标点少时比纯外观更稳。
- 仍继承 P2B 的固定局部搜索。  
**Limitations**: Box-aware 匹配不能补回裁剪前遗漏的点。

### Hui et al. (2022)

**Research question**: 粗到细 Transformer 关系能否改善模板—搜索关联？  
**Methodology**: 多尺度 Point Transformer、cross-attention 与局部特征空间注意力。  
**Key findings**:
- 区域内关系建模有助于区分相似干扰物。
- 粗到细迭代改善定位。
- 仍采用固定点预算与局部搜索。  
**Limitations**: 无独立长时重捕获，目标不在 crop 时失效。

### Xia et al. (2022)

**Research question**: 能否用历史运动粗中心加轻量点投票替代 3D 检测器？  
**Methodology**: 历史中心预测、粗中心条件化与显式投票。  
**Key findings**:
- 运动中心必须由当前点云证据纠正。
- 局部精修器对中心扰动有有限捕获半径。
- 多种简单运动模型差异小于证据校正的收益。  
**Limitations**: 原始点仍来自上一预测附近的固定 2 m 区域。

### Zheng et al. (2022)

**Research question**: 显式相邻帧运动能否优于纯 Siamese 外观匹配？  
**Methodology**: 目标分割、4DoF 相对运动、上一框修正与运动辅助形状补全。  
**Key findings**:
- 上一框修正能抑制递归误差。
- 运动粗定位与点云精修互补。
- 固定局部裁剪仍是硬边界。  
**Limitations**: 大幅漂移时局部区域内没有目标点，两个阶段都无法工作。

### Zhou et al. (2022)

**Research question**: Relation-Aware Sampling 与 Transformer 能否保留稀疏目标点？  
**Methodology**: 模板关系距离采样、PRT 与粗到细定位。  
**Key findings**:
- RAS 明显改善小目标点保留。
- 混合 50% relation 与 50% random 更稳健。
- 错误模板会误导关系采样。  
**Limitations**: RAS 只能重排搜索区内部点。

### Ma et al. (2023)

**Research question**: 单分支 Transformer 能否同步特征提取、匹配与注意力采样？  
**Methodology**: 模板/搜索 token 联合编码与 APST。  
**Key findings**:
- 仅对搜索区做 attention sampling 最有效。
- APST 优于随机/FPS 保留目标相关点。
- 模板仍需稳定，否则注意力会失真。  
**Limitations**: 无法恢复裁剪阶段不存在的目标点。

### Wang et al. (2023)

**Research question**: 如何减少网络逐层下采样造成的稀疏恶化？  
**Methodology**: 相关金字塔、多层注意力与 XY/z 分解定位。  
**Key findings**:
- 多层特征保留优于单层。
- 输入稀疏与网络下采样稀疏应分开。
- 测试仍依赖上一预测中心。  
**Limitations**: 固定 2 m crop 仍受递归漂移限制。

### Xu et al. (2023a)

**Research question**: 目标周围上下文能否改善遮挡、稀疏与同类干扰？  
**Methodology**: target-centric transformer、级联 targetness 与中心嵌入。  
**Key findings**:
- 保留局部上下文有益。
- 中心嵌入帮助排除相似干扰物。
- 上下文作用于已有搜索区。  
**Limitations**: 固定递归 search area 无法覆盖大幅漂移。

### Xu et al. (2023b)

**Research question**: 有限历史记忆与首帧 box prior 能否改善遮挡恢复？  
**Methodology**: 外部记忆、几何/mask 解耦传播与 box-prior localization。  
**Key findings**:
- 三帧记忆优于单帧，继续增加会退化。
- 保守复用上一预测可避免低证据时的激进更新。
- 短期可靠历史有助于目标重新出现后的恢复。  
**Limitations**: 错误历史会污染记忆，仍无独立全局重捕获保证。

### Liu et al. (2024)

**Research question**: 多帧、多感受野和多级监督能否共同缓解稀疏与遮挡？  
**Methodology**: many-to-one 历史注意力、多尺度 range sampling 和中间监督。  
**Key findings**:
- 少量历史模板最有效。
- 更多历史会带来冗余与错误污染。
- 多感受野能利用目标邻域背景。  
**Limitations**: 固定 support 外的长距离重捕获未处理。

### Lin et al. (2024)

**Research question**: 历史点云与框序列能否共同学习长期运动模式？  
**Methodology**: sequence-to-sequence 点云/box token 编码与序列监督。  
**Key findings**:
- 历史框位置先验改善稀疏场景。
- 训练随机偏移需要覆盖真实递归误差。
- 长序列会因累计框误差和分布不匹配而退化。  
**Limitations**: 历史先验不能凭空生成当前帧回波，也不能保证大漂移下的 crop recall。

### Luo et al. (2024)

**Research question**: 如何连续利用多帧运动与历史特征而保持效率？  
**Methodology**: 在线 memory bank、全局/局部混合注意力、query 与干扰物对比增强。  
**Key findings**:
- 少量历史帧提升明显，更多帧收益饱和。
- 逐点监督和对比增强改善区内判别。
- 快速目标可能越出小型预定义 search range。  
**Limitations**: 没有针对大幅递归漂移的显式全局重捕获。

## References

Giancola, S., Zarzar, J., & Ghanem, B. (2019). Leveraging shape completion for 3D Siamese tracking. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://openaccess.thecvf.com/content_CVPR_2019/html/Giancola_Leveraging_Shape_Completion_for_3D_Siamese_Tracking_CVPR_2019_paper.html

Hui, L., Wang, L., Tang, L., Lan, K., Xie, J., & Yang, J. (2022). 3D Siamese transformer network for single object tracking on point clouds. *European Conference on Computer Vision*. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620284.pdf

Lin, Y., Li, Z., Cui, Y., & Fang, Z. (2024). SeqTrack3D: Exploring sequence information for robust 3D point cloud tracking. *IEEE International Conference on Robotics and Automation*. https://arxiv.org/abs/2402.16249

Liu, J., Wu, Y., Gong, M., Miao, Q., Ma, W., Xu, C., & Qin, C. (2024). M3SOT: Multi-frame, multi-field, multi-space 3D single object tracking. *Proceedings of the AAAI Conference on Artificial Intelligence*. https://ojs.aaai.org/index.php/AAAI/article/view/28152

Luo, Z., Zhang, G., Zhou, C., Wu, Z., Tao, Q., Lu, L., & Lu, S. (2024). Modeling continuous motion for 3D point cloud object tracking. *Proceedings of the AAAI Conference on Artificial Intelligence*. https://ojs.aaai.org/index.php/AAAI/article/view/28196

Ma, T., Wang, M., Xiao, J., Wu, H., & Liu, Y. (2023). Synchronize feature extracting and matching: A single branch framework for 3D object tracking. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Synchronize_Feature_Extracting_and_Matching_A_Single_Branch_Framework_for_ICCV_2023_paper.html

Qi, H., Feng, C., Cao, Z., Zhao, F., & Xiao, Y. (2020). P2B: Point-to-box network for 3D object tracking in point clouds. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_P2B_Point-to-Box_Network_for_3D_Object_Tracking_in_Point_Clouds_CVPR_2020_paper.html

Wang, M., Ma, T., Zuo, X., Lv, J., & Liu, Y. (2023). Correlation pyramid network for 3D single object tracking. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*. https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/html/Wang_Correlation_Pyramid_Network_for_3D_Single_Object_Tracking_CVPRW_2023_paper.html

Wang, Z., Xie, Q., Lai, Y.-K., Wu, J., Long, K., & Wang, J. (2021). MLVSNet: Multi-level voting Siamese network for 3D visual tracking. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://openaccess.thecvf.com/content/ICCV2021/html/Wang_MLVSNet_Multi-Level_Voting_Siamese_Network_for_3D_Visual_Tracking_ICCV_2021_paper.html

Xia, Y., Wu, Q., Li, W., Chan, A. B., & Stilla, U. (2022). A lightweight and detector-free 3D single object tracker on point clouds. *arXiv*. https://arxiv.org/abs/2203.04232

Xu, T.-X., Guo, Y.-C., Lai, Y.-K., & Zhang, S.-H. (2023a). CXTrack: Improving 3D point cloud tracking with contextual information. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://openaccess.thecvf.com/content/CVPR2023/html/Xu_CXTrack_Improving_3D_Point_Cloud_Tracking_With_Contextual_Information_CVPR_2023_paper.html

Xu, T.-X., Guo, Y.-C., Lai, Y.-K., & Zhang, S.-H. (2023b). MBPTrack: Improving 3D point cloud tracking with memory networks and box priors. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://openaccess.thecvf.com/content/ICCV2023/html/Xu_MBPTrack_Improving_3D_Point_Cloud_Tracking_with_Memory_Networks_and_ICCV_2023_paper.html

Zheng, C., Yan, X., Gao, J., Zhao, W., Zhang, W., Li, Z., & Cui, S. (2021). Box-aware feature enhancement for single object tracking on point clouds. *Proceedings of the IEEE/CVF International Conference on Computer Vision*. https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Box-Aware_Feature_Enhancement_for_Single_Object_Tracking_on_Point_Clouds_ICCV_2021_paper.html

Zheng, C., Yan, X., Zhang, H., Wang, B., Cheng, S., Cui, S., & Li, Z. (2022). Beyond 3D Siamese tracking: A motion-centric paradigm for 3D single object tracking in point clouds. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.html

Zhou, C., Luo, Z., Luo, Y., Liu, T., Pan, L., Cai, Z., Zhao, H., & Lu, S. (2022). PTTR: Relational 3D point cloud object tracking with transformer. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_PTTR_Relational_3D_Point_Cloud_Object_Tracking_With_Transformer_CVPR_2022_paper.html
