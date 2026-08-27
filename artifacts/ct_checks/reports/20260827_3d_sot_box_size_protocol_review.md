# Systematic Literature Review: 3D SOT 中的目标尺寸与在线评估边界

**Date**: 2026-08-27  
**Papers surveyed**: 10  
**Scope**: arXiv query `3D single object tracking`, category `cs.CV`, relevance排序；并辅以本地 SeqTrack、TrajTrack、P2B、CXTrack、PillarTrack 代码审计  
**Citation format**: APA 7th edition

## Executive Summary

3D 单目标跟踪的公开定义普遍只在第一帧提供目标三维框，之后的预测必须由点云、历史预测及其派生状态产生。所查方法虽在两帧匹配、序列建模、点级流、轨迹先验和长期记忆上不同，但没有论文把“每一帧当前 GT 尺寸”描述为合法在线输入；TrajTrack 更明确假定刚性目标尺寸在序列中不变，只预测位置与航向。代码审计显示 P2B、CXTrack、PillarTrack 和 TrajTrack 的尺寸通常从第一帧或上一预测框传播，而本地原 SeqTrack 将当前帧 GT `wlh` 送入 decoder，形成潜在 test-time label leakage。推荐 CT-SeqTrack 的正式协议继续使用第一帧固定尺寸，并将旧 SeqTrack 当前 GT 尺寸路径只保留为明确标注的 oracle/legacy 诊断。

## Methodology

本综述于 2026-08-27 使用 arXiv 查询 `3D single object tracking`、类别 `cs.CV`，按 relevance 取前10篇。语言模型代理从摘要中提取研究问题、方法、主要结论和限制，主代理进行跨论文综合。由于摘要通常不会说明 evaluator 的逐字段实现，另对工作区中的 SeqTrack、TrajTrack、P2B、CXTrack 和 PillarTrack 推理代码进行了只读审计。

**Limitations of this review**: arXiv 预印本不一定等于最终出版版本；摘要不足以验证每个仓库的全部推理分支。代码审计针对工作区当前快照，不能自动代表作者用于论文表格的私有运行版本。

## Themes

### Theme 1: 第一帧标注是在线 SOT 的唯一人工初始化

调查文献共同把 3D SOT 建模为：第一帧给定目标框，后续帧持续定位同一目标。BAT 直接把第一帧 GT 框称为可利用的强先验；ChronoTrack 摘要也以第一帧三维框为任务条件。TrajTrack 的正式定义进一步假定目标尺寸在序列中不变，因此只预测 `(x,y,z,yaw)`。

### Theme 2: 主流差异集中在如何利用历史，而不是请求新的 GT

PTT、GLT-T 和 PCET主要增强候选生成与特征匹配；STTracker、FlowTrack 与 ChronoTrack分别利用多帧特征、点级流和长期记忆；TrajTrack利用历史框轨迹。它们的共同在线边界是历史信息来自过去可用观测或预测，而不是当前帧标签。

### Theme 3: 固定尺寸是刚性目标 4-DoF 跟踪的常见工程选择

本地 P2B 和 CXTrack 的正式默认配置从上一预测框生成下一框，offset函数只改变平移和yaw，保留参考框尺寸。PillarTrack把参考/template框的 `object_dim` 拼入最终预测，随后将预测框设置为下一帧参考。TrajTrack也将 `ref_box.wlh` 作为模型输入。对车辆等刚体，这与第一帧尺寸传播一致。

### Theme 4: 论文协议与公开代码实现可能不一致

本地 SeqTrack在评估构造输入时读取当前帧 `3d_bbox.wlh`，尽管最终 offset box 仍继承参考框尺寸；这使当前标签尺寸影响网络内部几何查询。TrajTrack论文描述用 local/global proposal 之间的 IoU 做 refinement，但本地默认实现使用当前 GT IoU触发 refinement 并选择候选，构成更强的 GT-assisted evaluation。论文文字、公开代码和实际表格运行版本必须分别核验。

## Convergences and Disagreements

**Convergences**: 相关论文一致把历史建模视为从已发生帧提取信息；第一帧标注是合法初始化；刚性目标尺寸通常不作为每帧必须重新预测的自由度。代码层面，P2B、CXTrack、PillarTrack与TrajTrack的主路径都能以预测框递归传播尺寸。

**Disagreements**: SeqTrack本地实现把当前 GT 尺寸输入模型，与任务的严格在线边界不一致。TrajTrack论文的GT-free proposal agreement与本地默认GT-assisted refinement也不一致。这些差异更像实现/评估协议问题，而不是该领域认可的另一种标准协议。

## Gaps and Open Questions

- 当前 GT 尺寸相对第一帧尺寸对 SeqTrack3D Success/Precision 的净增益尚无隔离消融。
- nuScenes/KITTI 每个 tracklet 的标注尺寸抖动程度需要实际统计；若逐帧尺寸完全相同，泄漏在数值上可能近似无效，但代码边界仍不安全。
- 公开论文往往不说明尺寸来源、历史框来自GT还是预测、refinement是否读GT，复现报告应主动补充。
- 若研究对象包含真实尺度变化，应增加GT-free 7-DoF size head，而不是读取当前标签尺寸。

## Per-Paper Annotations

### Zou et al. (2020)

**Research question**: 能否利用2D跟踪约束3D搜索空间，提高稀疏点云跟踪效率与鲁棒性？  
**Methodology**: 以2D Siamese候选构造3D视锥，并在线验证搜索区域。  
**Key findings**: 缩小3D搜索范围；报告KITTI上的强性能；对遮挡和稀疏场景更稳健。  
**Limitations**: 依赖图像/LiDAR同步和2D tracker，摘要未说明尺寸协议。

### Shan et al. (2021)

**Research question**: 如何以点间Transformer关系增强P2B式3D SOT？  
**Methodology**: 在P2B中加入特征嵌入、位置编码和自注意力PTT模块。  
**Key findings**: KITTI Car报告约10%增益并保持约40 FPS。  
**Limitations**: 摘要未覆盖长期递归漂移或逐帧尺寸来源。

### Shan et al. (2022)

**Research question**: 如何在远距离稀疏、遮挡场景提高精度并保持实时？  
**Methodology**: 在投票和proposal阶段加入PTT以建模patch及前景/背景上下文。  
**Key findings**: KITTI和NuScenes验证；Car报告约10%增益；约40 FPS。  
**Limitations**: 未说明极端长时遮挡及 evaluator 尺寸细节。

### Wang et al. (2022)

**Research question**: 如何缓解候选分数/定位质量失配及稀疏目标信息不足？  
**Methodology**: Adaptive Refine Prediction聚合候选；Target Knowledge Transfer补充稀疏目标知识。  
**Key findings**: 报告KITTI和Waymo SOTA及较低成本。  
**Limitations**: 未说明候选聚合在强干扰下的稳定性和尺寸协议。

### Cui et al. (2022)

**Research question**: 如何从稀疏点云提取更充分的3D SOT信息？  
**Methodology**: 稀疏点转pillar、多尺度注意力、两阶段set prediction。  
**Key findings**: 在KITTI和NuScenes取得有竞争力结果。  
**Limitations**: 摘要未量化模块开销或递归误差。

### Nie et al. (2022)

**Research question**: 如何以全局—局部Transformer改善VoteNet投票？  
**Methodology**: 融合全局/局部线索并预测种子点重要性。  
**Key findings**: 消融支持组件有效；报告KITTI/NuScenes SOTA。  
**Limitations**: 未讨论长时漂移和尺寸来源。

### Cui et al. (2023)

**Research question**: 如何利用多帧历史点云改善两帧跟踪？  
**Methodology**: 输入多帧点云，以patch级稀疏注意力融合时空特征。  
**Key findings**: 报告KITTI 62.6%、NuScenes 49.66%。  
**Limitations**: 未说明历史预测污染和尺寸处理。

### Xu et al. (2024)

**Research question**: 如何避免点重采样的信息冗余/丢失？  
**Methodology**: 采用pillar表示、金字塔编码特征与Transformer骨干。  
**Key findings**: 报告KITTI/NuScenes上的显著基线提升。  
**Limitations**: 未量化速度、显存及尺寸鲁棒性的具体贡献。

### Li et al. (2024)

**Research question**: 如何结合点级局部运动与历史目标信息？  
**Methodology**: 估计point-level flow，以可学习目标特征汇聚历史，再聚合成实例运动。  
**Key findings**: 报告KITTI提升5.9%、NuScenes提升2.9%。  
**Limitations**: 未说明有效历史长度、监督和长期遮挡稳定性。

### Yoo et al. (2026)

**Research question**: 如何以紧凑记忆实现长期3D SOT？  
**Methodology**: 使用memory tokens、时间一致性和记忆循环一致性损失。  
**Key findings**: 报告SOTA和RTX 4090上42 FPS。  
**Limitations**: 未说明token数量、有效历史跨度和长期遮挡恢复。

## References

Cui, Y., Li, Z., & Fang, Z. (2023). STTracker: Spatio-temporal tracker for 3D single object tracking. arXiv. https://arxiv.org/abs/2306.17440

Cui, Y., Shan, J., Gu, Z., Li, Z., & Fang, Z. (2022). Exploiting more information in sparse point cloud for 3D single object tracking. arXiv. https://arxiv.org/abs/2210.00519

Li, S., Cui, Y., Li, Z., & Fang, Z. (2024). FlowTrack: Point-level flow network for 3D single object tracking. arXiv. https://arxiv.org/abs/2407.01959

Nie, J., He, Z., Yang, Y., Gao, M., & Zhang, J. (2022). GLT-T: Global-local transformer voting for 3D single object tracking in point clouds. arXiv. https://arxiv.org/abs/2211.10927

Shan, J., Zhou, S., Cui, Y., & Fang, Z. (2022). Real-time 3D single object tracking with transformer. arXiv. https://arxiv.org/abs/2209.00860

Shan, J., Zhou, S., Fang, Z., & Cui, Y. (2021). PTT: Point-track-transformer module for 3D single object tracking in point clouds. arXiv. https://arxiv.org/abs/2108.06455

Wang, P., Ren, L., Wu, S., Yang, J., Yu, E., Yu, H., & Li, X. (2022). Implicit and efficient point cloud completion for 3D single object tracking. arXiv. https://arxiv.org/abs/2209.00522

Xu, W., Zhou, S., Xiong, J., Zhao, Z., & Yuan, Z. (2024). PillarTrack: Boosting pillar representation for transformer-based 3D single object tracking on point clouds. arXiv. https://arxiv.org/abs/2404.07495

Yoo, J., Lee, S., Jeon, Y., Lee, M., & Heo, J.-P. (2026). Temporally consistent long-term memory for 3D single object tracking. arXiv. https://arxiv.org/abs/2604.13789

Zou, H., Cui, J., Kong, X., Zhang, C., Liu, Y., Wen, F., & Li, W. (2020). F-Siamese tracker: A frustum-based double Siamese network for 3D single object tracking. arXiv. https://arxiv.org/abs/2010.11510

