# CT-SeqTrack 数据通路与耦合建议

2026-09-05；当前 v26 源码复核。结论：保留双流学习、模块梯度边界和唯一状态写入者；优先统一前向数据语义、机制 observation 的运行模式、原始点身份与部署闭环。源代码、配置和 output 未修改。

**当前结构可保留，但“数据耦合”需要比“梯度耦合”更充分、更一致。**

当前 observation 流使用四视图训练 B0；机制流用 canonical 递归样本，在同一步参数下执行 B1 prepass、点云获取、无梯度 B0 observation、带梯度 B1/B2/B3，然后合并损失交给统一 Adam。机制训练写入 observation；selective 部署写入实际接受的有界结果。跨帧是预测数据递归，没有跨整条轨迹的 BPTT。

| 损失 | 应更新的参数 | 建议保持隔离的上游 |
| --- | --- | --- |
| B0 四视图损失 | B0 主干和原有头 | 不含插件 |
| B1 物理 mean 损失 | 时序编码器、mean 分支 | B0、递归历史 |
| B1 sigma 损失 | 当前 sigma 分支 | mean 与 detached context |
| acquisition 损失 | 边界 head，建议独立的小特征分支 | 物理 mean、B0 |
| B2 relation/targetness/vote/raw | B2 encoder、adapter、attention 和预测头 | B0 features、B1 prior |
| B3 动作收益损失 | B3 projection、risk/gain heads | B0/B1/B2 输入 |

数据仍能跨上述边界传递，且下游参数正常学习。统一 Adam 的 named groups 本身不会让梯度跨越 detach。当前 formal 为 precision=32、gradient_clip_val=0；不应把共享 AMP 溢出或全局 clipping 当作已发生的差异根因。若以后引入全局 norm clipping/AMP，需重新验证优化器层面的隔离。

**必须修改：B1 learned 输出与实际 acquisition 记录分开。**

`models/seqtrack3d.py:2546` 的 prepass 白名单丢 acquisition margin，而 `:2650` 的 unbatch 强制检查它，造成 learned prepass 无效并退回 CV。后续 `:8446` 带梯度重算却可有效。

还有一处混用：`:2813–2822` 的 `MotionPriorOutput` 把二次 learned center/direction/sigma/valid/margin 与实际裁剪 `search_v3_prior_source_id` 拼在一起。可能形成 learned geometry + CV source；`pipeline_contracts.py:178` 又只在 source=1 时检查几何对齐。

B2 可以同时读取 learned prediction 和实际 support，但应显式区分：

- `learned_prior`：物理均值、统计 sigma、方向、预测有效性；带梯度版本用于 B1 自身 loss，detached 版本供下游条件。
- `acquisition_record`：实际使用的 endpoint、corridor/shell 几何、margin、来源、fallback 原因、anchor 和有效时间。它描述真实获取行为，不能用后续重算覆盖。

二者共享同一份历史输入构造，并记录 model step 与 input fingerprint。当前两次 B1 之间未发生 optimizer 更新，不应归因为旧权重缓存；但 prepass 与 sampler 使用完整旋转/纯 yaw 两条构造路径，宜统一几何语义。非平面姿态差异尚未在真实数据量化。

保留“无梯度 prepass 用于裁剪 + 带梯度重算用于训练”。缓存 detached 输出不能代替后者，否则 B1 自身梯度消失。CPU crop、top-k 和集合运算也不会仅因删除 detach 就成为可微获取策略。

**建议优先验证：机制 B0 用部署模式生成 observation 和递归状态。**

`_forward_safe_mechanism()`（seqtrack3d.py:8419）只加 no_grad；外层 :8610 只将 B0 BatchNorm 切到 eval。B0 的 `Seq2SeqFormer` 在 :1086 未覆盖默认 dropout，而 `models/attn/Models.py:86` 默认 dropout=0.2。因此普通机制前向期间 Transformer Dropout 仍开启；H3 `_shadow_forward():7958` 和正式评估则关闭。

这个差异意味着 B1/B2/B3 的历史框、observation 与收益标签可能混入部署时没有的 Dropout 噪声。它可以被解释为训练扰动，但现有方案未证明该扰动有益，尤其 H3 当前/未来步骤运行模式不统一。

建议将机制中 B0 observation 生成临时设为完整 eval + no_grad，结束后精确恢复各子模块 training flag。B1/B2/B3 随后恢复各自训练模式；B0 仍在 observation 流正常 train/backward/更新，不冻结任何启用参数。这里优先使用 no_grad，因为所得特征随后供可训练的 B2 adapter 使用，不应未经检查改成 inference_mode。

CPU 使用真实 `_forward_safe_mechanism` 的 B0 调用前缀和真实 BN 隔离 helper，在合成 host 上确认 grad=False、BN training=False、Dropout training=True，连续两次输出不同。见 [coupling_probe.json](coupling_probe.json)。这是控制流与框架语义复现，尚未运行完整模型/真实数据，也不证明关闭 Dropout 必涨分。应以相同机制输入、同预算、独立 scratch 版本比较状态噪声和最终性能。[PyTorch 对 eval/no_grad 的说明](https://docs.pytorch.org/docs/stable/notes/autograd.html#evaluation-mode-nn-module-eval)。

**必须修改：原始点索引、有效性与数量贯穿每个阶段。**

用 `(frame_id, original_point_index)` 标识实际点；同时保留原始数量、采样槽 valid mask、来源 bitmask。一次变换到明确坐标系，再按索引做各支持域 mask/并集/差集；padding 和复制采样都不产生新物理点。

这解决浮点 XYZ key 导致同点冒充 novel、FPS 重复索引、raw/sample 数量混淆等问题。局部 `(x,y,z)` 与 nuScenes `wlh` 的转换应统一为 `(l,w,h)`；memory role 和动作 IoU 标签共用几何语义。

现有 `regularize_pc` 对 <=2 原始点输出全零，而部分 base mask 固定为 1；`base_model.py:1892` 又通过 XYZ 代数总和判断是否绕过整条恢复链。应显式区分：B0 当前输入无效但历史/extension 可用；所有测量均不可用。前者构造明确 observation fallback、mask 无效特征并继续合法 B2/B3；后者回退。所有 endpoint 都导出状态与原因，不能因缺点消失于机制统计。

**保留为后续候选：新增测量不必只来自 crop 外。**

现有差集是 support 减去整个 B0 raw crop（sampler.py:804、:1351；ct_search.py:1210）。所以 crop 内有目标点但未被 1024 采样保留时，这些点也被永久排除于 B2。

当前定义适合证明“空间扩张获取”的贡献。若诊断证明采样遗漏是瓶颈，可以另立协议：

`P_new = (P_outside_crop ∪ P_inside_crop_unsampled) \ P_B0_seen`。

维持总 768→256 预算，对 crop 外与 crop 内补采样做明确来源分配；任何 B0 当前已输入的原始点都不能作为 novel。此方向改的是采样来源，仍可只允许 B0 未见点投票。不要直接让 base 点投票后继续称 extension-only；若改联合精修，需 base-only refiner 对照。

首先检查 eval 的 `base_raw_target_count>0 && base_sampled_target_count==0`，以及低采样覆盖分层。字段在 base_model.py:865–875，实际参数传入在 :2916。sampled 目前统计槽位，可含重复，不能用 raw-sampled 得到未见点数量；训练 `ct_acquisition_base_target_count` 来自 sampled 槽，而 eval `base_target_count` 来自 raw，必须统一命名。目标计数通常按 bb_scale 标签框，不能一律称 exact 实例点数。

本轮已读 Full epoch60 的 1928 条诊断：raw 含目标 1553 行，其中 sampled 为零仅 2 行，两者 raw point count 都为 2。排除 raw 点数<=2 后，1526 个 raw 含目标 endpoint 中 sampled 全漏为 0。全部诊断里 raw 点数<=2 有118行。该结果不支持把 crop 内补采样设为目前主改动，更直接支持先修 <=2 点置零和有效性；但没有排除部分目标点保留不足。此前清点发现该CSV缺251/2179个非首帧endpoint，不能向缺失帧外推。结果见 [crop_sampling_probe.json](crop_sampling_probe.json)。

**应加强：B0 到 B2 的表征条件，放到选点之前。**

当前 base/memory 读取 FeaturePointNet 第二个 1×1 Conv+BN+ReLU 的 64d 输出，未拼全局池化，亦不是最终 Transformer/decoder 表征（pointnet.py:286；seqtrack3d.py:3607）。当前帧 1024 点作为 base，前三帧各抽 8 框内+4 context，共36 memory。输入未被 segmentation prediction mask 清空，因此已有点和背景已经能给 B2 提供条件。

建议保留这些点对齐特征与 detach，先加 B2 自己的局部几何 adapter，并分开 target/context 条件，在 relation 选点前融合。可以后续另试 detached B0 高层摘要，但不可直接把 decoder token 当成逐点特征；需要说明位置对应和身份语义。没有必要为了获得更强条件就开放 B2 loss 回传 B0。

**必须补齐：状态闭环与校准分布。**

训练在 seqtrack3d.py:8099 强制 observation commit；部署在 base_model.py:1950 写实际 candidate。B3 接受动作会改变下一帧 crop、B1 历史和 memory。这是明确的 shadow-training 选择，不等同于完整 on-policy 训练。

保持一个公共 canonical commit 接口，携带 frame/time、state version、proposal source、quality 和 policy。H3 克隆的两个状态只生成标签，不回写主状态。校准阈值固定后，必须完整执行 selective dev rollout，再用未参与设计的测试数据评估。

现有 calibration/dev 仅对 mechanism 流留出；B0 observation 使用整个 train_split，且同场景不同轨迹仍相关。若主张整个模型严格 held-out，需在所有训练流之前统一按场景留出，各 arm 使用一致 B0 训练集合。若维持当前训练数据使用方式，应明确写 mechanism-held-out，不能声称整个模型未见，并保留独立完整模型测试。partition seed 应统一进入 manifest，而不是一处用训练 seed、一处用固定分区 seed。

现阶段先修 H3 结构有效样本覆盖、保持 shadow 算力、补 selective 闭环，不直接部署训练中未成熟的 B3。只有确认上游候选与 selector 有收益，才另立 policy-state mechanism 训练协议；所有策略选择使用训练内部数据，cal/dev/test 不回流训练，仍遵守新版本 scratch。

**推荐实施顺序。**

1. 修接口和基础语义：prepass 字段、独立 acquisition record、point ID/valid mask、几何、完整 endpoint。
2. 统一机制 B0 运行模式与输入构造，验证 B0 参数/BN/RNG 隔离；保留当前双流与梯度边界。
3. 加 acquisition 自身质量条件及 B2 选点前身份条件，分别消融。
4. 根据 crop 内采样遗漏数据，决定是否注册“B0 未见测量”补采样臂。
5. 冻结阈值的 selective dev 闭环；再决定是否需要训练期 policy-state 方案。

本轮没有新的 GPU/nuScenes 闭环结果。代码事实、CPU 控制流复现与研究候选在上文分别标明；不能以架构判断承诺分数提升。
