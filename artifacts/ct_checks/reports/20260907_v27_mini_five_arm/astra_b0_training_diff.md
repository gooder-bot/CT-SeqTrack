# `astra重构` 对 B0 有梯度训练的专项审计

日期：2026-09-08。范围：`a43a8ee^ → a43a8ee`，并检查当前五臂实际 observation 调用路径。只读生产源码、git 和 output；未改环境、训练代码或历史输出，未读取 TrajTrack。

## 结论

`astra重构` 确实修改了 B0 的**训练数据集合、稀疏样本接纳、点输入数值/身份和点级损失归约**，不是只改 B1/B2/B3。但它没有重写 B0 的 PointNet/Transformer 主干、中心/yaw 回归目标、四 view 权重或 Adam 拓扑。把所有近期低分说成“astra改坏了B0网络”超出证据。

本次新增的受控证据进一步缩小了范围：相同稠密全有效 batch、相同 B0 参数/BN 状态、相同 CPU RNG，仅切换 `ct_enable_v27`，21 个共同 forward Tensor 完全一致，最终 observation 完全一致，四 view 加权目标数值一致。新 CE mask 归约产生很小的浮点梯度差，不能说是逐位无变化，也不能据此证明20分以上崩坏。

确定的新增训练行为中，最应关注的是：**取消历史观测不足的样本重采样后，接纳的空/稀疏样本更多，但 forward 并未完整消费这些新 valid mask；框级目标仍正常监督。** 这是“数据接纳扩大、模型未同步适配”的风险，实际贡献需要真实数据按样本可观测性分层，不能在没有统计时声称它是唯一根因。

## 真实 observation 路径

1. `main.py` 构造 observation config，关闭插件训练分支、使用 `seqtrack_core` payload。
2. `get_dataset → MotionTrackingSamplerMF.__getitem__ → _build_view → motion_processing_mf`，训练历史来自 GT/独立 candidate 扰动，而不是 mechanism 的递归预测。
3. `StatelessCandidateBatchSampler` 保持四 candidate 协议。
4. `SEQTRACK3D.training_step` 的 observation transaction 临时关闭 `use_ct_joint_full/use_b1motion_v3/ct_enable_b1/2/3`。
5. `forward → compute_loss → _ct_candidate_weighted_observation_loss`，四 view 返回 B0 transaction，作为同一个 Adam 的 B0 更新来源。
6. 之后 mechanism 只给启用插件提供梯度；此次加入完整 `eval()+no_grad()` 改的是 mechanism 读取 B0 的方式，不是冻结 B0 的 observation 更新。

所以大部分 B1 acquisition、B2 labels/votes、B3 gain/H3 的改动不进入纯 B0 有梯度图；它们不能解释纯 B0 臂本身的低分。

## 提交中实际生效的 B0 训练变化

| 修改 | 旧行为 | astra/v27 行为 | 判断与必要性 |
|---|---|---|---|
| mini 参数训练集合 | 8个 mini_train 场景 | 6训练/1校准/1dev；实际B0每epoch由1262步变1057步 | 计划规定的实验协议变化；更新预算减16.24%，需与同6场景reference比较，不能直接拿旧8场景分数定量归因 |
| dataset/cache | split场景筛选、旧cache | 显式manifest场景筛选、role+manifest+IDs新cache键 | 需要，避免旧缓存混入；不是网络结构变化 |
| 历史目标点不足检查 | 三历史均无目标点时 assert，dataset以确定性retry换样本 | `online_recursive_state is None and not v27`，v27 observation也取消此拒绝 | 确定改变训练样本分布；可保留困难样本，但必须适配有效性和可观测目标，不能只删assert |
| N=1/2 regularize | 当作全零padding | 重复真实点到1024槽 | 修正真实观测被丢弃；应保留。但每个重复槽仍进BN/pooling/点loss，新unique mask没有全面进入B0前向 |
| N=0 regularize | 零padding，无显式身份 | 零padding+invalid身份 | 合理数据合同；其time/prior/BC不保证全零，不能假设模型自动忽略 |
| crop坐标 | crop到support局部，再回world，再到anchor | 同一membership的原始ID直接转换到anchor | 同一数学坐标变换的数值简化，合成旋转案例ID完全相同、误差仅1.8e-15；没有发现轴变换反向错误 |
| segmentation loss | 对所有槽CE | 排除padding槽，以剩余类别权重总和归约 | 合理修复，改变含空槽batch的权重；全有效数学上相同但浮点归约不是逐位相同 |
| box-cloud loss | 所有槽SmoothL1 | 排除padding槽后按有效槽数归约 | 合理修复；重复真实点仍反复计入，未改框级监督 |
| observation payload | 原核心字段 | 额外保留 IDs/valid/unique/rawcount/current-valid | 身份不直接输入学习器；新增mask用于loss/统计，前向完整接入仍缺失 |
| 时间CLI优先级 | 旧ct_time_mode优先 | dynamics_time_mode优先 | 修复控制语义；本次true运行的B0原始主干order-time设置没有因此改成另一套时钟 |
| mechanism Dropout模式 | 只隔离BN统计，仍可能training Dropout | 临时完整eval()+no_grad再恢复逐module flag | 符合预测递归合同；不进入B0梯度图；纯B0没有机制模块更新，不能用此归因其崩坏 |
| checkpoint回调 | 旧epoch-end顺序 | Checkpoint marker、严格boundary、resume RNG恢复 | 同run durability修复，不改scratch第一个稠密B0 forward/目标；本次五臂60epoch已完成 |

补充：`regularize_pc` 的 `>2 → >0` 是公共工具函数修改，未包在v27 flag下，因而“在新代码上只把ct_enable_v27关掉”不等于完整重放astra父提交；需要明确这一实验边界。

## 没有在 astra 新引入的 B0 问题/设计

- 帧/通道 `reshape` 缺少轴交换：astra未修改该行，原SeqTrack已经存在。CT的逐点特征绑定增加了其下游危害，但不是9月5日首次出现的B0新错误。
- `Seq2SeqFormer.forward(valid_mask)` 不使用历史mask：astra没有改 attention 模块，问题早已存在。
- B0 box-cloud transaction 双计：来自更早的tensor alias；8月24日高分v25同样经过该路径。astra只是将其基础点loss替换为masked版本，并没有新写这次双计。
- 四view `.5,1/6,1/6,1/6`、首帧尺寸、统一Adam、B0和插件梯度隔离、GT/noise observation与预测 mechanism分流：均已存在于高分v25。
- hard argmax分割/运动类别、按点槽顺序压缩128token、框loss没有按缺历史mask归约：旧主干限制。稀疏样本接纳变化可能加重它们，但缺少相同数据A/B不能量化。

## 受控探针

文件：[probe_astra_b0_training.py](probe_astra_b0_training.py)、[astra_b0_training_probe.json](astra_b0_training_probe.json)。运行真实 Seg/Mini/FeaturePointNet、Transformer、sampler、compute_loss；仅外部包入口沿用现有CPU合同测试fixture，未替换模型数学运算。

### 稠密固定 batch：v27 flag关/开

- 4行batch，对应candidate0/1/2/3，所有point valid=1。
- 两次执行恢复同一模型state_dict（含BN）、相同torch RNG，均training模式。
- 21个共同输出Tensor最大绝对差 **0**，observation框差 **0**。
- 实际四view加权目标两次均 **12.7884817123**。
- segmentation loss：**0.7032962441 → 0.7032963037**。
- 236个有梯度参数Tensor中37个有逐位差异；最大绝对差 **2.1569e-6**，整体梯度差相对L2 **6.2467e-8**。

解释：相同输入下没有发现稠密B0前向结构变化；新CE归约存在预期的浮点微差。**本探针不是完整父提交训练复现，不代表CUDA必定逐位一致，也不证明微差不会经训练累积。** 但不能把这样的微差直接升级为“已找到全部掉分根因”。

### 三历史全空、当前有点

同一真实sampler payload：

- v27关闭：拒绝，`not enough valid box`。
- v27开启：接受，四帧有效槽为 **[0,0,0,1024]**，框label仍为有限运动 **[0.3980,-0.03993,0,0]**。

这是可复现的训练接纳变化。当前forward缺历史mask意味着三份padding可影响表示；不能只依靠点loss排除padding。当前点仍可能支持绝对定位，因此也不能一律把这类样本删掉或称作所有目标不可学习，应分任务区分监督来源。

### crop anchor变换

取三个旋转/平移support历史框，以最新框为anchor。旧roundtrip与新direct transform三次crop点ID与点数完全相同，最大坐标差 **8.9e-16至1.8e-15**（float64合成数据）。因此没有证据说新canonicalize把点云旋错了；真实float32边界仍可能有数值差，应以ID membership和标签几何合同检查。

## 对本轮根因定位的建议

1. 先由主审计并列历史高分、astra之前低分和当前dev结果，避免将“时间上靠近astra”误判成“从此首次崩坏”。
2. v27独有的预测history hard prior、稀疏empty策略与输入重建属于最直接的新路径，优先用同checkpoint同轨迹做输入敏感性诊断。推理部分由主审计报告展开。
3. 对有梯度训练新增统计：三历史目标可见数、当前目标可见数、有效槽、独立点数，以及每类的seg/BC/center/motion/ref loss。特别统计“旧assert会拒绝、v27接纳”的样本占比和梯度贡献。
4. 修复BC双计、帧/通道布局、历史mask等确定错误，但不要把原先也能高分的bug说成astra新增原因。
5. 先完成匹配集合的reference/修复B0，再回到五臂；在B0未建立可解释闭环前，更多B2/B3创新无法回答纯B0失效的问题。

以上为审计和受控CPU探针，没有修改生产代码，没有训练新checkpoint。
