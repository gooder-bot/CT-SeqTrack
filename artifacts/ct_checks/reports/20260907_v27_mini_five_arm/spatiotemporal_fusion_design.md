# B0 的时空特征通路与融合位置：补充审计

日期：2026-09-07。仅新增分析和合成复现，不修改模型、历史配置或 output。本文区分确定的数据合同问题、既有架构限制和待实验验证的设计。

## 1. 先修真正的特征对应，再增加融合

当前 SeqTrack 已有时空融合：逐帧 FeaturePointNet → 每帧 local Transformer + 全部帧 global Transformer → box-corner cross decoder。`models/attn/Models.py:147–161` 已实现此结构，因此“再加一个时序 Transformer”不足以构成有针对性的修改。

### 1.1 明确的帧/通道打包问题

`models/seqtrack3d.py:3464–3469` 将 `[B,4096,5]` 的点输入转置，拼接9维 box-aware 特征，得到 `[B,14,4096]`。第3752行直接执行 `x.reshape(B*L,-1,chunk_size)`，无法把 channel-first 数据转成逐帧 `[B*4,14,1024]`。正确次序应为：

```python
solo_x = x.reshape(B, C, L, chunk_size).permute(0, 2, 1, 3)
solo_x = solo_x.reshape(B * L, C, chunk_size)
```

设3个历史帧与当前帧为 H1/H2/H3/T，通道为 x/y/z/time/prior/BC0…BC8，当前四个“帧”实际组成如下：

| 伪帧 | 14个输入通道实际来自 |
|---|---|
| 0 | 四帧的x、四帧的y、四帧的z、H1/H2的time |
| 1 | H3/T的time、四帧的prior、四帧的BC0、四帧的BC1 |
| 2 | 四帧的BC2、四帧的BC3、四帧的BC4、H1/H2的BC5 |
| 3 | H3/T的BC5、四帧的BC6、四帧的BC7、四帧的BC8 |

因此伪帧中同一槽号实际上拼接了不同帧的第i个采样点；这些点并没有跨帧对应关系。既非真正的帧内编码，也非显式构造的跨帧对应。标号张量复现中，仅3.57%的元素与正确逐帧打包位置相同。

**原 SeqTrack 的 `../seqtrack/models/seqtrack3d.py:141` 也使用这一写法。** 它是继承实现的语义问题，不能直接作为“v27相对原baseline掉分多少”的因果证据；模型也可能在这种固定排列上学到其他相关性。修复会改变前向和BN分布，必须重新从头训练，不能将旧checkpoint直接切换布局后的分数称为修复收益。

CT新增问题更明确：第3765–3767行把这些伪帧的第二层特征重新解释成 `[B,L,1024,64]`；第2818–2840行以真实 history_points/ref_box/mask/point ID 构造 memory，B2将 `[:, -1]` 作为当前base点特征。**特征与原始点身份已错配。** 检查shape和显式base点等于raw当前点，并不能证明特征确实属于对应点。局部几何、role和identity similarity必须建立在此合同修复之后。

更直接的复现：只改变当前帧XYZ、保留全部其他输入，最后一个伪帧输入的变化为0，正确最后帧输入的最大变化为123。由于最后伪帧全由box-distance通道构成，在eval固定BN状态下，从这里得到的所谓 `current_base_features` 没有当前点XYZ的直接编码路径；当前XYZ反而进入被标为“历史”的伪帧。此结论针对该64维point-aligned接口，不能误写成整个B0都不使用当前XYZ，B0的其他伪帧和coarse分支仍会使用。

### 1.2 历史有效位确实未用于attention

`Seq2SeqFormer.forward` 第125行接收 `valid_mask`，却未使用。帧内encoder、跨帧encoder和decoder都没有传入mask。`FeaturePointNet.forward` 也不接收点级valid/unique mask。

复现使用生产配置的真实Transformer、随机权重、eval模式：

- 同一输入把历史mask `[1,0,0]` 改成全1，输出差严格为0；
- 保持 `[1,0,0]`，仅替换两个无效历史块的source feature，当前框输出最大变化0.552515。

这证明无效历史具有真实输出影响路径；0.552515是合成夹具数值，不能解读为checkpoint上的实际误差。该mask遗漏同样继承自原SeqTrack。详见 `reproduce_b0_feature_contract.py` 和 `b0_feature_contract_reproduction.json`。

修复须覆盖global encoder的K/V、decoder的K/V、帧内padding与无效query输出，并处理全无效key：不能把所有logit设为同一个大负数后仍得到均匀softmax。历史有效性与点云观测有效性是两个维度；不存在的历史帧应屏蔽，存在但当前点为空的预测框仍可属于合法运动历史。

### 1.3 信息压缩与硬判决：限制，不是已证实掉分原因

- `models/backbone/pointnet.py:299–304` 将第二层64维点特征按**采样数组序号** AdaptiveMaxPool 到128槽，再拼接1024维全局max。每槽约8个输入点，不是基于XYZ的局部邻域；不同特征通道的max还可能来自不同物理点。后续512个token不具有明确的空间中心。不能把它们直接当作128个有point ID的点。
- B0 seg和FeaturePointNet里的BN/pooling不消费新增valid/unique mask；padding和重复采样仍参与表征/统计。v27 loss层排除padding，不代表forward已排除padding。
- `seqtrack3d.py:3479–3490` 的seg argmax只硬筛**coarse motion分支**；FeaturePointNet使用原始x，没有被seg硬清零。不能误写成“分割失败后整个B0都看不到点”。第3721–3722行motion state argmax又使coarse motion二值化。它们可能影响稀疏输入初始box query，但目前缺少coarse/final/error分层的实际证据，不能先把所有硬门一并删掉。

## 2. 最适合当前问题的插入位置

**如果目标是阻止B0在已包含少量目标点时早期漂移，首选正确逐点特征之后、128槽压缩之前的几何与时间条件融合。** 若仅增强B2 extension cross-attention，当前base已有目标而extension为空的情况仍没有改善路径。二者解决不同问题。

| 位置 | 直接解决 | 不能解决 | 建议 |
|---|---|---|---|
| B0第二层逐点64d特征后、压缩前 | 当前少量点借助历史身份/几何得到更好的当前帧表示，提前避免漂移 | 当前全云无目标；crop外目标 | 作为独立可关的ST模块，优先小规模验证 |
| B0现有512token之后再堆Transformer | 继续混合压缩特征 | 已丢失的点对应与几何；无效token污染 | 当前不优先 |
| B2现有256 query→base+36 memory的cross-attention处 | 新取得点的实例身份与背景排除 | B0-only分数；无extension时的早期稀疏失误 | 修特征合同后替换/增强现有attention，不再并列堆一套 |
| B1框序列后拼接全局特征 | 调整获取先验 | 同一crop内的细粒度点身份 | 现有B1先保持运动/获取分工，不把图像式融合随意塞到这里 |

### 建议的ST最小设计

1. 正确拆帧后获得 `P:[B,4,1024,3]`、`F:[B,4,1024,64]`、`valid/unique:[B,4,1024]` 和有效时间。每个F必须与对应P、frame UID、point ID严格同步。
2. 当前帧1024点作为query；从过去3帧唯一有效点抽取每帧8个框内、4个上下文key，形成36个history token。槽预算可先复用现有设计，历史角色来自预测框，质量来自历史预测分割/点数，当前GT仅用于损失。
3. 在真实点级表示上加入小型空间邻域聚合（64d、k≤16、尺寸归一化半径）。跨帧关系使用相对局部坐标、预测框运动对齐、effective age、role和质量；保留对齐前后的坐标语义，背景点不强制按目标运动变换。稀疏时不要只用很小硬半径排除所有历史key，可保留身份prototype的全局通路。
4. 当前query对36个历史token做一次cross-attention，得到64d残差 `F'_t = F_t + g * DeltaF_t`。g由当前有效唯一点数、历史质量等连续输入决定，不做presence硬拒绝；可用小的非零初始门，保证首步就有上游梯度。有效历史全空时残差为0。
5. 用融合后的真实点特征继续原PointNet后段和128-token压缩，再进入现有local/global Transformer与box decoder。若改为FPS中心+空间pooling形成128个带坐标token，作为下一项独立变化验证，避免同时改融合、采样和全部Transformer而无法解释收益。

精确插入应发生在 `FeaturePointNet.forward` 第298行得到第二层特征之后、其后深层point MLP与第299/300行pool之前；目前函数一次性完成4帧，需要拆成 `encode_points`、可选ST、`aggregate_tokens` 三个清楚步骤。仅修改返回的 `second_layer_out` 不会影响B0 decoder，不能称为增强了B0。

### 梯度与模块化合同

- ST属于观测网络的可选模块，和B0参数一起从epoch0由观测loss训练；同一forward内的历史点特征可参与梯度，历史预测框/采样索引detach。不通过跨时刻递归状态建立长BPTT。
- B2/B3消费的B0或B0+ST特征仍detach；B2/B3自身一直训练。不要调用现有 `build_box_memory_tokens` 后误以为历史特征可回传梯度：其第100行明确detach。ST应有共享几何选择器与不同梯度策略，不能让B0模块悄悄继承插件的detach合同。
- 固定比较“修复后的B0”和“同样修复+B0-ST”；若后续比较B1/B2/B3，则所有待比较臂采用同一个观测干路定义。原始SeqTrack实现保持只读，并明确区分原实现参考与公共修复对照。
- 新ST结构从头训练，禁止旧checkpoint改结构后继续作为正式实验，也不冻结B0后单训ST。

## 3. 当前空观测路径会丢弃任何新增融合输出

`models/seqtrack3d.py:3831–3835` 在eval中将 `ct_current_observation_valid=0` 的最终B0局部框设为零偏移，也就是reference box。这是v27原计划明确选择的空输入语义，不是数值NaN崩溃。

所以，仅给B0增加ST，**全空current crop时即使历史预测出了位移，也会被这里覆盖**。单纯“加入memory就能恢复空帧”不成立。应分别处理：

- 当前有1/2等稀疏真实点：通过有效mask、历史身份融合提高正常observation；
- base为空但extension合法：继续B1获取→B2候选→B3动作，先修现存presence合同；
- 全部当前证据为空：保持reference是一种既定策略；若要学习短期纯历史运动延续，必须作为显式动作/模式加入训练与评测，记录为无观测预测，而不是伪造点或绕过valid。全云无目标的帧不应期待空间特征凭空恢复真实位置。

## 4. 最少必要的实验

先做小规模顺序消融：正确拆帧与mask的公共修复B0 → +ST（仅新增一次点级历史融合）→ 在同一ST干路上接修复后的B1+B2/Full。原版reference另以匹配数据协议训练。每个变体从头训练，mini工程阶段可先短跑再正式60轮。

核心指标除S/P外要报告：首次距离>2m的时刻；当前全局/B0 raw目标点分桶；有目标的1–8/9–32/更多点时的coarse与final误差；重获目标所需帧数；crop全空连续长度；token有效数；invalid-history perturbation不变性；point ID和编码特征逐点对应。这样才能判断提升来自少量点识别、恢复动作还是更大的搜索支持域。

新结构尚未实现，不能据此保证涨分。确定的结论是：**当前“已经做了时空融合”与“拥有正确的时空点特征合同”并不等价；先修合同，再在压缩前增强真实点级时空关系最有针对性。**
