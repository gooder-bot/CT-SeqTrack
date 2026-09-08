# B0 训练目标、样本分布与 SeqTrack 对照审计

日期：2026-09-07。只读对照 CT 当前源码和冻结的 `../seqtrack`，未读取 TrajTrack，未修改模型或 output。复现使用现有 CPU 合同测试提供的真实 sampler、Seg/Mini/FeaturePointNet、Transformer 和 compute_loss；仅未安装的外部包入口由原测试 fixture 提供占位接口。

复现文件：[reproduce_b0_distribution.py](reproduce_b0_distribution.py)、[b0_distribution_reproduction.json](b0_distribution_reproduction.json)。

## 结论及优先级

目前已找到两个可以直接落实的 B0 问题，不能只将低分归结为 dev 比 mini_val 难：

1. **CT 五臂实际 B0 目标将 box-cloud loss 加了两次**。配置 `bc_weight=1` 的实际系数是 2；普通 SeqTrack/reference 目标系数是 1。这是确定的 tensor alias 错误，CPU 真实模型已经验证数值与梯度。
2. **v27 将全部递归预测历史输入标成了 0/1 的硬 prior**，而原 SeqTrack 对第二个及以后预测 endpoint 使用 0.2/0.8。v27 统一 sampler 只检查 `candidate_id`，忽略框来自 GT 还是预测；推理永远 candidate0，导致语义回归。

同时确认帧/通道 reshape 错误在原 SeqTrack 中已经存在，CT 继承到了 B0 和 B2 的逐点特征链，需由空间通路审计一并修复。**继承问题不能被称为 v27 相对原实现下降的唯一原因**。

修正上述错误后，还需要针对 B0 的有梯度训练分布做受控实验：当前 B0 只训练 GT 或 ±0.3m/±1.5° 微扰的历史，递归有误差的 mechanism 轨迹不训练 B0；加之 canonical 真 GT 分支占 50% loss，模型没有直接学习数米漂移后的闭环恢复。

## 1. 已确认：BC loss 双计

### 错误位置

`models/seqtrack3d.py:5809`：

```python
b0_transaction_loss = loss_total
```

两变量持有同一个 Tensor。随后 `7498–7500`：

```python
loss_total += loss_bc * self.config.bc_weight
b0_transaction_loss = b0_transaction_loss + loss_bc * self.config.bc_weight
```

第一行原地修改了共享 tensor，因此 B0 transaction 已经包含了一次 BC；第二行再次添加同一项。设不含 BC 的目标为 L，则最后：

- `loss_total = L + λ_bc L_bc`
- `loss_b0_transaction = L + 2 λ_bc L_bc`

主审计 git blame 已确认：相关 alias 和重复添加来自 `25586b27`（2026-08-13）的 CT 损失分离改动，v27 仍继承此问题，不能称为 9 月 5 日才引入。

2026-09-08 历史回查补充：实际较高分的 `049de82` v25 B0（mini_val S/P=50.6904/59.2801）也通过 safe-auto 四 view helper 读取这个双计 transaction。因此该错误需要修正，但**不能单独解释后续 v25 跳水或 v27 跨臂差异**；canonical 0.5、首帧尺寸也已存在于该高分运行。详细提交—运行对照见 [b0_git_changes.md](b0_git_changes.md)。

### 为什么这次五臂四个 view 都双计

需要按真实训练路径判断，不能只分析裸 Full 的 compute_loss：

- `main.py` 创建 observation dataset 时将插件关闭，并将 `ct_observation_payload_mode` 设为 `seqtrack_core`。
- `prune_seqtrack_observation_payload` 不保留 `ct_b0_auxiliary_only`。
- `_b0_auxiliary_batch` 读取的是该字段，**不是 candidate_id**。这次四 view 的普通 observation 都不走旧辅助微批的提前返回路径。
- `training_step` 在整个 observation transaction 中临时关闭 `use_ct_joint_full / use_b1motion_v3 / ct_enable_b1/2/3`。所以 B1/B2 其他 loss 分支不会在中间打断 alias。
- `_ct_candidate_weighted_observation_loss` 对每个 view 读取 `loss_b0_transaction`，故四个 view 都含 2×BC。

因此实际目标是：

`0.5 × (L0 + 2BC0) + [(L1+2BC1)+(L2+2BC2)+(L3+2BC3)] / 6`，其中 L_i 不含 BC。

不是“canonical 2×BC、auxiliary 1×BC”；后者仅适用于旧 online 辅助微批路径。也不是“仅 B0 双计、插件臂正常”；本次五臂 observation 路由完全相同。

### 真实模型复现

同一 pruned observation batch、相同初始化、实际训练路由：五臂每个 view 的 `(loss_b0_transaction - loss_total) / loss_bc` 全部为 **1.0**。五臂实际加权目标都为 `13.0184516907`。

固定同一 forward，单独把 BC 系数由 0 改到 1：

| 数值 | 结果 |
|---|---:|
| BC loss | 3.2603111267 |
| `loss_total` 增量 | 3.2603106499 |
| `loss_b0_transaction` 增量 | 6.5206217766 |
| transaction 增量对 pred_bc 梯度 / 单份 BC 梯度的范数比 | 2.0 |
| 该梯度与 2×BC 梯度的最大绝对差 | 0.0 |

v27 ordinary reference 的 `ct_unified_auto=false`，正式 training 返回 `loss_total`，因此仍只含单份 BC。原 `../seqtrack/models/seqtrack3d.py` 也只对普通 loss_total 添加一次 BC。

这足以证明 CT 相对 reference 的训练目标不符计划，但**不能据此推算修复后的 S/P 提升幅度**。五个 CT 臂共享这个错误，所以它也不能解释 CT 跨臂第一次 optimizer 后的 hash 分叉；那个问题仍需逐层数值诊断。

### 应怎样改

将模块目标各自独立构造，最后显式相加；避免共享 Tensor 别名后再 `+=`。先修正 B0 transaction 的 BC 系数，再检查所有模块 transaction 是否与声明的加权损失逐项一致。不要用 `bc_weight=0.5` 隐藏实现问题。

应增加有意义的损失合同：四 view 普通 observation、辅助微批及 reference，transaction 相对明确公式数值/梯度均一致。此前“loss 有限”“五 CT 臂相同”“参数有梯度”都会通过，无法捕获五臂一起双计的错误。

## 2. 已确认：预测历史的 prior 语义被 candidate_id 替代

原实现 `../seqtrack/models/base_model.py:259–268`：frame1 历史来自初始化 GT，使用 0/1；`frame_id != 1` 后改用 0.2/0.8，表示历史来自预测。

v27 `utils/v27_input.py` 总是以 `candidate_id=0` 调用 `motion_processing_mf`，该 sampler 在 `datasets/sampler.py:1614–1621` 只对 `candidate_id != 0` 软化 prior，未考虑 `online_recursive_state`、历史来源或 `_ct_inference`。于是从 frame2 到轨迹结束，即使历史框已经漂移，B0 输入仍是 0/1。

已用真实 sampler 复现：frame8、三帧历史框带 0.2m 预测误差，三个历史块输入第5通道唯一值均为 `[0,1]`；原实现应是 `[0.2,0.8]`。该通道直接进入 `seg_pointnet`，也进入 `feature_pointnet`，不是只用于诊断的字段。

这个变化与“统一训练/推理构造器”并不矛盾：机制流和推理现在确实一致，但 B0 **有梯度**训练中的硬 prior 对应真 GT 历史，推理的硬 prior 却对应预测历史。相同构造函数不能自动保证相同语义。

修复方向：以历史框来源/有效性决定 prior 编码，区分初始化、已预测、无效 padding，而不是根据 candidate_id 推断。先对现有 checkpoint 做恢复历史软 prior 的同轨迹敏感性评测，随后在所有 scratch 训练与评测中统一该语义。这是明确输入行为的修复；是否涨分仍需实测。

## 3. 样本与归约：相同的部分、刻意改变的部分

| 项目 | 原 SeqTrack | CT v27 observation | 判断 |
|---|---|---|---|
| 历史长度 | 3 | 3 | 一致 |
| 每帧点数 | 1024 | 1024 | 一致 |
| 当前 crop 中心 | 最近历史 GT/扰动框 | 最近历史 GT/扰动框 | 训练时相同，无当前 GT crop 泄漏 |
| candidate0 | 历史 GT，无扰动 | 历史 GT，无扰动 | 样本语义相同 |
| candidate1–3 | 各历史独立 ±0.3m、±1.5° | 相同幅度、按 seed/epoch/frame 确定 | 非共享递归误差，分布幅度基本相同 |
| 四 candidate 归约 | 普通 batch mean | branch mean 后 0.5/1⁄6/1⁄6/1⁄6 | GT 分支权重从约25%升至50% |
| 主时间特征 | 固定 order、current=0.1 | `main_time_source=order, current=0.1, raw` | B0 没有改成真实秒时钟 |
| 点坐标归一化 | anchor rigid transform | 同 anchor rigid transform，按 ID 直接映射 | 几何等价，避免反复变换浮点误差 |
| B0分割标签框尺度 | 1.25 | 1.25 | 一致，B2独立使用1.0 |
| BC监督 | 当前/历史GT点到框距离 | 同定义、padding剔除 | 定义一致，实际系数有上述bug |
| 预测尺寸输入 | 使用当前 GT 尺寸 | 使用首帧尺寸 | 计划中的消除泄漏，应该保留 |
| 无GT目标历史过滤 | 3帧均小于1目标点时重采样 | v27允许该样本进入 | 训练可见性分布改变，需计数分析 |

`ct_history_training_mode=correlated_candidate` 容易误导阅读：它构造的是 CT motion/search 辅助历史，不会把 B0 observation 的 `ref_boxs` 替换成相关递归误差框。正式 observation 仍是 `candidate_trajectory_mode=independent`；mechanism 的预测历史也没有通过梯度训练 B0。

### Canonical 加权的影响应实验验证

原 baseline 中约25%样本是精确 GT 历史、约75%是扰动历史；v27按损失将两者变成50%/50%。每个view分4行组成batch16时，单行权重分别为1/8和1/24，按权重平方计算的名义有效样本量为12，而普通等权16行是16。不能将这看成“损失总和仍为1所以优化完全不变”。

此外 motion 正样本归约、带类别权重的 segmentation 归约都先在4行分支内部执行，再按分支加权，和原16行全局归约也不完全等价。它是计划明确选择的训练设计，并非代码报错；修复 BC 后应优先用匹配 reference 对照其效果，不宜先把 CfC 换成 GRU作为低分解释。

## 4. 闭环误差与空输入的训练覆盖

### 有梯度 B0 没见到真实漂移历史

`main.py` 明确将 observation 的 `ct_online_recursive_training=false`，插件 mechanism 另走预测递归；机制 B0 为 `eval()+no_grad()`。所以 B0 只从 GT 或小扰动样本更新，机制里观察到的数米漂移、空 crop、错误 foreground 历史，均未成为 B0 自身的闭环修正训练。

此设计遵守原计划 detach/独立观测学习的边界，却也限制了 B0 对递归误差的适应。建议修正确定错误后，先做共享 B0 的预测历史训练覆盖实验：所有臂使用同一 observation 采样协议、同更新预算，并从 epoch0 scratch 训练；不冻结、不加载其他实验 checkpoint。真实预测历史与人工相关漂移增强应分别标记，不能在评测时用 GT reseed 掩盖恢复能力。

### 空 current 的训练输出与部署输出不同

v27会保留空 current crop 样本。真实 sampler 复现：`ct_current_observation_valid=0`、当前1024个point-valid全0，但每个 padding 输入仍含 `[0,0,0,0.1,0.5]`。分割和BC loss剔除padding，但 feature encoder/BN/池化未统一排除这些无效槽；box/motion等目标仍监督真实GT。

`models/seqtrack3d.py:3831–3836` 只有在 eval 时才将空 current observation替换成local零框，即reference保持。训练时仍对网络预测的空current框回归GT。这是明确的训练/部署目标条件差异，但没有真实训练空current占比，不能宣布它解释了全部低分。

应先记录 current 空crop、历史空crop、GT不可见、背景可见、唯一有效点数的训练样本比例，再统一部署动作和监督条件：保留所有endpoint与合法extension恢复，但避免用部署时永远不会采用的空current动作主导框回归；无效点在特征/池化中的处理也需与真实稀疏点区分。

## 5. 继承的问题与不应回退的修复

- `solo_x = x.reshape(B*L,-1,chunk_size)` 在 CT 和原 SeqTrack 都存在。x布局为[B,C,L×N]，直接reshape没有先交换channel/frame轴，名义“每帧特征”实际混合了时序与通道。空间通路报告已有独立最小复现；它还使B2的按物理点位置配对特征不成立，修复价值不止B0分数。
- `center_label_prev = box_label_prev[:, :3]` 这个可疑切片在损失中未被使用；实际历史框中心目标使用 `ref_label[:, :, :3]`。没有证据将前者认定为当前训练错位根因。
- 当前帧GT用于标签和诊断，不用于当前crop选择；v27正式输入的first-frame尺寸与预测历史合同已通过GT替换不变性测试。不能为了接近历史分数重新引入当前GT尺寸或GT历史重置。
- N=1/2保留真实点、传播原始point ID、padding loss排除、完整endpoint计数等公共修复应保留。问题在额外目标/张量语义，不是这些基本有效性合同本身。

## 下一步的最小有序实验

1. 修正BC双计、历史prior来源语义、帧/通道布局及B2特征对齐；补充损失数值/梯度、按帧点特征、预测历史prior的合同验证。
2. 用100-step诊断确认CT五臂实际B0目标与模块边界，继续定位跨臂首次数值分叉；该工作不应再混用普通loss与transaction loss。
3. 在同场景同预算下先跑一个干净B0与匹配SeqTrack reference，检查逐轨迹漂移/当前目标漏出crop的时间点。BC修复不能在旧checkpoint上通过改配置“补回来”；需要新的scratch训练证据。
4. 基础行为确认后再验证B2/B3，保留现有结果作错误链条的工程证据。对 canonical权重、预测历史覆盖或更大扰动的改动逐项登记，避免一次更换多个设计后无法解释收益。
