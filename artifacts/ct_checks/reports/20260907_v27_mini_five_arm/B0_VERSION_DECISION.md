# v25 / v26 / v27 B0：版本判断与修复取舍

2026-09-08。本轮核对实际运行提交、原始TensorBoard事件、checkpoint审计记录、逐函数AST、真实resolved config，并参考SeqTrack3D、CXTrack、MBPTrack论文。未读取TrajTrack，未修改生产代码、配置、参考仓库或历史output；没有新的nuScenes训练结果。

**建议在当前v27工程树上修复B0，以v25/v26共有的SeqTrack观测核心为数学参照，保留v27正确的数据与模块合同。** 历史单次最高的v25是`049de82`，适合做回溯对照，但不足以认定“v25架构最好”。v26没有换成一个更弱的B0网络；v27新增了真实的输入/输出语义问题，还扩大了旧主干未完整适配的稀疏样本范围。直接继续当前v27或整仓回滚v25，都没有解决完整问题。

## 1. 版本成绩必须先对齐含义

| 实际版本 | 提交 | epoch60 Success / Precision | 参数训练 | 日志评价 | B0总更新 |
|---|---|---:|---|---|---:|
| v25，高分运行 | `049de82` | 50.690 / 59.280 | 8个mini_train场景 | 官方mini_val，106轨迹/2285帧 | 75,720 |
| v25，后续低分运行 | `b8222bb` | 29.870 / 30.398 | 同上 | 同上 | 75,720 |
| v26 | `b445ecd` | 26.903 / 25.601 | 同上 | 同上 | 75,720 |
| v27 | `8b8b8d9` | 17.952 / 15.553 | 6个mini训练场景 | 内部dev，12轨迹/321帧 | 63,420 |

前三行可在相同评价集合下描述成绩差别，但仅一次运行不足以证明架构因果。第四行同时换了训练与评价集合，不能将`26.90→17.95`叫作同协议下降8.95点。旧8场景训练包含本次dev，旧模型在dev上重评也不能作为独立泛化证据。共同官方mini_val适合检查输入/模型差异，训练数据不匹配的限制仍需保留。

本轮新增原始事件抽取结果如下，v26没有第5轮记录，不插值：

| 运行 | epoch5 S/P | epoch20 S/P | epoch60 S/P |
|---|---:|---:|---:|
| 高分v25 | 34.67 / 36.94 | 44.01 / 55.87 | 50.69 / 59.28 |
| 低分v25 | 20.79 / 20.58 | 32.93 / 34.81 | 29.87 / 30.40 |
| v26 | — | 26.41 / 25.16 | 26.90 / 25.60 |
| v27，独立dev口径 | 23.57 / 21.32 | 23.78 / 23.24 | 17.95 / 15.55 |

低分从训练早期已经出现；不是只在保存或加载最终checkpoint时才坏。当前五臂完整结束60轮、最终权重/Adam状态有限。这里的“崩坏”是递归跟踪失效，现有证据不支持数值NaN或未训练完成的解释。

原始来源：[历史运行与checkpoint重查](historical_b0_recheck.json)、[固定轮次曲线](b0_version_curves.json)、[曲线采集脚本](collect_b0_version_curves.py)。

## 2. v25与v26究竟改了什么

### 高分v25到低分v25

实际resolved config的差异是验证频率1→5、tag以及8个B1相关字段。纯B0关闭这些B1分支。PointNet、Transformer、sampler和points_utils对应函数AST没有变化；forward/loss中新增的是受关闭B1门控制的逻辑。优化器相关代码增加hash/梯度审计，没有改变Adam定义。

两次checkpoint记录同一个B0初始化hash、相同前100个observation输入指纹和相同最初view0损失，但首次更新后的B0参数hash不同。首次可见view0差异出现在step3，15.680490与15.677980。这将问题定位到训练数值路径或未被现有摘要覆盖的执行状态，但尚未定位具体梯度算子及其对最终成绩的影响。

**相同标量loss不等于相同梯度；微小参数差异也不能自动解释20多点的最终分差。** 下一次要保存真实逐层梯度和Adam前后变化，先同臂重复，再跨臂比较。不得继续将未证实参与本次B0计算的PointNet2 atomicAdd作为已找到的根因。

验证频率专项核查也排除了一个容易混淆的解释：三版均`num_sanity_val_steps=0`，首次参数分叉前尚未验证；训练/验证DataLoader有独立generator，验证使用确定性点采样，StepLR按epoch更新，没有best checkpoint回灌。正常validation的eval切换与旧mechanism遗漏eval是两件事。没有证据表明将验证频率改回1即可恢复高分。详见[验证频率与状态审计](b0_validation_schedule.md)。

### 低分v25到v26

逐函数比较结果：

- `configure_optimizers`、四view加权helper、`_finalize_observation_output`的AST完全相同。
- `training_step`只增加接受v3 batch schema。
- `forward`只在B1输出新增两个acquisition margin字段。
- `compute_loss`只在B1分支增加margin监督和诊断。
- PointNet和Transformer源文件的函数AST相同。
- 核心参数均为1024点、历史3帧、batch16、FP32、Adam 1e-4、StepLR(20,0.1)、相同中心/yaw/seg/BC权重及canonical权重。

因此，**v25/v26应视为同一条B0观测实现谱系，不能根据版本名将26.90分归因于“v26的B0架构比v25弱”。** v26的B1/B2升级不等于其损失进入纯B0。

详见[数学图与运行配置对照](b0_versions_math.md)、[可复现AST/config证据](b0_versions_math_evidence.json)。

## 3. v27新增变化：必要修复与真实退化入口混在一起

| 变化 | 应怎样处理 | 原因与代价 |
|---|---|---|
| 统一sampler，却让预测历史prior全程0/1 | 保留统一sampler，修来源语义 | 旧eval首query后为0.2/0.8；candidate0不代表历史框真实可靠。当前错误会强化对漂移框的信任 |
| 当前crop为空，eval强制reference | 修改成明确的观测/历史支持预测合同 | 旧代码可在历史有点时继续预测；新覆盖丢弃历史输出，训练却仍回归该网络动作。不能恢复XYZ求和判空 |
| N=1/2保留真实点、N=0 padding、point ID | 保留 | B1/B2恢复和物理证据计数需要这些合同。重复真实点可填槽，不应被当成多个物理点 |
| 不再拒绝三历史均缺目标点的样本 | 保留难例方向，补全前向及任务有效性 | 删除不可观测性的粗糙过滤有意义，但当前mask主要进入loss，未同步进入表示学习 |
| mini6/1/1、full350/150与内部17/18 | 保留 | 已注册的数据协议；改变数据量不是bug，参考运行必须匹配 |
| dataset实际长度生成更新预算 | 保留 | 不重新硬编码1262；60轮与75,720次更新在新数据量下不是同一预算 |
| mechanism完整eval/no_grad并恢复flags | 保留 | 修正Dropout状态，不冻结B0有梯度训练 |
| 原始ID、完整endpoint、有效时间、resume合同 | 保留 | 新增证据链和可复现实验的基础，整体回滚会破坏这些能力 |

实际父提交eval构造器与新sampler的稠密输入对照中，XYZ、时间、ref_boxs、candidate_bc、首帧尺寸及valid字段一致或仅有极小浮点误差，主要差异是后续query的prior。第1query prior相同，所以它不能解释所有第一步失跟。

同一组2000个框，新旧benchmark_compat逐帧S/P贡献完全相同，不能把分差简单归因于评价公式缩小。相同稠密batch、相同权重/BN/RNG的生产B0探针也没有发现大幅forward变化。真正值得优先处理的是上述来源、稀疏输入和动作语义。

详见[astra提交判定](ASTRA_COMMIT_B0_VERDICT.md)。

## 4. 三个版本共有的问题，不能因高分运行曾经存在就保留

### BC被计算两次

当前`models/seqtrack3d.py:5809`的`b0_transaction_loss = loss_total`保留Tensor别名；第7498行先原地向loss_total加BC，随后又向transaction加BC。普通observation四view都走这一条实际优化路径。

令A为其他B0项、C为加权BC项，声明目标每view是`A+C`，实际是`A+2C`。高分v25也执行该错误，所以它不是高/低v25差值的充分解释；但CT与声明公式/原reference不一致是已证实问题。

**修法：显式分别构造B0/B1/B2/B3 loss，BC只加一次，再合成总loss。** 不用bc_weight=0.5掩盖alias，不通过冻结BC分支处理。

### 帧/通道打包错误，继而破坏B2点特征身份

`x`是`[B,C,L*N]`，直接`reshape(B*L,C,N)`不能拆出各物理帧。正确布局是：

```python
solo_x = x.reshape(B, C, L, N).permute(0, 2, 1, 3).reshape(B * L, C, N)
```

原SeqTrack本地代码及v25/v26也有旧写法。当前“最后帧”64维特征实际由一组错误打包的box-distance通道产生，再绑定真实当前点ID；这一接口不满足B2要求的逐点对应。它不是整个B0完全看不到XYZ，而是该逐帧/逐点分支语义错位。

**此项在CT中应修，不宜为复现某次高分继续继承。** 原参考实现保留原样并明确口径；CT公共修复后的B0作为插件增益分母。布局修正后必须scratch，旧权重即使shape兼容也不代表输入语义兼容。

### 有效性不能只传入不用，也不能统一成一个mask

Transformer接收history valid mask却没有使用；history box loss也对缺失历史槽直接mean。PointNet/BN/pooling未完整消费point padding mask。v27加入稀疏样本后，这些旧缺口更值得处理。

建议分四个概念：

1. **history existence**：该历史时刻真实存在，约束历史box query及ref loss。
2. **point validity**：当前槽是不是实际采样点，约束点loss、pooling、attention和空帧前向。
3. **physical uniqueness**：重复槽是否来自同一物理点，约束证据计数、质量统计与B2 memory。
4. **prediction quality**：历史框/targetness的预测可靠程度，控制软prior与记忆权重。

真实历史帧即使没有点，预测框仍可提供运动信息；不存在的历史帧不能因为重复填了首帧而获得额外注意力/监督票数。全无效key需有显式空输出，不能全masked后softmax仍均匀聚合。B0对真实重复采样槽的训练权重与B2独立物理证据权重也应分别定义，避免一次改动把全部归约都换掉。

## 5. 为什么应优先修B0，而不是先换B1后端

当前B0 dev有309个非首帧endpoint：117帧base为空，231帧base没有目标点，260帧误差超过2m。11条轨迹曾首次跨过2m，其中8条在首次失跟时，全局可见目标点仍全部包含于base。

例如轨迹7第1query有8个目标点且全部在base，仍产生2.64m误差；第2query有25个目标点且全部在base，误差增至9.24m；之后才出现目标裁丢。这里是“B0对已有证据定位错误→递归锚点漂移→后续证据丢失”的真实链条。

这解释了修复优先级：扩大获取区域不能直接纠正所有有目标base上的B0失败。B1-only正式输出仍是observation，GRU/CfC当前分差又混有不同B0权重，不能替代B0修复。

但B0也不是下游全部低分的唯一原因。B2的旧presence硬门、动作半径频繁裁剪，以及Full未安装校准artifact仍需各自修复/补评。当前同状态raw proposal存在有益方向，有界动作却几乎消除了净收益；不能承诺只修B0就自动得到高分Full。

## 6. 推荐底座与论文设计

| 选择 | 好处 | 代价 | 本轮判断 |
|---|---|---|---|
| 直接回滚高分v25 | 有实际较高分checkpoint和旧输入参照 | 共同bug仍在；丢失v27的数据/ID/稀疏/恢复合同；同版本已有低分 | 用作回溯，不能直接认定为最终底座 |
| 改用v26 | 保留部分获取/证据升级 | B0数学图没有新的优势，无法解释或解决分叉 | 无需单独追求这一版本号 |
| 当前v27继续训练 | 工程与模块功能最全 | 明确B0语义错误仍在，反复长跑难以解释收益 | 先修复再训练 |
| **v27工程树＋修正的SeqTrack观测核心** | 保留模块化链条，修清公共训练/输入/身份合同 | 需要重新训练与共享B0验证，不能承诺旧分数 | **推荐正式基础** |

五个CT臂必须共享完全相同的修正B0定义和训练分布。原SeqTrack reference保留原网络/原loss归约，使用匹配场景、预算和评价器。reference→CT B0的差别单独披露；B1/B2/B3的贡献用CT B0作直接对照，避免把公共bug修复计作插件创新。

暂保留注册的canonical 0.5权重，先将实际loss修到声明公式；之后若要靠近原baseline，单独比较原batch归约，而不是只把四个权重改成0.25。按view先归约再平均，与整个batch的类别加权CE/移动行归约一般不等价。

当前训练主要是GT/小独立扰动历史，mechanism在eval/no_grad下生成递归状态，B0未从这些实际漂移状态学习。可以在公共错误修复后，引入同一B0 observation轨迹生成的预测历史训练样本或更贴近递归误差的扰动；所有CT臂共用来源策略，历史预测detach、当前GT仅作监督，所有模块仍从epoch0训练。该设计是新的训练消融，不是已证实需要立即替换全部observation训练的bug修复。

### 文献对本项目的具体启示

[SeqTrack3D](https://arxiv.org/html/2402.16249v1)本来已经有帧内/跨帧编码和历史box序列约束；其动机包括稀疏甚至无目标点时利用历史运动。论文将历史/当前框loss设为1:10以控制历史误差积累，并采用GT随机扰动训练。当前v25/v26/v27的ref/current中心0.2/2、角度1/10已对应这个比例，优先修mask和输入来源，而非盲目加大历史loss。论文描述与本地公开实现的具体骨干存在差别，本报告以实际代码为准。

[CXTrack](https://arxiv.org/html/2211.08542v1)对局部上下文与targetness传播的设计支持“保留少量目标线索”的方向；其消融讨论了过度依赖不准确mask导致的信息损失。[MBPTrack](https://arxiv.org/html/2303.05071v1)将几何与targetness传播分开并使用历史memory。它们给本项目的启示是：在正确的物理点特征上区分几何和身份信息，而不是继续强化已经漂移的硬框prior。这是针对CT的设计推断，论文不构成本项目已获收益的证据。

### 时空融合放哪里

若增加B0-ST，首选**FeaturePointNet第二层64维真实逐点特征之后、128槽压缩之前**。当前base点作query、过去三帧目标/上下文memory作key/value，加入相对几何、有效时间和质量，通过连续残差增强当前点特征，再进入原PointNet后段和SeqTrack decoder。

仅改返回给B2的64维特征不会增强B0主输出；仅在B2扩展点attention加模块也覆盖不到无extension的B0失败。现有memory构造有detach，B0-ST应共用几何选择逻辑但保留同forward历史特征梯度；B2/B3读取时继续detach。所有启用参数从头学习，无冻结、无旧checkpoint分阶段初始化。

当前空crop强制reference还会覆盖融合输出，若目标包含无观测运动延续，必须同步定义可部署动作。ST作为修复后独立可关的研究模块，先比较修正B0与同底座B0-ST，再决定是否进入所有模块臂。不将“再加Transformer”本身列为新贡献。详细接入见[时空融合设计](spatiotemporal_fusion_design.md)。

## 7. 最有效的下一步

1. **先做已有权重的输入诊断**：v27 B0同checkpoint、同集合下比较旧/来源正确prior及hold/允许历史预测的2×2策略。先同状态单步，再完整递归闭环。恢复旧网络输出只是诊断条件，未经重新训练验证不能直接作为正式部署策略。
2. **补共同官方mini_val结果**，保留dev与mini_val的不同含义；不在官方val选后端、epoch或阈值。历史权重只用于诊断，正式参考从头匹配6/1/1协议。
3. **实现公共B0修复**：独立loss组装、正确拆帧、source-aware prior、历史/点有效性及空观测训练推理一致。保留原始ID等必要合同。记录旧拒绝/新接纳样本比例及各任务损失贡献。
4. **做100-step共享B0核对**：同卡顺序B0重复两次、再B1-GRU一次，比较输入、forward、BN、gradient、Adam实际更新。先定位同臂重复误差范围；不要求凭hash猜具体算子。
5. **先跑匹配reference与修正B0的完整mini**，固定60轮和late-3，看首次失跟、少目标点分桶、空crop连续长度及实际S/P。若基础定位仍弱，再加点级ST或预测历史训练作为独立消融。短跑用于实现检查，不替代正式60轮。
6. B0达到可解释、可复核的水平后，再重跑修正后的五臂及各checkpoint的Full校准。完整nuScenes保持全部350 train_track训练、150官方val评测，阈值仅在训练内部固定子集拟合。

正式修复实验继续采用首帧尺寸、全endpoint评价、所有启用参数从epoch0学习的合同。历史高分说明这条任务路径有可行性，适合论文的底座还应满足输入正确、训练部署一致、模块对照可解释。

本轮新增的是版本审计、曲线与决策文档；上述生产修复及ST尚未实施。
