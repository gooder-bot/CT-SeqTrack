# astra重构是否是本轮B0崩坏的起点

2026-09-08。审计 `a43a8ee^ → a43a8ee`，并核对五臂实际提交`8b8b8d9`。本次只读生产源码、git和原始输出，新增CPU实际模型/函数的合成复现。

## 判断

**astra重构确实改了B0；不能将这次提交描述为“只改插件、B0保持不变”。** 它更换了B0训练集合、部分样本接纳/点损失规则、推理输入构造、空current输出规则和验证集合。

但“历史B0第一次低分从该提交才开始”不成立：重构前v26 B0已为26.903/25.601；更早v25曾从50.690/59.280变为29.870/30.398。当前v27的17.952/15.553来自另一个dev集合，不能与旧mini_val分数直接减法比较。

准确的定位是：**这次提交是本轮五臂共同运行条件变化的主要来源，其中有新增语义错误和适配缺口；现有证据尚不能量化它们各自造成了多少掉分。** 新CPU对照没有发现稠密B0网络被重写或大幅坐标错换，最直接的新问题是历史prior与空/稀疏观测合同。

## 1. 为什么名为B0也受影响

`27_b0.yaml`仅关闭`ct_enable_b1/b2/b3`，仍继承`27_formal_base.yaml`的`ct_enable_v27: true`。公共开关还控制场景筛选、sampler空历史接纳、B0点级损失、推理构造/评价和空current输出覆盖。

所以关闭插件不会恢复旧B0。这是五臂共同低分的可能入口，不需要假设CfC、GRU、B2、B3各自独立崩坏。

## 2. 提交真正改变的B0行为

| 改动 | 提交前 | 提交后 | 判断 |
|---|---|---|---|
| 训练集合 | 8个mini_train场景，5051帧 | 6场景，4231帧 | 原计划要求的协议变化，真实改变学习数据 |
| B0每epoch更新 | 1262 | 1057 | 总更新75720→63420，减少16.24%；不是中断 |
| 验证集合 | mini_val：106轨迹/2285帧 | 内部dev：12轨迹/321帧 | 分母、场景难度改变 |
| 推理历史prior | 首query后0.2/0.8 | 全程0/1 | **新增来源语义回归** |
| 空current输出 | 全历史+当前XYZ判空，否则可网络推理 | current raw crop为空就强制reference | **覆盖历史预测能力，训练/部署动作不一致** |
| 1/2点采样 | 清成零输入 | 重复真实点到固定槽 | 应保留的修复，但改变稀疏分布 |
| 历史目标观测不足 | assert后retry换样本 | v27接纳 | 难例增加，前向mask适配不足 |
| seg/BC损失 | 包含padding槽 | 按有效槽归约 | 修复loss，但改变空槽batch权重 |
| crop数值路径 | support→world→anchor往返 | 原始ID直接到anchor | 几何对照一致，仅小数值差 |
| mechanism读取B0 | no_grad+BN隔离 | 完整eval+no_grad再恢复flags | 正确修复Dropout，不冻结B0训练 |
| 主干层、中心/yaw目标、四view权重、Adam | 原有定义 | 主要定义保持 | 未发现稠密主干重新设计 |

训练6/1/1是原计划要求，不能称为意外bug；但reference需要相同协议。YAML里的`val_split: mini_val`不能代表实际训练验证集合，`main.py:828`会按dev角色重新构造dataset。

## 3. 新问题一：预测历史被赋予GT式硬prior

旧`base_model.py`在`frame_id != 1`时将历史框内/外prior软为0.8/0.2。新推理走`v27_input.py`，固定candidate0；训练sampler却仅对candidate非0执行软化。canonical view被错误等同于GT式历史。

与训练组合后的错配是：有梯度训练中硬prior主要配GT历史，推理中硬prior配可能漂移的预测历史。函数统一不代表来源语义统一。

**父提交真实输入构造器对照确认：** 第1query新旧prior相同；第2/3/8query其余稠密核心输入相同，历史prior由0.2/0.8变成0/1。所以它可放大后续递归错误，但**不能解释第1query就发生的全部失跟**。

应保留公共sampler，按初始化/预测/无效历史来源构造prior。先用已有checkpoint评估恢复软prior的敏感性，再统一新训练规则。

## 4. 新问题二：空current覆盖历史分支预测

`seqtrack3d.py:3831`在eval对无current点的行把Transformer预测置零局部偏移，相当于reference。旧代码是全部历史和当前XYZ总和判空；旧判空本身不正确，但“current为空、历史有点”时通常仍能利用历史。

新条件更强：只要current crop为空，历史分支预测的位移也丢弃，训练模式却仍用网络输出计算框回归loss。

当前B0有117/309帧base全空，其中70帧全局仍有目标点。一部分是预测锚点漂移后的局部裁丢；停在reference又让后续crop维持错位。插件可能恢复这些帧，但旧presence门和较小动作半径仍是额外限制。

“空输入返回reference”是原计划明确选择的策略，属于需要重评的设计。应区分当前观测定位与历史支持的无观测预测，使训练和部署一致，不能简单恢复XYZ求和判空。

## 5. 新问题三：稀疏接纳扩大，旧主干未同步适配

保留N=1/2、取消三历史均无目标时的拒绝、增加point mask，能保留真实难例。但Transformer没有消费历史valid mask，PointNet BN/pooling也没有完整消费point valid/unique mask。loss排除padding不等于forward不受padding影响。

真实sampler复现：旧分支拒绝的三历史全空、当前有点样本，v27接受，有效槽为`[0,0,0,1024]`，框目标正常存在。当前点可能足以支持定位，不能一律删除；空历史前向和不同监督任务需要分别适配。

本地没有nuScenes点云，日志未保存“旧规则会拒绝、新规则接纳”的训练样本比例。因此已经证实接纳变化及适配缺口，不能定量声称它解释全部掉分。

## 6. 新增受控证据排除了哪些过度归因

### 稠密B0前向

真实B0、同稠密全valid batch、同参数/BN/CPU RNG，切换v27开关：21个共同输出Tensor逐位相同，observation差0，四view总loss均12.7884817123。新CE归约引起小浮点梯度差：相对L2为6.25e-8，最大绝对差2.16e-6。

这不是父提交完整训练重放，也不保证CUDA长期轨迹相同。但没有证据把它描述为“大幅改写稠密B0前向”。尤其`regularize_pc`是无flag公共改动，新代码关闭v27并不等于完整回滚。

### 坐标、时间和box-aware输入

父提交真实eval构造函数与v27真实sampler对比query1/2/3/8，包含非零大坐标、不同历史yaw及±π跨界：XYZ/time/ref_boxs/candidate_bc/bbox_size/valid_mask/Δt相同或在很小数值误差内，主要稠密差异是prior。直接canonical crop与旧变换保留同一ID集合，没有发现新anchor错换或新的GT输入混入。

### 指标计算

父提交原`estimateOverlap/estimateAccuracy`与新benchmark_compat对比2000组相同z-up 3D框：IoU最大差2.13e-14、距离差0、逐帧S/P贡献差0。不能将低分简单归因于评价公式整体缩小；实际评价集合变化仍然很大。

### 不是astra首次引入的旧问题

BC事务双计来自`25586b27`，高分v25也执行；`solo_x`帧/通道错排和Transformer忽略历史mask也早已存在，astra没有修改对应主干行。这些应修复，可与新增稀疏样本发生交互，但不是该提交新引入的独立退化开关。

`a43a8ee → 8b8b8d9`后继修复只涉及早期/成熟slot诊断字段一致性及测试，没有重写B0网络、损失或优化器。

## 7. 最小因果定位顺序

先不加入新ST模块，也不整体回滚：

1. **同一v27 B0 checkpoint、同一评测集合做2×2推理诊断**：当前/来源正确的prior × 当前hold/允许可靠历史支持的空观测预测。所有帧参与评价，记录S/P、首次失跟、空crop长度与重获目标帧数。
2. 同状态单步对照定位直接动作差异；完整闭环检验后续crop变化。单步改善不能代替闭环效果。
3. 补共同官方mini_val评价。旧8场景训练包含本次dev，旧权重dev分数不能作独立泛化比较。
4. 匹配6场景和更新预算，统计旧拒绝/新接纳样本及其梯度贡献；正确接入mask，修正BC目标，再验证训练分布改动。
5. 针对已知首次更新分叉，记录同输入梯度、BN和Adam输入/输出；先同臂重复再跨臂，不以未证明的CUDA算子猜测收尾。
6. 修改后的正式训练全部scratch、所有启用参数学习；B0通过验证后再做ST与五臂。

现有证据足以说明这次重构必须修正B0合同，但不足以承诺撤销某一行就恢复历史53分。完整nuScenes仍保持350 train_track/150官方val协议。

## 证据文件

- [历史原始日志与checkpoint重查](historical_b0_recheck.json)
- [训练专项diff](astra_b0_training_diff.md)、[实际B0探针](astra_b0_training_probe.json)
- [推理几何专项](astra_eval_geometry.md)、[父提交构造器对照](astra_eval_geometry_reproduction.json)
- [新旧指标对照](astra_metric_equivalence.json)
- [总体根因与模块设计](B0_ROOT_CAUSES_AND_FUSION.md)

以上为CPU合成输入/实际模型或函数验证，不是服务器nuScenes反事实训练。生产代码和历史output未修改。
