# v25、v26、v27 B0 数学图与优化定义比较

日期：2026-09-08。只读比较实际运行提交 `049de82`（高分v25）、`b8222bb`（低分v25）、`b445ecd`（v26）、`8b8b8d9`（v27）。本报告聚焦有梯度的B0路径，避免把插件源码增加或配置字段增加直接当成B0模型变化。未修改生产代码、配置、output或参考仓库。

## 最重要的新结论

**v25→v26没有发现纯B0核心数学图、B0损失公式或Adam/StepLR定义的改变。不能把v26称为比v25更弱的一种B0网络。** 两者的低分差异仍需要从实际训练轨迹、数据/运行状态与同集合闭环分析，不能靠回滚插件代码解释。

这不是只看提交说明：本次按Python AST逐函数比较，读取每次真实 `run_provenance.json.resolved_config`。可复现证据：[脚本](compare_b0_version_math.py)、[完整JSON](b0_versions_math_evidence.json)。AST相同能排除语句变化，但不能保证整个运行环境、CUDA执行或训练轨迹逐位相同。

## 1. 高分v25→低分v25：不能从所改B1公式推出B0目标变坏

两次实际运行配置差异只有：

- 验证频率 `1→5`；
- tag；
- 8个B1后端/不确定性相关字段，包括GRU/CfC与aux NLL。

两次纯B0均关闭B1/B2/B3。B1 NLL、tail loss等分支不会进入普通observation目标。`configure_optimizers`新增的内容只是记录Adam状态hash，Adam参数/调度定义没有变化；`on_before_optimizer_step`和`on_train_batch_end`新增梯度最大值、状态hash等审计。

`models/backbone/pointnet.py`与`models/attn/Models.py`没有变化。sampler和points_utils的函数AST也没有变化。

更严格地说，提交并非“只修改插件文件”：`forward/compute_loss`改动位于关闭的B1分支，B0优化器、step hooks还增加审计hash和梯度日志，main和配置增加后端字段及校准校验。但未发现普通B0 observation的可执行前向、目标或Adam数值定义被改写。不能把新增审计代码与新增B0目标等同。

因此：

- “修改B1 NLL后，B0损失被该NLL直接训练坏了”不符合当前纯B0调用路径。
- 已有checkpoint证据中第一次更新就出现hash分叉，而验证频率的实际执行差异尚未发生。验证频率不能解释首次分叉的起点，但其后实际运行是否完全隔离仍需逐阶段核查。
- 不能据相同seed/输入指纹宣称过程确定，也不能只看到hash不同就认定这解释了20多点差距。

## 2. 低分v25→v26：逐方法差异非常具体

| 函数/组件 | `b8222bb→b445ecd` 的实际变化 | 纯B0是否改变 |
|---|---|---|
| `FeaturePointNet`、`SegPointNet`、`MiniPointNet` | 整个pointnet.py的函数AST相同 | 否 |
| `Seq2SeqFormer`、Encoder/Decoder | attn/Models.py的函数AST相同 | 否 |
| `SEQTRACK3D.forward` | 仅在B1输出中新增main/aux acquisition margin两个字段 | B1关闭时否 |
| `compute_loss` | 仅在B1分支新增margin pinball及其日志/事务校验项 | B1关闭时否 |
| `configure_optimizers` | 函数AST完全相同 | 否 |
| `_ct_candidate_weighted_observation_loss` | 函数AST完全相同 | 否 |
| `_finalize_observation_output` | 函数AST完全相同 | 否 |
| `training_step` | 唯一变化是接受 `ct_seqtrack.train.v3` schema | 不改变已存在的observation执行 |
| `__init__` | 增加B1 margin、B2 relation/corridor/consensus、B3字段及构造参数 | 纯B0核心层不变 |
| `motion_processing_mf` | 新增corridor及768 prepool，位于插件获取路径；增加诊断字段 | observation_only核心采样路径没有对应结构变化 |
| `crop_pc_axis_aligned` | 原strict membership六个不等式提取成公共函数 | 真实nuScenes Box的数学判定相同；没有新增闭区间或wlh轴改写 |

v26配置的 `ct_memory_mode: real`、`ct_relation_aware_sampling: true`并不意味着B0使用了memory或relation。必须结合 `ct_variant=b0` 的最终开关和observation临时关闭插件判断。未发现这些插件参数通过loss或特征梯度注入B0的证据。

真实resolved config里B0共用：

```text
batch_size=16; point_sample_size=1024; FP32
lr=1e-4; StepLR(step_size=20, gamma=0.1)
seg_weight=0.1; center_weight=2; angle_weight=10
motion_cls_seg_weight=0.1; ref_center_weight=0.2; ref_angle_weight=1
bc_weight=1; box_aware=true
ct_runtime_protocol=safe_seqtrack_auto_v1
ct_b0_loss_reduction=candidate_weighted
candidate weights=[0.5,1/6,1/6,1/6]
```

验证频率从5变2，最后三轮checkpoint保留策略增加；它们改变记录/执行时机，不能被叫作B0网络结构退化。

## 3. v27才新增了影响B0语义的公共路径

v27 PointNet/Transformer源文件仍不变，但改了它们接收的数据、loss有效性和推理输出策略。这些属于真实B0变化。

| 内容 | v25/v26 | v27 | 建议 |
|---|---|---|---|
| 历史预测prior | 首次查询后0.2/0.8 | candidate0导致全程0/1 | 修正来源语义；保留统一构造器 |
| 当前crop为空 | 未必阻断历史预测 | eval强制reference | 拆分观测和历史运动模式；同checkpoint先做敏感性诊断 |
| 历史目标观测不足 | observation采样拒绝并retry | 接受 | 保留困难样本方向，补齐可观测性与mask |
| N=1/2 | 清零 | 保留真实点并重复 | 保留；补齐unique有效性 |
| seg/BC loss | 所有槽归约 | 排除padding | 保留数学修复，明确各任务归约 |
| 参数训练集合 | 8个mini_train | 6个mini训练子集 | 按既定协议保留，比较时匹配reference |
| B0每epoch更新 | 1262 | 1057 | 使用实际dataset预算，不恢复硬编码 |
| 日志验证集合 | 官方mini_val | 内部dev | 显式命名并补官方mini_val，不能直接相减 |

已完成的真实CPU探针证明：相同稠密全有效batch，当前代码仅切换v27开关，21个共同forward输出完全相同；CE归约产生约 `6.25e-8` 的整体相对梯度差。该探针不是完整旧提交训练重放，尤其 `regularize_pc` 的改动不受v27开关控制。

## 4. 共同的B0训练目标到底是什么

每个view的主体目标可写为：

```text
A = 0.1*seg_CE + 0.1*motion_cls_CE
    + 2*(coarse_center + final_center + moving_motion_center)
    + 10*(coarse_sin_yaw + final_sin_yaw + moving_motion_sin_yaw)
    + 0.2*past_ref_center + 1*past_ref_sin_yaw
C = bc_weight * box_cloud_SmoothL1
```

其中motion regression在开启motion class时按moving行归约；当前所有正式B0默认 `use_motion_cls=True`。中心/yaw loss的具体SmoothL1与每个head的坐标合同均沿用该路径。

过去框监督的center/yaw权重为0.2/1，当前框为2/10，已经保留了每种分量的1:10比例，与SeqTrack对过去/当前预测的相对权重设计一致。历史mask缺失是语义问题，不能据此泛化成“历史loss太弱，应继续加大权重”；先修有效性，再对新增时空模块是否需要辅助监督作独立实验。

CT事务alias使实际每view训练目标成为 `A+2*C`；四view归约后是：

```text
0.5*(A0+2*C0) + ((A1+2*C1)+(A2+2*C2)+(A3+2*C3))/6
```

这在高分v25、低分v25、v26、v27中都存在。原普通reference目标是单次BC。必须修复事务构造，而不是配置bc_weight=0.5补偿。该错误解释了“声明的公式和实际优化不同”，不能独立解释版本之间的分差。

**四view各0.25也不自动等同于原SeqTrack的普通batch loss。** 当前先按view分别计算class-weighted CE、moving行均值，再平均；普通batch CE的类别权重分母和moving样本分母是在整个batch计算。各view的目标点/移动行数量不同时，两种归约不同。若要原baseline训练语义，应提供显式 `reference_batch_reduction`，不能只把四个权重改为0.25就宣称完全复现。

canonical权重0.5扩大了准确GT历史view的梯度占比，有可能减少对扰动历史的适应，但高分v25也采用它。建议在公共错误修复后单独比较，不把它当作已证明的根因。

## 5. 不能只修history attention而忽略历史框监督

`compute_loss`使用全部 `updated_ref_boxs` 和 `box_label_prev` 计算mean SmoothL1，没有按历史是否真实存在的mask归约。此问题贯穿v25/v26/v27。

应区分两个维度：

1. **该历史时刻是否存在**：短轨迹前缀用第一帧填充的虚构时间槽，不应作为额外真实历史监督，也不应产生额外attention票数。
2. **该真实历史帧是否有点**：真实帧空点时，历史框仍可能由递归状态提供；框监督是否成立应依据任务信息来源，而非一律按point_count=0删除。

因此建议历史frame mask进入过去框loss；point mask进入点级loss/聚合；unique ID负责物理点数量与重复证据权重；历史置信控制预测prior和记忆可靠性。它们不能合并成一个0/1开关。

BC、mask和frame/channel布局修正会改变学习问题。对旧checkpoint仅做输入/输出敏感性分析；正式学习从头开始。不能在旧权重上改reshape后重评，再把下降解释为修复无效。

## 6. 哪个B0值得作为下一轮基础

应拆成三种用途，不选一个历史版本同时承担全部职责：

- **历史可恢复性参照**：`049de82`高分v25，有真实较高分checkpoint，适合复查相同输入、首个梯度/更新和同集合推理；但只有一次高分且有已知错误，不是“最优架构已证实”。
- **数学图参照**：v25/v26共有的B0核心，与现有SeqTrack主干衔接稳定。v26没有显示出新的纯B0数学优势，也没有证据表明其插件新增使纯B0更差。
- **正式下一轮实现基础**：当前v27工程树，保留point ID、实际数据预算、epoch-boundary恢复、所有endpoint与公共构造器，修正hard prior、empty策略、BC alias、帧/通道布局与mask。把结构增强（ST、soft segmentation或新tokenizer）与公共修复区分开做消融。

不建议整仓回滚v25：会丢失B2依赖的点身份与稀疏合同，撤回采样/恢复修复，并且并不能解释为什么相同v25 B0目标曾同时高分和低分。也不建议直接给当前坏链条追加一个Transformer后重跑五臂。

## 验证边界

本报告增加静态AST和真实配置可复现证据，没有新的GPU训练结果。历史首次更新分叉仍需逐层gradient、BN、Adam step前后tensor定位；不引用未执行的PointNet2 atomicAdd作为已知原因，不作环境审计。
