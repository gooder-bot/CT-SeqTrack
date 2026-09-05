# CT-SeqTrack v27：时间获取、身份条件证据与整体跟踪效用

v27 在共同修复的 SeqTrack 观测路径上训练模块化模型。它是待正式实验验证的方法实现，尚无 v27 涨分、SOTA 或时间/memory 因果收益结论。v24/v25/v26 配置、提交历史和输出保留为历史证据，不能初始化 v27。

## 1. 数据通路与所有权

```text
完整当前帧（原始 point ID）
  ├─ B0 原 crop → 1024 采样槽 → SeqTrack observation
  └─ B1 endpoint/tube + causal corridor
       → 减去整个 B0 raw crop，按原始 ID 合并来源
       → 768 novel prepool → 局部几何与身份条件 relation
       → 128 relation + 96 coverage + 32 exploration
       → extension-only cross-attention / targetness / vote
       → 有界实际动作 → B3 单一效用策略 → B0 host 提交递归状态

B0 当前唯一点特征 + 3×(8框内+4上下文)历史 memory ──detach──→ B2 context
```

B0 是唯一递归状态写入者；B1 提供获取先验，B2 提供新增测量候选，B3 决定是否接受实际有界动作。B1 不直接混入最终框。训练使用 observation-recursive 状态，机制流仅处理 canonical view0；B0 观测流四视图损失仍为 `0.5*L0+(L1+L2+L3)/6`。

启用参数全部从 epoch0 同时训练，无预训练初始化、冻结或分阶段解冻。`safe_seqtrack_auto_v1`、一个带 B0/B1/B2/B3 named groups 的 Adam、各组初始学习率 `1e-4` 保持不变。v27 batch schema 为 `ct_seqtrack.train.v4`；`ct_b0_steps_per_epoch=0` 使用当前数据集真实长度，关闭 recursive GT reseed。

B0/B1 到 B2 的特征、框、时间与尺寸输入 detach；B3 的所有上游输入再 detach。mechanism 中重复 B0 前向使用 no-grad/BN 隔离；这些是梯度与统计量边界，不是冻结模型参数。

机制流构造、取batch、结束检查与训练事务均恢复调用方的Python/NumPy/torch全局RNG；DataLoader自己的generator独立推进。各插件使用命名初始化域；GRU/CfC公共projection、context与输出head按层名初始化，后端参数数量差异不能移动公共decoder的初始权重。

v27机制sampler逐epoch完整遍历所有合法轨迹的frame1至末帧，每个endpoint一次；slot负载不等时继续输出剩余slot，允许尾部batch只含1个slot。机制tick超过观测步数时，`ct_seqtrack.train.v4`按固定比例将多个有序子batch分配给同一观测事务；逐tick提交因果状态，按endpoint数量加权插件loss，再进行一次共同Adam更新。不得以丢弃机制尾部或增加B0步数满足预算。历史schema继续使用原有固定slot/drop-tail行为。

## 2. 公共点身份与稀疏输入修复

原始身份为 `(physical frame uid, raw point index)`；裁剪与坐标变换只传播 ID，多个 source 的同一 ID 合并一次。source bitmask 保持 endpoint=1、tube=2、corridor=4，padding ID=-1 且 valid=false。不能继续用浮点 XYZ 键判断新测量。

空间 extension 仍是扩展 support 减去**整个 B0 raw crop**。B0 crop 内未采样点不纳入默认 v27；不得把重复处理已见点称为 novel evidence。1024 是 B0 输入槽预算，重复上采样槽不代表独立测量。另存 unique mask，供 B2 context、memory 和诊断使用。

公共采样语义：0 点零填充且全 invalid；1/2 点重复真实测量填足输入槽；大于2点沿用固定预算采样。五个 CT 臂和外部参考使用相同公共修复。当前 B0 无点时使用明确 observation fallback；若 extension 有效，仍允许恢复链运行。所有 noninitial endpoint 必须有诊断行，包括空点、fallback 和无法执行动作的行。

FPS 显式排除已选索引，以原始 ID 稳定处理并列。合法同 XY 的不同点不会重复选择同一索引，候选不足时用无效 padding，不复制成额外证据。

## 3. B1：GRU/CfC 共同的获取任务

GRU 与 CfC 共用物理历史、时间、运动先验和损失定义；五臂中的 B1-GRU/B1-CfC 是独立 scratch 对照，集成臂默认 GRU。比较后端须固定共同 endpoint/历史，报告 learned-vs-CV 误差、不确定性校准、支持域召回/背景/体积及计算成本。不能把不同 B0 轨迹上的总跟踪分差当成后端增益。

统计不确定性与 acquisition margin 分路。margin 读取 detached temporal context 和17维因果获取信息：三帧观测质量、质量有效位、递归年龄、首帧尺寸与时间间隔。两个轴的范围仍为平行 `[2,6]m`、垂直 `[1,3]m`，末层 bias=-4.6，q=0.90 pinball loss 权重0.05。GT 支持域网格目标只进入 loss/诊断，不能替换在线预测或裁剪。

prepass 与训练主分支共享输入构造和字段合同；margin 必须真实影响在线 support。无效 B1 使用有界 CV/fixed margin fallback。统计 sigma 不直接充当 support 尺寸。

## 4. B2：单层局部几何和身份条件

保持768→256预算，不增加第二个 backbone、长 memory 或额外 decoder。输入 extension encoder、B1 geometry encoder 与 source embedding 相加为64d。

选点前新增一层 `131→64→64` EdgeConv 残差，输入为 `[h_i,h_j-h_i,(p_j-p_i)/r]`；k=16，query 分块计算邻域。首帧尺寸明确按 nuScenes `wlh` 转换为 `(l,w,h)`，使用统一规则：

```text
r_local = clip(0.5*min(l,w), 0.25, 1.0) m
```

邻域不足时只使用合法近邻；没有近邻则保留自身特征，禁止补远处点。该局部半径不会随 Truck/Trailer/Bus 的车长无限扩张；Pedestrian 的半径较小，但邻域几何本身不保证实例身份。

memory 维持三帧36 slots，预测框内8个、近邻上下文4个。修正局部 x=length、y=width；按原始 ID 去重，invalid 点不能成为 token。metadata 包括局部位置、物理 age、相对 yaw、role 和帧身份。

relation context 分 current base、memory foreground、memory context 三路 mean/max，`384→64`；再拼接 extension64、context64、两个 role cosine 与两个 role-valid，共132d，`132→64→1`。只有 predicted history role 可作为输入，当前 GT 身份仅用于监督。

relation top128、XY coverage96、stateless exploration32 之后，使用原有一次 extension-query attention，base/memory 仅为 K/V。relation 用于 acquisition，不再作为最终 vote 的第二个乘法门。最终 vote 权重仅为 refined targetness sigmoid；它是判别分数，不能视为已校准概率。

vote head 为 `68→64→2`：enriched64、`log(l,w,h)` 与 `log(r_vote)`。跨类别投票半径：

```text
r_vote = max(4, 0.5*sqrt(l*l+w*w)+0.5) m
vote_xy = extension_xy + r_vote*tanh(offset)
```

Car 通常仍使用4m；长车头尾测量的中心目标不再被固定±4m截断。候选只修正 XY，z/yaw 保持 observation。vote 半径不是最终动作半径。

K=3 Huber consensus 保留。除 covariance、inlier ratio、模式间 margin 外，输出 top-mode 的有效质量、唯一点数量、平均 targetness 和平均身份 margin。这四项同时进入134d selected-presence head 和 detached B3 输入；consistency 只表示几何共识，不能单独当作身份置信度。

## 5. B2 监督

B2 单独采用 `ct_evidence_label_box_scale=1.0`，与 B0 的 `bb_scale=1.25` 分开。relation 标注768 prepool，targetness标注256 selected；presence 的部署语义是 **selected 中含目标测量**。prepool 是否含目标只作获取诊断。base-presence 使用当前 B0 唯一有效点。

默认总损失：

```text
L_B2 = .25*L_relation + .2*L_targetness + L_vote + L_raw
       + .1*L_base_presence + .1*L_selected_presence
```

relation 与 targetness 各自按对应 population 累计类别平衡；计数器进入 checkpoint，缺类时保留初始中性权重。presence 使用不加类别权重的 BCE。vote 仅对真实目标点、raw 仅对 selected target-bearing 行施加米单位 SmoothL1；纯背景/空行不获得 GT 中心梯度。

## 6. B3：以整体跟踪效用选择有界动作

Full−B3 使用 `bounded_always`：结构有效时执行与 B3 相同的有界候选，不设置 learned-presence 硬门。Full 使用 `selective`；缺失或错配 artifact 时精确 observation fallback。训练仍统一 observation-recursive。

实际动作半径保持 `min(.5+.5*dt,2.0)m`。B3 显式读取该动作的归一化 dx/dy、raw/radius 与截断比例，监督针对**实际有界动作**。主效用为预测 Success/Precision 单帧贡献增益的平均值：

```text
q = .5*(expected_success_gain + expected_precision_gain)
apply = structural_valid and (q > threshold)
```

presence 是连续输入，不是独立门。helpful/harmful heads 作为辅助监督保留。H3 shadow 每2步最多1个有效槽，选取不依赖当前 presence 阈值。`ct_router_weight=.2`。

v27 policy artifact schema 为 `ct_seqtrack.action_calibration.v27`。calibration 先由静态 endpoint 选出少量候选策略，再真实闭环运行，按整体跟踪效用选策略；显式包含 always/never。锁定策略后 dev 仅作诊断，不设置旧 v26 risk-promotion 阈值。artifact 绑定 checkpoint、resolved config、代码、scene manifest、score 与 metric 定义；不能跨 checkpoint 或类别复用。

主训练/比较采用 `benchmark_compat` 指标，保持公开基线定义；同时输出 `geometry_exact` 作为几何复核，二者不得混表。历史几何兼容偏差不能在同一对照中只修正其中一臂。

## 7. 场景、类别与五臂

唯一场景用途实现是 `utils/v27_protocol.py`；按 scene ID 与 split seed42 的 SHA256 稳定分配，不受训练 RNG 影响。

| 数据 | 参数训练 | 阈值拟合 | 锁定策略诊断 | 最终评估 |
|---|---:|---:|---:|---:|
| mini | mini_train 中6 scenes | 其余1 scene | 其余1 scene | 官方 mini_val 2 scenes |
| full | train_track 全350 scenes | train_track 内17 scenes | train_track 内另18 scenes | 官方 val 150 scenes |

mini 的6/1/1互斥，B0和mechanism均不得接触另外2个scene。full 的17/18是**参与过参数训练的内部拟合/诊断子集**，350个训练scene全部使用；artifact 记录 `parameter_training_overlap=true`，不能称作未见数据校准或分布无关风险保证。官方评估scene不参与训练与阈值选择。

先运行 mini Car 的五臂，再进入 full。full 主类别为 **Car、Pedestrian、Truck、Trailer、Bus**，每类每臂独立 scratch、seed42、60 epoch；复用同一组 full YAML，以 `--category_name` 设置类别，不混用类别 checkpoint。

| 臂 | mini YAML | 评测输出 |
|---|---|---|
| B0 | `27_b0.yaml` | observation |
| B1-GRU | `27_b1_gru.yaml` | observation |
| B1-CfC | `27_b1_cfc.yaml` | observation |
| Full−B3 | `27_full_minus_b3.yaml` | bounded_always |
| Full | `27_full.yaml` | calibrated selective |

对应 full 配置加 `_nuscenes_full`。`cfgs/27_seqtrack_reference.yaml` 与 `cfgs/27_seqtrack_reference_nuscenes_full.yaml` 是另外的 SeqTrack 架构参考：independent candidate、普通原始 loss、所有 CT 模块关闭，只共享公共修复与场景用途；不调用 CT variant 配置。它与 CT B0 四视图加权目标不同，结果分别说明。

单入口仍为 `main.py`。所有臂禁止 `--init_checkpoint`；同一运行 epoch-boundary resume 或评估才可传 `--checkpoint`。正式比较固定 final=60 与 late-3=58/59/60，每个 Full checkpoint 单独拟合 policy。

`tools/run_ct_v27_matrix.py --stage mini|full --path DATA_ROOT --output NEW_ARTIFACT_DIRECTORY` 在训练服务器生成配置快照、六运行/三十运行清单及 final/late-3 待执行命令。默认不启动进程；显式 `--execute` 仅顺序启动训练。每类独立配置快照同时供训练、校准与评估使用，避免校准意外回到默认 Car。Full 的三个 late-3 checkpoint 各自拟合策略；late-3 为三个独立 epoch 指标的算术平均，不是选择最好 checkpoint 或合并权重。

## 8. 必报验证与局限

本地验证覆盖原始ID、0/1/2点、FPS、wlh、padding/置换、梯度所有权、长车vote可达性、scene分区、policy复核。真实 nuScenes batch、CUDA性能、B0更新一致性和resume等价仍须服务器验证，CPU单测不能替代。

报告完整漏斗、每类 local neighbor 数与vote saturation、presence、raw/bounded/selected的配对收益、全帧S/P、latency/memory。新模块不是已经验证的涨分结果。半径公式和结构默认值是预先固定的研究起点，不能在评估集反复调参。

文献定位（核对原论文标题与方法，2026-09-05）：

- [CXTrack: Improving 3D Point Cloud Tracking with Contextual Information](https://arxiv.org/abs/2211.08542)：target-centric transformer 传播目标线索，X-RPN 使用局部 transformer 聚合与 center embedding，对应目标条件和局部上下文的参考依据。
- [MBPTrack: Improving 3D Point Cloud Tracking with Memory Networks and Box Priors](https://arxiv.org/abs/2303.05071)：DeFPM 将几何与 targetness 传播分为共享 attention map 的两路；历史 memory 与首帧尺寸先验提供跨帧和跨尺寸建模依据。
- [StreamTrack — Modeling Continuous Motion for 3D Point Cloud Object Tracking](https://arxiv.org/abs/2303.07605)：memory bank 与混合 attention 提供跨帧关系和局部几何思路；contrastive sequence enhancement 通过加入干扰轨迹及辅助对比损失区分错误目标。本轮仅借鉴问题定位，不额外叠加其完整对比学习结构。
- [HVTrack — 3D Single-object Tracking in Point Clouds with High Temporal Variation](https://arxiv.org/abs/2408.02049)：relative-pose-aware memory、base-expansion cross-attention 与 contextual point guided self-attention 分别应对形变、扩大搜索域中的同类干扰和背景噪声，是扩张搜索与 memory 上下文的关键近邻。
- [Hasani et al., Closed-form continuous-time neural networks, Nature Machine Intelligence 4, 992–1003 (2022)](https://www.nature.com/articles/s42256-022-00556-7)：CfC 使用显式时间依赖的门控与闭式近似以避免数值 ODE 求解；正式 DOI 为 `10.1038/s42256-022-00556-7`。这是 B1 连续时间后端的来源，其在本项目中相对 GRU 的收益仍需共同输入和任务下的配对实验检验。

v27 的候选研究贡献在于固定新增测量预算下的时间获取、身份判别和实际动作效用联结，需要匹配实验支持。
