# CT-SeqTrack 正式实验协议

## v31 当前协议（2026-09-19）

用户批准的联合模型替代旧共享B0插件协议。本次 mini Car 三臂固定B0、Full-CfC、Full-GRU，物理GPU1/0/0；从零60轮、每个端点四分支等权暴露、batch16、workers4、seed42、FP32。B0允许被启用模块的同帧梯度共同优化。
Adam lr1e-4、betas(.5,.999)、eps1e-6、wd0、foreach/fused false；StepLR20轮×.1。每5轮完整闭环验证，固定final60、late3=58/59/60。Full无旧标定/接受门，直接在有效四假设中选最高质量；不得人为改分或强求三组不同。
窗口前10轮升至短3/长8；只在预定窗口起点用过去GT初始化，内部预测递归，推理只首帧GT。统一原始点ID、坐标和数据总体；同一状态分支相邻端点必须串行提交。
只支持相同v31身份的epoch边界续训，不加载旧版、别臂或工程权重初始化。首版不做完整BPTT，不加H3训练。CfC是主方法，GRU是用户要求的真实时间输入对照。
目标为Full的final60 S/P超过同版B0并超过历史mini40.473742/47.840262；late3及获取/模式/错误写入/恢复/成本一并报告。达到mini目标再推进full/KITTI及固定模块消融。目前尚无v31正式结果。
服务器本轮仅只读。启动与参数见 [v31运行说明](CTSEQTRACK_V31_MINI_LAUNCH.md)，下方旧版协议仅适用于旧版复现。

## v30 当前协议（2026-09-15）

本节覆盖下方旧版本排程。实施仅在本地完成；本次不连接服务器、不修改服务器文件、不启动或停止已有任务。
最新启动安排：GPU0 Full-GRU、GPU2 Full-CfC，GPU1并行B0有效BN反向重算false/true两个任务；
不把重跑全量测试、全套哈希或工程报告作为新增启动门。CUDA专项尚未执行的证据边界仍保留。
命令与日志查看见[mini四任务启动专页](CTSEQTRACK_V30_MINI_BN_AB_LAUNCH.md)。
两个Full均保持重算false；主三臂比较使用B0 false，B0 true是额外的执行方式对照。
两份B0只改变实验名称与`ct_b0_masked_bn_recompute`，不改变有效性监督、BN统计公式或训练超参数。
同卡并行的耗时受资源竞争影响，不作为两执行方式独立速度差异的证据。
三臂统一 scratch60、batch16、workers4、FP32、strict deterministic，Adam
lr=1e-4、betas=(0.5,0.999)、eps=1e-6、weight_decay=0、foreach/fused=false；StepLR 每20轮×0.1。
每5轮验证，保留058/059/060；final 固定60，late-3为58/59/60算术平均，Full每个权重独立拟合策略。

**进入 nuScenes full 和 KITTI 的条件只有：至少一个预登记 Full 的 final60 S、P 都严格超过同版本 B0。**
late-3只报告，不设额外门槛；不要求额外 seed 前置实验。通过后仍比较 B0、Full-CfC、Full-GRU 三臂。
`tools/summarize_ct_v30_mini.py`读取完整官方测试端点报告，核验身份、人口和指标口径，输出结果而不启动训练。

三个实验臂共享新的 B0：有效点 mask 贯穿损失、BN、池化及注意力，真实速度动静标签，概率加权 coarse motion。
四候选总体与单 batch 归约保持 v29；请求分布25% teacher、62.5%最多3步、12.5%最多8步 roll-in。
长窗口由稳定哈希选择，短轨迹自然截短。teacher不因空测量换端点；roll-in局部初始化后只接受预测，不因漂移重置GT。
所有启用模块自epoch0学习；BN隔离与detach是所有权合同，不能冻结参数。B0始终独占递归状态写入。

B1 相对实际 B0 crop 外扩，21维获取上下文，带宽[min=.25/.25,max=4/3,init=.75/.5]m；
actual/max/9×9标签共用几何，需求/无需求平衡，超出最大范围的目标不当作缩小范围负例。
B2 768→256预算、128/96/32配额、36 memory tokens，输出3个确定性模式；仅前景vote回归，模式质量软监督。
B3 比较3模式×半幅/全幅及observation；六动作同状态标签，按合法动作再按行归约，B0/B1/B2输入全部detach。
机制流行为比例25% never、12.5%证据首模式全幅、12.5%合法动作探索、50% q最大且q>0。
校准按never每行max-action-q构造阈值，never/always/6阈值及最多2中点共不超过10次拟合闭环；dev只锁定诊断。

mini全部8个mini_train用于参数训练，内部calibration/dev各1场景；官方2个mini_val只评测。
full全部350个train_track训练、内部17/18拟合/诊断、官方150val评测；内部拟合与参数训练重叠，明确记录。
KITTI有标签官方training序列0000–0016训练、0017拟合、0018诊断、0019–0020最终测试；
sensor_relative、原始帧号×0.1s、所有角色preload_offset=-1，无OXTS不宣称世界坐标运动。

mini固定六项消融：旧获取几何、关闭需求平衡、单模式、关闭模式质量监督、证据首模式全幅、关闭8步roll-in。
额外紧带宽[2,1.5]单独登记。所有消融独立从头训练，不使用不影响最终输出的B1伪消融。
时间实验另登记nuScenes间隔1/2/4、KITTI间隔1/2/5，均对照true/fixed/shuffled。
统一记录global novel→最大可达→support→768→256→三模式→被选动作→闭环结果，含unique点数、事件/条件分母及稀疏、速度、年龄、失跟分组。

数据和方法身份进入配置、checkpoint、策略；B0修复与KITTI标定是共同实现基础，不计为插件创新。
旧配置/历史输出保持可读。不得用训练loss、presence AP或本地测试冒充S/P、SOTA、时序因果收益。
详见[v30实现](CTSEQTRACK_V30_IMPLEMENTATION.md)与[运行手册](V30_DATA_AND_RUNBOOK.md)。

## 以下为 v29 性能与历史协议

2026-09-11性能实施补充：用户允许昂贵纯诊断抽样，训练输入、监督、Adam更新和模块递归耦合仍保持v29。
新增独立`*_nuscenes_full_perf.yaml`；旧配置/旧输出保留。执行细则、诊断缺测标记、短测速与逐位对照及
三GPU命令见[性能记录](CTSEQTRACK_V29_PERFORMANCE.md)。不增加全套哈希门禁，不把本地CPU验证当成CUDA/速度证明。

## v29 当前协议（2026-09-10）

最新启动调整：用户要求本次仅轻量核对、直接提供三组独立后台命令，不做全套哈希或强制工程报告检查。
最新重启要求：停止当前三臂，将workers从12统一改为4，在新日期output目录从头训练，不跨workers续训。
v29工作进程数校验允许4或12，配置新默认为4；需先同步`utils/online_contract.py`和
`cfgs/ct_seqtrack/29_formal_base.yaml`，避免旧workers12硬校验阻止启动。历史日志和checkpoint保留。
每两轮保存完整checkpoint，额外保留059以维持58/59/60协议；验证仍每5轮。CUDA/长程恢复检查仍属未完成证据，
不能把直接启动写成已完成工程验收。此前“工程检查后启动”的排程由本条最新要求覆盖。

最新用户批准实施v29，并运行 **B0、Full-CfC、Full-GRU，完整nuScenes Car、seed42、60epoch**。
三臂比较模块组合及两种时序后端；不据此宣称每个模块的独立贡献。新配置均为`29_*_nuscenes_full.yaml`。
350个train_track场景训练，150个官方val验证/评测；batch16、workers4、每5轮验证、FP32、
Adam(1e-4, betas=(0.5,0.999), eps=1e-6, wd=0, foreach/fused=false)、StepLR20×0.1。
从头训练，所有启用模块从epoch0首个合法事务学习，无冻结、零学习率阶段或跨run初始化。

B0采用适应化语义，不冒用v28 reference兼容声明。原始四候选总体shuffle/drop_last，整batch一次loss，
candidate0保留teacher与既有重抽；其余从局部起点短递归生成当前模型历史。额外no-grad前向单独计时，
Adam更新数仍按真实完整数据长度计算。GT仅作合法首帧/窗口初始化和监督，预测历史与GT标签分离。
机制流以never/always/threshold0行为访问accepted状态，与部署共用纯转移，不能回灌共享B0流。

先进行真实CUDA、同卡100步逐位一致性和epoch边界恢复检查，再正式三臂60轮；不强制再跑mini60。
完整数据默认不preloading，正式输出新建日期output目录，工程checkpoint不得初始化正式运行。
final=60，late-3为58/59/60三个分数的算术平均；两Full每个checkpoint分别拟合策略。
17个内部场景拟合、18个锁定诊断，与参数训练重叠；官方150不选阈值/后端/epoch。
闭环候选never、always、最多3个筛选阈值；补同权重observation和bounded-always闭环对照。
“训练内部策略拟合”不等于概率校准、安全保证或全局重定位。得分恢复与涨分以实际结果验收。

具体语义、继承限制见[实施合同](CTSEQTRACK_V29_IMPLEMENTATION.md)，执行与未完成验收见
[服务器命令](CTSEQTRACK_V29_SERVER_RUNS.md)和[本地验收](CTSEQTRACK_V29_LOCAL_VALIDATION.md)。

## v28 历史协议（2026-09-10；旧排程已被上述v29替代）

9月10日后续指令：进行一组 **Full / Car / seed42 / 完整nuScenes / 从头60epoch** 诊断，
取得完整数据结果后集中分析。使用现有`28_full_nuscenes_full.yaml`，batch16、workers12、
每5轮官方val验证，不预加载点云；输出新建于
`output/YYYYMMDD-HHMMSS-28_full-nuscenes_full_car_seed42_60ep_bs16/`。
此单组诊断安排替代“mini必须恢复后才能进行任何full运行”的顺序，分数未恢复的事实保留，
不自动展开五类多臂矩阵。训练及逐58/59/60校准/评测见
[单组完整数据诊断](CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。以下三组mini安排为已完成记录。

上轮服务器启动按用户后续指令更新为三组mini并行：GPU1 B0 seed42、GPU2 B0 seed52
（`28_b0_seed52.yaml`独立注册）、GPU3 Full seed42。三组均从头训练60轮、batch16、
workers12、每5轮官方mini_val验证；这次启动安排替代原“首轮仅B0”的排程。
数据/数值/模块合同与分数验收要求保持一致；Full训练后的正式评测仍需各checkpoint独立校准。
正式输出新建于 `output/YYYYMMDD-HHMMSS-28_模块-mini_car_seedXX_60ep_bs16/`，
每组保存自己的 `train.log`、`train.pid` 和训练产物，不覆盖历史目录。
两次CUDA执行阻断及修复见 [CUDA排错记录](CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。
9月10日已核实三组60轮完成；B0/42=45.200/47.243，B0/52=52.876/64.478，
Full/42未校准=45.200/47.243。seed42未达恢复目标，late-3尚缺，不能直接放行完整矩阵；
证据与下一步见 [三组结果分析](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。

本节替代下方历史v27/v26/v25/v24中与v28冲突的规定；旧配置和结果不覆盖。
详细实现与阶段验收见 [v28实施](CTSEQTRACK_V28_IMPLEMENTATION.md) 与
[服务器流程](CTSEQTRACK_V28_SERVER_RUNS.md)。

- 已完成B0 seed42、B0 seed52、Full seed42三组mini Car，各自scratch、60epoch、
  batch16、workers12，每5轮官方val；seed52使用独立登记的`28_b0_seed52.yaml`。
  所有工程checkpoint丢弃；完整结果固定final60与late-3=58/59/60，禁止挑不同best。
- 每帧1024槽、3帧历史；Adam lr=1e-4、betas=(0.5,0.999)、eps=1e-6、weight_decay=0，
  StepLR每20轮乘0.1、无梯度裁剪。四候选总体自然drop_last，当前mini Car应为1262步/轮、
  75,720次更新；按实际长度校验，不能重复或补齐数据凑数。full按其实际长度计算。
- B0观测合同为seqtrack_reference_compatible_v1，采样为seqtrack_original_slots_v1。
  完整四candidate数据总体随机排列，batch不强制候选均衡；reference_batch在完整batch
  一次归约，BC只计一次。v28不消费历史[.5,1/6,1/6,1/6]目标或逐view均值。
- 保持safe_seqtrack_auto_v1、train.v4、unified Adam和B0唯一递归写入权。
  B1/B2/B3在第一个合法tick开始训练，所有启用参数从头学习；不冻结、不跨run初始化。
- mini全部8个mini_train训练，官方2个mini_val只评测；固定内部calibration/dev各1场景
  与训练重叠，明确标记training_internal，不声称held-out。full全部350 train_track训练，
  内部17/18拟合/诊断，官方150 val评测。官方val不用于选择后端、阈值或超参。
- 严格FP32、TF32关闭、cuDNN benchmark关闭/确定性开启、Adam foreach/fused关闭，
  CUBLAS工作区:4096:8；不允许warn_only降级。观测与数值合同绑定resume身份，
  同run恢复核对实际环境。合法原始ID及extension-only点证据不随旧槽合同被撤回。
- 分割CE合同固定`ct_b0_ce_contract=class_axis_logsoftmax_flat_nll_v1`：保留原三维
  class-axis log_softmax，再展平为二维NLL，维持全batch加权分母，避免空间NLL归约阻断。
  AP统计使用整数cumsum后转回浮点，避免浮点CUDA cumsum阻断；两项均不关闭严格确定性。
- B2读取seg_second64_v1真实逐点特征；结构合法性独立于presence，bounded-always/B3/
  导出/校准共享同一候选。Full每个checkpoint重新真实闭环拟合、锁定内部dev诊断后评官方val。
  v28 policy绑定checkpoint/config/source/scene/metric，缺失或失配则observation fallback。
- v28 reference命名为共享实现架构参考：ctseqtrack+b0与28 B0的网络、损失和采样相同，
  仅参考标签不同，不作“不同原始架构”的算法对照。原冻结SeqTrack结果仅作历史参照。
- 100-step同卡B0/B0/GRU逐位一致及真实Full连续/恢复是工程门禁，不能替代60轮有效性。
  后续mini五臂+reference及full五类30次仍待B0验收；不从已注册配置推断收益。
- 历史健康参照S/P=50.986/59.962，首轮分别要求S≥49.986、P≥58.962。
  严格“下降≤1个百分点”需随后同协议reference的final及late-3分别比较S/P；
  两次同样低分不能声明恢复。未通过则继续定位，不换seed、挑epoch或扩训练预算。

## v27 当前协议（2026-09-05）

本节及 [CTSEQTRACK_V27_METHOD.md](CTSEQTRACK_V27_METHOD.md) 定义当前轮次；
下方 v24/v25/v26 段落保留为历史协议。

- 顺序：mini Car 五臂 B0、B1-GRU、B1-CfC、Full−B3、Full → 完整nuScenes
  Car/Pedestrian/Truck/Trailer/Bus，每类每臂独立从epoch0训练60epoch、seed42。
- 配置：`cfgs/ct_seqtrack/27_*.yaml`；单入口`main.py`；禁止跨run/seed/类别/版本
  checkpoint初始化，resume只允许同run epoch边界。所有启用模块从第一步参与训练。
- B0四视图权重`[.5,1/6,1/6,1/6]`，机制流只用canonical view0；运行合同保持
  `safe_seqtrack_auto_v1`和unified Adam，batch schema升级`ct_seqtrack.train.v4`。
  `ct_b0_steps_per_epoch=0`按数据长度运行；禁用GT reseed；B0/B1/B2/B3 detach所有权不变。
- 场景：mini_train八scene按稳定seed42拆6训练/1阈值拟合/1锁定诊断，官方mini_val
  两scene只评测；full的train_track350scene全部参与参数训练，其内17scene拟合阈值、
  另18scene诊断，官方val150scene只评测。full明确记录参数训练重叠，不能称为未见数据校准。
- B1两个backend共用获取接口、监督和预算；集成默认GRU，后端诊断不能从不同B0轨迹总分归因。
- Full−B3评测`bounded_always`，Full评测`selective`；两者训练均observation-recursive。
  B3只有整体S/P效用策略，不再使用v26的presence双门或risk-promotion门。
- v27校准在calibration候选策略上进行真实闭环比较，再锁定到dev诊断；policy显式包括
  always/never，artifact绑定checkpoint/config/code/scene manifest/metric定义。
  artifact缺失或不匹配则observation fallback。
- 固定final=60与late-3=58/59/60；不挑选每臂不同best epoch。每个Full checkpoint独立校准。
- mini五个CT臂本次启动使用workers=4、preloading、每5轮内部dev诊断；
  last按完整epoch保存、late-3额外保存，不依赖验证间隔。具体后台命令见
  [mini启动说明](CTSEQTRACK_V27_MINI_LAUNCH.md)。完整数据配置保留原12 workers/每轮诊断默认值。
- 另行报告`cfgs/27_seqtrack_reference{_nuscenes_full}.yaml`外部架构参考：公共修复、
  independent candidate、普通SeqTrack loss、全部CT模块关闭，与CT B0加权目标区分。
- 主指标`benchmark_compat`，另报`geometry_exact`审计；全臂同口径，所有endpoint包括
  空点/fallback必须进入分母。尚未完成实验前不声明涨分、SOTA或因果时间/memory收益。

## 以下为历史 v24/v25/v26 协议

> 2026-08-28：当前正式轮次升级为 v26。本轮请求运行 B0、B1-GRU、
> B1-CfC、Full-B3、Full 五个 scratch-only 臂；集成主臂固定 GRU，CfC
> 仅作为 B1 backend 诊断，SeqTrack-strict 保持为单独登记的外部参考。
> held-out calibration/dev promotion 和结果边界以
> [CTSEQTRACK_V26_METHOD.md](CTSEQTRACK_V26_METHOD.md) 为准。下文 v24/v25
> 章节作为冻结历史协议保留，不得用于初始化或改写 v26。

本文档是 mini、完整 nuScenes、校准和消融实验的唯一协议来源。README 只提供入口，`need_to_do.md` 只记录状态。

> 2026-08-24 起，论文正式训练协议升级为 Safe-SeqTrack v25。
> `25_b0/b1/full_minus_b3/full*.yaml` 是新的可运行配置；本文后面的
> v24 内容作为冻结失败证据保留，不得初始化或续训 v25。v25 的完整
> 运行合同见 [SAFE_SEQTRACK_V25_PROTOCOL.md](SAFE_SEQTRACK_V25_PROTOCOL.md)。

v25 固定 B0 四候选损失
`0.5*L0 + (L1+L2+L3)/6`。只有 observation stream 使用四候选；
mechanism stream、B1、B2、B3 始终只读取 canonical candidate0。
v25 使用无状态 observation RNG、`ct_seqtrack.train.v2` envelope、一个
带 B0/B1/B2/B3 命名参数组的 Adam 和 Lightning automatic optimization。
所有实验仍从 epoch 0 随机初始化，启用模块不冻结，首帧尺寸和预测框
递归历史的无泄露合同不变。

以下 v24 章节不再定义当前可运行协议。

## 1. 不可变训练合同

- 所有实验从 epoch 0 随机初始化，各臂独立训练。
- 禁止 `--init_checkpoint`，禁止从另一实验或另一 seed 的 checkpoint 初始化。
- `--checkpoint` 仅用于同一运行的 epoch-boundary resume 或评测。
- 所有启用模块都参与训练，不冻结 B0、B1、B2 或 B3 参数。
- B0 辅助视图的 BN running-stat 隔离，以及 B1/B2/B3 的既有 detach，属于耦合合同，不是参数冻结。
- 正式比较使用 final epoch 和 late-3，不为不同实验挑选不同 best epoch。

## 2. 固定 candidate 协议

本轮不再执行 candidate1/candidate4 选择。四个正式实验统一固定为：

```text
B0 observation stream: full mini_train, 4 independent candidates
  each candidate is a normal SeqTrack sample in the shuffled batch
  B0 updates once per batch (1262 steps/epoch; 75720 steps/60 epochs)

Mechanism stream: online-recursive train partition
  canonical endpoint only -> B1 -> B2 -> B3 -> observation state commit
  one complete mechanism pass is embedded uniformly per epoch

B2: 1 view
  只读取 canonical view0；不把三个 B0 辅助视图送入 B2
```

B0 目标保持：

```text
L_B0 = 0.5*L_candidate0 + (L_candidate1 + L_candidate2 + L_candidate3)/6
```

固定配置字段为：

```yaml
num_candidates: 4
ct_recursive_candidate_views: 4
ct_b0_candidate_views: 4
ct_b0_candidate_weights: [0.5, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666]
ct_b2_candidate_views: 1
ct_recovery_candidate_policy: "off"
ct_training_topology: dual_stream
ct_b0_training_protocol: safe_seqtrack_auto_v1
ct_b0_candidate_mode: independent
ct_b0_steps_per_epoch: 1262
ct_mechanism_stream: online_recursive
ct_mechanism_passes_per_epoch: 1
ct_mechanism_b0_view: canonical_only
```

`24_b0_candidate1_control.yaml` 仅作为保留的协议对照配置，不属于当前四臂正式实验，也不能初始化其他实验。

## 3. mini 四臂实验

以下四臂分别从 epoch 0、seed42 独立训练 60 epoch：

| 配置 | 启用模块 | 训练/评测输出 |
|---|---|---|
| `25_b0.yaml` | B0 | observation |
| `25_b1.yaml` | B0+B1 | observation；B1 作为 prior/shadow 机制评估 |
| `25_full_minus_b3.yaml` | B0+B1+B2 | `raw_search`，用于直接衡量 B2 |
| `25_full.yaml` | B0+B1+B2+B3 | 未校准时 fail-closed 为 observation；校准后 selective |

不设置训练前 preflight、kill-test 或中途停止门禁。Acquisition、presence、target-bearing、retention 等字段在训练中持续记录，只在 final 和 late-3 完成后分析。B1 backend promotion 是完整训练后的独立机制指标决策，不停止或改写训练中的 run。

四臂之间不共享权重。临时验收 checkpoint 不得作为任何正式实验的初始化。

## 4. 必报指标

- 跟踪：Success AUC、Precision AUC、final、late-3 和运行成本。
- B1：learned mean vs CV RMSE、NLL、二维 coverage、support recall/volume，并按 gap、稀疏度和 recursive age 分层。
- B2：base/expansion/pool/sampled supply、target-bearing retention、presence AP/ECE、raw candidate gain、oracle headroom、harm 和 no-extension counterfactual。
- B3：action coverage、harmful rate、center/IoU gain、tracklet-bootstrap 区间和 risk--coverage 曲线。
- 公平性：共享 prefix 的初始化、step1、step100 和 epoch-end 参数 hash。

## 5. B3 校准

Full 完成 scratch 训练后，才可在与 train/dev/test 分离的 calibration tracklets 上选择 presence/action 阈值。Artifact 必须绑定 final checkpoint、正式配置、tracklet manifest、score 定义和内容 SHA。

缺失、失败、过期或错配时必须 fail-closed，输出 observation。校准不回灌训练，也不作为其他实验的初始化。

## 6. 服务器验收状态

真实 batch 前向/反向、GRU/CfC 有限非零梯度、100-step B0 参数/Adam 状态哈希、resume 等价和逐帧点框检查是正式长跑前的服务器门禁。本地无完整 nuScenes/CUDA 环境时只能标记为“未执行”，不能用 CPU 单测替代；所有 smoke checkpoint 必须丢弃。

## 7. 完整 nuScenes

mini 结果证明方案值得继续后：

1. 使用对应的 `25_*_nuscenes_full.yaml` 从头运行四臂 seed42；
2. seed42 保持正结果后，补四臂 seeds 52、62；
3. 报告 final-epoch 均值、标准差、tracklet paired CI、risk--coverage 和失败案例。

其他数据集和新创新点在本轮四臂 mini 验证完成前不加入。

## 8. v25 B1 修复后的实验顺序（2026-08-25）

1. mini seed42 分别以 `--b1-backend gru` 和 `--b1-backend cfc` 从 epoch 0
   跑满 `25_b1.yaml`。B1-only 部署输出仍是 observation，不能用 tracking score
   选择时序骨干。
2. 每个后端都必须在独立 dev tracklets 上通过 learned-vs-CV 的 tracklet
   paired-bootstrap 和 B1 calibration 门槛。CfC 只有在 CfC-minus-GRU RMSE 的
   95% CI 上界 `<0` 且 NLL/coverage 不劣时才晋升，否则主方案保持 GRU。
3. 以胜出 backend 重新从 epoch 0 训练 `25_full_minus_b3.yaml`，不得加载
   B1-only checkpoint。与 matched B0 比较 final/late-3 Success、Precision，
   并在同一 checkpoint 上评估 `force_b1_invalid` 和 `shuffle_b1_signal`。
4. mini 通过后，完整 nuScenes 先独立训练 B0、B1、Full-B3、Full seed42；
   保持正结果后补 seed52、seed62。所有 arm、seed 都从随机初始化开始。
5. B1/B3 calibration 都发生在对应 scratch 训练结束后，只生成 evaluation-only
   artifact/checkpoint，不得恢复训练或初始化后续实验。额外数据集优先使用现有
   KITTI-HV 时间间隔协议。

在上述实验完成前，不宣称涨分、SOTA、CfC 优越或物理时间的因果收益。
