# CT-SeqTrack 正式实验协议

## v28 当前协议（2026-09-08）

本次服务器启动按用户后续指令更新为三组mini并行：GPU1 B0 seed42、GPU2 B0 seed52
（`28_b0_seed52.yaml`独立注册）、GPU3 Full seed42。三组均从头训练60轮、batch16、
workers12、每5轮官方mini_val验证；这次启动安排替代下面“首轮仅B0”的排程。
数据/数值/模块合同与分数验收要求保持一致；Full训练后的正式评测仍需各checkpoint独立校准。

本节替代下方历史v27/v26/v25/v24中与v28冲突的规定；旧配置和结果不覆盖。
详细实现与阶段验收见 [v28实施](CTSEQTRACK_V28_IMPLEMENTATION.md) 与
[服务器流程](CTSEQTRACK_V28_SERVER_RUNS.md)。

- 首轮正式仅28 B0，mini Car、seed42、60epoch、batch16、workers12，每5轮官方val。
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
