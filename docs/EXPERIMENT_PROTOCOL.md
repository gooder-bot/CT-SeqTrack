# CT-SeqTrack 正式实验协议

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
