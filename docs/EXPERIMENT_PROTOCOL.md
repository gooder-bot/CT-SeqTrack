# CT-SeqTrack 正式实验协议

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
L_B0 = (L_candidate0 + L_candidate1 + L_candidate2 + L_candidate3) / 4
```

固定配置字段为：

```yaml
num_candidates: 4
ct_recursive_candidate_views: 4
ct_b0_candidate_views: 4
ct_b0_candidate_weights: [0.25, 0.25, 0.25, 0.25]
ct_b2_candidate_views: 1
ct_recovery_candidate_policy: "off"
ct_training_topology: dual_stream
ct_b0_training_protocol: seqtrack_d86990c
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
| `24_b0.yaml` | B0 | observation |
| `24_b1.yaml` | B0+B1 | observation；B1 作为 prior/shadow 机制评估 |
| `24_full_minus_b3.yaml` | B0+B1+B2 | `raw_search`，用于直接衡量 B2 |
| `24_full.yaml` | B0+B1+B2+B3 | 未校准时 fail-closed 为 observation；校准后 selective |

不设置 preflight、promotion、kill-test 或中途停止门禁。Acquisition、presence、target-bearing、retention 等字段在训练中持续记录，只在 final 和 late-3 完成后分析。

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

真实 batch 前向/反向、100-step/resume 和逐帧点框检查仍是推荐验收项。本轮用户明确选择不执行服务器 smoke，因此它们只能标记为“未执行”，不能标记为“通过”。为避免破坏现有耦合，依赖这些检查才能安全删除的兼容宿主源码继续保留。

## 7. 完整 nuScenes

mini 结果证明方案值得继续后：

1. 使用对应的 `24_*_nuscenes_full.yaml` 从头运行四臂 seed42；
2. 补 B0 和 Full seeds 43、44；
3. 报告 final-epoch 均值、标准差、tracklet paired CI、risk--coverage 和失败案例。

其他数据集和新创新点在本轮四臂 mini 验证完成前不加入。
