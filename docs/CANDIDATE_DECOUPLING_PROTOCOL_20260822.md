# CT-SeqTrack Candidate 解耦协议

本协议以 `main@63fcbcde855e72bf87ee87cecb42fb50aa185b36` 为结构基线，
不引入 v25/f320 的多时间角色或辅助插件梯度。唯一移植项是 B1 prior 到 B2
anchor 的 SE(2) 坐标重表达与 identity/round-trip 测试。

## 数据与梯度所有权

每个在线端点由 sampler 产生四行，四行共享同一递归状态快照：

```text
b0_view_id=0: clean B0 -> B1 -> B2 -> B3 -> observation state commit
b0_view_id=1: jitter B0 -> stop
b0_view_id=2: jitter B0 -> stop
b0_view_id=3: jitter B0 -> stop
```

辅助视图使用已有的 coherent `shared_se2` 小扰动：XY 各 `[-0.3,0.3] m`，
yaw `+-5 deg`。随机身份由 seed、epoch、tracklet、frame、view 构成，不读取当前
GT 来选择扰动方向或难度。辅助前向在独立 RNG 中运行，并冻结 BN running
statistics；B0 affine/weight 参数仍正常接收梯度。

B0 的唯一优化事务为：

```text
L_B0 = 0.5 L_clean + (L_view1 + L_view2 + L_view3) / 6
```

B1/B2/B3 事务权重均为 canonical `1.0`。每个活动模块每个在线端点只执行一次
optimizer/scheduler step。所有正式实验从随机初始化开始；禁止
`--init_checkpoint`，也不允许冻结启用模块。

## 固定配置

正式四臂使用：

```yaml
num_candidates: 4
ct_recursive_candidate_views: 4
ct_b0_candidate_views: 4
ct_b0_candidate_weights: [0.5, 0.1666667, 0.1666667, 0.1666667]
ct_b2_candidate_views: 1
ct_recovery_candidate_policy: "off"
ct_recursive_reseed_enabled: true
ct_b0_rng_shift_control: true
ct_targetness_class_weight_source: online_canonical_preflight
```

`cfgs/ct_seqtrack/24_b0_candidate1_control.yaml` 是候选增强选择阶段的 view1
对照；`24_b0.yaml` 是 view4 proposed。旧 2x2 YAML 已标为
`historical_2x2_do_not_run`，不得作为新实验配置。

## 运行顺序

1. view1 control 与 view4 proposed 各自从 epoch0 独立 scratch 跑满 60 epoch。
2. 只有 proposed 的 final Success/Precision 均提高，且 late-3 不低于
   `-0.3/-0.5`，才固定 candidate4。
3. 固定 B0 candidate 协议后，B0、B1、Full-B3、Full 各自从 epoch0 scratch
   跑满 60 epoch；不要求 preflight 或 B2 promotion artifact。
4. B2 targetness 在实际 candidate0 训练流中累计正负有效点，并使用与 preflight
   相同的逆频率公式自动平衡。缺少任一类别时暂用 1:1，不停止训练。
5. acquisition、target-bearing、retention、presence AP/ECE 等指标持续记录，
   仅在 final 与 late-3 结果产生后分析，不作为中途停止条件。
6. mini 最终结果分析后再跑完整 nuScenes seed42，随后补 B0/Full seed43、44。

mini 使用原四臂 YAML；完整 nuScenes 使用对应的
`24_*_nuscenes_full.yaml`。seed43/44 用 `--seed 43` / `--seed 44` 独立启动，
不得通过 checkpoint 续接或换臂初始化。

```bash
# Candidate protocol comparison（两个独立目录、均从 epoch0跑满60 epoch）
python main.py --cfg cfgs/ct_seqtrack/24_b0_candidate1_control.yaml \
  --path DATA_ROOT --epoch 60 --tag candidate1-final60-seed42
python main.py --cfg cfgs/ct_seqtrack/24_b0.yaml \
  --path DATA_ROOT --epoch 60 --tag candidate4-final60-seed42
```

恢复契约为 `ct_seqtrack.online_resume_contract.v5`。B2 promotion v4 与 acquisition
preflight v3 工具保留为可选的实验后分析工具，不参与初始化且不阻断训练。旧
candidate/targetness 语义 checkpoint 不允许跨协议恢复。
