# B2-v3：状态对齐的 Motion-Conditioned Selective Search

## 论文主线与边界

主线名称为 **State-Aligned Motion-Conditioned Search with
Action-Consistent Closed-Loop Routing**。创新合同不是“首次使用历史运动”、
“首次 proposal-refine”或“首次 memory”，而是三件可验证的事：

1. B1 motion prior 与 B2 endpoint evidence 在同一次 forward 中读取完全相同的
   因果 history/mask/Δt；
2. 当前帧 overlap/extension/context evidence 由候选监督直接训练，并作为 B1
   prior 的有界证据修正；
3. router 对 `2 candidates × 3 steps` 六个可执行动作逐一学习 signed H=3
   gain，选择、执行和 calibration 使用同一动作。

本版不实现 B1-conditioned 二次裁剪，不解冻 B1，也不主张长期 memory 创新。
真实物理时间仅保留为 paired fixed/shuffled 与 irregular-cadence 控制。

## 已修复的数据与梯度合同

- candidate0 使用 canonical history；非零 candidate 由
  `seed/tracklet/frame/candidate` 的 SHA-256 稳定 50/50 分配 correlated 或
  recursive history。该选择不消耗 point-sampling RNG。
- sampler 只构造一次 shared history。`motion_main_ref_boxs` 与
  `b2_v3_history_ref_boxs` 必须 byte-identical；valid mask 和 effective Δt 也在
  forward 中逐项 `torch.equal`。
- 在线路径让两个字段直接引用同一组递归 `results_bbs` tensor。
- B0 的 1024 点采样和增强没有改变；状态对齐严格限定为 B1/B2。
- v2.2 的未训练 `source_fusion` 不存在于 V3。V3 evidence 是
  `[overlap token, extension token, motion/observation context]` 的 384 维拼接；
  presence head 直接消费它，router 持有自己的 384→context projection。
- presence probability 只是监督与 router feature。硬 validity 只由点数、几何、
  finite 和 B1 validity 决定，低于 0.5 的结构有效样本仍进入 rollout。

旧 v2.2 类、字段、配置和工具保持不变。

## 严格初始化与 refiner 训练

先用固定的两个 epoch60 checkpoint 合成初始化：

```bash
python tools/build_b2_v3_init_checkpoint.py \
  --base-checkpoint <b2-v2-epoch60-last.ckpt> \
  --search-checkpoint <b2-v2.1-full-epoch60-last.ckpt> \
  --output <b2-v3-init.ckpt>
```

builder 要求 B1 恰好包含 14 个 tensor，并记录 B1 prefix hash、两个源文件
SHA-256、33 个迁移 tensor 的完整 target key 集。训练加载器进一步拒绝缺 key、
shape mismatch、迁移 key 集变化、B1 hash 不一致或任何未加载的冻结参数。
optimizer 前、最初两个 step 后及每个 epoch 末都会复核 B0/B1 prefix hash。
hash manifest 必须完整覆盖 `motion_state_mlp` 在内的全部 B0/B1 前缀；旧的
不完整 manifest 不能用于正式续训。

```bash
python tools/ct_v2/run.py train \
  --variant b2_v3_refiner \
  --init-checkpoint <b2-v3-init.ckpt> \
  --epochs 20 --batch-size 16 --seed 42
```

正式训练前先在同一服务器、同一数据路径上从原始 init 做两步预检（预检产生的
checkpoint 不得用于正式训练）：

```bash
python tools/ct_v2/run.py train \
  --variant b2_v3_refiner \
  --init-checkpoint <b2-v3-init.ckpt> \
  --epochs 1 --limit-train-batches 2 \
  --check-val-every-n-epoch 2 --batch-size 16 --seed 42
```

预检必须看到严格初始化报告，完成两个 optimizer step，并且没有 history contract、
非有限梯度或 frozen hash 异常；随后正式训练仍从同一个原始 init 重新开始。

固定使用 epoch20 `last.ckpt`。`save_top_k=0`，所以恒定的 observation tracking
metric 不参与 checkpoint 选择；epoch5/10/15/20 的候选诊断写入
`candidate_diagnostics/epoch_XX.csv`。训练期最终框必须与 observation bitwise
相同。

## 两轮 closed-loop rollout

第一轮只收集 observation-policy 状态：

```bash
python tools/export_b2_v3_rollouts.py \
  --checkpoint <b2-v3-refiner-epoch20-last.ckpt> \
  --output <round0-dir> --state-policy observation --round 0 \
  --split mini_train --horizon 3 --seed 42 --path <nuscenes-root>

python tools/train_action_router_v3.py \
  --rollouts <round0-dir> --output <provisional-router.pt> \
  --threshold-partition dev --seed 42
```

第二轮使用 provisional router 产生 on-policy state；它的 threshold 只能来自 dev：

```bash
python tools/export_b2_v3_rollouts.py \
  --checkpoint <b2-v3-refiner-epoch20-last.ckpt> \
  --output <round1-dir> --state-policy router --round 1 \
  --router-sidecar <provisional-router.pt> \
  --split mini_train --horizon 3 --seed 42 --path <nuscenes-root>

python tools/merge_b2_v3_rollouts.py \
  --observation-rollouts <round0-dir> \
  --on-policy-rollouts <round1-dir> --output <merged-dir>

python tools/train_action_router_v3.py \
  --rollouts <merged-dir> --output <final-router.pt> \
  --threshold-partition calibration --min-selected-count 100 --seed 42
```

每个 H=3 counterfactual 只在首帧强制候选/step；其余 H-1=2 帧显式使用
`POLICY_OBSERVATION=-2`，不依赖冷 router。tracklet 始终按稳定 hash 做
70/15/15 train/dev/calibration 分区，两个 round 的 checkpoint/config/policy/data
hash 都写入 manifest。

工具链会强制阶段一致性：round1 router 必须来自同一 checkpoint/config/seed 的
round0，并且只使用 dev threshold；merge 必须保持 config hash；最终 calibration
router 只能在 merged round0+round1 上从头训练。

## 打包与四种无歧义评测

```bash
python tools/package_b2_v3_checkpoint.py \
  --candidate-checkpoint <b2-v3-refiner-epoch20-last.ckpt> \
  --router <final-router.pt> --output <b2-v3-final.ckpt>

python tools/ct_v2/run.py test --variant b2_v3_selective \
  --checkpoint <b2-v3-final.ckpt> --proposal-mode obs_only
python tools/ct_v2/run.py test --variant b2_v3_selective \
  --checkpoint <b2-v3-final.ckpt> --proposal-mode obs_vs_motion
python tools/ct_v2/run.py test --variant b2_v3_selective \
  --checkpoint <b2-v3-final.ckpt> --proposal-mode obs_vs_refined
python tools/ct_v2/run.py test --variant b2_v3_selective \
  --checkpoint <b2-v3-final.ckpt> --proposal-mode obs_vs_all
```

打包器只接受最终 calibration partition 通过的 router，并在替换全部 router key
前后复核 B0/B1/refiner hash。强制 `MOTION/REFINED + step` 只用于 rollout 与
诊断，不作为 leaderboard 模式。

`b2_v3_selective` 还会在加载时强制验证 package schema、完整 router key、
protected hash 和 calibration threshold，因此误用 refiner `last.ckpt` 会立即失败，
不会静默退化为 observation-only。单候选评测只改变 action-allowed mask；router
看到的 intrinsic structural validity 与 `obs_vs_all` 保持一致。

## Seed42 promotion gate

- valid-foreground 上 refined RMSE 同时低于 B1 motion 与 raw search；
- calibration：helpful precision ≥75%，harm ≤10%，coverage 5%–25%，且至少
  100 个状态；
- mini_val：helpful precision ≥70%，harm ≤10%；
- `obs_vs_all` >54.132/64.755，不低于两个辅助模式，并相对同 checkpoint
  `obs_only` 至少 +0.5 Success / +1.0 Precision；
- Seed42 通过后才运行 43/44；三种子平均增益为正且无灾难性单种子下降。

把四种模式与 mini_val routing 统计写成下面的 JSON 后，可机械执行 gate：

```json
{
  "seed": 42,
  "modes": {
    "obs_only": {"success": 0, "precision": 0},
    "obs_vs_motion": {"success": 0, "precision": 0},
    "obs_vs_refined": {"success": 0, "precision": 0},
    "obs_vs_all": {"success": 0, "precision": 0}
  },
  "mini_val": {"helpful_precision": 0, "harm_rate": 0}
}
```

```bash
python tools/check_b2_v3_promotion.py \
  --router <final-router.pt> \
  --candidate-diagnostics <proposal_endpoints.csv> \
  --metrics <seed42-metrics.json> --output <promotion.json>
```

实现正确性由 `tests/test_b2_v3.py` 覆盖；旧 v2.2 regression 保持通过。
