# CT-SeqTrack B1–B4 重构实现说明

本实现把论文主链固定为 B1 概率运动先验、B2 base-preserving 非对称双查询搜索、B3 观测锚定动作路由。B4 是独立 decoder-token 短测，不再进入旧 PFTC 点匹配路径。

## 已实现的数据合同

```text
recursive history boxes + effective physical time
    -> frozen B1 pre-pass (mu / calibrated covariance / velocity / feature)
    -> unchanged B0 crop UNION B1 tube (128 independent evidence tokens)
    -> shared encoder
       -> q_obs from final decoder state
       -> q_search = stopgrad(q_obs) + bounded zero-init motion residual
    -> observation / motion / raw_search
    -> scalar-first six-action q10 router
    -> recursive history
```

- Transformer 默认返回值不变；`return_decoder_state=True` 才返回 `[B,L,64]` state。
- `SEQTRACK3D.predict_motion_from_history(...)` 是 box-only B1 公共接口；在线 pre-pass 不读取当前 `3d_bbox`。
- B1 输出唯一的 `basis_velocity_xy/direction_xy`、预测 `velocity_xy`、运动坐标系中的平行/垂直 sigma、XY covariance 和低速各向同性结果。NLL、校准、Mahalanobis 和 support 禁止重新推导方向；校准尺度以 persistent checkpoint buffer 保存。
- 在线 B2 保留原 base crop 和采样路径，只额外裁取 prior tube。B1 无效时依次回退 `fallback_cv` 和 `base_only`。
- `raw_search_xy` 是正式 Search candidate；`legacy_clipped_search_xy` 仅用于历史诊断。
- B2 structural availability 不依赖 B1 valid；presence 和 utility 是独立监督。双查询不会向 B0/B1 反传。
- B3 的 source/step 预测与执行是一套六动作。先执行 `0.25/0.5/1.0` 比例，再只在 B3 做一次 `0.20/0.35 m` normal/gap 安全裁剪；z/yaw 始终来自 observation。
- B4 仅在 decoder states 上使用 EMA projector + online stop-gradient representation、cosine invariance 和 variance/covariance guard；它不是完整 EMA encoder teacher。配置 19/20 为 `experimental_only`，运行时必须显式传 `--allow-experimental-b4`。

## 训练与实验入口

新配置：

- `15_b1_calibrated.yaml`：B1 mean + NLL，不改变 observation 输出。
- `16_b2_asymmetric_dual_query.yaml`：P2 hand-support 双查询与 raw Search。
- `17_b1_b2_replay_support.yaml`：P4 fixed-margin B1 support、replay、10-D uncertainty geometry。
- `18_b1_b2_b3_selective.yaml`：校准 sigma + dynamic support + scalar q10 B3，正式评估专用。
- `19_b4_decoder_alignment.yaml` / `20_b4_decoder_anticollapse.yaml`：B4 两个五 epoch 实验臂；matched B0 使用配置 01。

P2 的严格初始化需要在原 v3 builder 上增加 `--dual-query`；它保留 5-D
observation statistics 的迁移权重，并将旧 256-D observation-query 权重按四列
均值确定性压缩为 64-D final-decoder `q_obs` warm start。双查询 residual 的末层
仍精确零初始化，因此初始 `q_search == q_obs`，但首批 residual 梯度不会被下游
零权重切断：

```powershell
python tools/build_b2_v3_init_checkpoint.py --base-checkpoint b1.ckpt --search-checkpoint b2_v21.ckpt --dual-query --output b2_dual_init.ckpt
```

关键命令：

```powershell
# B1 calibration split (stable train-tracklet partition only)
python tools/export_b1_calibration.py --config cfgs/ct_v2/15_b1_calibrated.yaml --checkpoint b1.ckpt --output b1_calibration.npz --partition calibration
python tools/calibrate_b1_uncertainty.py --input b1_calibration.npz --output b1_calibration.json --checkpoint b1.ckpt --output-checkpoint b1_calibrated.ckpt

# Frozen recursive replay cache. Use the exact P4 initialization checkpoint;
# resume checkpoints remain valid only while their frozen B0/B1 hashes match.
python tools/export_recursive_replay_cache.py --config cfgs/ct_v2/17_b1_b2_replay_support.yaml --checkpoint b2_geometry10_init.ckpt --b0-checkpoint b0.ckpt --output replay/mini_train --split mini_train

# P4 paired ablations before B3 packaging:
# fixed B1 support only
python tools/ct_v2/run.py test --variant b1_b2_replay --geometry-off --checkpoint b2.ckpt --proposal-mode raw_search
# promoted dynamic sigma support only
python tools/ct_v2/run.py test --variant b1_b2_replay --geometry-off --dynamic-sigma --checkpoint b2.ckpt --proposal-mode raw_search
# geometry/replay stages use the default 10-D config 17 (and --replay-cache for training)

# Recompute the promotion evidence from the five frame-level CSVs.  The two
# controls must use the same sampling/input identity as observation.
python tools/build_b2_v3_five_mode_metrics.py --candidate-checkpoint b2.ckpt --observation observation.csv --motion motion.csv --raw-search raw_search.csv --legacy-clipped legacy_clipped.csv --selective selective.csv --forced-invalid forced_invalid.csv --shuffled-b1 shuffled_b1.csv --support-calibration support_mini_train.csv --calibration-split mini_train --output five_mode_metrics.json
python tools/check_b2_v3_promotion.py --candidate-checkpoint b2.ckpt --frame-diagnostics observation.csv --metrics five_mode_metrics.json --output b2_promotion.json

# forced-invalid.csv is produced with the same checkpoint/seed/input sampling:
python tools/ct_v2/run.py test --variant b1_b2_replay --dynamic-sigma --force-b1-invalid --checkpoint b2.ckpt --proposal-mode observation
python tools/ct_v2/run.py test --variant b1_b2_replay --dynamic-sigma --shuffle-b1-signal --checkpoint b2.ckpt --proposal-mode observation
# support_mini_train.csv must be generated on the training split; the metrics
# builder itself retains only the stable calibration tracklets.
python tools/ct_v2/run.py test --variant b1_b2_replay --dynamic-sigma --split mini_train --checkpoint b2.ckpt --proposal-mode observation

# P4 training; loader rejects manifest/content mismatch or GT-bearing records
python tools/ct_v2/run.py train --variant b1_b2_replay --init-checkpoint b2_geometry10_init.ckpt --replay-cache replay/mini_train

# Five attribution modes
python tools/ct_v2/run.py test --variant b1_b2_replay --checkpoint candidate.ckpt --proposal-mode observation
python tools/ct_v2/run.py test --variant b1_b2_replay --checkpoint candidate.ckpt --proposal-mode motion
python tools/ct_v2/run.py test --variant b1_b2_replay --checkpoint candidate.ckpt --proposal-mode raw_search
python tools/ct_v2/run.py test --variant b1_b2_replay --checkpoint candidate.ckpt --proposal-mode legacy_clipped
python tools/ct_v2/run.py test --variant b1_b2_b3 --checkpoint packaged.ckpt --proposal-mode selective
```

从 P2 的 9-D dual-query checkpoint 进入 P4 前，先合入独立校准的 B1，
再扩展 10-D geometry：

```powershell
python tools/compose_calibrated_b1_b2_checkpoint.py --b2-checkpoint dual_query.ckpt --calibrated-b1-checkpoint b1_calibrated.ckpt --output b1_b2_composed.ckpt
python tools/expand_b2_geometry_checkpoint.py --checkpoint b1_b2_composed.ckpt --output b2_geometry10_init.ckpt
```

新增列以精确零初始化，因此启用 Mahalanobis 通道前输出保持不变。

B2 promotion 与 B3 artifact 是强制链路：

```powershell
python tools/check_b2_v3_promotion.py --candidate-checkpoint candidate.ckpt --frame-diagnostics proposal_endpoints.csv --metrics five_mode_metrics.json --output b2_promotion.json
python tools/export_b2_v3_rollouts.py --checkpoint candidate.ckpt --promotion b2_promotion.json --output rollouts/round0 --round 0 --state-policy observation
python tools/train_action_router_v3.py --rollouts rollouts/round0 --output router_dev.ckpt --threshold-partition dev
python tools/export_b2_v3_rollouts.py --checkpoint candidate.ckpt --promotion b2_promotion.json --output rollouts/round1 --round 1 --state-policy router --router-sidecar router_dev.ckpt
python tools/merge_b2_v3_rollouts.py --observation-rollouts rollouts/round0 --on-policy-rollouts rollouts/round1 --output rollouts/merged
python tools/train_action_router_v3.py --rollouts rollouts/merged --output router_screened.ckpt --threshold-partition calibration
python tools/calibrate_b3_router_recursive.py --candidate-checkpoint candidate.ckpt --promotion b2_promotion.json --router router_screened.ckpt --output router_final.ckpt
python tools/package_b2_v3_checkpoint.py --candidate-checkpoint candidate.ckpt --promotion b2_promotion.json --router router_final.ckpt --output packaged.ckpt
```

Replay schema v2 校验 dataset/split、replay-relevant config、commit、当前
B0/B1 state-prefix、B1 calibration 和 records hash；sampler 还把 cached B1
传入 forward，对 `mu/direction/log_sigma/gap/valid` 做 `1e-5` 同历史复算。
B3 sidecar v4 固定完整 scalar 名称/顺序、p1/p99、mean/std、动作顺序、配置、
candidate checkpoint 与 promotion hash。router 训练阶段的 H=3 counterfactual
阈值只用于筛选扫描网格，不能直接打包；最终阈值必须由
`calibrate_b3_router_recursive.py` 在 calibration tracklets 上逐阈值真实递归，
按 Success 最大且 harmful intervention `<=5%` 冻结。无非空安全动作或不优于
observation 时校准和 packaging 都失败。

## 必须遵守的晋级边界

代码只提供可复现机制，不替代真实实验结论。sigma 未通过 ECE/coverage/NLL 门槛时，配置 17 保持 fixed residual margins；B2 未通过 presence/reachability/oracle 门槛时不得训练正式 B3；B4 未同时通过 std、effective-rank、速度和 matched tracking 门槛时不进入论文主表。
