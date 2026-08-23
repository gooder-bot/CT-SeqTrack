# Safe-SeqTrack v25 运行协议

## 1. 正式配置

mini 四臂为 `25_b0.yaml`、`25_b1.yaml`、
`25_full_minus_b3.yaml`、`25_full.yaml`；完整 nuScenes 使用对应的
`25_*_nuscenes_full.yaml`。所有臂从 epoch 0、seed42、batch16 训练
60 epochs，每 epoch 验证。`24_*` 只读且不能续训到 v25。

固定运行身份：

```yaml
ct_runtime_protocol: safe_seqtrack_auto_v1
ct_optimizer_topology: unified_auto
ct_observation_rng_mode: stateless_seqtrack
ct_validation_rng_mode: stateless_tracklet_frame
ct_batch_schema: ct_seqtrack.train.v2
ct_candidate_policy: b2_raw
```

## 2. B0 与随机事务

B0 保留 `1+3` 历史和四个 candidate。candidate0 恒等，candidate1--3
按历史帧独立生成 SeqTrack 分布的均匀扰动。每个 batch 四个分支数量
相同，分支内样本顺序独立打乱，损失为：

```text
L_B0 = 0.5*L_candidate0 + (L_candidate1+L_candidate2+L_candidate3)/6
```

shuffle、扰动、历史点采样和当前帧点采样均由
`seed/epoch/tracklet/frame/candidate/role` 派生，禁止依赖 worker 数量、
iterator 构造顺序或启用的实验臂。validation 使用独立生成器；验证频率
不得移动训练事务。

observation payload 只保留 SeqTrack 前向、损失及候选审计字段。
mechanism payload、监督和 online-recursive 顺序保持 v24 合同。B0 使用
四候选，B1/B2/B3 只使用 canonical candidate0，B2 固定一个 view。

## 3. 优化与耦合

v25 使用一个 Adam：betas `(0.5, 0.999)`、eps `1e-6`、weight decay 取
配置值。参数组顺序固定 B0、B1、B2、B3，每组当前 LR 均为 `1e-4`；
StepLR 每 epoch 由 Lightning 自动更新。未调度模块梯度为 `None`，Adam
跳过该参数组。正式训练为 FP32，无手动 scaler、无跨模块梯度裁剪。

mechanism 事件首先在 `no_grad` 和 BN-stat 隔离下生成 B0 contract，随后
只为 B1/B2/B3 建图。detach 边界、B0-only recursive writer、B2
extension-only、B3 calibrated/fail-closed 均不改变。

默认 `b2_raw` 保持现有晚耦合。代码保留 `prior_backoff`：B2 candidate
无效时可退到物理半径约束的有效 B1 candidate，否则精确退回 observation；
该策略不是首轮四臂默认，只有完成既定 B1/B2 门槛后才能新建第五个
scratch mini 实验。

## 4. 审计和验收

checkpoint 写入 v7 resume contract、resolved-config SHA、候选/RNG 协议、
命名参数组、模块更新次数和 B0 initialization/step1/step100/epoch-end
hash。只允许同一协议、同一实验、epoch-boundary resume。

本地必须通过 pytest 和 compileall。服务器正式训练前必须完成：相同安全
batch 的 SeqTrack/CT-B0 输入、输出、loss、梯度与一次 Adam 更新对照；
四臂 100-step B0 hash 对照；validation cadence 1/5 对照；连续与 resume
等价检查；分阶段 CUDA allocated/reserved/peak 报告。smoke checkpoint
必须丢弃。

mini 只比较 final 和 late-3。完整 nuScenes 的进入条件、B1/B2/B3 机制
指标和 B3 held-out calibration 门槛继续遵循正式实验计划，不能以历史
高分替代。
