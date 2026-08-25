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

checkpoint 写入 v8 resume contract、resolved-config SHA、候选/RNG 协议、
命名参数组、模块更新次数、各启用模块最大梯度范数，以及 B0
initialization/step1/step100/epoch-end 参数 hash。initial/step1/step100 同时记录
Adam `step/exp_avg/exp_avg_sq` 状态 hash。只允许同一协议、同一实验、
epoch-boundary resume。

本地必须通过 pytest 和 compileall。服务器正式训练前必须完成：相同安全
batch 的 SeqTrack/CT-B0 输入、输出、loss、梯度与一次 Adam 更新对照；
四臂 100-step B0 hash 对照；validation cadence 1/5 对照；连续与 resume
等价检查；分阶段 CUDA allocated/reserved/peak 报告。smoke checkpoint
必须丢弃。

mini 只比较 final 和 late-3。完整 nuScenes 的进入条件、B1/B2/B3 机制
指标和 B3 held-out calibration 门槛继续遵循正式实验计划，不能以历史
高分替代。

## 5. B1 修复语义与可切换后端

v25 直接采用修复后的 B1，不新建 v26。GRU 是默认正式实现；CfC 只替换
chronological transition aggregator，其余 step projection、物理特征、固定
kinematic anchor、context、均值头、sigma 头和输出合同完全共享。命令行使用
`--b1-backend gru|cfc`；未选择的后端不实例化。正式尺寸下 GRU 为 74,496
参数，CfC 为 74,537 参数。

B1 均值监督在运动坐标系内学习归一化残差。envelope 内使用
`SmoothL1(beta=0.25)`；envelope 外只施加 margin 0.9、权重 0.25 的方向
hinge。sigma 使用 `0.1 + softplus(raw)`，上界为 `exp(2.5)m`，读取 detached
context；beta-NLL 的均值误差和方向同样 detach。训练 NLL 只把每轴有限误差
winsorize 到 12m，报告的 Gaussian NLL 与 coverage 使用未截断误差。

main、gap2、gap4 的均值与 sigma 损失必须全部进入同一个 B1 transaction；
代码运行时会把 transaction 与记录的四项加权损失逐值核对。归约先按 query
gap，再按 recursive age `[0,1] / [2,3] / [4,7] / [8,+∞)` 对非空组等权。
reset 边界年龄归零；缺失年龄标记为 invalid。

B2 geometry 固定：`search_v3_use_dynamic_sigma=false`，parallel/perpendicular
margin 始终为 `2m/1m`。B1 sigma 只允许作为诊断和 detached 下游 feature，
校准 artifact 不得写回或改变 B2 margin。

B1 calibration v3 必须同时绑定同一 scratch checkpoint 的两个原子分区：
calibration tracklets 只拟合 parallel/perpendicular log-scale，dev tracklets
只计算 promotion。promotion 要求 coverage ECE `<=0.05`、coverage95 `>=0.90`、
校准 NLL 优于 fixed-sigma，以及 learned-vs-CV tracklet paired-bootstrap 95% CI
上界 `<0`。生成的 post-hoc checkpoint 仅供评估，并显式禁止 resume。
