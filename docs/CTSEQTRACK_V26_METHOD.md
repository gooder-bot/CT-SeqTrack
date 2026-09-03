# CT-SeqTrack v26：Observation-Anchored Bounded Evidence Recovery

v26 是当前待实验验证的正式实现；它不改写 B0，而是在 B0 之外增加有界、可拒绝的证据恢复链。历史 v24/v25 配置与失败证据保持只读，不能用其 checkpoint 初始化 v26。

```text
B0 stable observation
  -> B1 bounded adaptive shell + causal history corridor
  -> deterministic 768-point novel pre-pool
  -> relation/spatial/exploration 256-point selection
  -> mode-consistent robust voting
  -> B3 held-out calibrated selective action
```

## 1. 所有权与训练边界

- B0 crop、1024 点、四候选 RNG、`0.5/1⁄6/1⁄6/1⁄6` 损失和网络输出保持原合同；B0 仍是唯一递归状态写入者。
- B1/B2/B3 只读取 canonical view0。启用的 B0/B1/B2/B3 参数从 epoch 0 同时进入 unified Adam，各自使用 named parameter group；没有模块冻结。
- detach 与 BN-stat isolation 是耦合边界，不是冻结。B1/B2 不能直接写框，B3 不能反向更新上游输入特征。
- 训练与未校准评估使用 observation-recursive 状态。只有通过 calibration 与不相交 dev promotion 的 checkpoint 才允许 selective evaluation。

## 2. B1 获取几何

B1 acquisition-margin head 读取 detached temporal context，分别输出运动平行/垂直半边界：

```text
parallel      = 2 + 4 * sigmoid(head_parallel)
perpendicular = 1 + 2 * sigmoid(head_perpendicular)
```

末层 bias 初始化为 `-8`，所以训练初期近似 `2m/1m`。该 head 使用 q=0.90 pinball loss，权重 `0.05`；B1 mean、direction、GT residual 都在该 loss 中 detach。统计 `log_sigma` 仍只服务不确定性学习，不控制 v26 crop。

B1 无效时使用有界 CV endpoint/tube 和固定 `2m/1m` margin；CV 也无效时只保留 B0。B2 structural availability 定义为“因果 support 有效且 novel pre-pool 非空”，不再额外乘 `b1_valid`。

## 3. Causal backup corridor

Corridor 只读取最近三帧递归框、真实物理间隔和第一帧固定尺寸。速度、加速度、外推位移、总长度和宽度上限分别为 `20m/s`、`8m/s²`、`12m`、`16m`、`6m`。近期 transition 违反加速度约束时，从较老的一致 anchor 外推；否则使用两段速度的有界平均。

以下任一条件成立时尝试 corridor：时间间隔异常、endpoint/B0 尺寸比超限、B0 raw crop 少于 64 点、未扩张 inner core 少于 3 点、B1 无效或约束发生 clipping。Corridor 只增加采样支持域，不能产生候选框或写状态。

## 4. Extension-only sampling 与 voting

CPU 端以 `1e-6` XYZ key 精确减去 B0 raw crop并合并来源；source 是 bitmask：endpoint=`1`、tube=`2`、corridor=`4`。0.2m voxel round-robin 构造 768 点 deterministic pre-pool：local 512、corridor 256，来源不足时借用剩余预算，且不消费 B0 RNG。

GPU 端关系 head 使用 detached B0 current/history/memory summaries 与 extension geometry。relation BCE 权重为 `0.25`。去重互补后固定选择 relation top-128、XY FPS coverage-96、stateless exploration-32；只有这 256 点进入 cross-attention、targetness 与 vote head。

每点 vote 权重为 refined targetness × relation probability。K=3 hypothesis 使用 0.75m seed NMS、1.0m inlier radius与 3 次 Huber IRLS。排序分数为：

```text
consistency = normalized_mass * inlier_ratio * exp(-trace(covariance))
```

只融合与 top1 相距不超过 0.75m 的 hypothesis。B3 读取 detached consistency、2×2 covariance、inlier ratio、top1-top2 margin 和 compatible-hypothesis count。空 novel pool 精确返回 B0；单点 covariance 与 margin 为 0。

## 5. Counterfactual schema v3

在线 crop 与诊断使用同一 strict box-membership 函数，nuScenes `wlh` 明确对应 local `(x,y,z)=(length,width,height)`。诊断禁止旧的含混 `target_bearing`，只使用 `raw_target_bearing` 与 `novel_target_bearing`。

固定漏斗为：

```text
global observable -> B0 raw/sample -> support raw -> novel pool
  -> 768 pre-pool -> 256 selected -> consensus inliers
  -> raw candidate -> B3 action
```

正式同行反事实为 `fixed_2_1`、`adaptive_local` 与 `adaptive_dual_support`。`tools/report_ct_b2_v26.py` 会验证 schema、candidate0、集合单调性、PSD covariance，并报告三种支持域的 raw/novel recall、背景、体积、截断和来源。

## 6. B3 held-out calibration v2

B3 保留 helpful、harmful、expected center gain 与 expected IoU gain 四个 head：

```text
action_score = sigmoid(help_logit) * (1 - sigmoid(harm_logit))
```

共识特征只作为 B3 输入，不引入手工阈值。`ct_seqtrack.action_calibration.v2` 先在 calibration tracklets 选择 presence/action 阈值，再锁定阈值到不相交 dev tracklets 做 promotion。两份集合均要求至少 30 个 selected tracklets、100 actions、coverage≥1%、harmful-rate 单侧 95% 上界≤5%，center/IoU gain 单侧 95% 下界≥0。

缺失、未通过、schema/hash/checkpoint/config/code/manifest 任一不匹配时不安装阈值，B3 精确返回 B0。每个 final/late-3 checkpoint 单独导出两份 rows 并生成自己的 artifact，禁止跨 checkpoint 复用。

## 7. nuScenes 注册臂

本轮最新实验顺序先运行 mini、Car、seed42、60 epoch；mini 完成并分析后再从 epoch 0 运行匹配的完整 nuScenes 五臂。集成主臂固定 GRU，B1-CfC 只作为 backend 诊断。mini 注册命令为：

```bash
python main.py --cfg cfgs/ct_seqtrack/26_b0.yaml --path MINI_NUSCENES_ROOT --tag ct26_b0_mini_seed42
python main.py --cfg cfgs/ct_seqtrack/26_b1_gru.yaml --path MINI_NUSCENES_ROOT --tag ct26_b1_gru_mini_seed42
python main.py --cfg cfgs/ct_seqtrack/26_b1_cfc.yaml --path MINI_NUSCENES_ROOT --tag ct26_b1_cfc_mini_seed42
python main.py --cfg cfgs/ct_seqtrack/26_full_minus_b3.yaml --path MINI_NUSCENES_ROOT --tag ct26_full_b3_mini_seed42
python main.py --cfg cfgs/ct_seqtrack/26_full.yaml --path MINI_NUSCENES_ROOT --tag ct26_full_mini_seed42
```

匹配的完整 nuScenes 注册命令为：

```bash
python main.py --cfg cfgs/26_seqtrack_strict_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_seqtrack_strict_seed42
python main.py --cfg cfgs/ct_seqtrack/26_b0_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_b0_seed42
python main.py --cfg cfgs/ct_seqtrack/26_b1_gru_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_b1_gru_seed42
python main.py --cfg cfgs/ct_seqtrack/26_b1_cfc_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_b1_cfc_seed42
python main.py --cfg cfgs/ct_seqtrack/26_full_minus_b3_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_full_b3_seed42
python main.py --cfg cfgs/ct_seqtrack/26_full_nuscenes_full.yaml --path FULL_NUSCENES_ROOT --tag ct26_full_seed42
```

所有命令均不允许 `--init_checkpoint`。同一运行硬件中断时只能用自己的 epoch-boundary `--checkpoint` 恢复。训练期每 2 epoch 验证一次；独立的 epoch-end callback 仍保存 58/59/60 三个 checkpoint，因此 late-3 不依赖验证频率。

对 Full 的每个 final/late-3 checkpoint：

```bash
python tools/export_ct_action_rows.py --config cfgs/ct_seqtrack/26_full_nuscenes_full.yaml --checkpoint CKPT --path FULL_NUSCENES_ROOT --partition calibration --output artifacts/ct_checks/CKPT_calibration.csv
python tools/export_ct_action_rows.py --config cfgs/ct_seqtrack/26_full_nuscenes_full.yaml --checkpoint CKPT --path FULL_NUSCENES_ROOT --partition dev --output artifacts/ct_checks/CKPT_dev.csv
python tools/calibrate_ct_actions.py --rows artifacts/ct_checks/CKPT_calibration.csv --dev-rows artifacts/ct_checks/CKPT_dev.csv --checkpoint CKPT --config cfgs/ct_seqtrack/26_full_nuscenes_full.yaml --tracklet-manifest artifacts/ct_checks/CKPT_calibration.csv.manifest.json --dev-tracklet-manifest artifacts/ct_checks/CKPT_dev.csv.manifest.json --output artifacts/ct_checks/CKPT_action_calibration.json
```

通过后使用该 checkpoint 自己的 artifact、两份 manifest SHA 和 `--proposal_mode selective --test`。未通过时 Full 正式输出就是 observation/B0。

## 8. 结果边界

机制最低门槛为 globally-observable need rows 的 novel-pool target-bearing≥15%、selection row recall≥90%、point recall≥70%。论文主性能要求 Full 相对 B0 的 Success 与 Precision 均不下降，且至少一个指标的 tracklet-paired 95% CI 下界大于 0。

当前实现与本地合同测试不能证明涨分、SOTA、跨 seed 稳定性或分布无关风险保证。若有界双层支持仍取不到 novel target evidence，应升级为长期/全局重检测问题，而不是继续无界放大 shell 或让 B1 直接改框。
