# CT-SeqTrack `delta_t` 可辨识性、HTV 论文边界与执行计划

更新时间：2026-07-22

状态：**研究与实验设计决策，不是性能结果。** 本文回答三个问题：标准 nuScenes 上很小的 `delta_t` 波动是否会掩盖物理时间创新；受控丢帧能否形成论文；M2 formal 和从头训练对照完成后应如何决策。

## 1. 结论先行

1. **标准协议不适合作为物理时间创新的唯一主战场。** 当前 standard 的 `delta_t` 几乎恒为 `0.5 s`，时间条件函数接近常量，很容易被普通权重吸收；即使总体分数上涨，也不能自动归因于正确物理时间。
2. **standard 仍然必须保留。** 它的论文角色是正常 cadence 下的性能 guardrail，而不是要求时间模块必须在近常量时间上显著涨分。
3. **受控丢帧可以支撑论文。** HVTrack 已在 ECCV 2024 用不同 frame interval 构造 KITTI-HV；因此不能再宣称“首次 HTV/首次 skip-frame”，必须强调同一 tracklet 内不规则间隔、matched endpoint 时间干预、单 checkpoint 跨 cadence 和 held-out schedule 泛化。
4. **当前六组旧 HTV 结果只是 pilot。** random20 的旧 A2 为正，但 gap1124/burst-drop 为负；它们还是单 seed、mini、按协议分别训练且总 optimizer steps 不同，不能进入最终方法主表。
5. **当前先等三个训练任务结束，不继续写 M3/M4。** 先回答 M2 是否有效、收益是否依赖 A1 初始化、增加新模块本身是否优于 matched W0；只有 online 指标和时间负对照同时成立，才升级方法路线。

## 2. 本地 `delta_t` 证据

统计来自冻结诊断 CSV：

- `output/diagnostics/reliability_signals/standard_p0b3/reliability_endpoints.csv`
- `output/diagnostics/reliability_signals/gap1124_p0b3/reliability_endpoints.csv`
- `output/diagnostics/reliability_signals/burst_drop_p0b3/reliability_endpoints.csv`

| protocol | endpoint | mean (s) | std (s) | CV | P50 (s) | P95 (s) | max (s) | `dt > 0.75 s` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 4246 | 0.4974 | 0.0228 | 4.59% | 0.4999 | 0.5491 | 0.5999 | 0.00% |
| gap1124 | 2127 | 0.9870 | 0.5818 | 58.94% | 0.9500 | 2.0012 | 2.1499 | 52.99% |
| burst-drop | 2098 | 1.0006 | 0.6267 | 62.63% | 0.5005 | 2.0012 | 2.1499 | 41.85% |

standard 中约 `86.55%` endpoint 位于 `0.5 ± 0.01 s`。这不是“样本不够多”能单独解决的问题，而是时间变量的实验激励不足。

若模块为：

```text
h'_t = h_t + g(delta_t) * r_t
```

且 `delta_t ~= 0.5`，则：

```text
g(delta_t) ~= g(0.5) = constant
```

该常量可被线性层、bias、归一化或其他结构吸收。由此产生三种风险：

- 模型直接忽略时间分支；
- `true/fixed/shuffled` 输出几乎相同，机制不可辨识；
- standard 上的涨点来自参数量、初始化或正则化，而不是时间与运动的正确对应。

P0-B3 已出现相同警告：standard-only calibrator 中 raw `current_delta_t` 跨协议外推时导致过触发，删除它后 gap/burst AUROC 恢复到 `0.865/0.872`。这说明当前可靠性信号主要来自 observation quality，而不是已经成立的 timestamp mechanism。

## 3. 受控丢帧的论文合法性与新颖性边界

nuScenes 的 LiDAR 原始采样频率为 20 Hz，但标准人工标注关键帧为 2 Hz；标准 3D SOT 关键帧间隔因此约为 `0.5 s`。来源：[nuScenes，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)。

HVTrack 已构建 KITTI-HV，通过不同 frame interval 模拟 skipped tracking、边缘设备处理不足和高动态场景，并在 interval 2/3/5/10 下评测；这是“受控降采样可以形成同行评审论文”的直接先例。来源：[HVTrack，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php)。

MambaTrack3D 预印本进一步在 KITTI-HTV 与 nuScenes-HTV 上研究状态空间模型，说明该方向仍然活跃，也提高了新颖性要求。来源：[MambaTrack3D，arXiv 2025](https://arxiv.org/abs/2511.15077)。

因此论文不能使用以下 claim：

- 首次提出 HTV 3D SOT；
- 首次通过丢帧评测 3D tracker；
- 人工丢关键帧等同真实 LiDAR packet loss；
- 一个 random20 结果即可证明跨采样率泛化。

更可防御的区别是：

1. **within-track irregular cadence**：同一 tracklet 内 `delta_t` 不断变化，而不是整条序列固定 interval；
2. **matched causal controls**：相同 endpoint、点云、candidate、sampling seed 和 checkpoint，只改变 effective time；
3. **one-checkpoint generalization**：一个 checkpoint 直接测试 standard、seen cadence、unseen cadence；
4. **dual-clock modeling**：order clock 保留 SeqTrack3D 顺序语义，physical clock 只控制动力学/创新；
5. **failure diagnosis**：同时报告 crop reachability、首次失控、连续失败、fallback 和长 gap 分桶。

丢关键帧应表述为：

> virtual-rate / irregular-observation stress test，模拟感知计算跳过、处理延迟或低频运行。

只有引入真实系统丢帧统计、原始 sweeps 与可信中间标签后，才应进一步声称“真实 sensor packet-loss benchmark”。

## 4. 正式协议矩阵

### 4.1 训练域

| train regime | 用途 | 必须回答的问题 |
| --- | --- | --- |
| standard-only | 最严格的零适配泛化 | 正常 0.5 s 训练能否直接适应不规则 cadence |
| mixed-cadence | 主方法训练 | 模型能否利用不同 `delta_t`，而不只是记住一种 schedule |
| per-cadence specialization（可选上界） | 与既有 HTV 工作对齐 | 分别训练的上界有多高，不作为主要泛化 claim |

mixed-cadence 必须预先留出至少一种 pattern 和一种 drop 强度，不得训练后再挑 unseen schedule。

### 4.2 测试域

| test protocol | 推荐定义 | 论文角色 |
| --- | --- | --- |
| standard | 原始约 0.5 s keyframe | guardrail |
| fixed skip | `K=2/3/4`，约 1.0/1.5/2.0 s | 与固定 interval HTV 对齐 |
| gap1124 | `0.5/0.5/1.0/2.0 s` 循环 | 轨迹内确定性不规则 |
| random drop | `p=0.2/0.4`，manifest 固定 | 独立随机缺失 |
| burst drop | 连续丢 1–3 帧，manifest 固定 | 连续中断 |
| unseen schedule | 训练完全未出现的 pattern/probability | 跨 cadence 泛化主证据 |

### 4.3 时间因果控制

每个 M2/M3/M4 final checkpoint 都必须在同一 endpoint 上评测：

```text
true-dt      : 正确物理时间
fixed-dt     : 所有 transition 使用 0.5 s
shuffled-dt  : 保留 dt 边际分布，破坏 dt 与具体运动 transition 的对应
```

`fixed/shuffled` 是 **same-checkpoint evaluation-only controls**，不能分别训练。只有 `true` 同时超过两者，才可写“正确物理时间对应关系产生收益”。

### 4.4 公平性硬约束

- train regime 内 optimizer steps 相同；不能因丢帧减少而缩短训练预算；
- checkpoint 规则固定为 final/last，不按 test 选择 best；
- endpoint、当前点云、candidate 数、crop/search budget 和 point-sampling seed 匹配；
- manifest 按 split 冻结，保存 selection/mapping hash；
- 评测 retained endpoints；若要预测被删除帧，必须另行定义“无当前输入预测”任务，不能混入同一表；
- 至少 3 seeds，使用 tracklet-level bootstrap，不能把帧当独立样本；
- 最终补 full nuScenes 和第二数据集/官方 HTV 协议；mini 只作筛选。

## 5. 2026-07-22 训练任务登记

以下状态来自服务器已回传日志和用户当前运行说明；尚未拉回的输出不得写成完成结果。

| ID | 训练 | 初始化 | 目的 | 当前状态 |
| --- | --- | --- | --- | --- |
| R1 | M2 formal true-dt，60 ep / 75720 steps / workers12 | 冻结 A1 `last.ckpt`，SHA256 `a2fbff...a82` | 主方法、低方差 continuation、与 A1 同域比较 | **运行中**；E6 preflight PASS，输出根为 `output/m2_formal_true_seed42_473738f_20260722_112536` |
| R2 | M2 full scratch，60 ep / matched budget / workers12 | 随机初始化 | 检查 M2 是否只能依赖 A1 表征 | **用户报告运行中**；结果/精确 OUT_ROOT 待回传 |
| R3 | W0 scratch matched baseline，60 ep / matched budget / workers12 | 随机初始化 | 隔离“从头训练本身”和新增 M2 结构的净效应 | **用户报告运行中**；结果/精确 OUT_ROOT 待回传 |

R1 回答“在已知稳定 A1 解附近加入 zero-init 时间创新是否有效”；R2-R3 的差值回答“从头训练时 M2 相对同预算 W0 是否有结构净收益”。不能用 `R2 - R1` 单独评价方法，因为两者同时改变了初始化与优化轨迹。

建议结果符号：

```text
Delta_continuation = R1_M2 - frozen_A1
Delta_scratch      = R2_M2 - R3_W0
Init_effect_M2     = R1_M2 - R2_M2       # 只作初始化敏感性，不是方法主效应
```

## 6. 三个任务完成后的执行顺序

### Step 1：先验收训练完整性，不先看涨跌调参

对 R1/R2/R3 分别固定并拉回：

- `console.log` / launch log / `training_exit_code.txt`；
- `last.ckpt`、epoch、global step、SHA256；
- resolved config/hparams、seed、batch、workers、optimizer steps；
- run provenance、formal contract、artifact manifest；
- TensorBoard event 文件或逐 epoch metric CSV；
- R1 自动生成的完整 formal archive 和 `.sha256`。

任何任务若不是 epoch60/预定总 step、发生 resume 漂移或缺少 final checkpoint，不进入正式比较。

### Step 2：先做 standard final 指标，再做 R1 正式 controls

1. 统一用 `last.ckpt` 评测 R1/R2/R3 standard；
2. 对 R1 运行现有 `tools/run_m2_formal_time_controls_gpu3.sh`，导出 standard/gap1124/burst-drop 的 M2 `true/fixed/shuffled` 与 matched A1；
3. R2/R3 第一轮只做 matched `M2 scratch - W0 scratch`；只有这个差值为正，再扩它们的 strong-cadence/time-control，避免无目的增加评测矩阵；
4. 从 endpoint CSV 计算 per-tracklet paired delta、bootstrap CI、`delta_t`/位移/稀疏度分桶、首次失控和连续失败。

### Step 3：按预注册门槛决策

方法路线至少要求：

- standard guardrail：相对 matched baseline 不明显退化；
- strong cadence：gap1124/burst-drop 至少一项达到 `+1 Success / +2 Precision`，另一项不得为负；不允许只靠 random20；
- causal time：`true > fixed` 且 `true > shuffled`；
- 置信区间：主要 paired effect 的 tracklet bootstrap 95% CI 支持正效应；
- scratch fairness：`R2_M2 > R3_W0`，否则 A1 continuation 的涨点不能证明从头训练时新增结构有净价值。

当前沿用的最低 promotion 门槛为：

```text
min(true - fixed, true - shuffled)
    >= +0.5 Success
    >= +1.0 Precision
```

standard 可作为 guardrail，不强求显著上涨；non-inferiority margin 现在冻结为相对 matched baseline `Success -0.5 / Precision -1.0`，不得在查看 R1 final 后更改。

### Step 4：路线分叉

| 结果组合 | 决策 |
| --- | --- |
| R1/R2 都优于 matched baseline，且 true controls/strong cadence 通过 | **方法路线 GO**：补 seed43/44、full nuScenes、第二数据集；之后才考虑 M3 |
| R1 为正、R2-R3 不为正 | 暂称 A1 continuation specialization；先诊断优化/初始化，不作通用 M2 claim，不进入 M4 |
| M2 指标为正但 `true ~= shuffled/fixed` | 只能称结构/正则收益，不能称 physical-time method；转 benchmark 或非时间方法叙事 |
| 只有 random20 为正，gap/burst 为负 | 论文证据不足；停止增加模块，优先修复 strong-cadence 泛化/搜索可达性 |
| 多个 tracker 都在不规则协议系统退化，M2 因果收益不稳 | **benchmark/diagnosis 路线**：扩多模型、多数据集、失效曲线和公开 manifests |

## 7. M3/M4 解锁边界

- **M3**：只有 R1 的 true-time causal gate 与 standard guardrail 通过后，才实现 asymmetric canonical-teacher -> irregular-student distillation。
- **M4-0**：只有 frozen M2 predicted history 在固定点预算下表现出 tube-only crop complementarity，才做 trajectory tube oracle。
- **M4-1 以后**：fixed-Q/R GT-free filter 必须先为正；calibration 通过后才允许 learned covariance。
- 任一 gate 失败，不用更大 Transformer/Mamba/ODE、learned Gate 或扩大 crop 来掩盖失败。

## 8. 投稿证据底线

方法论文最低需要：

- full nuScenes + KITTI-HV/Waymo 等第二数据集；
- SeqTrack3D/W0、HVTrack、至少一个运动/轨迹方法的公平基线；
- standard、固定 skip、随机、burst、held-out schedule；
- 3 seeds、same-checkpoint time controls、tracklet bootstrap；
- 参数量、FLOPs、FPS、显存；
- 公开 manifest、时间干预定义和 failure diagnostics。

如果 M2 机制未通过，但形成了跨模型、跨数据集的可靠退化规律，则可转为 benchmark/diagnostic paper；此时贡献中心必须是规范化协议、系统失效分析和可复现资产，而不是宣称当前时间模块有效。
