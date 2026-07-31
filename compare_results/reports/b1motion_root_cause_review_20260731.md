# B1motion 根因复核：为什么当前版本远差于 Dyn / M2

日期：2026-07-31  
范围：nuScenes-mini Car、seed42；当前 B1motion-v2、旧 B1、A2-Dyn、M2；代码、训练指标与相关论文交叉复核。

## 结论

当前 B1motion-v2 的失败不是单一超参数问题，也不是训练没跑完。它是三个训练合同错误串联后的结果：

1. **首要问题：35% irregular sampling 替换了整个 SeqTrack3D 主训练视图。** 历史点云、历史框、motion label、coarse box query 和 Transformer 序列同时改变，但主干仍使用不含真实 gap 的 `order` token。Adapter 尚为严格零输出的 epoch1-2，核心 center/motion loss 已经明显分叉，说明破坏先发生在 B0 主路径。
2. **轨迹辅助目标含不可识别项。** Ordered encoder 只看相对 candidate-anchor 的历史轨迹，target 却是 `current GT - candidate anchor`；其中的最新 anchor 误差无法由相对历史唯一恢复。75% 非 candidate0 样本因此把定位修正噪声当成物理 motion 监督。
3. **epoch3 以后，未限幅的 feature adapter 把上述有噪轨迹表示注入 coarse motion。** `normal_scale=0.1` 只是乘数，不是 correction norm 上限；极弱的 `1e-4` L2 无法维持 B0 identity，随后错误 coarse proposal 又进入 SeqTrack3D 的 box-sequence decoder，形成递归放大。

Pre-crop search 没有救回模型：它在全部训练样本上的有效率只有约 3.9%，折算到最多 35% irregular 样本中也仅约 11%；而 `valid` 只表示扩区内至少有 16 个额外点，不表示这些点包含目标。

因此应判定为：

> **NO-GO 当前 B1motion-v2；但不能外推为 motion / trajectory 方向无效。**

## 当前实验事实

| 模型 | Final Success | Final Precision | Best S / P | Late-3 S / P |
|---|---:|---:|---:|---:|
| B0 baseline（历史） | 53.360 | 64.382 | 54.135 / 64.382 | 52.905 / 63.104 |
| Legacy B1, alpha=0 | 47.049 | 49.184 | 49.876 / 58.691 | 46.828 / 49.669 |
| Legacy B1, alpha=0.25 | 29.581 | 28.862 | 35.027 / 41.130 | 29.472 / 28.849 |
| **B1motion-v2** | **20.618** | **19.830** | **30.196 / 34.990（epoch5）** | **21.777 / 21.195** |

B1motion-v2 相对 B0 final 下降 32.742 Success / 44.551 Precision。训练有完整的 75,720 step、12 次验证和 epoch60 checkpoint；最佳点出现在 epoch5，随后总体下降，所以不是截断、坏 final checkpoint 或“再训一会儿就会恢复”。

完整原始表见：

- [`b1motion_v2_seed42_20260730_validation.csv`](../data/b1motion_v2_seed42_20260730_validation.csv)
- [`b1motion_v2_seed42_20260730_training_epochs.csv`](../data/b1motion_v2_seed42_20260730_training_epochs.csv)
- [`b1motion_v2_seed42_20260730_integrity.csv`](../data/b1motion_v2_seed42_20260730_integrity.csv)

## 根因 1：Adapter 开启前，B0 主路径已经被 mixed cadence 破坏

配置以 35% 概率选择 query gap 2/4 和不规则 transition gaps：

- [`02_ct_motion.yaml`](../../cfgs/ct_v2/02_ct_motion.yaml) 中 `trajectory_training_irregular_probability: 0.35`；
- [`sampler.py`](../../datasets/sampler.py) 的 `_sample_history_offsets()` 产生 irregular offsets；
- 同一 offsets 随后直接传给 `_build_view()`，所以替换的不是独立 trajectory view，而是完整主视图；
- [`nuscenes_mini_base.yaml`](../../cfgs/ct_v2/nuscenes_mini_base.yaml) 仍固定 `main_time_source: order`。

这造成两个同时发生的变化：

1. 主干面对更大的点云形变、目标位移、crop miss 和历史缺失；
2. 物理 query gap 没有作为主干时间条件输入。相同 order token 可以对应不同 query gap，尤其 query gap 2/4 与历史 transition pattern 独立抽样时，主干不能从 token 确定预测时域。

最强证据来自 warmup：

| Epoch | 指标 | B0 | B1motion-v2 | 比值 |
|---:|---|---:|---:|---:|
| 1 | `loss_center` | 0.1816 | 0.4907 | 2.70x |
| 1 | `loss_center_motion` | 0.3377 | 0.9869 | 2.92x |
| 2 | `loss_center` | 0.1144 | 0.4180 | 3.65x |
| 2 | Adapter correction | - | **严格为 0** | - |

`trajectory_adapter_warmup_epoch=2`，代码在 epoch1-2 把 adapter scale 设为 0。初始化检查也证明 320 个共享 B0 tensors 完全一致。因此在这两轮中，新增 motion 参数不能通过 adapter 改变 coarse head；核心损失分叉只能来自被替换的训练视图、相应的随机采样序列及其更困难的标签分布，而不是 adapter 本身。

另一个旁证是 `valid_history_ratio`：B0 约 0.903，B1motion-v2 约 0.854。跳帧在 tracklet 前段制造了更多 padding/无效历史，进一步削弱了 SeqTrack3D 原本依赖的连续序列。

这里仍保留一个因果边界：B0 来自历史 commit `d86990c`，B1 run 来自 `5f260e7` 加 dirty patch。共享初始化 identity 大幅降低了混杂，但**精确归因仍需要当前最终代码上的 same-code B0 和 data-shift-only 对照**。

## 根因 2：Trajectory target 在数学上含不可观测的 anchor error

当前 ordered history 先以最新 candidate crop anchor 表达：

```text
r_i = history_box_i - candidate_anchor
```

轨迹 target 则由 [`anchor_relative_trajectory_targets()`](../../utils/candidate_utils.py) 定义为：

```text
y = current_GT - candidate_anchor
  = (current_GT - latest_GT) - anchor_error
  = physical_motion - anchor_error
```

对所有 candidate history 与 anchor 同时施加共同平移 `c`，相对输入 `r_i` 不变；但在 current GT 不变时，target 变为 `y-c`。因此只接收相对历史的 OrderedTrajectoryEncoder 不可能唯一恢复 `anchor_error`。

这不是“坐标系格式不一致”，而是**监督目标包含输入中不存在的信息**。当前 `num_candidates=4`，candidate0 为 identity，约 75% 样本是非零 candidate；它们持续把候选框误差混入 velocity、displacement 和 trajectory NLL。训练日志也符合这一点：epoch1 到 epoch60，`loss_velocity` 仅从约 0.202 降到 0.176，`loss_dynamics_displacement` 仅从约 0.176 降到 0.156，远没有学成一个可靠 motion prior。

当前点云 observation feature 的确可以感知“目标相对 crop anchor 在哪里”，所以**完整跟踪器**有能力修正 anchor error；问题在于这项修正被错误地监督给了**只看历史框**的 trajectory head。正确分工应是：

- trajectory head：预测 candidate-independent 的 canonical physical motion；
- observation/refinement head：读取当前点云证据，估计 anchor correction；
- 最终模块：在可观测性/一致性门控下组合两者。

## 根因 3：Recursive candidate 模拟把人工漂移直接当成速度

[`ct_history.py`](../../utils/ct_history.py) 的 `recursive_candidate` 会让 older history 的 correlated error 随年龄按 `sqrt(age) * 4.0` 放大。Ordered encoder 又把最近 transition 速度作为零初始化时的 `base_rate`，即：

```text
base_rate = (newer_candidate - older_candidate) / delta_t
prediction = (base_rate + learned_residual) * query_gap
```

因此 candidate history 中的人工差分误差会直接进入冷启动速度。它既不是纯物理运动，也不等于真实 B0 recursive rollout 中与遮挡、crop miss、点云稀疏相关的误差。训练和推理虽然都叫“recursive”，但误差生成机制并未匹配：

- 训练：独立采样后做相关化、按年龄固定放大；
- 推理：误差由上一帧网络输出、当前点云质量和 crop 共同产生，并带强时序相关。

这使 trajectory 表示在进入 adapter 前已经带有系统性伪速度。

## 根因 4：Adapter 的“0.1”并不安全

[`ZeroInitTrajectoryAdapter`](../../models/ct_v2/motion.py) 的实际形式是：

```text
correction = MLP(inputs) * sample_scale
adapted_feature = observation_feature + correction
```

`normal_scale=0.1` 没有约束 `MLP(inputs)` 的大小，网络可以通过增大权重抵消这个 0.1。实测：

| Epoch | Mean correction L2 | Raw mean norm-squared |
|---:|---:|---:|
| 1-2 | 0 | 0 |
| 3 | 1.859 | 22.18 |
| 5 | 2.619 | - |
| 60 | 2.072 | 19.017 |

epoch60 的 adapter penalty 乘 `trajectory_adapter_l2_weight=1e-4` 后只有约 0.0019，几乎没有约束力。此时 `loss_center_motion` 为 B0 的 8.47 倍，而 refinement center loss 只为 1.87 倍，说明主要损伤首先发生在 coarse motion / box query，再被后续 decoder 部分修补但无法挽回。

这也解释了为什么 zero-init 检查全部通过仍然会失败：zero-init 只保证 step0 恒等，不保证训练后的函数仍接近恒等。

## 根因 5：Pre-crop 分支覆盖率和证据质量不足

当前 search 只有在 gap trigger 成立、预测位移非静止、扩区相对 baseline 至少多出 16 个点时才标记 valid。训练均值：

- `trajectory_search_valid`：约 3.9%-4.1%；
- irregular sampling：35%；
- 因此即使假定所有 valid 都来自 irregular，conditional coverage 也只有约 11%-12%。

此外，`valid` 检查的是额外点数量，不检查目标 recall、foreground ratio 或与 local proposal 的一致性；扩区点仅经轻量 PointNet/max-pool 编码。高时间变化文献指出，扩大搜索区会同时引入相似目标和背景噪声，HVTrack 为此分别设计了 base-expansion cross-attention 和背景抑制，而不是只增加一个扩区池化特征。[HVTrack, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php)

所以这条分支目前既覆盖太低，也没有证据证明有效样本真正包含目标；它不是首要崩溃源，但无法承担 irregular robustness 的补救作用。

## 为什么以前的 Dyn 没有这样掉

历史 A2-order-dyn 的最终结果是：

| 模型 | Success | Precision | 相对 SeqTrack baseline |
|---|---:|---:|---:|
| SeqTrack baseline | 50.986 | 59.962 | - |
| A2-order-dyn | 50.962 | 63.314 | -0.024 / +3.352 |
| A2-order-dyn-disp | 50.542 | 63.848 | -0.444 / +3.886 |

但它与 B1motion-v2 有四个关键差异：

1. **主训练视图仍是连续 cadence**，没有以 35% 概率替换完整 SeqTrack3D 输入；
2. 主干保留稳定 `order` token，真实时间只进入 DynamicsEncoder；
3. `dynamics_motion_mode=feature` 时，`z_dyn` 与 point feature 拼接，由下游 MLP 自行选择，模型可以弱化或忽略不可靠 dynamics；
4. 没有 trajectory feature 对 coarse motion 的无上限 residual 改写，也没有把 synthetic anchor correction 当作 trajectory endpoint。

早期 raw-time A1 从 50.986/59.962 崩到 28.277/27.429，A2 Dynamics 恢复到 45.266/58.832。它主要说明 Dynamics 能帮助一个已被 raw-time 主分支破坏的模型恢复，而不是证明任何 motion 注入都会独立涨点。

Cand1 的 26.677/24.499 也提醒：旧 Dyn 的稳定性部分依赖四候选训练预算/正则化；cand1 同时少了约 4 倍 optimizer steps，不能简单解释为“去掉 candidate noise 更好”。

## 为什么以前的 M2 看起来更好

M2 的三个训练结果：

| Run | 初始化 / 对照 | Success | Precision |
|---|---|---:|---:|
| R1 | A1-init + M2 再训练 60 epoch | 55.303 | 67.182 |
| R2 | scratch M2 | 53.318 | 62.503 |
| R3 | scratch matched W0, shared-SE(2) | 28.999 | 28.023 |

这些结果是真实的，但不能解释成“正确物理时间带来全部增益”：

- R1 比历史 A1 多训练了 60 epoch，没有 A1-init W0 continuation；
- R2/R3 的巨大差值主要说明 M2 能救回一个严重塌陷的 shared-SE(2) W0，对照本身不是历史 SeqTrack3D；
- 同一 R1 checkpoint 的 true/fixed/shuffled 时间评测几乎相同。Standard true-fixed 为 +0.031/-0.010，true-shuffled 为 +0.068/+0.085；gap1124 的 shuffled 反而比 true 高 0.318/0.209。现有证据否定了“正确 physical delta_t 是 M2 正信号来源”的因果主张。

即便如此，M2 比当前 B1 更安全：

1. normal 主视图没有被 irregular sampling 替换；
2. velocity/displacement 使用 candidate-independent canonical target；
3. adapter 有 5-epoch warmup；
4. proposal innovation 在坐标空间有明确半径上限 `R(delta_t)`，而不是无界 feature correction；
5. 其结构更接近“observation proposal -> bounded motion correction”。

原始 M²-Track 本身也是先用两帧点云分割和相对运动生成 coarse box，再用 motion-assisted shape completion refinement，而不是让只看历史框的轨迹分支独立承担 observation correction。[M²-Track, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.html)

## 与相关研究的一致性

当前失败并不反常，反而与几篇关键论文的边界条件高度一致：

- SeqTrack3D 报告 1+3 历史优于 1+1，但扩到 1+7 又接近 1+1；作者直接把退化归因于训练 random offsets 难以模拟测试条件及历史框累积误差。这与当前 `recursive_error_scale=4` 加 mixed cadence 的问题同源。[SeqTrack3D](https://arxiv.org/abs/2402.16249)
- HVTrack 明确指出大 frame interval 会同时破坏平滑运动假设、增加点云形变、扩大搜索区并引入相似目标/背景噪声，因此使用专门的相对位姿记忆、base-expansion cross-attention 和背景抑制模块。[HVTrack](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php)
- TrajTrack 的 v3/ICRA 2026 采用明确的 “propose-predict-refine”：先由当前两帧点云形成 local proposal，再由历史框轨迹形成 global prior，最后根据两者一致性做 proposal 级选择/修正。它没有把不可观测 anchor error 直接监督给 trajectory-only encoder。[TrajTrack](https://arxiv.org/html/2509.11453v3)
- STTracker 和 SeqTrack3D 都让历史点云、历史框与当前点云共同参与时空融合；这比“只用历史轨迹预测完整当前 anchor-relative target”具有更完整的观测条件。[STTracker](https://arxiv.org/abs/2306.17440)

## 最小定位实验：不要再直接跑一轮 60 epoch

建议按以下顺序执行，每项先做 10-15 epoch kill test：

1. **E0 same-code B0**  
   当前最终代码、seed42、完全相同数据选择和初始化，纯连续 cadence。先消除 commit/patch 混杂。

2. **K1：`irregular_probability 0 / 0.35 x adapter off / on`，search 全关**  
   这是首要 2x2 factorial。若 adapter off 时 0.35 已显著拉高 epoch1-2 core loss，主因“主视图替换”成立；若 0 cadence + adapter on 也掉，则 adapter 本身独立有害。

3. **K2：Target contract A/B**  
   固定 continuous view、adapter off，仅比较：
   - canonical physical motion target；
   - 当前 anchor-relative target。  
   记录 candidate0 与 candidate1-3 的 endpoint RMSE。若问题判断正确，anchor-relative 的非零 candidate error 会出现明显 irreducible floor。

4. **K3：分离 physical prior 与 observation correction**  
   主视图永远保持 B0。Irregular history 只作为 paired auxiliary view。Trajectory head 预测 physical motion；correction head 必须读取当前点云/local proposal，并使用 stop-gradient prior 或 proposal disagreement 作为输入。

5. **K4：真正受限的 residual**  
   Normal cadence 永久 exact identity；irregular residual 使用相对上限，例如 `||delta_f|| <= beta * ||f_obs||`，再乘 GT-free evidence gate。不能再用可被网络权重抵消的固定 0.1。

6. **K5：Search observability audit**  
   分开记录 geometric trigger、extension available、目标点/GT recall（仅离线诊断）、foreground evidence、最终 applied。先证明扩区提高 target recall，再决定训练；不要直接降低 `min_points`。

晋级标准仍应严格：normal 相对 same-code B0 达到 Success >= -0.3、Precision >= -0.5，epoch1-2 core loss 不分叉，之后才允许 60 epoch、random20/gap1124 和多 seed。

## 最终判断

真正的问题不在“GRU 不够强”“NLL 为负”或“训练轮数不够”，而在模块分工和训练分布：

```text
当前 B1:
irregular view 替换主路径
    -> gap-blind SeqTrack3D 主干先受损
    -> trajectory head 学不可识别的 physical motion + anchor error
    -> 无界 feature adapter 注入有噪 prior
    -> coarse box / Transformer query 被污染
    -> recursive validation 持续漂移

应修改为:
连续 B0 主路径保持不变
    + 独立 irregular auxiliary trajectory view
    + candidate-independent physical motion prior
    + 读取当前点云的 bounded observation correction
    + 仅在可观测、长 gap 且 search 有证据时启用
```

因此，下一版 B1 的目标不应是“让轨迹分支更强”，而应是**让轨迹分支只预测它能够观察到的量，并保证任何不可靠运动先验都无法破坏 B0 主路径**。
