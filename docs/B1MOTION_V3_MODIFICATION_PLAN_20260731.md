# B1motion-v3 修改方案：保护 B0 的物理运动先验插件

日期：2026-07-31

## 决策

当前 `cfgs/ct_v2/02_ct_motion.yaml` 应冻结为失败复现配置，不再通过调小
`trajectory_adapter_scale`、降低 irregular probability 或增加 search points
继续修补。

下一版建议命名为 **B1motion-v3 / physical-proposal plugin**，第一版只做：

1. 使用 same-code B0 checkpoint，并冻结 B0；
2. 连续 cadence 永远走 B0 原路径，输出必须逐元素完全相同；
3. irregular view 只训练独立的物理运动 prior；
4. prior 与 B0 observation 只在 box/proposal 空间做有界 innovation；
5. 第一阶段只修正平面 `x/y`，`z/yaw` 保留 B0；
6. search 暂时关闭，等 prior-only 通过后再独立解锁。

这不是保守调参，而是修正三项根本合同：训练视图、监督目标和融合位置。

## 为什么不能继续修 B1motion-v2

当前失败是结构性的：

- `trajectory_training_irregular_probability=0.35` 替换整个 B0 历史视图，
  但主干仍使用 gap-blind `order` token；adapter 还为零时主损失已经分叉。
- ordered encoder 只看到 candidate-anchor 下的相对历史，target 却是
  `current GT - candidate anchor`，其中混入了无法从相对历史恢复的 anchor
  平移误差。
- `ZeroInitTrajectoryAdapter` 把有噪轨迹表示注入 256 维主特征；
  `normal_scale=0.1` 不是范数硬上限，epoch3 correction L2 已达 1.859。
- pre-crop extension 的训练有效率只有约 3.93%；其 `valid` 只表示背景/前景
  混合的额外点达到 16 个，并不表示目标被召回。

因此仅把 scale 从 `0.1` 调成 `0.01` 只会把灾难性失败变成较慢失败，不能
恢复可识别的 target，也不能证明 motion 的贡献。

## 方法定位

B1motion-v3 应成为 SeqTrack3D 外围的 **可关闭、可限幅、可归因 motion
plugin**：

```text
历史递归框 + Δt ──> physical-motion prior d_prior
                              │
当前点云 ──> 冻结 B0 ──> observation proposal d_obs
                              │
                 GT-free eligibility / reliability
                              │
 d_final = d_obs + α · clip(d_prior - stopgrad(d_obs), R(Δt))
```

核心不变量：

- normal cadence：`alpha == 0`，`d_final == d_obs`，不是“接近 B0”；
- invalid prior：`alpha == 0`；
- correction：`||d_final-d_obs|| <= alpha_max * R(delta_t)`；
- irregular loss 不得更新 B0 参数或 BatchNorm running statistics；
- 轨迹 prior 只学物理位移，不学 candidate-anchor 定位误差；
- 所有在线 gate 输入均为测试时可得量。

这与项目的 dual-clock 方向一致：稳定的 order-time B0 保留；真实
`delta_t` 只进入可干预的外围 motion branch，并继续接受
`true/fixed/shuffled` 因果控制。

## 1. 正确拆分 target

记：

- `A`：最新 candidate / 递归预测 anchor；
- `G0`：最新已观测历史帧的 GT box，仅训练时用于构造 label；
- `Gt`：当前 GT box；
- `R(A)`：anchor yaw 对应的旋转矩阵。

当前 B1motion-v2 学的是：

```text
d_box = R(A)^T [center(Gt) - center(A)]
```

它可以拆为：

```text
d_box =
    R(A)^T [center(Gt) - center(G0)]       # physical motion
  + R(A)^T [center(G0) - center(A)]        # anchor correction
```

第二项是 anchor 定位误差。仅看相对历史轨迹的 encoder 无法唯一恢复它。

B1motion-v3 的 prior target 改为：

```text
d_phys = R(A)^T [center(Gt) - center(G0)]
yaw_phys = wrap(yaw(Gt) - yaw(G0))
```

应用 prior 时：

```text
center(B_prior) = center(A) + R(A) d_phys_pred
```

这样 prior 保留 anchor 本身的平移误差；当前点云 observation branch 仍负责
`anchor correction`。两条分支在融合时都转换成同一个 anchor-frame
displacement，因此可以直接比较。

第一版只训练和应用 `d_phys[:2]`：

- `z` 的帧间变化小、监督噪声占比高；
- yaw 的 wrap、候选 yaw error 和 motion-state mask 会引入第二套语义问题；
- 先证明平面中心能涨点，再分别解锁 z/yaw。

### 训练 augmentation 限制

motion branch 不再使用 `ct_history_recursive_error_scale=4.0`。

优先级如下：

1. 最优：使用冻结 same-code B0 在 mini-train 上导出的真实递归历史；
2. 预研：candidate0 canonical history；
3. 只作鲁棒性增强：小幅、时间相关的真实 rollout residual 重采样。

不要把现有 `shared_se2` 直接当作 motion-error 模型。若只旋转历史候选而不
同步旋转当前物理 label，它会再次制造“输入不变、target 改变”的不可识别
样本。几何坐标增强可以使用 shared SE(2)，但必须对历史、当前框和点云一起
施加同一变换；预测误差增强则应来自真实 B0 rollout。

## 2. 训练视图拆分

复用现有 paired-view 基础设施，但新增独立模式
`use_b1motion_v3_paired_training`，不要复用 TWC/M3 的完整 loss 逻辑。

每个样本生成：

```text
view_main:
    current frame 相同
    offsets = [1, 2, 3]
    原 B0 candidate / point sampling 合同

view_motion:
    current frame 相同
    latest history offset = query_gap ∈ {2, 4}
    后续 history 由 transition gaps 构造
    使用真实 timestamp / effective timestamp
```

需要删除 B1-v3 对“两个 view 必须共同以 t-1 为 anchor”的限制；这是 TWC 的
假设，不适用于 gap tracking。

推荐训练方式：

### 首选：冻结 B0 plugin training

- 从当前代码重新训练并冻结一个 B0 checkpoint；
- `view_main` 只做 identity audit；
- `view_motion` 的 B0 forward 使用 `no_grad`，同时冻结 BatchNorm running
  stats；
- 只有 `physical_motion_encoder` 和后续 fusion/gate 参数进入 optimizer；
- irregular final-box loss 可以反传到新模块，但不能进入 B0。

`main.py::load_initial_weights()` 已支持把 baseline 权重匹配到含新模块的模型，
`BaseModel::configure_optimizers()` 已过滤 `requires_grad=False` 参数，可以
直接复用。

### 后续可选：受保护的 joint fine-tuning

只有 frozen-plugin 已经涨点后才解锁：

- B0 只从 `view_main` 接受原始 supervised loss；
- `view_motion` 在 B0 observation 输出处 detach；
- irregular loss 仍只更新新模块；
- 不再使用 `0.5*(loss_a+loss_b)` 或 M3 的 full-loss 加权平均。

## 3. prior encoder

可以复用 `OrderedTrajectoryEncoder` 的 ordered GRU 和 causal finite
difference 框架，但应新建 `OrderedPhysicalMotionEncoder`，不要在旧类上
继续叠兼容分支。

输入合同：

```text
motion_ref_boxs: [B, H, 4]
motion_delta_t: [B, H]
motion_current_delta_t: [B]
motion_valid_mask: [B, H]
```

输出合同：

```text
motion_prior_xy: [B, 2]
motion_prior_log_sigma_xy: [B, 2]
motion_prior_valid: [B]
motion_prior_gap_ratio: [B]
```

规则：

- recent finite-difference velocity 是 cold start；
- residual head 继续 zero-init；
- invalid row 的 `gap_ratio` 固定为 `1`，不再出现约 120 的无效均值；
- `log_sigma` 先只作训练诊断，不立即控制 gate；
- 必须按 candidate0 / nonzero candidate、gap、history-valid ratio 分桶记录
  endpoint RMSE。

## 4. proposal-space 融合

删除 B1-v3 对 `ZeroInitTrajectoryAdapter` 的使用。不要再修改
`motion_feature`。

直接复用并小幅扩展 `models/dynamics.py::apply_proposal_innovation()`：

```text
innovation_raw = d_prior - stopgrad(d_obs)
innovation = clamp_norm(innovation_raw, R(delta_t))
d_final = d_obs + alpha * innovation
```

第一阶段只让 `x/y` 进入 innovation：

```text
d_prior_xyz = [d_prior_x, d_prior_y, stopgrad(d_obs_z)]
```

从而 `z` correction 精确为零，yaw 完全沿用 B0。

### eligibility

先用确定性、GT-free gate，不立即训练新 gate 网络：

```text
irregular = gap_ratio >= 1.5
prior_ok = motion_prior_valid
obs_weak = foreground_count < N_min
        or foreground_score < P_min

eligible = irregular and prior_ok and obs_weak
alpha = eligible * alpha_fixed
```

第一轮建议只预注册一个安全值：

```yaml
b1motion_v3_alpha: 0.25
b1motion_v3_radius_base: 0.25
b1motion_v3_radius_per_second: 0.50
b1motion_v3_radius_max: 1.25
```

这些是安全起点，不是最终最优参数。通过 A1 后最多比较
`alpha=0.25/0.50` 两臂；不做大网格搜索。

`foreground_count/score` 已可由当前 observation statistics 构造。阈值必须
在一个固定 calibration split 上预注册，并在 frozen validation 上只评一次。

如果固定 gate 已有正信号，再把以下测试时可得统计送入小型 gate：

- `log1p(current_delta_t)`、gap ratio；
- valid-history ratio；
- baseline foreground count / mean confidence；
- `||d_prior-d_obs||`；
- 后续 search evidence。

学习 gate 仍必须乘 hard eligibility mask，且 `alpha <= alpha_max`。

## 5. loss

先分两步训练，避免把 prior 可学性与 fusion 混在一起。

### V3-A0：prior-only

```text
L_prior = SmoothL1(d_prior_xy, d_phys_xy)
```

只检查 prior 是否真的比 constant velocity / latest-velocity 更好。若 prior 在
candidate0、nonzero/rollout history 和 gap2/gap4 上都不能降低 endpoint
RMSE，不进入 tracker 融合。

### V3-A1：bounded fusion

```text
L_final = SmoothL1(d_final_xy, d_box_xy)
L_total = L_prior + lambda_final * L_final
```

B0 已冻结，因此 `L_final` 只优化 prior/fusion。`lambda_final` 先用 batch
preflight 令两项 loss 同数量级，不从历史配置照搬。

在不确定性校准稳定后再加：

```text
L_nll = 0.5 * error^2 * exp(-2 log_sigma) + log_sigma
```

不能只看 NLL；必须同时报告 RMSE、sigma mean、68%/95% coverage。

## 6. search 的正确解锁方式

V3-A 阶段：

```yaml
use_trajectory_search: false
```

原因是 search 当前同时有“几何覆盖不足”和“特征语义过弱”两个问题，和 prior
一起修改无法归因。

V3-B 前先做纯离线 crop audit，把当前 `valid` 拆成：

- `search_geometry_triggered`；
- `search_extension_available`；
- `target_center_in_baseline`（仅离线 oracle）；
- `target_center_in_extension`（仅离线 oracle）；
- `target_foreground_points_in_extension`（仅训练/诊断）；
- `search_branch_applied`。

只有在“baseline miss”样本中，extension target recall 有明确提升，才实现
search feature branch。

届时不再对 extension raw points 直接全局 max-pool。最小可行版本：

1. point-wise encoder；
2. point-wise targetness head，用 extension segmentation label 监督；
3. targetness-weighted pooling；
4. 输出独立 `d_search` 与 `search_confidence`；
5. 只在 base observation 弱且 search confidence 高时，用第二个有界
   proposal innovation 把 `d_obs` 拉向 `d_search`；
6. 再由 physical prior 对得到的 local proposal 做最终有界融合。

pre-crop 几何先继续使用训练/推理一致的 analytic
`build_ordered_trajectory_search_box()`。不要在 dataset worker 与在线模型
中分别复制一个 learned prior；若未来要用 learned prior 决定 crop，必须把
crop 调度移入同一个可复用推理入口或离线缓存其训练预测。

HVTrack 对高 temporal variation 的有效设计也支持这一路径：扩大区域后需要
显式 base-expansion cross-attention / background suppression，而不是把更多
背景点压成一个无监督全局特征。

## 7. 文件级修改清单

| 文件 | 修改 |
|---|---|
| `cfgs/ct_v2/02_ct_motion.yaml` | 保持冻结，不改语义 |
| `cfgs/ct_v2/02_ct_motion_v3.yaml` | 新增 frozen-B0、paired irregular、prior-only/proposal-fusion 配置 |
| `utils/candidate_utils.py` | 新增 `physical_motion_targets()`；保留 legacy `anchor_relative_trajectory_targets()` |
| `datasets/sampler.py` | 新增 B1-v3 paired sampler；main 永远连续，motion view 独立 irregular；返回明确的 `motion_*` 字段 |
| `utils/ct_history.py` | 新增 rollout history 输入/校验；V3 禁止 scale=4 synthetic recursive error |
| `models/ct_v2/motion.py` | 新增 `OrderedPhysicalMotionEncoder`；V3 不实例化 `ZeroInitTrajectoryAdapter` |
| `models/dynamics.py` | 复用 innovation；可增加 xy-only mask 与逐轴诊断 |
| `models/seqtrack3d.py` | B0 proposal 生成后做 V3 proposal fusion；新增专用 paired loss 与 gradient isolation |
| `models/base_model.py` | 在线构造与训练一致的 `motion_*` 字段；normal eligibility 强制为 0 |
| `utils/ct_search.py` | V3-B 才增加 recall/evidence 诊断，不在 A 阶段改 crop |
| `main.py` | 使用现有 `--init_checkpoint`；只需补充冻结清单和 init report 校验 |
| `tests/test_ct_v2.py` | 增加 target invariance、normal identity、norm cap、gradient isolation、train/eval coordinate tests |

建议使用新字段名，避免继续把两种语义都叫
`trajectory_displacement_label`：

```text
motion_physical_displacement_label
motion_box_displacement_label
motion_prior_xy
motion_observation_xy
motion_final_xy
```

## 8. 必须新增的测试

### 数学合同

1. 对 candidate anchor 添加共同平移：
   `physical_motion_target` 不变，`box_displacement_target` 按预期改变。
2. history 与当前框一起做刚体坐标增强：
   physical target 与输入按同一旋转变换。
3. yaw wrap 在 `+-pi` 附近连续。

### 安全合同

4. normal cadence 下 module on/off 输出 bitwise equal。
5. invalid history 下输出 bitwise equal。
6. `||applied correction|| <= alpha_max * R(delta_t)`。
7. xy-only 阶段中 z/yaw correction 精确为 0。
8. irregular loss backward 后，所有 B0 参数 grad 为 `None` 或精确 0。
9. irregular forward 不改变 B0 BatchNorm running mean/variance。

### 数据/在线一致性

10. `view_main` offsets 永远为 `[1,2,3]`。
11. `view_motion` 第一 offset 可以是 2/4，不被 paired TWC validator 拒绝。
12. 同一 synthetic boxes/timestamps 经 sampler 与 `base_model` 得到相同
    motion input 和 target coordinate。
13. rollout history 不得读取当前/未来 GT。

### search 后续测试

14. `valid`、target recall 和 branch applied 三个概念分离。
15. extension 无前景时 search correction 精确为 0。

## 9. 实验阶梯和晋级门槛

### E0：same-code B0

- 当前 commit、seed42、mini Car、60 epoch；
- 导出 normal、random20、gap1124；
- 保存 per-tracklet / endpoint 结果；
- 后续所有 module on/off 均从该 checkpoint 比较。

### K0：现有根因短实验

在 search off 下做 10–15 epoch：

```text
irregular probability 0 / 0.35
×
adapter off / on
```

目的只是确认退化归因，不作为 v3 模型。

### V3-A0：prior-only

- frozen B0；
- canonical 与真实 B0 rollout 两套 history 诊断；
- gap2/gap4 分桶；
- 对比 constant velocity、latest velocity、learned prior RMSE；
- 不改 tracker output。

晋级条件：learned prior 在 rollout history 和较大 gap 上都优于最强解析
baseline，且非 candidate0/rollout 桶没有系统性崩溃。

### V3-A1：bounded proposal fusion

- search off；
- xy-only；
- fixed GT-free gate；
- 10–15 epoch screen 后再决定是否跑 60 epoch。

硬门槛：

- normal module on/off：同一 checkpoint 逐 endpoint 完全相同；
- normal Success/Precision 聚合也完全相同；
- random20 或 gap1124 至少一个协议达到预注册提升；
- correction clip rate、applied rate、prior/obs/final error 都可解释；
- 无 NaN、无 invalid gap-ratio 污染。

建议的首轮 promotion 线：

```text
normal: exact identity
gap1124: >= +1.0 Success and >= +2.0 Precision
random20: non-negative on both metrics
```

### V3-A2：时间因果控制

用同一个 checkpoint 做 `true/fixed/shuffled`：

- 若 true 未稳定超过 fixed/shuffled，只能声称 trajectory/proposal robustness；
- 只有 true 的 paired bootstrap CI 为正，才能恢复 physical-time causal claim。

### V3-B：search

只有以下条件同时满足才解锁：

- V3-A1 tracker signal 已成立；
- baseline-miss 样本的 extension target recall 明显提升；
- search 前景 evidence 在固定 split 上有可用的 precision/recall；
- search-only、prior-only 和 combined 三臂都能独立评估。

### V3-C：joint fine-tune / yaw

最后才尝试：

- B0 仅由 continuous main view 更新；
- irregular gradient 仍隔离；
- yaw 单独一臂；
- 多 seed 只在前述门槛全部通过后启动。

## 10. 推荐的首个配置骨架

```yaml
_base_: nuscenes_mini_base.yaml

experiment_name: ct_b1motion_v3_physical_proposal
use_b1motion_v3: true
use_ordered_trajectory_encoder: false
use_trajectory_adapter: false
use_trajectory_search: false

b1motion_v3_freeze_baseline: true
b1motion_v3_xy_only: true
b1motion_v3_use_paired_training: true
b1motion_v3_main_offsets: [1, 2, 3]
b1motion_v3_query_gaps: [2, 4]
b1motion_v3_transition_gaps: [1, 1, 2, 4]
b1motion_v3_history_source: b0_rollout

b1motion_v3_gate_mode: fixed_gt_free
b1motion_v3_gap_trigger: 1.5
b1motion_v3_alpha: 0.25
b1motion_v3_radius_base: 0.25
b1motion_v3_radius_per_second: 0.50
b1motion_v3_radius_max: 1.25

b1motion_v3_prior_weight: 1.0
b1motion_v3_final_weight: 1.0
b1motion_v3_nll_weight: 0.0
```

阈值 `foreground_count/score` 不写死在设计文档里，应由 calibration export
确定后冻结进 formal YAML。

## 11. 与历史 dyn/M2 的关系

历史 A2-order-dyn 能做到接近 baseline，主要因为：

- continuous B0 主训练视图没有被 35% irregular history 替换；
- dynamics 只是 concat feature，后续网络可以学会忽略；
- 没有直接、无界地把 correction 写入主 feature。

M2 R1 的 tracking signal 说明 proposal-space innovation 值得保留，但它同时
混入 A1 continuation、candidate path 等变化，且 true time 没有超过
fixed/shuffled；它不是现成的净模块增益。V3 应复用其**有界 proposal
innovation primitive**，而不是复用其历史结论。

TrajTrack 的论文结构支持 propose-predict-refine，但本地 evaluator
`pre_w_refine()` 读取当前 GT overlap 决定何时 refine，并用 GT overlap 选择
proposal；本项目只能借鉴“两 proposal 的关系”，不能复制该 oracle 逻辑。
在线 gate 必须使用 proposal agreement、点数、分割置信和 prior uncertainty。

SeqTrack3D 自身的 time-window ablation 也表明历史不是越长越好：Car 从
`1+3` 的 `56.73/66.01` 降到 `1+7` 的 `51.15/60.07`，作者明确归因于训练
random offsets 难以模拟测试和历史框累计误差。这与当前 B1 的 failure mode
直接一致。

## 12. 最小实施顺序

1. 新增 target helper 与不变量单测；
2. 新增 V3 paired sampler，先只打印/验证张量，不接模型；
3. 新增 prior encoder，跑 V3-A0；
4. 冻结 B0，并验证 normal bitwise identity / BN identity / grad isolation；
5. 接 existing proposal innovation，跑 V3-A1；
6. 完成 same-checkpoint 三协议评测；
7. 通过后再做 search audit；
8. 最后考虑 learned gate、yaw 和 joint fine-tune。

首个有效工程目标不是“直接超过 B0”，而是先同时证明两件事：

1. motion prior 在真实递归历史上预测的是可识别的物理位移；
2. 开启该 prior 后，normal B0 路径仍是结构性的 exact identity。

只要这两个合同没有同时成立，就不应再启动新的 60-epoch B1。

