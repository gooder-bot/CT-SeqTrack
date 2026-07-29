# CT-SeqTrack v2 架构

## 设计约束

- 基线是同仓库、同训练流程的 SeqTrack3D。
- 推理只依赖历史预测框、历史/当前点云和 timestamp。
- 不读取当前 GT、oracle reachability 或未来帧。
- 搜索总点数固定为 1024，避免用额外算力换指标。
- 旧实验开关默认关闭，并与 v2 做 fail-fast 互斥检查。

## 数据流

### 1. Continuous-Time Motion Prior

`ContinuousTimeMotionEncoder` 读取最近到更早的历史框：

```text
velocity_i = (box_i - box_i+1) / delta_t_i
query_displacement = predicted_velocity * current_delta_t
```

训练时不再只使用无噪声 GT 历史。candidate0 保持精确 canonical；
非零 candidate 对更早历史框施加相关系数 `rho=0.75` 的平滑轨迹误差，
同时把最近 motion anchor 固定为 canonical，因此噪声只改变输入历史，
不会改变 canonical displacement/velocity 标签。递归评测仍使用历史预测框。

### 2. Time-Guided Search Expansion

`utils/ct_search.py` 用历史轨迹的有界速度外推当前中心，并构造连接最近框与预测中心的 tube：

```text
search = baseline crop ∪ bounded trajectory tube
sample = 75% baseline + 25% expansion-only
```

当历史不足、时间无效或目标近似静止时，精确回退到 baseline crop。tube 长度、宽度、速度和最大位移都有上限。
扩展区域少于 32 个独立点时同样回退，避免把极少量背景点重复采样成 25% 的输入。
训练时 correlated search history 的最近框与实际 candidate crop anchor
一致；更早误差平滑递推，避免使用 GT anchor 构造 oracle tube。

Search-only A1 的训练中约 3.460% 样本使用 expansion，平均扩展 token 占
0.865%；这与 B2 基本一致，证明模块并非未执行。但当前验证 event 没有记录
递归阶段逐 endpoint 的 search 激活与扩展点数，因此无法直接观察 tube 在
预测历史误差下何时抽入背景。这个日志缺口是下一轮机制诊断的重点。

### 3. Adaptive Proposal Fusion

SeqTrack3D 先产生 observation proposal，motion encoder 产生 dynamics proposal。门控读取：

- observation/dynamics feature；
- 当前点云数量、前景置信度和历史有效率；
- proposal disagreement；
- 当前 effective `delta_t`；
- search expansion ratio。

最终结果为：

```text
innovation = clip(d_dyn - stopgrad(d_obs), R(delta_t))
d_final = d_obs + alpha * innovation
```

`alpha` 上限默认为 0.75，初始化为 0.25；这与已有 M2 的有效固定系数保持可比较，同时仍由时间相关半径限制实际修正。`ct_fusion_alpha` 记录不含安全掩码的 nominal gate 输出，`ct_fusion_alpha_applied` 记录乘过 history/search valid mask 和 warmup scale 的实际系数。少于 3 个可用搜索点、无有效历史或 warmup 未结束时，applied alpha 为零并回退 observation proposal。

CT-v2 的 gate、motion encoder、innovation radius 和 search tube 统一读取
`current_delta_t_effective`。真实 `current_delta_t_real` 只用于监督与诊断，
不会在 fixed/shuffled 控制下进入 gate。

### 4. Δt-aware Point-Feature Temporal Consistency（Δt-PFTC）

第四模块是从 B0 独立派生的训练目标，不继续叠加到已经失败的 B3。它不注册
可学习参数，不增加 memory token，也不改变推理图：

```text
每帧 1024 个采样点
  -> FeaturePointNet 前两层
  -> 64-D point-aligned feature ---------> Δt-PFTC（仅 train）
  -> 原有 MLP + AdaptiveMaxPool(128)
  -> Transformer 128 tokens
```

不能在 128-token 输出上做点对应，因为 AdaptiveMaxPool 后的槽位不再对应具体
XYZ。训练时按照固定存储位置把 `[t-1,t-2,t-3,t]` 重排为
`[t-3,t-2,t-1,t]`；true/fixed/shuffled 只改变时间值，不参与排序。

每帧先用 `seg_label` 过滤 1.25-scale GT 前景，再用同一 anchor 坐标系中的
`box_label_prev + box_label` 去中心并消除平面 yaw。完全重复的采样 XYZ 会先
合并并平均特征。所有有效早帧到晚帧执行单向 canonical XYZ 最近邻，只保留
距离 `<0.3 m` 且至少 3 个对应的帧对，特征目标使用 raw SmoothL1。距离和索引
在 float32/no-grad 前景子集上计算，因此梯度只进入 FeaturePointNet 前两层。
loss 先在每个样本内部按有效帧对平均，再在有效样本间平均。

加权臂使用：

```text
raw_w(i,j) = clamp(abs(t_j - t_i) / 0.5 s, 0.5, 3.0)
w(i,j) = raw_w(i,j) / mean_sample(raw_w)
L = L_supervised + ramp_5ep * lambda_pftc * L_dt_pftc
```

样本内权重均值固定为 1，避免把“使用真实时间”偷换成“放大辅助 loss”。
unweighted 臂令所有帧对等权。PFTC 与旧 paired-view TWC、M3 EMA
distillation 首版互斥并 fail-fast；技术上允许和 CT-v2 组合，但正式实验只接
B0。

## 配置和兼容

`main.py` 支持 `_base_` YAML 继承，新消融配置只写差异。所有旧 YAML 仍可直接传给 `main.py`，旧 checkpoint 的参数命名不变。v2 新增参数只在 `use_ct_v2: true` 时注册。

`ct_history_training_mode` 默认为 `canonical`，因此旧 YAML 行为不变；
B1–B3 显式选择 `correlated_candidate`。相关历史只改变输入数据，不新增
模型参数或 state-dict key。

### 消融初始化边界

当前 `ct_proposal_fusion` 在 `motion_mlp`、`feature_pointnet` 和
`Transformer` 之前实例化；开启 gate 会先消耗随机数，使后续共享层即使使用
同一全局 seed 也不再拥有相同初始权重。B0/B1 之间插入 motion encoder 时也有
同类问题。因此本轮单 seed 结果足以否决大幅退化的设计，但不能作为严格共享
初始化下的论文级模块效应。下一轮必须从同一初始化 checkpoint 加载共享 key，
或隔离可选模块的 RNG，再比较新的最小消融。

## 当前不包含

- TWC/M3 teacher：延后到三模块晋级以后。
- Kalman/state filter：现有 M4 代码保留为 legacy，不属于 v2。
- 额外 memory backbone/Mamba：会扩大变量和训练成本，不符合当前最小消融目标。
- 搜索点 source channel：当前只把 expansion ratio 提供给 gate，避免改变 SeqTrack3D 点特征维度和 baseline checkpoint 接口。

PFTC 通过后才重新评估 compact memory 和 MCC；二者不属于本轮实现。

### 2026-07-30 Δt-PFTC 实现审计

首个 seed42 artifact 只到约 epoch23.05，不是完整 60 epoch。代码审计还发现
当前 `canonicalize_points` 的 yaw 方向与项目已有几何约定不一致。对中心化后的
列向量，当前实现使用：

```text
[x', y'] = R(+yaw) [x, y]
```

但 `generate_subwindow`、`transform_pc` 和 `get_offset_points_tensor` 都使用
box rotation 的逆，即 object-local 应为：

```text
x_local = cos(yaw) * x + sin(yaw) * y
y_local = -sin(yaw) * x + cos(yaw) * y
```

现有 yaw 单测按当前公式的逆构造输入，未覆盖项目真实的 local→shared 约定。
因此当前 PFTC 运行只作为失败诊断，不能视为上述架构的有效实现。

训练曲线同时表明 raw SmoothL1 正对应目标存在明显的特征尺度收缩：epoch1 到
epoch20 的前景 feature std 从 `0.0947` 降至 `0.0210`，而匹配距离和对应数量
基本不变。修订版必须加入明确防坍缩路径（train-only projector 加 variance
floor；必要时加入 spatial negatives 或 stop-gradient teacher），并记录同定义
B0 feature std 与两项 loss 的 gradient norm/cosine。当前逐样本/逐帧对循环还
使训练慢约 10.2 倍，正式重跑前必须消除循环内 GPU `.item()` 同步并批量化或
预计算 correspondence。

### 2026-07-30 Motion fixed-alpha 接入审计

alpha0/0.25 两组 scratch 60 epoch 完整复核表明，当前
`proposal_innovation` 的失败不只是 alpha0.75 过强：

| alpha | final Success | final Precision | late-3 S/P |
|---:|---:|---:|---:|
| 0 | 47.049 | 49.184 | 46.828/49.669 |
| 0.25 | 29.581 | 28.862 | 29.472/28.849 |
| 0.75（旧 B1） | 26.021 | 24.972 | 26.080/25.299 |

alpha0.25 warmup 后 effective alpha 均值为 0.184，73.7% 样本应用，
correction norm 均值只有 0.083 m，仍相对 alpha0 final 下降
17.468/20.322。结构上存在三层放大：

1. 训练时 DynamicsEncoder 读取 GT/correlated 合成的
   `ct_motion_ref_boxs`，eval 时读取 recursive `ref_boxs`；
2. `dynamics_valid` 只表示 transition 存在，不表示 proposal 方向可靠；
3. 修正发生在 coarse `aux_box` 和 Transformer box-corner query 之前，
   会同时改变局部坐标和后续 refinement 查询。

alpha 增大时 epoch60 training loss 从 0.223 降到 0.217/0.215，但递归验证
反向下降，因此当前 loss 与 closed-loop stability 不对齐。alpha0 是
`apply_proposal_innovation` 的精确零回退；其 dynamics 辅助损失只训练独立
DynamicsEncoder。它低于 B0 不能证明“motion 即使不融合也有害”，因为插入
DynamicsEncoder 会消耗 RNG 并改变后续共享层初始化。

正式边界为 `NO_GO_FIXED_GLOBAL_MOTION_INNOVATION`。在同 checkpoint
alpha on/off 2×2 和逐 endpoint proposal attribution 完成前，不再训练更小
alpha 或恢复 adaptive gate。完整复核见
[`Motion alpha 报告`](../compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md)。

进一步的逐行审计确认了两个结构性限制。第一，DynamicsEncoder 对 transition
特征只做 mean+max pooling；交换同一组速度 transition 的先后顺序会得到完全
相同的 `z_dyn/velocity/displacement`，因此当前函数族无法区分加速与减速。
第二，当前帧 crop 在 model forward 之前已经围绕上一预测框完成，motion
correction 只发生在 post-crop coarse query；目标已经离开 crop 时，该分支无法
召回缺失点。完整代码、研究与决策审计见
[`B1motion 深度审计`](../compare_results/reports/b1_motion_module_deep_audit_20260730.html)。

## 当前实证状态

2026-07-27 的 seed42 normal-mini 复核表明：

- B0 baseline 为 53.360 Success / 64.382 Precision。
- B1 固定 0.75 motion fusion 降至 26.021 / 24.972，当前实现不通过。
- B2 search expansion 恢复到 47.973 / 52.088，但仍低于 B0。
- B3 完成 60 epoch 后仅为 25.537 / 24.707，late-3 为
  26.321 / 25.104，重新退回 B1 水平。
- A1 Search-only 为 27.036 / 25.596，late-3 为 27.933 / 26.400；
  相对 B0 final 下降 26.324 / 38.786，当前 search 不能独立晋级。
- B3 nominal alpha 从 epoch5 的 0.250 升到 epoch6 的 0.707，并在
  epoch7 达到 0.749；epoch60 的 batch-min mean 仍为 0.749998。当前
  adaptive gate 已退化为几乎恒定的最大权重，未学到条件可靠性。

这意味着当前三模块 bundle 和独立 search 都被 normal-mini 否决。A1 末轮
training loss 为 0.2221，接近 B0 的 0.2208，而递归验证从第一次评测起就处于
低位；现有证据更支持 train/recursive-search mismatch 或强模块交互，而不是
训练不足。B2 的正增量只能表述为 search 对失败 B1 的交互恢复。

下一步使用现有 B0/A1 checkpoint 做 Search 开/关 2×2，并补验证阶段
endpoint diagnostics；结果出来前不训练 A2。B0–B3 复核见
[`B0–B3 报告`](../compare_results/reports/ct_v2_ablation_seed42_20260727.md)，
Search-only 复核见
[`A1 报告`](../compare_results/reports/ct_search_only_seed42_20260727.md)。

## 2026-07-30 B1motion-v2 修正

旧固定-alpha B1 已冻结到 `02_ct_motion_legacy_fixed.yaml`。当前
`02_ct_motion.yaml` 改为有序 GRU、crop 前第二搜索区域、独立扩区点编码和
zero-init feature residual；它不再修改 observation proposal。B0 的 320 个
共享 state tensors 已验证 step-0 exact match，新 adapter 初始 correction
严格为零。2026-07-30 的二次复核进一步统一了训练坐标合同：ordered
history、endpoint target 和 velocity target 均位于实际 candidate crop
anchor 坐标，覆盖递归推理中最新预测框自身带误差的情形，避免把 GT-anchor
motion feature 与 candidate-anchor observation feature 直接拼接。完整设计、
外部代码审计、训练命令与晋级门槛见
[`B1motion-v2 设计`](B1MOTION_V2_ORDERED_PRECROP_20260730.md)。
