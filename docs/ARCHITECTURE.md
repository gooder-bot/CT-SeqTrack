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

## 配置和兼容

`main.py` 支持 `_base_` YAML 继承，新消融配置只写差异。所有旧 YAML 仍可直接传给 `main.py`，旧 checkpoint 的参数命名不变。v2 新增参数只在 `use_ct_v2: true` 时注册。

`ct_history_training_mode` 默认为 `canonical`，因此旧 YAML 行为不变；
B1–B3 显式选择 `correlated_candidate`。相关历史只改变输入数据，不新增
模型参数或 state-dict key。

## 当前不包含

- TWC/M3 teacher：延后到三模块晋级以后。
- Kalman/state filter：现有 M4 代码保留为 legacy，不属于 v2。
- 额外 memory backbone/Mamba：会扩大变量和训练成本，不符合当前最小消融目标。
- 搜索点 source channel：当前只把 expansion ratio 提供给 gate，避免改变 SeqTrack3D 点特征维度和 baseline checkpoint 接口。
