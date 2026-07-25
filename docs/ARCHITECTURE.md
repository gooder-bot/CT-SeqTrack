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

训练时使用相对最近历史 GT anchor 的 canonical 历史轨迹，避免 independent candidate jitter 被错误解释为速度；递归评测使用相对最近预测 anchor 的历史预测框。两侧都遵循“历史轨迹相对当前 anchor”的语义。

### 2. Time-Guided Search Expansion

`utils/ct_search.py` 用历史轨迹的有界速度外推当前中心，并构造连接最近框与预测中心的 tube：

```text
search = baseline crop ∪ bounded trajectory tube
sample = 75% baseline + 25% expansion-only
```

当历史不足、时间无效或目标近似静止时，精确回退到 baseline crop。tube 长度、宽度、速度和最大位移都有上限。
扩展区域少于 32 个独立点时同样回退，避免把极少量背景点重复采样成 25% 的输入。

### 3. Adaptive Proposal Fusion

SeqTrack3D 先产生 observation proposal，motion encoder 产生 dynamics proposal。门控读取：

- observation/dynamics feature；
- 当前点云数量、前景置信度和历史有效率；
- proposal disagreement；
- 当前 `delta_t`；
- search expansion ratio。

最终结果为：

```text
innovation = clip(d_dyn - stopgrad(d_obs), R(delta_t))
d_final = d_obs + alpha * innovation
```

`alpha` 上限默认为 0.75，初始化为 0.25；这与已有 M2 的有效固定系数保持可比较，同时仍由时间相关半径限制实际修正。无有效历史或空搜索时回退 observation proposal。

## 配置和兼容

`main.py` 支持 `_base_` YAML 继承，新消融配置只写差异。所有旧 YAML 仍可直接传给 `main.py`，旧 checkpoint 的参数命名不变。v2 新增参数只在 `use_ct_v2: true` 时注册。

## 当前不包含

- TWC/M3 teacher：延后到三模块晋级以后。
- Kalman/state filter：现有 M4 代码保留为 legacy，不属于 v2。
- 额外 memory backbone/Mamba：会扩大变量和训练成本，不符合当前最小消融目标。
- 搜索点 source channel：当前只把 expansion ratio 提供给 gate，避免改变 SeqTrack3D 点特征维度和 baseline checkpoint 接口。
