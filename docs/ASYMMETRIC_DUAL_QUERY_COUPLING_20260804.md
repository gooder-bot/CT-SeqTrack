# CT-SeqTrack 不确定性感知非对称双查询耦合

更新时间：2026-08-04

状态：B1–B3 下一版耦合技术规范

上位决策：[B1–B4 连接重构与消融计划](B1_B4_REDESIGN_AND_ABLATION_PLAN_20260804.md)

## 1. 最终推荐

下一版不继续当前“前期弱耦合、候选阶段围绕 B1 强裁剪”的结构，也不原样采用
“Motion Prior → 单一 motion-guided query → 全部候选”的串行强耦合。正式推荐：

> **局部紧耦合、全局解耦的不确定性感知非对称双查询耦合**：B1 只紧耦合
> B2 的扩展支持域、相对几何和辅助 search query；SeqTrack3D observation query
> 与 observation box 保持独立；B3 最后以 observation 为锚点选择候选来源和
> 执行步长。

耦合边界：

```text
B1 -> B2 geometry/query : one-way, stop-gradient, locally tight
B0 observation path     : motion-independent, always available
B0/B1/B2 -> B3          : detached late decision
B2/B3 -> B1             : no gradient
```

该结构同时追求三个目标：

1. B1 正确时扩大搜索覆盖并聚焦目标证据；
2. B1 错误时不破坏原始 observation 路径；
3. 每条连接都能用单因素消融证明，不把涨点归因混在一个端到端黑盒中。

## 2. 当前实现与图示方案的差异

### 2.1 当前代码实际耦合

```text
results_bbs + timestamps
   ├── hand-coded CV/acceleration/yaw-rate -> endpoint crop -> 128 B2 points
   └── learned B1 -> μxy / velocity / log_sigma / motion feature

base crop -> B0 coarse point_feature -> Transformer -> observation box
endpoint points + coarse point_feature + B1 feature -> B2 raw vote
B2 raw vote -> clip around B1 -> refined candidate
observation / B1 / refined -> detached B3 source×step -> final box
```

关键代码事实：

- 在线 baseline 和 endpoint crop 在模型 forward 前完成，分别见
  [`base_model.py`](../models/base_model.py#L1063) 和
  [`base_model.py`](../models/base_model.py#L1200)；此时 learned B1 尚未运行。
- endpoint box 由
  [`build_trajectory_endpoint_search_box`](../utils/ct_search.py#L319)
  手工外推，尺寸原样复制 latest box，见
  [`ct_search.py`](../utils/ct_search.py#L385)。
- B1 的 learned prior 由
  [`OrderedPhysicalMotionEncoder`](../models/ct_v2/motion.py#L208)产生。
- B2 query 使用 pre-Transformer
  [`point_feature`](../models/seqtrack3d.py#L1430)，而最终 observation 来自
  [`Transformer`](../models/seqtrack3d.py#L1709)。Transformer 当前只返回 box，
  没有暴露 `l1` 后的 current decoder state。
- B2 对 observation/motion feature 使用 `detach`，见
  [`selective_innovation.py`](../models/ct_v2/selective_innovation.py#L351)；
  这有利于模块隔离。
- B2 最后把 raw vote 围绕 B1 裁剪，见
  [`selective_innovation.py`](../models/ct_v2/selective_innovation.py#L437)；
  当前实验证明这条连接伤害 raw candidate。
- B3 以 observation 为 reference，并保留其 z/yaw，见
  [`selective_innovation.py`](../models/ct_v2/selective_innovation.py#L1116)
  和 [`selective_innovation.py`](../models/ct_v2/selective_innovation.py#L1283)。

因此当前不是简单的“松耦合”，而是：

- history tensor 强制对齐；
- crop/proposal 来源不对齐；
- B0/B1 到 B2 是冻结的单向特征耦合；
- B2 raw 到 refined 对 B1 过度耦合；
- B3 是合理的晚期解耦决策。

### 2.2 图示原始方案

图示可解释为：

```text
history -> motion prior -> μ/σ/velocity
current points -> encoder -> geometry-aware decoder
motion-guided query -> joint candidate generation -> decision
```

优点是只有一个 motion prior，并能在 decoder/query 层提前聚焦目标。它与
[Motion-to-Matching](https://arxiv.org/abs/2308.11875) 的“历史框 coarse motion
+ 点云 geometry refinement”、[PTTR](https://arxiv.org/abs/2112.02857) 的
coarse-to-fine refinement、[MBPTrack](https://arxiv.org/abs/2303.05071) 的
box-prior reference localization 方向一致。

但原图缺少四个必要定义：

1. `μ/σ` 是否在 point crop 前生效；如果只进 decoder，crop 外目标无法恢复；
2. 是否保留 motion-independent observation candidate；若没有，B1 错误会污染
   全部候选；
3. `step candidate` 的语义错误：step 是动作比例，不是空间候选；
4. `μ/σ/velocity` 的职责重叠以及 sigma 的监督、校准、梯度边界未定义。

直接实现原图还会与 Motion-to-Matching 等已有两阶段方法过近。论文差异必须来自
连续物理时间的校准风险、base-preserving uncertainty support、双 query 以及
observation-anchored action-consistent decision，而不是“motion-guided query”本身。

## 3. 推荐目标数据流

```text
Historical boxes + physical delta_t
                  │
                  ▼
     B1 probabilistic motion pre-pass
   μxy, Σmotion, velocity, hmotion, valid
       │                 │
       │ geometry        │ dynamics context
       ▼                 ▼
base crop ∪ uncertainty-aware prior support
                  │
                  ▼
          shared point encoder/tokens
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
 independent q_obs    motion-guided q_search
          │                │
          ▼                ▼
 C_obs full box       C_search_raw XY
                           ▲
                           │ B1-relative geometry bias
 C_motion XY = μxy ────────┘
          │
          └────────────┬─────────────────┐
                       ▼                 │
        B3 observation-anchored source×step decision
                       │
                       ▼
                  final box/history
```

### 3.1 正确的候选与动作定义

候选来源只有三种：

```text
C_obs    = SeqTrack3D observation full box
C_motion = B1 μxy + C_obs[z, yaw]
C_search = B2 raw_search_xy + C_obs[z, yaw]
```

B3 动作为：

```text
keep observation
motion × {0.25, 0.5, 1.0}
search × {0.25, 0.5, 1.0}
```

`step` 不在 Candidate Generation 下生成；它由 Decision Head 与 candidate source
联合预测。当前代码只修改 XY、保留 observation z/yaw 的合同继续保留。在 XY
链路晋级前，不扩展 B1 到完整 z/yaw candidate。

### 3.2 B1 张量合同

输入：

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `ref_boxs` | `[B,H,4]` | newest-to-oldest 因果历史框 |
| `delta_t` | `[B,H]` | 历史物理间隔 |
| `valid_mask` | `[B,H]` | 历史有效性 |
| `current_delta_t` | `[B]` | 当前 query gap |

输出：

| 字段 | 形状 | 只允许的用途 |
| --- | --- | --- |
| `mu_xy` | `[B,2]` | motion candidate、prior support 中心 |
| `sigma_parallel/perpendicular` | `[B,2]` | support 宽度、几何 bias、B3 risk |
| `velocity_xy` | `[B,2]` | 定义运动方向、辅助 query dynamics |
| `motion_feature` | `[B,Dm]` | 只进入 `q_search`/B3，不进入 `q_obs` |
| `valid/gap_ratio` | `[B]` | fallback 与风险控制 |

推荐把当前 axis-aligned `log_sigma_xy` 改为 motion-aligned covariance：

```text
Σmotion = R(velocity)
          diag(σ_parallel², σ_perpendicular²)
          R(velocity)^T
```

低速或 invalid 时回退 isotropic/fixed support。sigma 必须使用 heteroscedastic NLL
训练并通过 coverage 校准；校准前只能使用预注册 fixed width，不能把当前
`log_sigma_xy` 接入正式方法。

### 3.3 双 query 合同

`q_obs` 是原 SeqTrack3D current-target decoder state，完全不读取 B1。当前
Transformer 在 `l1` 后得到 `[B,4,64]` state，应新增可选返回值并取
`decoder_state[:, -1]`，同时保持原 box 输出逐 bit 兼容。

`q_search` 是零初始化的 motion residual query：

```text
q_search = LayerNorm(
    W_obs stopgrad(q_obs)
    + g(Σ, valid) W_motion stopgrad(h_motion)
    + W_time [delta_t, gap_ratio]
)
```

约束：

- motion residual 分支零初始化，使初始 `q_search ≈ q_obs`；
- `g` 有上限，B1 invalid 时精确退化为 observation-conditioned search query；
- 训练 B2 时 `q_obs`、`h_motion`、`μ/Σ` 全部 stop-gradient；
- `q_obs` 继续单独产生 C_obs，不被 B2 loss 改写。

相邻的 3D MOT 工作 [DQTrack](https://openaccess.thecvf.com/content/ICCV2023/html/Li_End-to-end_3D_Tracking_with_Decoupled_Queries_ICCV_2023_paper.html)
说明单 query 承担不同任务可能出现表示冲突；这里借鉴的是 task-specific query
分工，不把 MOT 结论直接当作 3D SOT 贡献。

### 3.4 Geometry-aware evidence 合同

支持域必须保留 base：

```text
support = S_base(last prediction)
        ∪ S_prior(last -> μ, object size, calibrated Σ)
```

base 和 expansion 使用独立 token 预算与 source flag，避免扩展区域抢走 B0 点。
[HVTrack](https://arxiv.org/abs/2408.02049) 的 base-expansion cross-attention 与背景
抑制说明 expanded search 需要显式区分 base/context，而不是简单混合。

对点 `p_i` 增加：

```text
p_i - μ
trajectory-parallel / perpendicular coordinates
clipped Mahalanobis bias = -0.5 (p_i-μ)^T Σ^-1 (p_i-μ)
base/extension source
overlap / point-count / validity
```

Mahalanobis 项只作为 attention bias/feature，不作为硬 mask。否则 sigma 偏小会把
真实目标点直接删掉。

### 3.5 B3 梯度与执行合同

B3 延续当前 action-consistent router 的正确部分：

- B0/B1/B2 candidate producer 全部冻结并 `detach`；
- observation 为 reference；
- candidate source 与 step ratio 组成同一个被监督、被选择、被执行的动作；
- normal/gap step cap 保留；
- router 只在 raw Search 通过 candidate gate 后训练。

相对当前 B3 的候选/执行合同，唯一变化是第二个辅助候选从
`B1-clipped refined` 改为独立 `raw Search`。

## 4. 在线与训练数据传输

### 4.1 在线两阶段 pre-pass

当前 `build_input_dict` 在 forward 前已经裁点，learned B1 无法通过简单增加一条
模型内箭头控制 crop。正式在线链应改为：

```text
results_bbs + timestamps
 -> predict_motion_from_history()       # tiny B1, box-only
 -> motion_prediction {μ,Σ,v,valid}
 -> build_input_dict(..., motion_prediction=...)
 -> construct S_base ∪ S_prior
 -> full B0/B2/B3 forward
```

项目已有 M4 在输入构造前先 `predict`、再把 prediction 传入
`build_input_dict` 的模式，见 [`base_model.py`](../models/base_model.py#L613)。
B1 pre-pass 可以复用该调用结构，但不复用 M4 方法本身。

### 4.2 训练数据

不要在 DataLoader worker 内运行可训练 B1。推荐：

1. box-only 训练并冻结 B1；
2. 用冻结 B0/B1 在训练 tracklet 上生成 recursive replay；
3. cache 每个状态的 `μ/Σ/v/valid` 与 previous prediction error；
4. sampler 读取 cache 构造 base/prior support；
5. B2 训练只更新 search encoder、dual-query adapter 和 vote/presence/utility head；
6. B2 晋级后冻结全部 candidate producer，离线训练 B3。

训练、验证必须比较 previous-error、gap、support reachability、validity 和点数分布，
防止再次出现训练 structural-valid `30.58%`、递归验证 `5.79%` 的错配。

## 5. 与两种原方案的取舍

| 维度 | 当前实现 | 图示串行强耦合 | 推荐方案 |
| --- | --- | --- | --- |
| B1/B2 proposal 一致 | 否 | 是 | 是 |
| B1 错误时 observation 回退 | 有 | 图中没有 | 始终保留 |
| 当前帧独立纠错 | raw 有，但被 clip | 可能被 motion query 限制 | raw Search 独立 |
| query 语义 | pre-Transformer coarse | 单一 motion query | final `q_obs/q_search` 双 query |
| sigma 风险 | 未训练、链路断裂 | 监督未定义 | NLL 校准后进入 geometry/risk |
| 梯度归因 | 清楚 | 易互相共适应 | 单向 stop-gradient，清楚 |
| 理论涨分上限 | 较低 | 高但脆弱 | 较高且 fail-safe |
| 论文差异 | 模块像外挂 | 接近 MTM/PTTR | 连续时间风险+双路证据+闭环 |

在当前 B1 尚未稳定涨点、sigma 未校准的证据条件下，原图强耦合的 cascade risk
高于潜在收益。推荐方案把强耦合限制在 auxiliary evidence branch，是更合适的
涨分/论文折中。

## 6. 实现与实验顺序

### Stage 0：不训练，确认 headroom

- 同 checkpoint 导出 observation、motion、raw Search、clipped refined、
  oracle(obs/raw)；
- 以 tracklet bootstrap 判断 raw Search 是否有 final tracking headroom；
- 导出 base/endpoint reachability 和当前 B1 sigma coverage；
- 若 oracle 不足，暂停 B3。

### Stage 1：解除负连接

- 暴露 `raw_search_box`；
- 新增 `obs_vs_raw_search`；
- 旁路 B1-centered clip；
- 保留 observation-anchored step cap；
- 用原 checkpoint 复测，不先长训。

### Stage 2：先实现 final-decoder dual query

- Transformer 可选返回 `decoder_state`；默认返回与旧 checkpoint 完全兼容；
- B2 query 从 coarse `[B,256]` 改为 final current state `[B,64]`；
- 增加零初始化 motion residual，形成 `q_obs/q_search`；
- 保持现有 endpoint support 不变，隔离 query 因果效果；
- short run 通过后才进入 Stage 3。

### Stage 3：校准 B1 风险

- mean robust loss + motion-aligned heteroscedastic NLL；
- true/fixed/shuffled `delta_t`；
- 50/80/95% coverage、NLL/ECE、gap 分层；
- sigma 未通过 calibration gate 前不接动态 support。

### Stage 4：B1 pre-pass 与 uncertainty support

- 新增 box-only `predict_motion_from_history`；
- 在线先 B1、后 crop；训练使用 frozen recursive cache；
- `base ∪ object-size/fixed-width prior support` 作为第一对照；
- 再单独替换为 calibrated sigma width；
- 加 B1-relative/Mahalanobis geometry bias；
- 保留 base 独立 token 和 raw candidate。

### Stage 5：修复训练状态和 utility

- synthetic history → recursive replay；
- 加同类别 hard-negative tracklet；
- availability、presence、utility 分开；
- utility 监督 raw Search 相对 observation 的 signed advantage。

### Stage 6：最后启用 B3

- no-router / oracle / offline router；
- H1/H3、q50/q10、cooldown、scalar/full embedding；
- seed42 final/late-3 晋级后才运行 seed43/44。

## 7. 耦合消融

| 行 | Geometry coupling | Query coupling | Decision coupling | 唯一问题 |
| --- | --- | --- | --- | --- |
| C0 | hand endpoint | coarse query | none | 当前基准 |
| C1 | hand endpoint | final `q_obs` | none | final decoder state 是否更有效 |
| C2 | hand endpoint | final dual query | none | motion residual query 的净贡献 |
| C3 | B1 `μ` + fixed support | final dual query | none | proposal/search alignment |
| C4 | B1 `μ/Σ` support+bias | final dual query | none | calibrated uncertainty 的净贡献 |
| C5 | 同 C4 | 同 C4 | B3 source×step | 闭环选择的净贡献 |

必要鲁棒性控制：

- normal / shuffled `μ` / forced-invalid B1；
- fixed sigma / calibrated sigma / shuffled sigma；
- raw Search / B1-clipped Search；
- base-only / prior-only / base∪prior；
- synthetic / recursive replay。

若 shuffled/invalid B1 会显著破坏 C_obs，说明 observation 路径没有真正解耦，
不得把该实现登记为本规范的双查询方案。

## 8. 最近一个可关闭里程碑

当前只实现：

```text
五模式同-checkpoint attribution
+ raw Search 独立推理/no-clip
+ Transformer final decoder state API
+ q_obs/q_search 双查询短测
```

这四项完成前，不实现 learned B1 pre-crop，不训练 B3，不重启 B4。完成后根据
raw/oracle 与 dual-query 的 tracklet-level 结果决定是否投入 B1 calibration 和
动态 support。
