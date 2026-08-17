# CT-SeqTrack B1–B4 连接重构与消融计划

> **TrajTrack 参考状态（2026-08-16）**：自本标注起，TrajTrack 不再作为
> CT-SeqTrack 后续方法设计、Gate/proposal 机制选择、超参数设定或性能有效性的
> 参考依据；仅保留为必须引用的相关工作、历史审计对象和 GT-free 评测警示。
> 下文既有 TrajTrack 内容均为历史记录，不再驱动当前或未来方案。

更新时间：2026-08-04

状态：**当前唯一权威的模块决策与实验顺序**

> 本文覆盖 B1、B2、B3、B4 的当前完成度、连接问题、目标数据流、修复顺序、
> 晋级门槛和论文消融。旧文档继续承担历史实现与复现说明；凡是与本文冲突，
> 以本文为准。本文中的“完成”只表示工程合同或机制证据达到相应阶段，不等于
> 已经得到可发表的涨点结论。

## 1. 结论先行

CT-SeqTrack 目前不是“四个模块都已完成、只差一起训练”，而是：

| 模块 | 工程状态 | 科学状态 | 当前决定 |
| --- | --- | --- | --- |
| B1 连续时间运动先验 | 核心 encoder、物理 `delta_t`、递归接口基本完成 | 均值能学，但标准跟踪增益不稳定；`sigma` 未被正确训练和校准 | **保留均值主干，补齐不确定性合同** |
| B2 状态对齐双尺度搜索 | 数据采样、候选生成、refiner 训练和诊断链已完成 | raw candidate 有信号，但当前 B1-centered refinement 显著伤害 raw candidate；候选覆盖率和 presence 判别不足 | **当前第一优先级，先修连接再训练** |
| B3 闭环选择器 | 六动作、打包、rollout、观测锚定步长限制基本完成 | 因 B2 未通过 candidate gate，router 没有进入有效科学验证 | **冻结；B2 晋级后再启用** |
| B4 点特征时序一致性 | 首版已完整训练并有日志 | 当前几何、目标和表征层级均有较大问题，且成本过高 | **退出论文主线，最后单独重做** |

论文主线应收敛为：

> **B1 提供连续物理时间下的均值与校准风险；B2 通过模型前 motion pre-pass、
> base-preserving support 和独立 `q_obs/q_search` 提取当前帧证据；B3 以
> observation 为安全锚点选择 candidate source 与执行步长。**

正式耦合接口、张量、在线两阶段数据流和消融见
[不确定性感知非对称双查询耦合](ASYMMETRIC_DUAL_QUERY_COUPLING_20260804.md)。

B4 暂时不是主线贡献。若 B4 后续不能在正确几何、防坍缩目标和合理速度下独立
通过机制测试，则论文只写 B1–B3。

## 2. 证据边界

当前可直接支持决策的关键事实如下。

1. 新 B1 precursor seed42 final 为 `53.318/62.573`；历史 B0 为
   `53.360/64.382`。二者不是同 commit matched 对照，因此只能说明 B1 尚无
   稳定正证据，不能精确声明净下降。
2. B2-v3 refiner 的 endpoint 误差为：motion `2.9045`、raw search `2.6496`、
   refined `2.7344`。`refined - raw = +0.0848`，tracklet bootstrap 95% CI
   `[+0.0379, +0.1323]`；21 个发生改变的端点中 15 个变差、6 个变好。
3. B2-v3 训练 structural-valid 为 `30.58%`，验证仅 `5.79%`；验证集
   foreground-valid 只有 `66/2004 = 3.29%`，presence AUC 为 `0.497`。
4. 因 candidate gate 未通过，B3 router 没有正式运行。B3 当前只能登记为
   “工程管线完成”，不能登记为“模块完成”。
5. B4 final 为 `51.189/60.886`，相对历史 B0 为 `-2.171/-3.496`；前景 feature
   std 从 `0.0947` 收缩到 `0.0156`，训练耗时约为 B0 的 `8.24×`。

对应原始分析产物：

- [B2-v3 precursor 汇总](../compare_results/data/b2_v3_precursor_seed42_20260804_summary.csv)
- [B2-v3 refiner headline](../compare_results/data/b2_v3_refiner_seed42_20260804_headline.csv)
- [B2-v3 refiner 便携报告](../compare_results/reports/b2_v3_refiner_seed42_20260804/report.html)
- [B4 最终诊断](../compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md)

所有论文主表最终仍必须使用 same-code、same-commit、same-data-hash、same-seed 的
matched B0；历史 B0 只作为内部 guardrail。

## 3. 当前数据流为什么没有形成一个模块链

### 3.1 当前实际数据流

```text
history boxes + delta_t
    ├── B1 learned physical prior --------------------┐
    │                                                 ├── B2 refiner around B1
    └── hand-coded CV/yaw-rate endpoint box -> crop --┘

current point cloud -> SeqTrack3D observation box
pre-Transformer point_feature -----------------> B2 query

B2 raw vote -> clip around B1 -> refined candidate
observation / B1 / refined candidate -> B3 observation-anchored step

low-level time-bearing point features -> B4 raw pairwise consistency
```

这条链有四个断点：

1. **B1 与 B2 有两个运动模型。** B1 使用学习到的物理先验；B2 crop 中心仍由
   手工 CV/yaw-rate endpoint 生成。两者读取相同 history tensor，只证明了
   state tensor 对齐，不证明 proposal/search support 对齐。
2. **B2 先看到当前帧证据，随后又被拉回 B1。** raw search 已优于 motion，
   但 [`selective_innovation.py`](../models/ct_v2/selective_innovation.py#L437)
   把 raw residual 裁剪到 B1-centered radius，导致 refined 显著差于 raw。
3. **B2 的观测语义层级过早。** 当前 refiner 使用
   [`point_feature`](../models/seqtrack3d.py#L1430)，而 SeqTrack3D 最终 box 来自
   Transformer decoder。Transformer 当前只返回回归后的
   [`dec_output`](../models/attn/Models.py#L146)，没有暴露最终 decoder hidden，
   所以 B2 没有读取与最终 observation decision 同层级的 query。
4. **风险量没有贯通。** B1 输出
   [`log_sigma_xy`](../models/ct_v2/motion.py#L408)，但 B1 主损失仍是均值的
   [`SmoothL1`](../models/seqtrack3d.py#L3033)；B2 的搜索宽度来自另一套启发式，
   endpoint box 又原样复制目标尺寸；B3 却消费这个未校准 sigma。

### 3.2 B3 的“双重安全”问题

B2 已把搜索结果裁剪到 B1 附近，B3 又在 observation 坐标系中限制
`0.25/0.5/1.0 × step_cap` 的修正。前者会提前抹掉“当前观测纠正错误运动先验”
的能力，后者才是最终闭环真正需要的安全约束。

因此目标合同是：

- B2 负责生成**独立且可评价**的 raw Search candidate；
- B3 负责决定是否使用、使用哪个候选、走多大一步；
- 不再由 B2 和 B3 对同一误差重复裁剪。

### 3.3 训练状态分布与部署状态不一致

当前每个训练样本含 4 个 candidate，造成约 75% 的非零人工 history；训练
structural-valid `30.58%`，递归验证只有 `5.79%`。这不是普通 class imbalance，
而是 refiner/presence 学到的状态分布与部署闭环不同。presence AUC 约 0.5 是
该错配的直接表现之一。

后续必须用冻结 B0/B1 生成 recursive replay，并加入同类别干扰负样本；不能再
主要依赖独立的 synthetic candidate 扰动训练 router。

## 4. 目标数据流与模块合同

```text
history boxes + physical delta_t
        │
        ▼
B1 box-only pre-pass: μ_xy + calibrated Σ_motion + h_motion
        │                         │
        │ geometry               │ dynamics context
        ▼                         ▼
base crop ∪ uncertainty-aware prior support
        │
        ▼
shared point tokens
        ├── independent final q_obs ─────────> observation candidate
        └── motion-guided q_search
              + B1-relative geometry ───────> independent raw Search candidate

B1 μ_xy ────────────────────────────────────> motion candidate

observation / motion / raw Search
        │
        ▼
B3: observation-anchored keep / source × {0.25,0.5,1.0}
        │
        ▼
recursive prediction history
```

全局原则是“局部紧耦合、全局解耦”：B1 到 B2 geometry/query 是单向
stop-gradient 耦合；`q_obs` 和 observation box 不读取 B1；B3 冻结所有候选
生产器。B1 错误时，系统必须仍能精确退化到原 observation 路径。

### 4.1 B1：连续时间均值与可校准不确定性

输入仍是因果历史框、valid mask 和真实 `delta_t`。输出合同改为：

- `mu_xy`：当前 query gap 下的位移均值；
- `sigma_parallel/perpendicular`：运动方向坐标系下、同时间跨度的 aleatoric
  scale；低速时回退 isotropic/fixed scale；
- `valid`：历史与时间是否足够支持运动预测；
- 可选 `feature`：供 B2/B3 使用，但不替代明确的均值和风险量。

训练从仅均值 SmoothL1 改为 motion-aligned 二维 heteroscedastic NLL，并保留
均值稳健项。必须
报告 50%/80%/95% coverage、NLL/ECE 和按 gap 分层的误差。只有 coverage 基本
校准后，`sigma` 才能控制 B2 support 和 B3 风险；在此之前 B3 应忽略或冻结 sigma。

### 4.2 B2：双尺度支持域中的独立当前帧证据

B2 应保留完整 B0 base crop，并增加独立预算的扩展支持：

```text
support = base crop
        ∪ object-size-aware endpoint region
        ∪ B1 trajectory tube / sigma expansion
```

具体合同：

- endpoint/tube 的中心以 B1 `mu_xy` 为统一参考；若 B1 invalid，回退手工 CV，
  但必须记录 fallback，不能混成一个候选来源。
- 宽度同时受目标尺寸、位移跨度和校准 `sigma` 控制，不再只复制 latest box 的
  固定尺寸。当前固定尺寸实现在
  [`ct_search.py`](../utils/ct_search.py#L319) 和
  [`ct_search.py`](../utils/ct_search.py#L385)。
- 点特征显式编码 `point - mu`、沿轨迹/垂轨迹坐标、base/extension source、
  overlap 和有效点数。
- 在线必须先用 history 运行轻量 B1，再把 prediction 传给 `build_input_dict`；
  不能只在模型 forward 内增加箭头，因为当前点云已经在 forward 前完成裁剪。
- 保留 motion-independent `q_obs`，并从 Transformer `l1` 后暴露最终
  current-target decoder hidden；`q_search` 是以 `q_obs` 为基础、零初始化的
  bounded motion residual query。B2 训练时两路上游特征全部 stop-gradient。
- `mu/sigma` 只作为 support 和 clipped Mahalanobis attention bias，不作为硬
  point mask，避免错误 prior 删除真实目标点。
- B2 官方候选是 `raw_search_box`。可以保留 `b1_clipped_box` 作为诊断消融，
  但它不能继续替代 raw candidate。

### 4.3 B3：观测锚定的保守闭环决策

B3 只在 B2 通过 candidate gate 后训练。candidate source 与 step 必须分开定义：

```text
abstain / keep observation
motion candidate × {0.25, 0.5, 1.0}
search candidate × {0.25, 0.5, 1.0}
```

`step` 不是与 motion/search 并列的第三个 candidate；它是 Decision Head 对所选
candidate residual 的执行比例。motion/search 当前都只提供 XY，z/yaw 继续来自
observation。

最终步长仍以 observation 为 reference，并保留 normal/gap step cap。router 输入
优先使用少量可解释量：candidate disagreement、B1 calibrated risk、B2 validity、
targetness、presence、观测稀疏度和 gap；高维 embedding 只有在标量版本不够时
再加入。训练 rollout 必须与部署的递归历史和 cooldown 规则一致。

### 4.4 在线 pre-pass 与训练 cache

在线使用：

```text
results_bbs/timestamps
 -> predict_motion_from_history()
 -> build_input_dict(..., motion_prediction)
 -> base∪prior crop
 -> B0/B2/B3 forward
```

该调用方式可参考现有 M4 在输入构造前产生 prediction 的框架，但不复用其滤波
逻辑。训练时先冻结 B1，用 recursive tracklet replay 缓存 `mu/sigma/velocity`，
sampler 读取 cache；禁止在 DataLoader worker 内运行可训练 B1。

### 4.5 B4：独立研究支线

当前 PFTC 的
[`canonicalize_points`](../models/ct_v2/point_feature_consistency.py#L36)
存在 yaw 逆变换约定问题，且
[`raw SmoothL1`](../models/ct_v2/point_feature_consistency.py#L228)
允许常量表示坍缩。即使修正 yaw，也不能原样重跑。

若重做，必须同时满足：

1. 单元测试验证正/逆 yaw 与 box canonical 坐标完全一致；
2. 在 projector、memory token 或最终 decoder token 上做一致性，不直接压平
   B0 依赖的低层 time-bearing feature；
3. 使用 stop-gradient teacher，并加入 variance/covariance guard，或采用
   memory-cycle objective；点级匹配则至少改为 mutual NN；
4. 5-epoch 三臂机制实验同时监控 feature std、effective rank、tracking metric
   和 step time；速度必须恢复到 `< 2× B0` 才允许长训。

## 5. 执行顺序与晋级门槛

### Phase 0：零训练归因，先判断搜索是否值得救

对**同一个冻结 checkpoint、同一个递归 mini_val endpoint**导出：

1. `observation`；
2. `motion`；
3. `raw_search`；
4. `current_b1_clipped_refined`；
5. `oracle(observation, raw_search)`。

同时导出 tracklet id、GT、候选有效性、help/harm、B1 sigma、point count、crop
reachability、presence/targetness 和 gap。所有置信区间以 tracklet 为重采样单位。

内部预注册 gate：

- 若 oracle 相对 observation 不达到 `+0.5 Success` 或 `+1.0 Precision` 中至少
  一项，则暂停 B3，不训练 router；
- 若 raw search 相对 observation 没有可重复的 helpful 子集，先验证
  final-decoder-state `q_obs` 和零初始化 `q_search`；双查询仍无信号时再修
  pre-pass/support/replay，不调 router；
- 若 `raw_search` 明显优于 `b1_clipped_refined`，立即把 raw 设为官方候选；
- 同时完成 B1 sigma coverage 与“GT 是否落在 base/endpoint/tube support”审计。

### Phase 1：最短涨点路径——解除错误连接

只做一个机制改动：

- 暴露 `raw_search_box` 给推理和离线 oracle；
- 移除 B2 内部 B1-centered residual clip；
- 保留 B3 observation-anchored step cap；
- 同 checkpoint 复跑五模式归因。

这一步无需先重训 B1，也不引入新 router。它直接检验当前已观察到的
`raw < refined` 是否能转化为跟踪收益。

### Phase 2：先修 final-decoder dual query

在保持现有 endpoint support 不变的前提下：

1. 扩展 Transformer API，可选返回 `l1` 后的 current-target decoder state；
2. observation box 默认输出与旧 checkpoint 保持逐 bit 兼容；
3. B2 从 pre-Transformer `point_feature` 切换到 final `q_obs`；
4. 加入零初始化、risk-bounded 的 motion residual，得到独立 `q_search`；
5. B2 loss 不更新 B0/B1，`q_obs` 仍单独产生 observation candidate；
6. short run 只比较 coarse / final `q_obs` / final dual query。

先做 query 是因为它完全位于模型内，能够在不重构在线 crop 的情况下验证图示中
“motion-guided query”的净贡献。

### Phase 3：补齐 B1 风险链

在 box-only 小实验中比较：

- CV vs learned GRU/MLP mean；
- mean SmoothL1 vs mean + heteroscedastic NLL；
- `true dt` / dataset-mean fixed / within-dataset shuffled；
- stride/gap 1、2、4 与 held-out cadence。

只有当 `true dt` 在同数据集重采样和未见 cadence 中持续胜出，才把
“frame-rate-invariant physical time”写成贡献；否则 B1 只写成 trajectory prior，
不做过强因果声明。

### Phase 4：接入 B1 pre-pass、support 与递归训练状态

严格一次只改一个因素：

1. **Pre-pass**：新增 box-only `predict_motion_from_history`，在线先 B1、后 crop；
2. **Support mean**：fixed endpoint → `base ∪ B1 μ + fixed-width support`；
3. **Support risk**：fixed width → object-size + calibrated sigma width；
4. **Geometry bias**：加入 B1-relative/Mahalanobis feature，但不做硬 mask；
5. **State distribution**：synthetic candidates → frozen B0/B1 recursive replay/cache；
6. **Negatives**：加入同类别干扰 tracklet/sequence hard negatives；
7. **Presence/utility**：availability、presence、raw-vs-observation signed utility
   分开监督。

B2 晋级到 B3 的最低条件：

- foreground-valid 不再停留在当前 `3.29%`，且有效样本量足以按 tracklet 估计；
- presence AUC 明显高于随机，内部目标 `>= 0.65`；
- raw search 对 observation 存在正 oracle headroom，且 helpful/harm 可由
  validity/utility 特征区分；
- forced-invalid/shuffled B1 时 observation 路径基本不退化；
- same-checkpoint、tracklet bootstrap 下结论稳定。

### Phase 5：B2 通过 gate 后再训练 B3

先做离线 oracle/router 上限，再做一轮与部署一致的 recursive rollout。比较
H=1/H=3、q50/q10、with/without cooldown、scalar/full embedding。B3 的论文价值
不是“有一个 gate”，而是**候选独立、动作可执行、训练和闭环执行一致**。

### Phase 6：最后决定是否重做 B4

B4 不阻塞 B1–B3 论文。只有 B1–B3 主表已稳定、B4 三臂短测通过几何、
防坍缩和速度门槛，才启动一个 seed42 长训；否则删除主表 B4 行。

## 6. 消融实验的正确顺序

### 6.1 开发期机制筛选

开发期先回答因果问题，不把多个改造打包：

| 顺序 | 对照 | 唯一变化 | 回答的问题 |
| --- | --- | --- | --- |
| D0 | obs / motion / raw / clipped / oracle | 无训练 | 候选是否有 headroom，哪里被破坏 |
| D1 | raw vs B1-clipped | 只移除 clip | B1-centered refinement 是否是负连接 |
| D2 | coarse vs final `q_obs` | 只改 query 语义层级 | 是否需要最终观测语义 |
| D3 | final `q_obs` vs `q_obs/q_search` | 只加 motion residual query | 双 query 是否优于单 query |
| D4 | hand endpoint vs B1 μ fixed support | 只改 pre-pass/support mean | proposal/search alignment 是否有效 |
| D5 | fixed vs calibrated sigma support+bias | 只改 B1 风险 | uncertainty 是否真正帮助 geometry |
| D6 | synthetic vs recursive replay | 只改训练状态 | distribution shift 是否破坏 presence/utility |
| D7 | no router vs B3 | 只加闭环选择 | 选择器是否把 oracle headroom 变成跟踪收益 |

每个 D 实验先短程/冻结 checkpoint 机制测试，通过预注册 gate 后才做 60 epoch。
不要把 support、query、replay、router 一次全改，否则即使涨分也无法写清贡献。

### 6.2 论文主表

| 行 | 模型 | 论文解释 |
| --- | --- | --- |
| A0 | matched SeqTrack3D B0 | 同代码、同数据、同训练预算基线 |
| A1 | B0 + B2（CV support + dual query） | 隔离当前帧扩展搜索证据与双 query |
| A2 | B0 + B1 + B2 | learned physical prior 替代 CV，并控制 support/bias/risk |
| A3 | A2 + B3 | 闭环保守选择能否把候选 headroom 转为最终涨点 |
| A4 | A3 + redesigned B4（可选） | 只有 B4 独立正向且成本合理时保留 |

该顺序比“B0 → B1 → B1+B2 → B1+B2+B3”更严谨，因为 B2 必须先证明在没有
learned B1 时也提供独立 observation evidence；否则 B1 与 B2 的贡献不可分。

### 6.3 论文内部消融

- B1：CV/learned mean，true/fixed/shuffled `dt`，fixed/calibrated/shuffled sigma，
  gap 分层；
- B2：hand/B1 pre-pass，base/prior/base∪prior，coarse/final-single/final-dual query，
  raw/clipped，synthetic/replay，普通负样本/同类 hard negatives；
- B3：H1/H3，q50/q10，cooldown on/off，scalar/full embeddings；
- B4：独立表展示 yaw correct、anti-collapse、feature level 和时间开销，不与主线
  多因素捆绑。

主表报告 Success/Precision、late-3、per-category、gap/稀疏度分层和 FPS；所有
差值给 tracklet-level paired bootstrap CI。best checkpoint 只用于诊断，正式
判断使用预注册 final/late-3。

## 7. 文献借鉴与创新边界

| 工作 | 可借鉴点 | CT-SeqTrack 中的落点 | 不能直接声称的内容 |
| --- | --- | --- | --- |
| [SeqTrack3D](https://arxiv.org/abs/2402.16249) | 历史点云和框序列、local/global sequence | 保留 B0 order-clock observation 主干 | “首次使用历史运动/框序列” |
| [M²-Track](https://arxiv.org/abs/2303.12535) | motion-centric 两阶段定位 | B1 proposal 与 observation/search correction 分工 | “首次 proposal-refine” |
| [Motion-to-Matching](https://arxiv.org/abs/2308.11875) | 历史框 coarse motion + 点云 geometry refinement | 说明原图串行强耦合不是充分创新；改为 calibrated risk + dual query + fail-safe observation | “首次 motion-guided matching” |
| [PTTR](https://arxiv.org/abs/2112.02857) | coarse-to-fine refinement | B2 使用最终 decoder query 做 evidence refinement | “首次 coarse-to-fine” |
| [MBPTrack](https://arxiv.org/abs/2303.05071) | box-prior reference points、粗到细定位 | object-size-aware endpoint/reference support | “首次 box prior” |
| [HVTrack](https://arxiv.org/abs/2408.02049) | base-expansion cross-attention、背景抑制 | B0 base 与 extension 分源编码、同类干扰抑制 | “首次扩展搜索” |
| [DQTrack](https://openaccess.thecvf.com/content/ICCV2023/html/Li_End-to-end_3D_Tracking_with_Decoupled_Queries_ICCV_2023_paper.html) | 相邻 3D MOT 中的 task-specific decoupled queries | `q_obs/q_search` 的表示分工参考，不作为 SOT 直接贡献 | “DQTrack 已验证本方法” |
| [UA-Track](https://arxiv.org/abs/2406.02147) | 相邻 3D MOT 中 uncertainty-aware decoder/query initialization | sigma 进入 geometry/query 前必须受监督和校准 | “首次 uncertainty-aware tracking” |
| [StreamTrack](https://arxiv.org/abs/2303.07605) | memory/query、contrastive sequence enhancement | recursive replay 和同类别负 tracklet | “首次连续运动/memory” |
| [TrajTrack](https://arxiv.org/abs/2509.11453) | 显式 proposal + 隐式轨迹修正 | B1 轻量 box trajectory prior 的定位 | “首次轨迹建模” |
| [ChronoTrack](https://arxiv.org/abs/2604.13789) | memory token temporal/cycle consistency | B4 若重做则在 compact token 层做一致性 | “首次时序一致性” |
| [VICReg](https://arxiv.org/abs/2105.04906) | variance/covariance 防坍缩 | B4 projector 的 variance/covariance guard | 直接把 2D SSL 结论当成 3D tracking 贡献 |

可发表的差异应落在**连续物理时间的校准风险、base-preserving uncertainty
support、motion-independent/motion-guided 双 query，以及 observation-anchored
动作一致闭环**这一组合合同上，而不是单个已有组件的重新命名。

## 8. 当前禁止事项

- 不启动 B3 seed43/44 或长训，直到 B2 raw candidate 通过 gate；
- 不再以 B1-centered refined candidate 代表 Search candidate；
- 不把 shared history tensor equality 写成“搜索与运动先验已对齐”；
- 不使用未训练/未校准 B1 sigma 控制论文方法并声称 risk-aware；
- 不让 motion feature 进入唯一的 observation query，也不让 B2 loss 更新 B1；
- 不把 step 与 motion/search 画成并列 candidate；
- 不原样重跑 B4，不只修 yaw 就开始 60 epoch；
- 不用历史 B0、异常 SeqTrack control 或 best checkpoint 代替 matched final；
- 不一次同时修改 support、query、replay、router；
- 不在 true/fixed/shuffled 和 held-out cadence 证据前宣称物理时间因果收益。

## 9. 最近的可执行出口

下一个实现里程碑不是“训练 B3”或“动态 B1 crop”，而是完成以下四个产物：

1. 同 checkpoint 五模式 endpoint + recursive tracking 导出；
2. `raw_search_box` 独立推理路径，以及关闭 B1-centered clip 的消融；
3. Transformer final decoder state 的兼容 API；
4. 保持当前 support 不变的 `q_obs/q_search` 双 query 短测，同时完成 B1 sigma
   coverage 与 crop reachability 报告。

完成后只有两种分叉：

- **oracle/raw 与 dual query 有 headroom**：进入 B1 calibration，再接 pre-pass、
  uncertainty support、recursive replay 和 B3；
- **oracle/raw 无 headroom**：停止 router，先验证 final/dual query；仍无改善则删掉
  B2/B3，论文退回更小但可证实的 B1 或 SeqTrack3D 分析工作。
