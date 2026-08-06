# CT-SeqTrack 实验执行路线

更新时间：2026-08-05

本文档是当前实验主计划。目标不是继续增加模块，而是用一条可归因、可停止、可复现的实验谱系，回答三个核心问题：

1. B1 是否真的提供了优于简单匀速模型的历史运动先验？
2. B2 是否能利用额外点云证据产生优于 B0 observation 的独立候选？
3. B3 是否能把候选 headroom 转化为稳定的递归 tracking 增益？

权威架构说明：

- [B1–B4 连接重构与消融计划](docs/B1_B4_REDESIGN_AND_ABLATION_PLAN_20260804.md)
- [不确定性感知非对称双查询耦合](docs/ASYMMETRIC_DUAL_QUERY_COUPLING_20260804.md)

相关实验原则参考：

- [SeqTrack3D](https://arxiv.org/html/2402.16249)：使用历史框扰动模拟递归误差；在较小训练子集上做组件消融、完整验证集评估；历史长度不是越长越好。
- [M2-Track](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Beyond_3D_Siamese_Tracking_A_Motion-Centric_Paradigm_for_3D_Single_CVPR_2022_paper.html)：先证明第一阶段运动估计，再验证第二阶段 refinement，并在训练中模拟测试历史误差。
- [MBPTrack](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_MBPTrack_Improving_3D_Point_Cloud_Tracking_with_Memory_Networks_and_ICCV_2023_paper.pdf)：逐项消融 memory 和定位组件；历史信息增加过多会退化。
- [CXTrack](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_CXTrack_Improving_3D_Point_Cloud_Tracking_With_Contextual_Information_CVPR_2023_paper.pdf)：上下文、mask、center branch 和 query 结构分别验证；更强的门控不一定更稳定。
- [HVTrack](https://arxiv.org/html/2408.02049)：扩大搜索区域既提高目标召回，也引入相似目标和背景噪声；扩张范围和时间间隔必须单独分析。

---

## 0. 当前判断

```text
CORE_ARCHITECTURE_IMPLEMENTED
LATEST_CONFIG_15_TO_18_NOT_FORMALLY_VALIDATED

B0 = OBSERVATION_AND_FAIL_SAFE
B1 = BOX_HISTORY_MOTION_PRIOR, NOT YET PROMOTED
B2 = RAW_SEARCH_OFFLINE_SIGNAL_POSITIVE, RECURSIVE_GAIN_UNPROVEN
B3 = ENGINEERING_READY, SCIENTIFICALLY_BLOCKED_BY_B2
B4 = OUT_OF_MAINLINE

MAIN_RISK = TOO_MANY_B1_TO_B2_COUPLINGS_OPENED_AT_ONCE
MAIN_NEXT_STEP = BUILD_ONE_MATCHED_B0_LINEAGE_AND_RUN_GATED_EXPERIMENTS
```

已有证据：

- 历史 B0 为 `53.360 Success / 64.382 Precision`，但还缺少与最新代码完全 matched 的 B0。
- 新 B1 tracking 结果没有超过历史 B0；B1 是否优于 CV、是否真正利用真实时间仍需正式对照。
- B2-v3 endpoint 中，observation RMSE 为 `2.7396`，B1 motion 为 `2.9045`，raw Search 为 `2.6496`，B1-clipped refined 为 `2.7344`。
- raw Search 有离线正信号；B1-centered clip 相对 raw Search 变差约 `0.0848`，因此 clipped candidate 不进入主链路。
- validation structural coverage 约 `5.79%`、foreground-valid 约 `3.29%`，presence AUC 约 `0.497`；当前 B2 可靠性证据不足以训练复杂 B3。
- 旧 full v21 递归结果约 `26.754/26.876`，说明离线候选改善不能替代递归验证。

证据文件：

- [B2-v3 headline](compare_results/data/b2_v3_refiner_seed42_20260804_headline.csv)
- [当前决策记录](compare_results/data/b2_v3_refiner_seed42_20260804_decisions.csv)

---

## 1. 术语与实验控制

### 1.1 三种“固定”

1. **固定 B0 checkpoint**：后续所有实验加载同一个 `B0_seed42.ckpt`，不能各自重新训练不同 B0。
2. **冻结 B0/B1 参数**：模块仍参与 forward，但 `requires_grad=False`，不进入 optimizer，训练前后参数 hash 相同。
3. **fixed support**：额外搜索区域使用确定性的 CV 端点和固定边界，不使用 learned B1、动态 sigma 或 Mahalanobis geometry。

本计划中的 fixed support 统一定义为：

```text
support center/tube = constant-velocity endpoint
parallel margin     = fixed
perpendicular margin = fixed
dynamic sigma       = off
uncertainty geometry = off
```

具体 margin 沿用预注册配置，不在训练中根据验证结果反复调参。

### 1.2 模块角色

```text
B0 observation:
  当前点云证据产生 observation box 和 final decoder query q_obs。

B1 motion prior:
  历史框 + 时间戳产生 mu_xy、sigma 和 motion feature。
  第一角色是 support prior，不默认作为最终 tracking 输出。

B2 Search refiner:
  使用 base crop ∪ extra support 的点和 q_obs/q_search，产生独立 raw Search candidate。

B3 selector:
  在 observation 与 raw Search 之间决定 keep/update；不重新提取点，不反向更新 B0/B1/B2。

B4 consistency:
  B0 训练期正则，不是 B0+B1+B2+B3 后面的推理模块。
```

### 1.3 唯一 checkpoint 谱系

```text
B0_seed42.ckpt
│
├── E0: B0 observation baseline
│
├── D1: freeze B0 -> train B1 -> B1_seed42.ckpt
│                         │
│                         └── export matched recursive replay/cache
│
├── E1: freeze B0 -> train B2 with fixed CV support + q_obs
│
└── B1_seed42.ckpt
      ├── E2: freeze B0+B1 -> train B2 with B1 support + q_obs
      └── E3: freeze B0+B1 -> train B2 with B1 support + dual-query
                                 │
                                 └── freeze B0+B1+B2 -> rollout -> train E4 B3
```

E1、E2、E3 是独立训练对照，应使用相同 B0、相同 B2 初始权重、数据顺序、seed 和训练预算；不能简单把 E1 连续微调成 E2、再微调成 E3。

---

## 2. 训练前必须完成的工程检查

### 2.1 已完成的基础能力

- [x] B0 final decoder state 已暴露为 `q_obs`。
- [x] asymmetric dual-query 已实现，零初始化且 B1 invalid 时回退 `q_search=q_obs`。
- [x] 独立 `raw_search_box` 已保留，B1-clipped proposal 可作为诊断旁路。
- [x] B1 box-only no-GT online pre-pass 已实现。
- [x] B2 训练时 observation final output 保持不变。
- [x] B2 正式配置冻结上游并检查 trainable prefix、梯度和 frozen hash。
- [x] replay/cache、checkpoint hash 和候选来源诊断已具备。
- [x] B3 offline router 和 packaged inference 已具备。

### 2.2 正式服务器训练前的必修项

- [ ] 提交或标记一个干净 commit；记录 `git status`，不能用未标识的 dirty snapshot 长训。
- [ ] 为 config15 增加 B1-only 参数白名单：冻结 B0，只允许 B1 参数进入 optimizer。
- [ ] 为 B1 训练增加 B0 训练前后 hash 不变断言。
- [ ] 增加正式的 `q_obs-only` B2 配置；不能用 coarse query 代替此对照。
- [ ] config17 正式训练缺少 `--replay-cache` 时直接报错，禁止静默回退 hand-CV。
- [ ] 增加 B3 action saturation/effective unique action 统计。
- [ ] 建立 config01、15、16、17、18 对应的 full-dataset 配置。
- [ ] 移除/参数化 B3 工具中硬编码的 `mini_train`。
- [ ] 记录 train/val tracklet manifest、数据 hash、类别、endpoint 数和过滤规则。
- [ ] 明确 runner 中 `--variant full` 是旧 config04；新版组合入口是 `--variant b1_b2_b3`，不能混用。

### 2.3 Smoke test

每个新阶段开始长训前必须通过：

- [ ] 单 batch forward/backward，loss 和 gradient finite。
- [ ] 小样本 overfit 或 100–200 batch 训练，确认目标 loss 可下降。
- [ ] 一个短 tracklet 递归推理，检查首次失控帧。
- [ ] 固定输入重复推理，候选和 source count 可复现。
- [ ] observation-only 输出与对应上游 checkpoint 一致。
- [ ] checkpoint contract、replay hash、config hash 全部通过。

---

## 3. Gate A：建立 matched B0

目标：得到唯一、可复现、与最新代码完全一致的 observation 基线。

### 3.1 开发筛选

- [ ] 使用 mini 或固定 1/5 train 做工程检查；评估必须覆盖完整对应 validation。
- [ ] 不根据 mini 结果反复修改最终 test 协议。

### 3.2 正式 B0 seed42

- [ ] 使用正式 full train split 从头训练 B0 seed42。
- [ ] 使用当前 baseline 预注册的完整 epoch、batch size、optimizer 和 scheduler。
- [ ] primary checkpoint 使用预注册的 final/last；best checkpoint 只作辅助诊断。
- [ ] 保存 `B0_seed42.ckpt`、optimizer 信息、config、commit、数据 manifest 和日志。
- [ ] 在 normal、random20、gap1124 上运行 observation-only OPE。
- [ ] 报告总体和 category-wise Success/Precision。
- [ ] 按当前帧目标点数、gap、previous-error 分桶。
- [ ] 记录 FPS、峰值显存、参数量。

Gate A：

- 同一 checkpoint 重复评估结果稳定；
- 无 NaN、异常 tracklet 跳过或递归协议错误；
- observation 输出成为后续所有实验的共同 reference。

Gate A 未通过时，不训练 B1/B2/B3。

---

## 4. Gate B1：B1 资格验证

目标：判断 learned B1 是否比简单 CV 更适合作为 support prior；不先要求它直接超过 B0 observation。

### 4.1 训练

- [ ] 从 `B0_seed42.ckpt` 初始化 config15。
- [ ] 冻结 B0，只训练 B1 mean + NLL。
- [ ] 保存 B1 final/last checkpoint 和 calibration artifact。
- [ ] B1 训练期间最终 tracking 输出保持 observation；B0 OPE 应与 E0 一致。

### 4.2 第一轮必要对照

```text
D1-CV       constant velocity，无训练
D1-TrueTime learned B1，真实 delta_t
```

- [ ] 比较 endpoint mean/median RMSE。
- [ ] 按 normal、gap、类别、速度、history-valid 分层。
- [ ] 报告 target-in-support recall、support 体积和新增点数。
- [ ] 报告 NLL、50/80/95% coverage、ECE、sigma saturation。
- [ ] 以 tracklet 为单位 paired bootstrap。

### 4.3 仅在 TrueTime 优于 CV 后补时间对照

```text
D1-FixedTime
D1-ShuffledTime
```

- [ ] 固定数据、模型预算和 seed，只改变时间协议。
- [ ] 验证真实 `delta_t` 是否提供可重复增量。

Gate B1：

- learned mean 相对 CV 在 held-out tracklet 上方向稳定；
- 长 gap 或目标覆盖率至少有一项显示实际价值；
- sigma 与误差具有基本单调关系，coverage 不是明显失校准；
- fixed/shuffled 对照不能与 TrueTime 完全等价，否则不能声称 physical time 贡献。

停止规则：

- 若 TrueTime 不优于 CV：主线改用 CV fixed support；停止 dynamic sigma、Mahalanobis 和 B1 final-action 训练。
- 若 mean 有价值但 sigma 失败：允许 `B1 mu + fixed margins`，禁止 uncertainty-aware 声称。
- B1 motion 未证明直接优于 observation 前，不作为 B3 最终动作候选。

说明：`B0+B1` 是必要的训练/诊断阶段，但 config15 最终仍输出 observation，因此不把它伪装成主 OPE 增益行。B1 的价值放在独立 prior/calibration 表中报告。

---

## 5. Gate B2：先证明 raw Search

目标：先证明额外点证据和 final observation query 能产生有价值的独立候选，再讨论 learned B1 和 router。

### 5.1 E1：Search-CV / q_obs-only

```text
B0               = fixed checkpoint + frozen
support center    = hand CV
support margins   = fixed
q_search          = q_obs
learned B1        = off
dynamic sigma     = off
uncertainty geom  = off
utility loss      = off or weight 0
B3                = off
official candidate = raw Search
```

- [ ] 从与其他 B2 arm 相同的 B2 初始化训练。
- [ ] 保留完整 B0 base crop；extra support 使用独立点预算和 source flag。
- [ ] observation、raw Search、clipped 只作为不同输出模式评估，不改变 checkpoint。
- [ ] clipped candidate 只做失败机制对照，不进入最终候选。

### 5.2 E1 必报指标

- [ ] observation 与 raw Search 的 endpoint mean/median delta。
- [ ] raw win rate、help/harm rate、correction norm。
- [ ] oracle(observation, raw Search) headroom。
- [ ] structural coverage、foreground-valid coverage 和每个 tracklet 有效样本数。
- [ ] base/overlap/extension unique point count 和 GT reachability。
- [ ] short recursive 与完整 validation recursive Success/Precision。
- [ ] 首次失控帧、连续漂移长度和 tracklet-level paired bootstrap。

Gate B2：

- raw Search 相对 observation 有可重复的 paired 正信号，或存在明确且非极少数样本驱动的 oracle headroom；
- structural/foreground coverage 足以支撑 tracklet-level 判断；
- raw Search 的短递归结果没有再次出现旧 full 的灾难性崩溃；
- q_obs-only 确实优于或至少不弱于旧 coarse-query 对照。

停止规则：

- 若 oracle 也没有 headroom：停止 B1 support 和 B3，回到 B2 特征/监督问题。
- 若 endpoint 有提升但 recursive 崩溃：先检查 crop/replay/递归分布，不调 B3 掩盖。
- 若 foreground-valid 仍接近当前 `3.29%`：先修采样、真实 replay 和 hard negatives，不增加网络层。
- presence/utility 不能区分 help/harm 时，不继续叠加第二、第三个可靠性头。

---

## 6. Gate C：逐项接回 B1，不做全因子排列

前置条件：Gate B1 和 Gate B2 均通过。每一步只改变一条 B1→B2 路径。

### 6.1 E2：Search-B1 / support-only

```text
E1: CV support + q_obs
E2: B1 mu support + q_obs

其他条件全部相同：
fixed margins / geometry off / dynamic sigma off / B3 off
```

- [ ] 使用冻结 B0+B1 生成 exact recursive replay/cache。
- [ ] E1/E2 使用相同 B2 初始化、数据 manifest、seed、epoch 和 optimizer。
- [ ] 比较 target recall、背景点膨胀、raw Search paired delta 和 recursive OPE。

解释：`E2 - E1` 近似回答 learned B1 support 是否优于简单 CV support。

停止规则：若 E2 不优于 E1，最终模型保留 CV support，不接 dynamic sigma。

### 6.2 E3：Search-Dual / dual-query

```text
E2: B1 support + q_obs-only
E3: B1 support + q_obs + bounded B1 motion residual
```

- [ ] 保持 support、point features 和 replay 不变，只打开 dual-query。
- [ ] 检查 q_obs/motion 均 stop-gradient，上游 frozen hash 不变。
- [ ] 报告 gate 分布、residual norm、B1 invalid 时的 exact fallback。
- [ ] forced-invalid 和 shuffled-B1 时 observation 路径必须保持稳定。

解释：`E3 - E2` 回答 motion-conditioned query 的净贡献。

停止规则：dual-query 不产生稳定增量时，最终模型回到 q_obs-only。

### 6.3 可选 E3-Sigma：dynamic support

仅当 B1 calibration 通过且 E2 support 有增量时运行：

- [ ] fixed margin 与 dynamic sigma 只改变 width，中心和 B2 不变。
- [ ] 报告 coverage、target recall、背景膨胀和不同 gap 的收益。

停止规则：dynamic sigma 不优于 fixed margin 时，正式 full 保留 fixed margin。

### 6.4 可选 E3-Geometry：uncertainty geometry

仅当 dynamic support 已通过时运行：

- [ ] 在同一 support 上增加 parallel/perpendicular/Mahalanobis 特征。
- [ ] Mahalanobis 只作 soft feature/bias，不硬删除点。

停止规则：geometry 没有独立增量就不进入最终模型。

---

## 7. Gate D：B3 从二选一开始

前置条件：选出一个冻结的最佳 B2 candidate producer，并证明 raw Search 存在可识别的 help/harm 子集。

### 7.1 先做零训练上限和简单规则

同一 rollout 比较：

```text
R0 keep observation
R1 always raw Search
R2 oracle(observation, raw Search)          # 只作上限，不能部署
R3 one fixed conservative correction ratio
R4 simple threshold abstention
```

- [ ] 所有规则使用相同候选、step cap、tracklet 和递归历史。
- [ ] 阈值只在 train-tracklet calibration split 上选择，不用 validation 调阈值。

### 7.2 第一版 learned B3

第一版只保留：

```text
candidates = observation, raw Search
actions    = keep, one conservative Search update
B1 motion final action = off
three ratios            = off
utility feature         = optional, default off
```

- [ ] 冻结 B0/B1/B2，用 exact H=3 recursive rollout 离线训练 B3。
- [ ] 比较 no-router、always-raw、简单阈值和 learned B3。
- [ ] 报告 helpful precision、harm rejection、applied rate、calibration 和 OPE。

Gate B3：

- learned B3 必须超过简单阈值，而不只是超过 always-raw；
- validation recursive Success/Precision 有净增益；
- harm rate 满足预注册安全约束，normal/gap 不出现灾难性退化；
- 训练与部署使用相同候选分布和 step cap。

停止规则：

- learned B3 不超过简单规则：保留简单规则或移除 B3。
- binary router 不成立：不扩展到六动作。
- B1 motion 仍弱于 observation：不恢复 motion final action。

### 7.3 六动作仅作后续可选实验

只有 binary router 通过后才允许：

- [ ] 统计 `0.25/0.5/1.0` 经 step cap 后的 action-equivalence/saturation rate。
- [ ] 若大量动作裁剪为同一 correction，删除冗余 ratio。
- [ ] scalar-only 与 embedding router 只做一次受控对照。
- [ ] q10/q50、cooldown、H1/H3 不做完整排列组合，只在明确失败机制下单项测试。

---

## 8. 正式模型表与训练预算

### 8.1 主结果表

| ID | 模型 | 唯一新增变量 | 作用 |
|---|---|---|---|
| E0 | B0 observation | — | matched baseline |
| E1 | B0 + CV fixed support + B2(q_obs) | Search evidence | 证明 B2 本身 |
| E2 | B0 + B1 support + B2(q_obs) | learned support center | 证明 B1 support |
| E3 | E2 + asymmetric dual-query | motion-conditioned query | 证明 query coupling |
| E4 | 最佳 E2/E3 + simple/learned B3 | selective update | 转化候选 headroom |

B1 的 CV/True/Fixed/Shuffled 和 calibration 单独放在 B1 分析表，不用一个输出仍为 observation 的 `B0+B1` 行声称 OPE 增益。

### 8.2 三级预算

1. **Smoke**：单 batch、100–200 batch、短 tracklet，只验证工程正确。
2. **Screen**：固定 mini 或 1/5 train、完整 validation、seed42、约 5 epoch；只做方向判断。
3. **Formal**：只有通过 gate 的模型才使用 full train 和完整预注册 epoch。

### 8.3 多 seed

- [ ] 开发和第一轮 gate 只使用 seed42。
- [ ] 架构冻结后，仅对 E0、最终 E2/E3、E4 运行 seed43/44。
- [ ] 不对失败配置、所有 sigma/geometry/router 组合跑多 seed。
- [ ] 报告三 seed 均值、标准差及 tracklet paired CI。

---

## 9. 最终报告必须包含的指标

### Tracking 主指标

- Success / Precision，OPE recursive；
- overall weighted-by-frame 和 category-wise；
- final/last 为 primary，best checkpoint 为辅助；
- normal、random20、gap1124；
- 稀疏点数、gap、previous-error 分桶。

### B1 指标

- CV/learned endpoint RMSE；
- true/fixed/shuffled time；
- target-in-support recall；
- support 面积/点数膨胀；
- NLL、coverage、ECE、sigma saturation。

### B2 指标

- observation/raw paired error；
- raw win/help/harm rate；
- oracle headroom；
- structural/foreground-valid coverage；
- base/overlap/extension point count 和 GT reachability；
- first-failure 和 drift length。

### B3 指标

- applied rate；
- helpful precision；
- harm rejection；
- calibration；
- action saturation；
- 相对 keep-observation、always-raw、simple-threshold 的净增益。

### 工程指标

- FPS、step time、峰值显存、参数量；
- trainable parameter list；
- frozen hash；
- checkpoint/config/data/replay hash；
- fallback source 频率。

统计原则：所有 help/harm 和候选差异以 tracklet 为 bootstrap 单位，禁止把同一 tracklet 的 endpoint 当作独立 IID 样本。

---

## 10. B4 独立路线

B4 不进入当前 E0–E4 主链路。

原因：

- B4 是 B0 训练期表示正则，不是推理时可插拔模块；
- B4 改变 B0 后，会使 B1 checkpoint、B2 replay/candidate 分布和 B3 calibration 失效；
- 当前历史 B4 实现曾出现表示收缩和明显训练开销。

只有 E0–E4 主结果稳定后，才允许一次独立 kill-test：

- [ ] 新 B0 + no consistency；
- [ ] 新 B0 + invariance-only；
- [ ] 新 B0 + anti-collapse；
- [ ] tracking 不退化、表示不收缩、step time 可接受后，才重建 B1→B2→B3。

---

## 11. 当前明确不做

- [ ] 不直接运行旧 `--variant full` 代表新版 full。
- [ ] 不把 config18 当作端到端训练配置。
- [ ] 不同时打开 B1 support、dynamic sigma、geometry、dual-query 和 B3 后再做归因。
- [ ] 不让 B1/B2 loss 修改固定 B0 observation。
- [ ] 不恢复 B1-centered clipped candidate 作为正式候选。
- [ ] 不在 B1 calibration 失败时声称 uncertainty-aware。
- [ ] 不在 raw Search 未通过递归 gate 时训练复杂 B3。
- [ ] 不用 B3 掩盖候选 producer 的分布或递归问题。
- [ ] 不为了“完整模块化”强行把 B4 塞进最终 full。
- [ ] 不在 seed42 尚未通过时启动 seed43/44。
- [ ] 不做所有开关的全因子排列组合。

---

## 12. 服务器实际启动顺序

### 最近一个可关闭的里程碑

```text
MILESTONE-1
  clean commit + full-dataset configs
  -> matched B0 seed42
  -> frozen B1 TrueTime vs CV
  -> q_obs-only B2 with fixed CV support
```

立即执行：

- [ ] 完成第 2.2 节的五个必修代码/配置项：B1 freeze、q_obs-only、formal replay、full configs、B3 saturation diagnostic。
- [ ] 跑全部单元测试和三个阶段 smoke test。
- [ ] 从头训练唯一 matched B0 seed42。
- [ ] Gate A 通过后训练 frozen B1 seed42，并与 CV 比较。
- [ ] 同时准备 E1 q_obs-only B2 配置，但 B0 未固定前不启动正式训练。
- [ ] Gate B1/B2 结果写入本文件“实验登记”后，再决定是否运行 E2/E3。

MILESTONE-1 完成前，不运行 dynamic sigma、uncertainty geometry、learned B3、seed43/44 或 B4。

---

## 13. 实验登记模板

每个服务器任务启动前复制一份：

```text
Experiment ID:
Purpose / single changed variable:
Commit:
Git dirty status:
Config path + hash:
Dataset root / split / manifest hash:
Seed:
Parent checkpoint + hash:
Replay cache + hash:
Trainable parameter prefixes:
Frozen parameter hash:
Epoch / batch / workers / GPUs:
Primary checkpoint rule:
Primary metrics:
Pass gate:
Stop condition:
Output directory:
Result / decision:
```

所有实验必须能回答一个明确问题；无法填写“single changed variable”的任务不进入正式队列。

## 14. 2026-08-06 Joint Full 四组实验与 15 轮 B0 诊断新增问题

### 14.1 当前实验事实

#### 15 轮结果

| 实验 | Success | Precision |
|---|---:|---:|
| 当前提交纯 B0，epoch 5 | 29.3950 | 34.7330 |
| 当前提交纯 B0，epoch 10 | 41.7932 | 55.6608 |
| 当前提交纯 B0，epoch 15 | 46.7560 | 53.6510 |
| 历史 B0，epoch 15 | 49.7385 | 60.4158 |
| Full，epoch 15 | 26.3775 | 26.6641 |
| `-B1`，epoch 15 | 31.6444 | 32.7013 |
| `-B2`，epoch 15 | 29.9880 | 32.1083 |
| `-B3`，epoch 15 | 26.6050 | 26.3250 |

#### 60 轮结果

| 实验 | Success | Precision |
|---|---:|---:|
| Full | 29.4256 | 29.3249 |
| `-B1` | 29.1127 | 29.3479 |
| `-B2` | 30.5109 | 30.9267 |
| `-B3` | 26.0536 | 24.9814 |
| 历史 SeqTrack/B0 | 53.3600 | 64.3818 |

当前纯 B0 与四组 Joint 实验使用同一当前提交 `835f911`、相同 seed、数据选择、batch 规模和全局训练步数。工作区仅有未跟踪日志，`dirty_tracked=false`，因此不能把 Joint 的下降简单归因于代码版本不同。

当前纯 B0 在 epoch 15 比 Full 高 `20.38/26.99`，比 `-B2` 高 `16.77/21.54`。这足以排除“当前 B0 从训练开始就整体崩坏”，但当前 B0 仍比历史 B0 低 `2.98/6.76`，正式主表前仍需补齐当前提交的 60 轮 matched B0。

Joint 与纯 B0 的关键分叉出现在 epoch 5 到 epoch 10：epoch 5 时差距尚不极端，随后纯 B0 快速上升，而所有 Joint 变体下降或停滞。需要定位第一个发生分叉的 batch、tensor 或递归帧，不能继续仅依赖最终指标猜测原因。

### 14.2 已确认的问题

#### P0：Joint 公共路径破坏了 B0 的递归跟踪能力

- `-B2` 已严格输出 observation，但结果仍只有 `30.51/30.93`，远低于纯 B0。这说明主要问题发生在 B2/B3 最终校正之前，或发生在 Joint 公共训练/推理路径中。
- `-B1` 仍然严重下降，因此 B1 learned residual 不是唯一根因。
- 四组实验共同启用了 `use_ct_joint_full=True`，共同经过 Joint 数据构造、额外模块前向、随机数消耗、decoder state 返回以及递归历史更新路径。公共路径是当前最高优先级嫌疑对象。
- epoch 15 的 B0 observation center loss 非常接近：当前纯 B0 `0.04577`、历史 B0 `0.04534`、Full `0.04389`、`-B2` `0.04268`。训练期 teacher-forced 单步损失正常，而递归验证崩坏，说明训练 loss 不能代表实际闭环跟踪状态。

待排查的具体链路：

- Joint 与 baseline 是否构造了完全相同的 `points/ref_boxs/delta_T/valid_mask`；
- observation box 的坐标系、反归一化和 box transform 是否一致；
- Joint 是否改变了 B0 Transformer/dropout 的随机数流；
- CT loss 是否意外向 B0 参数传递梯度；
- decoder state、历史框和上一帧预测在递归推理中是否被不同地写回；
- 训练阶段使用的历史框与验证阶段递归历史之间是否存在严重分布差异。

#### P1：B1 内部有效，但尚未转化为最终涨点

- 运动学锚点 RMSE 约为 `0.313`，B1 prior RMSE 约为 `0.291`，内部改善约 `7.1%`。
- B1 残差饱和率约 `0.19%`，目前没有证据表明 dynamic bound 长期卡边界。
- B1 将 raw Search RMSE 从 `-B1` 的约 `0.583` 改善到 Full 的约 `0.546`。
- 但是 Full 相对 `-B1` 只有约 `+0.31 Success/-0.02 Precision`，不能宣称 B1 已经给最终跟踪带来稳定正收益。

结论：B1 的运动建模方向可以保留，但必须在修复 Joint 公共路径后通过真正的 `B1-only` 和 `B1+B2` 实验重新证明其贡献。

#### P1：B2 raw Search 当前为净负作用

- raw Search RMSE 约 `0.546`，observation RMSE 约 `0.248`，前者约为后者的 `2.21` 倍。
- expansion candidate valid 约 `83.72%`，低于计划的 `95%` 验收线。
- 实际 expansion 点数平均约 `33.96`，远低于配置目标 256，需确认这是候选不足、采样统计口径还是有效点过滤导致。
- Search 实际被使用约 `25.86%`，但 Full 仍比 `-B2` 低 `1.09 Success/1.60 Precision`。

结论：在修复公共路径前不能判断 B2 架构最终无效；修复后如果 matched `B2-only` 仍低于 B0，应优先简化或删除 B2，而不是继续增加补丁 head。

#### P1：B3 只能减少 B2 伤害，尚未产生净增益

- Full 比 `-B3` 高约 `3.37 Success/4.34 Precision`，说明 router 相比“始终应用完整 raw residual”具有安全保护作用。
- 但 Full 仍比 `-B2` 低约 `1.09/1.60`，说明 B3 没有把 B2 校正变成相对 observation 的净收益。
- sigmoid gate 即使较小也不等于严格 no-op，递归序列中小偏移可能累积成漂移。
- B3 依赖 B2 candidate，不存在有意义的 `B3-only`；应比较 `B2-only` 与 `B2+B3`。

#### P1：query reliability gate 尚未学到有效可靠性

- `alpha_q` 均值约 `0.424`，query-gate BCE 约 `0.691`，接近随机二分类的 `log(2)=0.693`。
- 当前软标签由 `sigmoid((kin_error-learned_error)/0.25)` 构成，而 B1 平均只改善约 `0.02m`，标签大量集中在 0.5 附近，门控容易退化为近常数。
- 需要检查 `alpha_q` 与 B1 相对增益、`delta_t`、历史有效性和最终 Search 改善之间的相关性；若无相关性，应重定义为带 dead-zone 的排序/置信监督，或先取消监督门仅保留保守规则门。

#### P0：`alpha_q=0` 时 `q_search=q_obs` 的契约没有严格成立

当前路径近似为：

```text
q0       = LN(stopgrad(q_obs))
q_search = LN(q0 + alpha_q * r)
```

第二个可学习 LayerNorm 使 `alpha_q=0` 时也不能数学保证 `q_search=q_obs`。实测 `-B1` 仍存在约 `0.110` 的 query shift norm。

虽然当前匹配 logits 使用 `obs_score + alpha_q * residual_score`，这未必是整体崩坏的唯一原因，但它违反消融和安全回退定义，必须修正并增加 exact equality 测试。

#### P1：candidate valid 的语义过弱

当前 candidate valid 更接近“至少存在有限点/可以计算”，却同时用于 router 和 correction loss。它没有表达“raw Search 比 observation 可靠”。必须拆分：

- `structural_valid`：输入和数值可计算；
- `evidence_valid` 或 `trustworthy`：点数、targetness、vote 一致性、候选置信度达到校正要求；
- 不可信时 B3 必须严格输出 observation，而不是保留非零软残差。

### 14.3 不能直接从当前消融得出的结论

- 当前 `-B1/-B2/-B3` 是已经崩坏的 Joint 公共路径内的减法消融，不等同于各模块的干净 standalone 实验。
- 历史 `motion_v3/search_v21` 是旧版独立实现，不是当前共享运动锚点和双参考系几何下的新 B1/B2，不能作为新模块的正式论文证据。
- 历史结果显示旧 B1-only 约 `53.318/62.573`，说明运动模块可以独立运行，但没有稳定超过历史 B0 `53.360/64.382`；旧 Search-only 约 `51.307/59.816`，可以运行但为负收益；旧 Motion+Search 约 `26.754/26.876`，进一步支持“耦合/传输链路可能导致崩坏”的判断。
- 15 轮当前 B0 可以用于诊断“是否从一开始全局崩坏”，不能替代正式 60 轮 baseline，也不能据此声称最终涨点。

### 14.4 下一步必须先完成的 P0 定位实验

#### A. 同一 B0 checkpoint 的跨路径递归推理

加载完全相同的 B0 权重，仅切换推理代码路径：

1. baseline path；
2. Joint `-B2` path，强制最终框严格等于 observation。

逐帧比较输入、observation 和写回历史：

- 第一帧就不同：优先检查输入构造、forward 分支、坐标变换或 decoder state；
- 前几帧相同、随后分叉：优先检查历史框写回和递归状态；
- 两条路径完全相同但都低：优先检查训练轨迹、随机数流或 checkpoint 本身；
- baseline 高、Joint `-B2` 低：可以直接确认数据传输/递归推理链路有问题。

验收标准：同一 checkpoint、相同输入和关闭全部 CT 校正时，两条路径的 observation 与最终框应在约定数值容差内逐帧一致。

#### B. baseline 与 Joint `-B2` 的首 batch 和前 100 step 等价性审计

使用同一初始化、同一 batch 顺序和同一数据，逐项记录并比较：

- `points/ref_boxs/delta_T/valid_mask`；
- B0 中间 feature、query 和 observation；
- 各项 B0 loss；
- 每个 B0 参数的梯度；
- optimizer step 后的 B0 参数；
- CPU/CUDA RNG state。

目标是找到第一个不一致的 tensor 和第一个不一致的 step，禁止只比较 epoch 平均 loss。

#### C. 梯度隔离审计

- 对所有 CT-only loss 分别反向，验证所有 B0 参数梯度严格为零或 `None`；
- 对 `L_B0` 反向，验证 B0 梯度与 baseline 在同一输入上数值一致；
- 验证 B1 feature、均值和不确定性仍能按设计接收 B2 梯度；
- 验证 B3 candidate detach 不会把 router loss 反向扭曲 B0/B2 坐标。

#### D. 随机数流隔离

辅助分支中的 dropout、采样和点选择不能改变 B0 的随机数序列。需要比较 baseline 与关闭 CT 模块后的 B0 dropout mask/输出；必要时为辅助模块使用独立 generator 或在 B0 完成后再执行随机操作。

#### E. 模块关闭语义修复

- B1 off：`delta_B1=0`、`alpha_q=0`，实际匹配严格等于 observation matching；
- B2 off：不构造会产生训练或 RNG 副作用的 Search 分支，最终严格等于 observation；
- B3 off 需要拆成两个明确实验：router no-op（输出 observation）与 always-raw（完整应用有界 residual），不能用同一个 `-B3` 名称混合两种含义；
- 历史失效、非有限数和证据不足时全部严格 no-op。

### 14.5 公共路径修复后的最小模块矩阵

只运行以下六组，避免无意义或重复实验：

| 编码 | B1 | B2 | B3 | 目的 |
|---|---:|---:|---:|---|
| `000` | 关 | 关 | 关 | matched B0 |
| `100` | 开 | 关 | 关 | B1-only，验证运动分支自身是否改善可用输出或明确内部指标 |
| `010` | 关 | 开 | 关 | B2-only，验证确定性 anchor/tube 的 raw Search |
| `011` | 关 | 开 | 开 | B2+B3，测量 router 对 Search 的净作用 |
| `110` | 开 | 开 | 关 | B1+B2，验证共享锚点与 query/geometry 耦合 |
| `111` | 开 | 开 | 开 | Full |

不存在 `001` 的 B3-only，因为没有 B2 candidate 时 B3 没有独立功能。

每组必须使用相同初始化、manifest、数据顺序、训练预算和 checkpoint 选择规则。先跑 1 batch、100 step、5 epoch 和 15 epoch 诊断，通过后再跑 60 epoch。

### 14.6 暂停条件与继续训练门槛

在以下条件满足前，暂停新的 60 轮 Full、其他类别、`gap1124/random20/burst_drop` 和大规模 seed 实验：

- baseline 与 Joint 全关闭路径通过逐帧/逐 step 等价性测试；
- CT-only loss 对 B0 的梯度隔离符合设计；
- B1 off 时 `alpha_q=0` 且匹配严格回退；
- B2 off 时 Joint 递归结果与 matched B0 一致；
- candidate availability 达到或合理接近 95%，并解释实际 expansion 点数不足的原因；
- raw Search 在有效样本上的平均误差低于 observation，或至少存在预先定义且可复现的可靠子集；
- B3 在该可靠子集上使 `B2+B3` 同时不低于 B0 的 Success 和 Precision；
- `alpha_q` 与 B1 的真实相对增益存在可解释相关性，而不是近常数输出。

如果公共路径修复后 B1-only、B2-only 或 B2+B3 仍不能产生正收益，应删除或进一步简化对应模块，不再通过增加新 head 或新损失掩盖问题。
