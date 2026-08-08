# CT-SeqTrack 在线递归训练与有效 Search 修复计划

更新时间：2026-08-08

## 0. 实验证据与总体判断

本计划针对以下三组 nuScenes v1.0-mini Car、seed42、60轮实验：

- Full：epoch 60 为 `48.30 Success / 55.98 Precision`；
- `-B3`：`48.72 / 62.66`；
- B2-only：`51.12 / 62.72`。

这些分数来自 mini_train 的稳定 15% tracklet-dev 分区，共 444 个非首帧，
不是完整 mini_val。三组训练中 `search_effective_rate`、
`evidence_valid_rate` 和 `search_applied_rate` 全程均为 0，因此当前实验不能证明
Search 或 B3 带来 tracking 增益。

已经保留的正确部分：

- candidate、crop、history、Search 使用同一个 `RecursiveTrackState`；
- candidate 0 是唯一允许写回的 view；candidate 1–3 不修改 canonical state；
- recovery view 的 crop/history/Search 使用同一连续变换；
- 第一帧目标尺寸和具名确定性 RNG 契约继续保留；
- B1 `OrderedPhysicalMotionEncoder` 与 mean prior 继续保留；
- 所有模块从 epoch 1 共同训练，不按 epoch 冻结或启停模块。

当前主要问题不是状态源再次分裂，而是模型从随机初始化开始沿完整 tracklet 进行
无干预的纯 on-policy rollout。状态虽然内部一致，却可能一致地偏离目标。

## 1. 修复在线递归状态分布

### 问题位置

- `utils/recursive_state.py:142-203`：四个 slot 按完整 tracklet 逐帧推进；
- `models/seqtrack3d.py:5694-5714`：candidate 0 的最终预测每帧无条件 detach 写回；
- `datasets/sampler.py:727-738`：下一帧历史和当前 B0 crop 以递归预测框为锚；
- `datasets/sampler.py:1351-1359`：B1 当前监督仍相对递归历史锚计算。

模型在训练初期尚未学会跟踪时，一次错误预测就会改变下一帧 crop。crop 离开目标后，
B0 看不到足够前景，B1 的历史锚和 GT 相隔数十米，Search endpoint 也随错误历史退化。
之后的监督虽然数值存在，但输入中可能已经没有可恢复目标证据。

### 修改

保持四个 tracklet slot 和四个 candidate view，但固定、跨 epoch 不变地划分状态模式：

- slot 0–1：纯 learner on-policy，始终写回部署动作；
- slot 2–3：safeguarded learner，最多连续递归 `H=3` 帧；达到窗口边界或检测到
  `target_in_b0_union_search=false` 后，下一帧状态从当前 GT box 加确定性 SE(2)
  噪声重新锚定；
- GT 干预只能在当前帧输入构造和 loss 完成后决定下一状态，不能进入当前帧输入；
- 同一个 slot/view 内 crop、全部history、Search几何和label必须由干预后的同一状态生成；
- candidate 1–3仍不写回canonical state；
- 所有B0/B1/B2/B3参数始终在同一optimizer中，从epoch 1共同更新。

新增配置建议：

```yaml
ct_recursive_state_policy: mixed_fixed
ct_recursive_pure_learner_slots: 2
ct_recursive_safeguarded_slots: 2
ct_recursive_max_learner_horizon: 3
ct_recursive_reset_noise_mode: deterministic_se2
ct_recursive_reset_on_unreachable: true
```

### 不修改的后果

- 冷启动误差继续沿完整 tracklet 自我放大；
- B0/B1的loss主要学习“目标在crop外”的不可恢复样本；
- B1 RMSE继续处于几十米量级；
- Search proposal位移和新增点数继续趋近0；
- 训练结果高度依赖seed和tracklet排列，Full消融无法归因。

### 验收

- 误差不再随rollout age单调爆炸；
- safeguarded slot的B1 kinematic/prior RMSE恢复至亚米级；
- pure learner slot仍保留真实部署分布；
- 干预前后的当前帧输入均无GT泄漏；
- 相同seed重复运行得到相同状态选择和重置位置。

## 2. 增加状态可见性与无掩码诊断

### 问题位置

当前主要记录 `ct_recursive_state_age` 和
`ct_candidate_state_consistency`。它们只能证明状态合同一致，不能证明目标仍位于可观察
区域。`models/seqtrack3d.py:4492-4498` 又使用 `search_effective` 掩码记录
raw-vs-observation gain；当 `search_effective=0` 时日志恒为0，无法区分“候选相等”与
“没有可评估样本”。

### 修改

新增按 candidate 0/recovery、pure learner/safeguarded 和 rollout age 分层的日志：

- `ct_target_in_b0_crop_rate`；
- `ct_target_in_b0_union_search_rate`；
- `ct_target_foreground_points_b0`；
- `ct_target_foreground_points_extension`；
- `ct_local_target_displacement_p50/p90/p99`；
- `ct_error_by_rollout_age/*`；
- `ct_recursive_intervention_rate` 和 `ct_recursive_intervention_reason/*`；
- structural-valid、support-valid、presence-valid、evidence-valid条件下各自独立的
  `raw_vs_obs_gain` 与 `raw_better_rate`；
- presence、extension-mass和vote-consistency每个硬谓词的单独通过率。

### 不修改的后果

- 只能看到最终Search为0，却无法确定是crop、history、proposal、points还是classifier
  首先失败；
- 继续调阈值可能掩盖上游无目标证据的问题；
- 无法为论文提供reachability、有效子集收益和失败覆盖率证据。

### 验收

- 任一Search失活样本都能映射到明确且互斥/可组合的失败原因；
- 即使 `search_effective=0`，structural/support子集的反事实误差仍可读取；
- 所有分层计数的分母同时记录，禁止只记录条件均值。

## 3. 修复 B2 presence 学习与阈值

### 问题位置

- `models/ct_v2/joint_full.py:167-173`：presence从概率0.1初始化；
- `models/ct_v2/joint_full.py:529-542`：使用全体Search点的汇聚特征并以0.5硬阈值
  决定 `search_effective`；
- `models/seqtrack3d.py:4308-4320`：标签是extension中是否存在至少一个前景点，
  但使用普通未加权BCE，并且只在稀少的support-valid样本上更新。

本轮三组presence最大概率仅0.091–0.108，所以0.5阈值直接阻断所有Search。

### 修改

- 分别汇聚extension-only targetness、vote、点数、voxel覆盖和质量统计，避免baseline
  背景淹没新增证据；
- 记录positive rate、AUROC、AUPRC、正负概率均值和可靠性分桶；
- 根据训练集正负比例使用有限上界的 `pos_weight` 或 focal BCE；
- threshold只在router-dev上选择，并写入checkpoint；不能直接把0.5降到0.1；
- structural/support invalid仍严格no-op，阈值校准不能绕过硬几何条件。

### 不修改的后果

- `search_effective`继续恒为0，B3和H3得不到有效候选；
- 若只粗暴降低阈值，接近初始化值的随机presence会触发有害Search；
- Full与`-B3`继续不是有效的router消融。

### 验收

- support-valid样本中存在足够的presence正负样本；
- AUPRC显著高于正样本先验，正负概率有稳定间隔；
- 校准后effective rate非零，且raw Search有效子集的中位数收益为正。

## 4. 修复 `alpha_q` 的反事实监督

### 问题位置

- `models/ct_v2/joint_full.py:444-467`：`query_gate`同时改变 `q_search` 和
  targetness logits；
- `models/seqtrack3d.py:4345-4359`：随后用这个已经被gate影响的raw Search与
  observation比较，反过来监督同一个gate。

标签依赖被监督gate自己的当前输出，是内生、自指的目标，并且还混入了B2点证据质量。
本轮AUROC约0.5，正负alpha均值几乎相同。

### 修改

对每个support-valid candidate生成两个不依赖预测alpha的detached反事实动作：

```text
raw_obs    = B2(points, q_obs, alpha=0)
raw_motion = B2(points, q_obs + delta_q, alpha=1)

Gq = error(raw_obs, GT) - error(raw_motion, GT)
yq = 1 if Gq > 0.05 m else 0
```

- 两个标签候选都必须使用相同点、相同B2权重和相同随机状态；
- 标签和两条候选detach，只允许gate loss更新query gate；
- 主部署路径仍使用预测alpha和严格invalid no-op；
- 分别记录helpful/harmful/dead-zone比例、AUROC和分桶可靠性。

### 不修改的后果

- alpha继续收敛到接近常数；
- `q_search`几乎等于`q_obs`，B1 query信息实际上没有被使用；
- 即使最终分数变化，也无法把增益归因于dual-query或运动条件。

### 验收

- gate AUROC至少0.65；
- positive/negative alpha均值差至少0.1；
- `alpha=0`、B1-off和Search invalid时继续严格满足 `q_search=q_obs`。

## 5. 让 B1 learned prior 真正控制 Search 几何

### 问题位置

`datasets/sampler.py:936-979` 在 Joint Full 中强制
`use_prepass_support=False`，在线训练又没有 `replay_b1`，所以
`resolve_b1_search_support`始终走kinematic fallback。B1目前只进入detached query
特征，不决定endpoint/tube。

### 修改

把在线输入构造拆成两段：

1. 由 `RecursiveTrackState` 构造不含当前GT的history/time/B0状态合同；
2. 在GPU上执行一次box-history-only B1 prepass；B1通过自身loss保留梯度，但传给
   非可微采样器的mean/sigma必须detach；
3. 用detached B1 mean/sigma和具名确定性RNG构造endpoint/tube；
4. B1-off、history invalid或数值invalid时才使用kinematic fallback；
5. 训练和推理必须调用同一个两段式构造接口。

禁止从其他实验checkpoint续训，禁止按epoch启用B1；所有模块仍从epoch 1共同更新。

### 不修改的后果

- Full和B2-only的Search几何本质上都由CV产生；
- B1即使学好，也无法改善目标reachability和新增点证据；
- 论文无法声称learned motion prior指导了Search support。

### 验收

- B1-on时日志明确记录 `support_source=b1`，B1-off严格记录fallback；
- train/inference相同state和seed下endpoint/tube逐元素一致；
- learned support相对CV报告target-in-support recall、体积和新增点数，而非只报告最终分数。

## 6. 修复 dev / mini-val 评估协议和指标命名

### 为什么本轮不是 mini-val

配置虽然包含：

```yaml
val_split: mini_val
test_split: mini_val
```

但 `main.py:633-637` 在 `ct_online_recursive_training=true` 时覆盖了 `val_data`：

```python
val_data = PartitionedTestTrackingSampler(
    train_data.dataset, config=cfg, partition='dev')
```

`train_data.dataset`来自 `mini_train`。`utils/recursive_state.py:30-39`再按完整tracklet
稳定hash切成70% train、15% dev、15% calibration。因此训练期间每5轮看到的是
mini_train-dev的444个非首帧，而不是mini_val的2,285帧。

`mini`只表示 nuScenes `v1.0-mini` 数据版本；该版本内部仍分别包含
`mini_train`和`mini_val`。只有 `main.py:725-738` 的 `--test` 分支才按
`cfg.test_split`加载真正的mini_val。

### 修改

保留dev用于checkpoint选择和router诊断，同时增加明确的最终评估阶段：

- 训练期指标改名为 `success/dev`、`precision/dev`，不再写成 `*/test`；
- checkpoint monitor改为 `precision/dev`；
- provenance记录wrapper后的有效partition、tracklet数、非首帧数和selection hash，
  不能只记录底层mini_train的5,051帧；
- 训练完成后，从dev选定唯一checkpoint，再自动或通过固定命令仅运行一次mini_val；
- mini_val指标命名为 `success/mini_val`、`precision/mini_val`，写入独立目录和provenance；
- mini_val不得用于router threshold、presence threshold、checkpoint或超参数选择；
- 增加同提交、同70/15/15训练协议的matched B0和motion-only，再统一到mini_val比较；
- 若为了工程观察启用中途mini_val，必须标记为diagnostic且不能继续据此调参。

新增配置建议：

```yaml
ct_checkpoint_selection_split: train_dev
ct_checkpoint_metric_namespace: dev
ct_final_evaluation_split: mini_val
ct_run_final_evaluation_after_fit: true
ct_forbid_final_split_calibration: true
```

### 不修改的后果

- mini_train-dev分数会继续被误认为mini-val分数；
- 新Full与历史B0/motion-only不在同一评估集和训练数据预算上，无法判断涨点；
- `precision/test`命名会掩盖实际的数据来源；
- provenance只记录底层dataset而非wrapper有效样本，复现实验时仍会产生歧义。

### 验收

- 日志名称、provenance和实际dataset逐项一致；
- dev只负责选checkpoint和诊断，calibration只负责阈值，mini_val只负责一次终评；
- 修改GT或dev阈值不会改变mini_val输入和checkpoint权重；
- Full、B0、motion-only最终比较使用相同训练partition、相同seed协议和同一mini_val。

## 7. 执行顺序

1. 先完成第2项诊断和第6项评估协议，确保下一次smoke能定位失败且分数口径明确；
2. 实现第1项固定混合递归状态，跑1–3轮smoke；
3. 状态可恢复后修改第3项presence；
4. presence产生非零有效候选后修改第4项alpha反事实标签；
5. 最后接入第5项B1 prepass Search几何；
6. 只有Search applied达到5%–25%、有效子集收益为正后，才重新验证B3/H3并跑60轮。

## 8. 简单提示词

```text
请检查并修复CT-SeqTrack当前Joint Full在线递归训练：模型从随机初始化开始对完整
tracklet纯on-policy写回，导致crop/history/Search一起漂移；当前又缺少target-in-crop
等可见性日志。presence使用稀少正样本的未加权BCE且0.5硬阈值使Search全程失活；
alpha_q标签依赖已被自身gate影响的raw candidate，形成自指监督；Joint Full还强制
关闭B1 prepass，使learned prior没有真正控制endpoint/tube。请保持所有模块从epoch 1
共同更新，采用固定跨epoch的learner/safeguarded状态混合，补齐无掩码诊断，修复
presence和alpha反事实监督，并让B1 detached prepass统一指导训练/推理Search几何。
同时修复评估协议：训练期mini_train-dev指标命名为dev，仅用于选checkpoint；训练结束后
对选定checkpoint一次性评估真正mini_val，并在provenance中记录wrapper后的实际样本与
hash。不得使用mini_val调阈值或反向传播，所有invalid条件继续严格回退observation。
```
