# B3 实际执行历史审计：v23—v29

日期：2026-09-15。当前 HEAD：`525eb543a321a36b5ae65545da967d53d3eb00c3`。只读 git 历史与代码；不改训练、配置、output，不参考 TrajTrack。来源节选保存在本目录，完整映射见 `source_manifest.json`。

## 结论

建议保留 **v29 的“有界动作即时效用 + 接受动作后的机制递归 + 真正闭环策略拟合”**作为主线，不整体回退 v24/v26 的双概率门和风险晋升，更不将兼容字段误解释为仍有 H3 残差头。若后续证据证明即时收益漏掉恢复机会，再把未来效用做成明确、独立的预测目标与消融。

这不是已证明 v29 涨分。旧版本与新版本的 B0、获取和数据协议同时变化，现有实验不能直接给出 B3 版本的因果排名。以下取舍来自目标一致性、递归状态匹配与可验证性。

## 可复核的版本表

| 阶段 / 提交 | B3 真正学什么 | 真正执行什么 | 实质评价 |
|---|---|---|---|
| v23，`25586b2`，2026-08-13 | B2 先学习 H1 utility；B3 `H3UtilityResidualRouter` 学额外标量 logit，`apply_logit=h1_utility_logit+h3_residual`；H3 BCE | 有界 XY 残差乘以二值 action，保留 B0 Z/yaw | 模块分工混杂：B2 同时做证据与即时效用，B3 只校正其 logit。这里 residual 是 logit 残差，不是学习几何校正，也不是 Gaussian residual NLL。 |
| v24，`96f0baf`，2026-08-17 | B3 统一学习 helpful、harmful、expected center gain、expected proxy-IoU gain。B2 去掉 utility/action head | `p_help*(1-p_harm)` 超阈值，且 presence 超阈值、结构合法、校准可部署；同样有界 XY | 分工更清楚，但 score 不是期望涨分；gain 幅度头不直接决定 gate。presence 与效用双硬门可能重复压低动作覆盖。 |
| v25，`62e1f90`，2026-08-24 | B3 上述目标仍在；主要恢复 B0 和正式训练协议 | 命名参数组的单 Adam，`unified_auto`；不是旧方法文档所写每模块独立 optimizer | 不能把 v25 高/低 B0 分数当作 B3 版本优劣证据。 |
| v26，`5225ff0`，2026-08-28；正式五臂登记 `b445ecd` | 增加共识一致性、协方差、inlier ratio、候选差距、兼容模式数等 B3 输入；保留上述 help/harm+gain 目标 | presence/action 阈值在 calibration 固定，再要求 dev 也通过风险晋升；不通过保持 observation | 增加了与实际动作相关的信息，但风险晋升约束强、数据少时易退化为无动作。不能把 bootstrap 上界称作分布外安全保证。 |
| v27，`a43a8ee`，2026-09-05 | 新 `B3UtilityUpdater`：tanh 的 Success/Precision gain 两头；q 为两者平均。H1 MSE；存在 H3 子集时再混合其 MSE；help/harm BCE 权重各0.1 | q 阈值，支持 never/always/threshold；显式加入 bounded dx/dy、raw/radius、clipping 和 mode-summary；不再以 help/harm 概率乘积为 gate | 相比 v26 更直接对齐评测效用，并使用实际执行动作信息。H1/H3 的总体与权重依赖调度，q 不再有统一的“即时”或“3帧 return”含义。 |
| v28，`69461f1`，2026-09-08 | B3 gain 目标仍继承 v27 | 清除 B2 导出的 `candidate_valid=availability & evidence_present` 残留门，v28 改成结构 availability；统一 host/导出/动作集合 | 这是必要的接口一致性修复，应保留。v27 B3 类本身已无 presence 门，不等于全链路当时已统一。 |
| v29，`608f963`，2026-09-10 | gain 两头只拟合同状态下 bounded 动作的即时 S/P；H3 退出 loss；help/harm 保留辅助监督 | 机制每轨迹稳定分配25%never、25%always、50%q>0，接受动作写入后续机制状态；共享训练/部署状态转移；B0独立观测流不读插件状态 | 当前最清楚、最易判断涨分的版本；接受动作后的分布得到部分覆盖。仍是混合行为，不是完整 on-policy；即时监督也不是长期恢复价值。 |
| v29 perf，`525eb54`，2026-09-11 | 目标与参数更新路径保持；H3合法调度事件只抽10%诊断 | 诊断采样，不是新动作策略或新价值定义 | 不应写成 B3 从 H3 训练改成10%训练；H3在v29已不进入loss，perf改变的是诊断覆盖。 |

## 旧风险版究竟在限制什么

v24/v26 的 H1 center gain 是米制中心误差下降，IoU gain 使用同尺寸、轴对齐 BEV proxy，并非正式完整 S/P 贡献。H3可用时每行 gain 标签取 H1/H3 的均值。helpful 要求即时 center 超 margin 且 IoU 不差，H3存在时未来也需如此；harmful 包括任一 center/IoU变差，**以及没有新增目标点**。因此“无目标证据”与“实际动作有害”被混入同一标签。

v24/v26 action score=`sigmoid(help)*(1-sigmoid(harm))`，两个 gain 头主要通过共享 trunk 辅助学习。它会弱化收益幅度的直接作用：帮助概率高不保证净收益高，帮助少但幅度大的动作也可能被舍弃。该乘积没有自动成为联合概率的保证；即使两头各自概率准确，也不能未经依赖关系验证就称联合校准概率。

旧校准还同时要求默认至少100动作、30轨迹、1%覆盖，bootstrap harmful-rate上界≤5%，center与IoU gain下界≥0。v26再对固定阈值的dev集合要求一次同类晋升。它适合“低经验风险覆盖”的研究目标，但不直接最大化完整轨迹S/P；动作较稀缺的数据上，可能只给出 observation fallback。v27改成实际闭环整体 `U=(S+P)/2` 排序，依次用S、P、较少动作打破并列，更贴合用户当前涨分目标。

## H1/H3变化的精确解释

`a43a8ee:utils/v27_training.py:62-75`：

```text
L1 = mean_valid (predicted - H1)^2
若本batch存在有限H3合法行：L = 0.5 L1 + 0.5 mean_H3_mask(predicted-H3)^2
否则：L = L1
```

两个标签训练同一头 **不天然是数学错误**。当总体和混合权重固定时，平方损失可视为拟合相应混合目标的条件期望。这里的问题是 H1 使用所有合法行、H3 使用调度后的小子集，batch是否有合法H3还改变H1权重；不统一的总体与时间权重让 q 的含义依赖采样。

更关键的是，H3是当前一次 bounded 动作与 observation 的双分支，在未来两帧都只使用 B0 回退续推，取未来两帧 S/P 差的均值。**标签不包含当前H1本身，也不是持续执行B3策略的长期回报**。v27 `.5*H1+.5*future_two_mean` 即使所有行都覆盖，也不是三帧等权平均。

v29只保留H1使目标易解释，但丢掉了“现在无收益、未来有恢复”的训练信号。这是真实能力取舍，不是免费改进。适合把未来价值作为独立头或明确固定视野目标来检验，不建议恢复随batch可用性混合的旧实现。

## 动作本身其实没有随这几代变大

v23到当前的核心动作都是 `raw_xy-observation_xy`，按 `radius=min(0.5+0.5*dt,2.0)` 限幅，仅改XY。典型dt=0.5时半径0.75米。v27引入实际bounded动作特征，v29保留该几何而改目标和递归访问。因此不能用“v29的即时头”解释远距离重定位能力不足：动作空间本来就是局部修正。

扩大动作半径会改变训练目标分布与闭环稳定性，应先区分raw正确/bounded太小与raw方向本来错误，再做受控动作幅度消融。仅换风险门、恢复旧H3或换“校准”名称都不能产生未被B2获取的目标证据。

## 状态耦合：v29比前版更适合验证完整组合

`a43a8ee:models/seqtrack3d.py:8334-8364`明确强制正式训练只提交 observation，B2/B3为影子学习；部署Full提交最终接受动作，存在访问状态差距。

`608f963:utils/v29_policy.py`和host机制事务改成共享转换函数，并在机制训练中提交混合行为接受后的框。此处是v29值得保留的主要变化。它不解决所有分布偏移：训练的25/25/50行为混合与最终单阈值策略仍不同，但比完全不访问接受动作后的状态更贴近实际部署。

## 名称和旧文档的陷阱

- `docs/CTSEQTRACK_B0_B3_METHOD.md`仍写`B3SelectiveUpdater`、presence双门、H1/H3、每模块独立optimizer、永远observation递归。它是历史方法记录，**不能用来解释当前29 perf实验**。
- 当前host在`ct_enable_v27`分支构造`B3UtilityUpdater`，不是仅凭`pipeline.py`导出的旧类推断执行链。
- 当前`ct_b3_h3_residual`、`ct_b3_h3_utility`只是q的兼容别名，没有因此产生独立H3 residual head。
- `b8222bb`的sigma/NLL/coverage修复属于B1不确定性；`3b8cc57`的二维NLL属于B0分割严格确定性。v23—v29 B3没有Gaussian残差NLL训练目标。二分类BCE可数学上视作Bernoulli NLL，但不能与几何残差概率建模混称。
- v29的`calibrated`缓冲区、`action_calibration`文件名仍保留兼容接口；artifact明确写`internal closed-loop policy selection; not probability calibration`。内部17/18场景与参数训练重叠，更不能称独立的概率/统计安全校准。

## 最小的版本取舍原则

1. 以同权重、同完整端点的 observation / bounded-always / fitted-policy 闭环S/P比较判断B3价值，gain loss/AP只作解释。
2. 分离“候选质量不足”与“排序/阈值选择不足”：候选本身几乎没有正增益时，不能优先堆B3价值头。
3. 先保留v29结构合法性、实际动作特征、accepted递归与闭环拟合；收益幅度q比旧双概率乘积更贴合当前目标。
4. 若真实事件显示H1中性/H3有益机会足够，单独检验固定视野未来增益头。保留明确总体、缺失标签mask与固定归约，避免恢复旧随事件混合的q定义。
5. 不把当前checkpoint重新解释成旧B3；不同头/目标的优劣需要重训或严格隔离的诊断头实验。不能把历代不同B0分数直接归因于B3变化。

## 复核方式

本目录每个节选都保留 `revision:path:line` 与行号；例如：

```bash
git show a43a8ee:utils/v27_training.py
git diff 69461f1^ 69461f1 -- models/ct_v2/evidence_memory.py
git diff 608f963^ 608f963 -- utils/v27_training.py models/ct_v2/action_v27.py
git show 608f963:utils/v29_policy.py
```

没有执行模型训练/测试，也没有凭CPU或源代码检查宣称得分提升。
