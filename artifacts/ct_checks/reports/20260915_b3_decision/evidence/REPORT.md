# B3 历史实验证据与取舍边界

审计日期：2026-09-15。只读本地已经同步的 output、既有报告及小型原始 CSV/provenance；没有改训练代码、配置、checkpoint 或运行状态。未访问其他跟踪项目。大 checkpoint 与完整训练日志不重复加载。

## 结论

**目前没有可核验的历史证据证明旧版 B3 已经通过真实闭环选择动作提高了 S/P。不能用 v25 Full 的 52.553/61.194 作为“旧 B3 优于 v29”的依据。** v25、v26、v27、v28 的被引用 Full 结果全部是 B3 未安装策略 artifact 时的 observation 回退；v25–v27 的跨臂 B0 又已经分叉。v29 尚只有前两轮训练诊断，没有第一次官方验证或拟合后 Full 分数。

当前证据能够帮助选择合理的设计目标、判断动作空间是否有机会；还不能按历史分数确定哪代 B3 最会涨分。

## 最小历史结果表

单位为百分数。每行只能在标出的评测与状态下解释；表内不同代的绝对分数不构成公平对照。

| 版本/结果 | B0 S/P | Full S/P | 实际 B3 部署动作 | 可否归因 B3 |
|---|---:|---:|---|---|
| v25 retryfix，60轮，官方 mini_val | 50.690/59.280 | 52.553/61.194 | 0；calibrated rate=0 | 不能。Full 是另一条 B0 训练轨迹，首次更新已分叉 |
| v26，60轮，mini_val | 26.903/25.601 | 48.802/54.888 | 0；无 calibration/dev artifact | 不能。B0 同样从 step1 分叉；minus-B3 在第10轮验证崩溃 |
| v27，60轮，内部 dev scene-0061，321帧 | 17.952/15.553 | 17.593/15.202 | 0；missing_calibration_artifact | 不能。只有单内部场景，且 B0 权重不同 |
| v28，60轮，官方 mini_val，2285帧 | 45.200/47.243 | 45.200/47.243 | 0；2179预测端点均 b3_calibrated=0 | 只能证明本次回退与同权重 B0 一致，不能判断拟合后策略 |
| v29 perf，full nuScenes，前两完整训练轮 | 尚无官方验证 | 尚无官方验证 | 训练混合行为包含动作；部署尚无结果 | 训练动作率不是部署策略结果 |

来源：[v25 原表](../../20260824_v25_four_arm/arm_summary.csv)、[v25 审计数据](../../20260824_v25_four_arm/analysis_summary.json)、[v26 审计数据](../../20260905_v26_mini_five_arm/analysis_summary.json)、[v27 报告](../../20260907_v27_mini_five_arm/REPORT.md)、[v28 Full核查](../../20260910_v28_mini_three_arm/full_review.md)、[v29 训练诊断](../../20260915_v29_partial/validation/REPORT.md)。

### 更早版本也不能凭名称认定成功

- `output/20260808-1707-21_ct_joint_full-eval_full_bestdev_minival/` 的原始 `proposal_endpoints.csv` 共1996条；`router_applied_gate` 全为0，虽然 `router_gate` 全为1。分类分数与执行标志不能互换；端点文件又未覆盖全部2285评价帧。
- `output/20260809-2214-22_ct_joint_repaired-ct22_full_calibrated_bestdev_minival_s42/` 只有 `run_provenance.json`，没有本地同步的 TensorBoard/CSV/评价结果。它指向 `epoch=29-step=25650-router-calibrated.ckpt`，目录名包含 calibrated，均不足以证明真实校准成功或动作有效；而且它选择的是 bestdev epoch，不是当前固定 final/late-3 协议。
- 对本地 output 全部文件名作 calibration/policy JSON/CSV/YAML/MD 检索，只发现早期 `observation_v1_local_recheck_calibrator.json`。其标签为 `current_obs_crop_miss`，是 observation 风险逻辑回归，不是当前 B3 的动作效用策略。此检索不能排除有 artifact 嵌在 checkpoint、使用其他名称或尚未同步。

## 真正能够支持 B3 设计取舍的局部证据

### v27：原始候选有局部潜力，固定小动作消掉了大部分即时 S/P 潜力

同一 Full 第60轮状态、321帧分母（无动作及首帧按零贡献）：

| 量 | 值 | 含义 |
|---|---:|---|
| raw candidate 平均净 ΔU | +5.148分 | 同状态单次候选反事实，不是闭环可实现涨分 |
| 实际 bounded candidate 平均净 ΔU | −0.0078分 | 当前动作相对 observation 的即时效用很弱 |
| 仅选即时正收益动作的 oracle ΔU | +0.1207分 | 固定状态的一步选择空间很小；不是完整策略长期上界 |
| 结构候选被截断 | 140/140 | 当前约0.75m半径在这些候选中全部生效 |
| bounded helpful/harmful | 8/6帧 | 其余绝大部分收益为零 |

原始主表还存在残留 presence 硬门；上述有界效用由候选 sidecar 恢复并交叉核对。对应报告已有可复算脚本和 JSON。此证据支持“若动作本身没有有效变化，改筛选器不能产生很多即时分数”；**不能因此直接把旧 raw 动作作为部署选择，也不能把半径增大当成必涨分**。

### v28：有界动作确实降低伤害，但新增目标证据几乎不存在

官方 mini_val 第60轮，567个结构候选：raw helpful/harmful=21/491，平均效用−0.36944；bounded=157/321，平均效用−0.07434；accepted=0。

本次全局可新增目标点22715个，但实际 support 去除整个 B0 raw crop 后仅取得4个，后续 pool→selected→共识全部保留这4点。因此恢复策略无法凭空弥补获取失败；同时这一代数据显示**直接采用较大 raw 动作可能明显更坏**，它与 v27 raw 潜力不同，不应只选对某个设计有利的一代证据。

### v29：H1 目标有足量即时有益标签，但当前无条件候选主要有害

第2轮以下为 epoch末汇总的抽样合法候选，两个后端均训练未成熟；并非官方val。

| 量 | CfC | GRU |
|---|---:|---:|
| 样本数 | 665 | 682 |
| selected含目标 | 103（15.49%） | 104（15.25%） |
| 即时U正/负/零 | 85/437/143 | 94/428/160 |
| bounded平均U标签 | −0.15284 | −0.14507 |
| help AUROC/AP | 0.7579/0.3719 | 0.7657/0.4006 |
| presence AUROC/AP | 0.9305/0.7097 | 0.9295/0.7052 |
| H3 executed且valid | 205 | 207 |

U标签使用0–1 S/P差；不能写成官方成绩下降15分。训练behavior accepted均值约12.06%/12.24%，混合never/always/threshold0，用于访问不同递归状态，不能作为 learned deployment policy 的接受率或效果。

这与 v27 的大量有界零收益并不完全相同：v29第二轮有正收益的合法候选占12.8%–13.8%，H1 不是完全无训练信号。但负例占62.8%–65.7%，保留所有合法候选负例并学实际收益，比恢复 presence 硬门或无条件更新更有直接依据。presence AP 改善也不意味着 action q 的收益排序已经可靠。

H3有效事件的回退两步S/P均值仍为负，且没有逐行H1/H3关联，不能据此判定重新把H3加回同一q头会更好。H3回答一次干预之后继续B0回退的结果，不等于完整选择策略持续运作的长期价值。

## 不同设计应如何用实验取舍

现有历史不能提供赢家；应在**同一已训练 Full checkpoint、同一数据覆盖、同一候选几何**上先做三种真实闭环：observation/never、bounded-always、拟合后selective。用同权重观察 B3 是否利用了有益动作并减少坏更新，再对新训练目标做独立scratch实验。

建议最小判断条件：

1. **动作可用性**：所有结构候选的 H1 正/负/零比例、正收益总量；按漂移、真实新增目标、共识目标支持、截断比例分桶。正收益空间几乎没有时，优先改获取/候选，不能用更深 B3 补救。
2. **收益排序**：除help/harm AUROC，还看按q从高到低的净效用—动作覆盖曲线、同状态 oracle 与 learned gap。概率校准指标不能代替收益排序或闭环跟踪效果。
3. **实际闭环**：同checkpoint selective 相对 observation 和 bounded-always 的全覆盖 S/P；首次失跟、恢复时延、动作后连续有害段。H1/H3反事实都不替代这个结果。
4. **时间目标一致性**：若比较 H1-only、H1+几何辅助、独立H1/H3两头，主预测量必须定义清楚且输入同一实际动作。不要把两个不同目标当成一个 q 的同义监督。
5. **公平选择**：只在注册的内部数据拟合策略，记录与参数训练重叠；官方val不挑阈值或best epoch。固定final60与58/59/60独立评测，不能用每代最好seed/epoch做选择。

## 目前缺少的证据

- v25–v28 对各自 Full 的有效策略 artifact、相同checkpoint的 observation/always/selective 闭环结果及动作覆盖；旧高分不补足此缺口。
- v29 首次完整官方验证，以及 q/H1/H3/原始候选/有界/accepted 的逐行关联。
- 当前数据上以统一合同对照 H1-only、H3相关目标、分类式门控或多幅度动作的新scratch实验。
- 独立于采样动作策略的 oracle-gap 和误筛/漏筛分解；只看分割/presence AP不能得到此结论。
- 跨seed、跨类别稳定性。当前full数据前两轮Car结果不支持论文级最终优劣。

以上是本地证据可支持的范围。实现更清楚、更符合效用目标可以作为选择 v29 基础的理由；“已经证明比旧 B3 更涨分”仍不成立。
