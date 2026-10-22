# v38：强 B1 与独立 B3

2026-10-22，按用户批准方案实施。新入口为 `main.py` → `models/ct_v38/entry.py`，宿主为 `models/ctseqtrackv38.py`。当前尚无 v38 正式成绩；本地合成检查不替代真实 nuScenes CUDA 检查。操作见 [三组启动页](CTSEQTRACK_V38_LAUNCH.md)。

## 主路与候选

完整复用已复现的 concat B1：B0、V35 decoder、PhysicalTimePrior、W=1/4/4/8 课程、采样、BN 和主路监督。原 q0 quality 和 acquisition head/loss 均保留。B2 扩展取点、身份检索、vote、mode、相关损失不执行；identity memory 只保留全空兼容张量，不初始化或更新点记忆。accepted 历史、真实时间、可信位置/速度以及 B0 历史观测保留。

B3 在单次 B1 forward 后读取 detached 候选、点特征、分割概率与状态，初始化放在独立 RNG scope 中。主路参数继续正常训练，不冻结。Full 的真实闭环允许因选择策略而与 B1 分叉；只要求相同输入/状态下主路数值与梯度合同成立。

| 槽位 | 几何来源 | 观测与可信资格 |
|---|---|---|
| 0 q0 | 原强 B1 最终输出 | 四帧非全空时是真实 q0，可申请可信写入 |
| 1 prior | 原始 `prior.box` | 仅传播；`prior.valid=False` 不否定有限 fallback 位置 |
| 2 trusted propagation | 原 `fallback_box` | 原可信 XYZ 按可信速度与真实时间外推，yaw 不变；速度无效时为最后可信位置 |

先判来源，再按 q0、prior、trusted propagation 顺序去重。XYZ 最大差 ≤1e-4m 且环绕 yaw 差 ≤1e-5rad 才合并，保留 alias；不转移写入权限。全空 q0 是 prior 别名，不参与观测竞争。框和分数非有限时报错。

B3 保留 69→64 局部点编码器、8 角各最多16个唯一点、512→64 box encoder、64维 fusion 和3个输出。点池仅使用 B0 当前帧已采样的唯一有效 raw IDs，不新增裁剪、点数或 backbone。候选附近无采样点仅表示当前输入缺乏证据。

统计从35维扩到45维：来源换为三种新候选，coarse参照换为最终q0；增加候选相对 trusted propagation 的 XYZ/yaw 共5维，以及上一输出是否传播、连续传播次数、连续未可信写入次数、速度有效性、速度年龄共5维。计数/年龄用固定 log1p；无效速度年龄为0并用valid区分。它们是风险上下文，不是已校准的位置协方差。

## 监督与决策

三个通道 S/P/T：S、P 学习现有 `benchmark_compat` 单帧 Success/Precision 贡献；T 学习真实 q0 的明确可信事件。旧 q0 quality 不改目标，也不解释为位置正确率。

实际支持集合 A 是 **当前 B0 唯一有效 raw IDs ∩ 最终 q0 框内 ∩ detached predicted-FG≥0.5**。T 标签为 exact oriented IoU≥0.5、支持点≥3、支持的 GT purity≥0.5 三条件同时成立。GT只生成监督或提交后的审计，不参与 A 的选取、选框或提交。包括未选中 q0 以及0/1/2支持点负例；传播 T 不监督。

总损失为原强 B1损失 + 0.125 L_S + 0.125 L_P + 0.1 L_ΔU + 0.5 L_T。S/P为soft BCE，T为端点等权普通BCE；无正负重加权。预测效用 U=(sigmoid(S)+sigmoid(P))/2，所有去重后有效候选的无序对用 `SmoothL1(beta=1)` 监督预测效用差与真实差。候选/候选对先端点内平均，再对有效端点平均；无有效对返回连图零。新增损失只训练B3。

**第1轮即执行策略**，不预热、不shadow、不加epoch。三个输出头权重/偏置全0，初始sigmoid=0.5。margin=0，仅严格更高的效用替换q0；同分顺序q0→prior→trusted propagation。无真实q0时两传播候选竞争，同分prior优先。不设低分保持、静止门、最大传播次数停止或低支持禁止输出。

## 单次提交事务

1. B3先选唯一输出，Adam后写入accepted；下一帧crop及历史严格跟随最终输出。
2. 只有选中真实q0、同一A支持≥3且T≥0.5，才写trusted位置和支持时间。阈值是固定规则，不是校准保证。
3. 位置写入后，仅当上一trusted来自相邻帧且原innovation transition合法，才按真实Δt更新速度；否则照常写位置，但速度清零、标无效。传播和未通过T的q0不刷新trusted/速度时间。

原B1运动边/CfC/fallback数学保持；传播选中不授予观测资格。`selected_quality/previous_quality` 在Full选真实q0时是旧q0quality，传播时为0；`output_utility` 与 `trusted_score` 分开。真实q0点量充足但T拒绝时可以刷新strong点量时间，不能刷新trusted支持时间。传播两者均不刷新。

## 研究与验收边界

这次只借鉴[校准研究](https://proceedings.mlr.press/v70/guo17a.html)对分数/正确率的区分、[Recursive KalmanNet](https://github.com/ixblue/RecursiveKalmanNet)对运动预测/累计状态误差的区分，以及[M2-Track](https://github.com/Ghostish/Open3DSOT/blob/main/models/m2track.py)对运动评价的重视。不引入新滤波器、协方差网络或运动分类门；未参考 TrajTrack。

固定比较Full−standard B1与hold−standard B1。报告final60及late-3(58/59/60)、原R/C门、三候选使用/错误接管/真实恢复、>5m/>10m/持续失跟；区分657个曾移动轨迹预测帧与171个运动端点，另报低运动和启停边界。T报告覆盖、错写、正负量、Brier和固定十分箱可靠性图，不据验证重选阈值。记录实际参数量、训练/验证/末三轮评测成本；不预先声称加速。

只减少写入、只改善低运动或单步oracle上升都不足以宣布成功。结果不足保留负结果，不自动加阈值搜索、种子或full训练。旧配置、checkpoint、output与artifacts保留；旧运行使用对应源码恢复。本次修改前源码为 `artifacts/ct_checks/v38_implementation_20261022/baseline_source_cc8458b.zip`。
