# B1 专项设计判断：GRU / CfC 与获取目标适配

审阅日期：2026-09-05。范围为当前本地 `motion.py`、`cfc.py`、`seqtrack3d.py`、`candidate_utils.py`、`ct_search.py`、sampler 与 v24→v26 配置继承；参考 CfC、β-NLL 及 Faithful Heteroscedastic Regression 原文/官方实现。没有参考 trajtrack，没有修改源码、正式配置或实验输出。以下把源码事实、设计判断及待实验候选分开。

## 结论

当前把 **显式使用物理时间的 GRU 保留为工程主模型，CfC 保留为后端对照** 更合理；这不是认定 GRU 理论上更强，也不是用现有 B1-only tracking 分数选赢家。最值得先修改的是 B1 到实际证据获取的接口、获取监督目标及可识别性；把 GRU 换 CfC 无法修复这些共同限制。

论文贡献应落在“因果物理历史如何改善有限预算下的新增目标测量”，后端胜负服从这个问题。若修复后的匹配实验中 CfC 稳定提高不规则时间下的新增目标点覆盖和闭环 Success，再晋升 CfC；若 GRU 相当或更好，采用 GRU 不妨碍论文成立。

## 1. 当前编码器到底做了什么

1. `24_formal_base.yaml:182` 为 `hist_num: 3`。历史 newest→oldest，构造两次相邻框 transition，再反转为 oldest→newest 编码。主流连续 keyframe 通常是 0.5 秒；aux 另有 query gap 2/4 和 transition gap 1/2 的真实帧下采样，不能说训练从未见过不规则间隔。
2. `motion.py:981` 附近，每步输入是 xy 速度、xy 位移、yaw 差的 sin/cos、`log1p(pair_gap/time_scale)`、query/pair ratio、valid，共 9 维。因此 **GRU 已经显式获得时间信息**。
3. GRU 与 CfC 共享上述投影、物理 anchor、residual head、sigma head、margin head。`motion.py:751` 的 CfC 每次 forward 从零 hidden 起步，最多执行两次历史 transition；没有跨 endpoint 持久化神经 hidden。
4. CfC 的 elapsed 是历史 transition 的 `pair_gap / time_scale`；`time_scale=0.5`，只是在时间单位上归一化。query gap 不作为最后一次 CfC rollout 步，而在每个 transition 的 query/pair ratio 和最终 context 中出现。
5. 两者 recurrent 参数已经近乎匹配：GRU(64,128) 为 74,496；当前 CfC(64,128,105) 为 74,537，差 41。不能把本项目 CfC 简单描述为“多很多参数”。融合 kernel、Python loop、pack/cpu lengths 等使延迟仍需实测，不能靠参数量推断。
6. 当前 `cfc.py:69` 后的 full-gated 形式，与作者官方 `torch_cfc.py:127–136` 对应：两个 tanh 候选，由显式时间 sigmoid 插值；不是把一个普通 MLP 随意叫成 CfC。官方 full-gated CfC 也不自动提供任意组合的严格 semigroup、物理守恒或零时长恒等映射保证。

上面说明当前 CfC 放置是合理的短历史时间编码器，但不是完整的连续时间状态估计器。仅有两次 transition、每次重新初始化，原文针对长序列/不规则采样的潜力不一定转化成此任务收益；这是设计推断，须以数据验证。CfC 原文的速度优势主要针对需要数值求解的连续时间网络，不能迁移成“本项目必然比 GRU 快”。

来源：[CfC 原文](https://www.nature.com/articles/s42256-022-00556-7)、[作者官方 PyTorch 实现](https://raw.githubusercontent.com/raminmh/CfC/main/torch_cfc.py)。

## 2. 共享约束比后端名字更影响效果

**均值只学习物理位移，这是合理的职责限制。** `candidate_utils.py:82` 的 target 为 `R_anchor^T(c_GT,current−c_GT,previous)`；最近 GT 框仅作监督位移原点，部署输入为递归预测框。它避免让纯相对框历史强行解释不可观测的 candidate 平移误差。

但 **获取 margin 目前用了同一个物理位移 target**。`seqtrack3d.py:5770` 把 main_target 直接送入 `acquisition_margin_pinball_loss`，拟合的是 `abs(d_GT−mu)` 的两个轴向分位数。预测运动终点 `c_hat,previous + R_anchor mu` 相对真实当前目标的误差为：

`e_acq = (d_GT − mu) + R_anchor^T(c_GT,previous − c_hat,previous)`。

这里的 `e_acq` 是**终点误差，不是 tube 中心误差，也不是应直接添加的最小 margin**。`ct_search.py:489–517` 的实际 tube 中心为 `c_hat,previous + 0.5 R_anchor mu`，未经尺寸上限截断时，平行半长为 `0.5|mu| + 0.5 object_length + margin_parallel`，垂直半宽为 `0.5 object_width + margin_perp`。tube 已覆盖从最近预测 anchor 到预测终点的走廊以及物体基础尺寸；其他 endpoint/corridor 支持还会参与实际获取。因此应根据真正支持域的中心、方向、基础半径、尺寸截断与并集计算目标框投影或可见目标点的额外覆盖需求，不能把完整终点绝对误差不加处理地当作 margin 标签。

现有监督未显式包含终点误差公式的第二项，也未直接优化真实支持域的新增目标点覆盖。目标点占据范围、遮挡后可见表面分布及固定点预算不在这个中心残差分位数目标中。分轴 90% marginal quantile 也不等于二维 90% joint coverage。此为**监督目标与下游用途不完全一致**，不应描述成 GT 泄漏 bug。

关键限制：若所有历史预测框共同平移同一个未知常量，相对历史不变，B1 无法辨识这个常量。单独改 loss 或延长网络不能凭空定位它。获取 head 可以学条件误差分布并适度扩大支持，但必须承认这是统计覆盖，而不是已知漂移方向。

**可学习均值的修正范围很小。** shared anchor 在直线匀速、query gap=0.5 秒时，两个 envelope 为 `(0.375m, 0.275m)`，tanh residual 不能越过它们。6m 级错误不可能靠换 recurrent cell 解决。固定边界能抑制历史抖动，也是恢复上限；应先按 recoverable/tail 分桶看收益，再决定放宽何种范围。不要统一无限制增大 residual。

另外 shared-anchor residual 是位移 envelope，包含与时间无关的常量项，因此源码没有硬性保证 `mu(Δt)→0` 当 `Δt→0`。当前数据不含无限接近零的间隔，这不是已证实的涨分瓶颈；若论文强调物理连续时间，应明确只是时间条件先验，或在新候选中采用有界速度/加速度残差并乘 Δt/Δt²，使单位和零时长边界成立。

## 3. 损失与 detach 应保留什么、补什么

`motion.py:729`：sigma head 读取 `context.detach()`；`physical_motion_uncertainty_loss` 的 sigma 目标误差也 detach。故 β-NLL 当前只优化 sigma 预测，不更新 temporal backbone 或 mean；均值通过 normalized SmoothL1/tail loss 学习。不能宣称此处 β-NLL 在解决“方差给均值梯度降权”的原论文问题，因为现有 detach 已经去掉那条路径。

这不等于 β-NLL 有错。它仍改变 sigma 分支各样本的梯度权重，可能帮助或损害校准。保持均值与统计尺度隔离，比较普通 NLL 与 β-NLL 的 held-out NLL、分层 coverage 和范围大小即可，没必要重新把 NLL 梯度送回均值。Faithful Heteroscedastic Regression 提供相关的均值/方差解耦思路，但本项目的有界 robust mean、截断 sigma error 等不同，不能直接继承其理论保证。

margin head 同样只读 detached 的、主要为 mean 学习的 context。于是它是一个浅层条件分位数读出，并未通过 acquisition objective 塑造自己的历史特征。建议采用独立小 MLP 输入 detached history/kinematic summary 与过去的 B0 quality、状态 age、空点或稀疏计数；其参数可从 epoch0 正常学习，禁止回写/冻结 B0，也不把 acquisition loss 反传给物理均值。

`bias=-8` 初始 margin 约 `(2.00134,1.00067)`，靠近最小边界。原始 TB 已确认 head 并未彻底不学习，部分 batch 的平行 margin 接近上界，因此不能仅用初始 sigmoid 导数判定“死头”。优先改监督目标；仅当修复后同桶 margin 长期贴边且 coverage 不足，再比较温和初始化或新参数化。

来源：[β-NLL 原文](https://arxiv.org/html/2203.09168)、[Faithful Heteroscedastic Regression，AISTATS 2023](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf)。

## 4. 建议主方案与两个候选

### 主方案：保留 GRU 物理均值，独立学习实际获取需求

- 修复 prepass 字段贯通及原始 point ID 集合，确保训练出的 mean/margin 真正决定新测量。
- mean 继续预测物理位移；acquisition 单独监督实际 support 对当前 GT 目标的额外覆盖需求。最小版本根据当前目标框在真实 tube 方向的投影、tube 中心及已有基础半径计算所缺 margin，再比较可见目标点覆盖目标；结合实际 endpoint/corridor 并集与尺寸上限核验，不能直接把完整终点偏差当所缺 margin。GT 只在训练标签使用，不进入部署输入。
- 采用独立小 acquisition 特征分支；引入过去可用的质量摘要与轨迹 age，保留纯 history mean 的职责。按预算固定 768→256 报告新增目标点 recall、背景比例、B2 positive vote quality，不能只追求更大支持体积。
- 对相同预算做固定 shell、CV support、GRU adaptive support 三者比较。收益目标是把真正新增目标点带进来，且不明显挤掉有效点。

这是较低改造成本的路线，但不能辨识共同平移。观察到的涨分取决于新增目标点和 B2/B3 消化能力，不承诺必涨。

### 候选 A：把 CfC 改为与 query gap 有直接联系的物理预测器

仅在不规则/较长间隔是论文核心且主方案获得可信信号后开展。保持 B0 历史协议不动，为 B1 单独读取稍长的因果 box-only 历史（例如 H=6 的先导实验），最后使用以 query gap 为条件的演化/decoder。输出有界速度或加速度残差并作物理积分，避免只把两个离散 transition 换成 CfC 就主张连续时间优势。

公平对照必须让 GRU 获得同样的历史长度、相同 Δt/query 输入和 decoder；不能同时延长 CfC 历史再归因给 cell。不要简单追加“假当前观测零 token”而不定义含义、missing mask 与零时长行为。长历史带来 stale/wrong history 污染及输入合同复杂度，需 reset/valid 处理；此候选尚无涨分证据，优先级低于获取修复。

### 候选 B：运行一次 B0 后，按当前观测质量分配新增测量预算

纯历史无法识别当前失配时，可让 acquisition/B2 使用 detached 的当前 B0 质量、稀疏度及 observation/prior 分歧，物理 mean 仍只读历史。B0 只运行一次；按质量在一组因果支持域之间分配固定预算，再由 B2/B3 完成修正。

这比仅添加更深 recurrent cell 更贴近恢复决策，但当前 raw crop 在 sampler/prepass 中提前生成，改为 B0 后获取会涉及 CPU/GPU 边界与原始点传输。较实用的先导是提前生成固定候选 support pool，B0 后仅在池内分配预算；它不能召回池外目标。真正动态裁剪需新训练/部署协议、同顺序实现和全流程延迟测量。当前 B0 可能自信地跟到邻车，单个 quality 分数不能解决身份问题，仍需 B2 身份证据。

## 5. GRU / CfC 公平选择实验

先做机制诊断，再做正式系统对照；两个阶段不要混写。

1. 在修复接口后，使用相同 endpoint、相同递归预测框/时间/valid 的因果历史集合比较 CV、GRU、CfC。以 tracklet 划分拟合/校准/选择，不拿测试轨迹调后端。若使用同一 B0 的固定 rollout 作为读入，这是后端诊断，不是更改正式 scratch_only 训练规定。
2. 固定 shared heads/参数预算/初始均值、采样与优化更新数，先对 shared anchor+GRU、shared anchor+CfC 各至少数个 seed 观察。真实 Δt 与 fixed/shuffled 控制须先证明确实生效。
3. 主指标：实际支持中的 novel-target recall@固定预算、全局可观测且 B0 缺证据条件下的召回、目标点占比；辅助指标：物理位移误差、相对 CV 的配对改善、分轴/二维 coverage、support 面积/点数、tail saturation。
4. 分层：query gap、历史间隔不规则度、速度变化、递归 age、历史质量、B0 目标点缺失程度。随机下采样须同时真实改变几何和时间；不能仅伪改时间后称为真实 irregular benchmark。
5. 按 tracklet 做 paired bootstrap/区间，报告固定最终 epoch 与真正 late-3。选用 final checkpoint 的标准预先固定，不能每个 arm 各挑最好点。
6. 最终在匹配 B0 学习轨迹和 scratch 正式合同下比较 Full-GRU 与 Full-CfC。B1-only 输出 observation，既不能用它的高 S/P 证明 B1 好，也不能用它不涨 S/P 宣布 B1 没用。CfC 的晋升条件是实际获取收益传递到闭环跟踪，同时标准 cadence 不显著退化、延迟可接受。

已有记录中 GRU 与 CfC 的 own-rollout RMSE/coverage 不同，上游 B0 历史和有效行也不同；仅可用于发现问题，不能作为后端胜负证据。
