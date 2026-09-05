# CT-SeqTrack v26 实现审计（2026-09-05）

范围：只读检查当前源码；未修改模型、正式配置或 output；未使用 trajtrack。复现脚本只读取源码定义并在 CPU 执行，不伪装完整 Lightning/nuScenes/CUDA 跑通。下述行号对应审计时工作树。

## 结论

模块边界已经比普通“给 SeqTrack 加几个分支”清楚得多：B0 的监督与递归状态、B1 的物理先验、B2 的扩展证据和 B3 的行动风险分开；但当前有已复现的接口、采样唯一性和几何语义错误，足以让模块的实验失去方法解释。应先修复这些，再判断网络容量或论文创新是否不足。

### P0：B1 prepass 的 margin 字段被丢弃，正常预测全被判无效

- 编码器实际输出 `acquisition_margin_parallel_perp`（`models/ct_v2/motion.py:1173`）。
- 公共 `predict_motion_from_history()` 在 `models/seqtrack3d.py:2546` 的白名单过滤中没有返回它。
- `_unbatch_motion_prepass_predictions()` 在 `models/seqtrack3d.py:2650` 强制该字段存在且有限，缺失即在 2661–2662 写 `valid=False/source_id=0`。
- 在线训练经过 7696–7740 的 batch prepass；评估经过 2667–2713 的单行 prepass，因此两边都受影响。
- `utils/ct_search.py:650` 仅在预测 valid 时采用 learned B1 support，否则用 CV fallback；“B1 学到了 margin”因此不等于“B2 实际用到了它”。同时模型后半段重新算 B1 后可以 valid=True，造成诊断/特征里的 B1 和 crop 实际来源不同。
- **CPU 复现**：假编码器明确输出 margin 且 valid=True，经真实公共方法和真实 unbatch 之后 margin 缺失、valid=False、source=0。
- **修改**：补齐公共返回字段；单行和批量训练/推理都增加接口集成测试，断言 learned source、margin 值和实际 support 尺寸联动；B1 无效样本继续使用现有 failover。修复会改变训练中 B2 看到的点集，需要新的 scratch 实验，不能把旧 Full 结果当作修复后的方法。

### P0：FPS 没排除已选行，合法不同 XYZ 点会重复索引并触发评估崩溃

- `models/ct_v2/evidence_memory.py:26` `_fps_indices()` 更新最近距离，但没有给已选索引设禁止值。
- v26 coverage 使用 XY FPS（330–334）；不同高度但相同 XY 的点会让所有剩余距离都为 0，于是反复 argmax 同一行。
- `_hybrid_select()` 的 `append_rows()`（309–319）只按调用前 selected_flag 过滤，传入同一数组内重复行不会被去重；selected_count 还会将重复项计入预算，影响后续借用。
- 评估 `models/base_model.py:1306` 发现 selected rows 不唯一即 RuntimeError。
- **CPU 复现**：300 个不同 Z、相同 XY 的有效点，选择 256 个槽位却仅有 161 个唯一索引；独立 FPS 前 8 个索引全为 0。
- **修改**：FPS 显式排除已选索引；append_rows 再做稳定去重；从仍未选行补足预算。测试覆盖相同 XY 不同 Z、完全重复点、少于各子预算和跨组借用；同时检查 memory FPS 是否重复填有效 token。

### P1：memory 使用 wlh 却按 xy 直接归一化，前景/上下文角色出错

- `datasets/sampler.py:1706` 传入第一帧 `.wlh`；评估 `models/base_model.py:2264` 的固定尺寸也来自同一 wlh 语义。
- `models/ct_v2/evidence_memory.py:106` 直接取前两个尺寸，119–120 使用 x/w、y/l；nuScenes 应当是局部 x/length、y/width。
- 这既改变 foreground/context 采样，也改变 memory metadata 的归一化坐标，不只是报告字段。
- `cfgs/ct_seqtrack/26_formal_base.yaml:41` 正式启用 `ct_memory_mode: real`，因此该问题实际影响 Full 与 Full-B3；不能只当作停用分支遗留问题。
- **CPU 复现**：wlh=(2,4,2)，yaw=0，(1.8,0,0) 应为前景却被取成 context；(0,1.8,0) 应为侧方 context 却进入前景。
- **修改**：统一显式 `size_xyz=(l,w,h)` 转换；增加非正方形、旋转框测试，直接检查 token 的 source point 和 role。现有 support-membership 的正确性不能证明 memory 路径正确。

### P1：B3 H1 的 IoU 标签会发生符号翻转

- `models/seqtrack3d.py:5197` 明确用同尺寸轴对齐 BEV IoU proxy；5200–5208 又把 wlh 前两维直接当 xy。
- help/harm 标签在 5237–5248 将 `iou_gain_h1` 正负作为必要条件；所以这不是只影响一个小的辅助 loss，而会影响最终 action score 的两个主分类器。
- **CPU 复现**：wlh=(2,4,2)，目标中心=(0,0)，B0=(1.5,0)，动作=(1,.5)。动作幅度0.707m，处于常规0.5s下0.75m B3半径内；代码 proxy gain=+0.13714，正确轴对齐 gain=-0.06324。
- 即使交换长宽，未考虑预测/目标 yaw、Z、尺寸的 proxy 与正式 oriented IoU 仍可能有符号差异；这里的已确定 bug 是长宽轴错配，额外的 proxy 偏差需要用真实 action rows 量化。
- **修改**：这些标签本来就需要 detach，不要求可微，优先和 H3/calibration 共用真实 oriented IoU 的标签函数；至少先正确处理 l/w，再按 yaw 和稀疏度统计 proxy/exact 符号冲突率。

### P1：在线“novel”点身份也可能被浮点变换破坏

- 这比已有 counterfactual float32/float64 诊断问题更接近创新核心。
- `datasets/points_utils.py:304` 到 317 对每个 sample box 分别执行 world→sample-local→world→anchor-local；`datasets/data_classes.py:84`、93 把每次平移/旋转结果回写 float32 点数组。
- 基线（`models/base_model.py:2325`）、endpoint（2592）、tube（2443）使用不同 sample box；同一个原始 LiDAR return 因变换路径不同可以出现微小局部坐标差。
- `utils/ct_search.py:1209` 以1e-6量化 XYZ 做 B0差集和source union。这能保证“key不重复”，却不能保证“原始物理点不重复”；落在舍入边界两侧的同一点会冒充 novel evidence。
- **CPU 复现**：加载真实 PointCloud、真实 crop 和真实 bounded_novel_support_pool，两个 support 都包含完全相同的100个原始点，理论 novel=0，实际 novel=8；最大局部差只有4.77e-7m。测试中心靠近原点、两个support yaw不同。大世界坐标的另一个样例未触发，故**不推断真实 nuScenes 中污染比例**。
- **修改**：把原始 point index 贯穿支持域 mask、集合差、来源 bitmask 与诊断；或者统一变换一次到固定 anchor frame 后按原始行索引选点。仅加大容差会把相邻真实不同点误合并。
- 已有 `base_model.py:799` 对 active pool 的“共用key函数重算”检查不能发现此问题，因为两边重复同一种量化身份假设。counterfactual 的另一条世界坐标float64路径（623–636, 724–733）也应统一到原始索引。

### P1：空点云判定绕过所有恢复分支，且判定条件不正确

- `models/base_model.py:1892` 判断的是全部 B0 历史+当前 XYZ 的**代数总和**是否为0，随后直接用 reference box，跳过 B2/B3和诊断行。
- 这不是“当前 raw B0 crop 为空”的精确判据；对称非空点云也可能总和为0。所有 B0 点为padding时，即使 extension有效，模型仍不会执行恢复。
- 现有 endpoint metrics 可以保留这些帧，但 proposal/acquisition CSV 缺行，会改变机制统计分母。
- **修改**：使用显式 raw count/valid mask；定义无B0点时 observation fallback 的框和统计，但仍允许独立 extension+memory 检查；每个 endpoint 无论是否forward都导出明确状态/原因。不能仅删一行 early skip 而不定义 B0 空输入行为。

## 已落实的结构，以及不应误判为 bug 的部分

- B0→插件：`pipeline.py:16` 的 ObservationOutput detached，`seqtrack3d.py:2743` 实际使用。
- B1均值、sigma、margin分别监督；sigma与margin读取 detached context（`motion.py:730`,738），不通过采样间接向均值传播 GT。
- B2最终坐标只从extension点加 bounded point vote得到（`evidence_memory.py:625`），base/memory只作为上下文；无有效extension精确返回observation（647–654）。
- B3再次detach上游（858起），只训练行动风险和预期收益；缺失 calibration不部署动作（953–960）。
- 正式使用unified Adam named groups（`seqtrack3d.py:3163`），不是冻结 B0 或多套手工优化器。机制前向no_grad加BN隔离（8419,8610），训练最终写observation递归状态（8099–8126）。
- consensus有离散 seed/top-mode/inlier选择，但被选模式内的 vote/weight 仍有梯度；CPU复现确认两者梯度有限且非零。不能笼统称为“投票不可训练”。
- raw_search输出经 `_forward_ct_contract_v3:3025` → `forward:3984` → `_finalize_observation_output:3251` → evaluator `base_model.py:127`；诊断1157也取同一字段，未发现 finalize 再次覆盖成B0。若 Full-B3 CSV final全等B0，应检查运行时 mode和candidate availability/presence分布。

## 修改错误后再验证的设计风险（推断，不是现有结果证明）

1. **获取远处目标与只能小幅行动之间可能不匹配。** corridor长度上限16m，B3常规dt=.5时radius=.75m且max2m（`evidence_memory.py:941`）。B2还继承B0的z/yaw（654）。先在真实状态上测 raw oracle与bounded-action oracle增益差、按XY/Z/yaw分解失败；没有bounded oracle headroom时，继续调risk head没有意义。
2. **强背景共识不等于目标共识。** 当前 consistency=normalized_mass×inlier_ratio×exp(-trace covariance)（444–446），本质上奖励集中模式，不能单独证明identity正确。需要看 targetness/relation的前景排序、背景模式占优、target vote偏差与raw harm；在证据标识修复前不要直接扩大池子或K。
3. **presence监督和实际投票证据不完全一致。** presence target按768 prepool含目标定义（`seqtrack3d.py:5165`），raw/vote supervision按256 selected含目标定义（5167）。当selection漏掉全部目标时presence仍可为正。应拆分“pool里可能有目标”和“selected voting里有可定位目标”，或至少报告该分层的误判率。
4. **H3训练样本由当前固定presence阈值选择。** `_attach_h3_shadow_labels:8005` 使用ct_search_candidate_valid，初始presence=.1时大多无H3标签；校准后若采用低于.5的presence阈值，会部署到未充分覆盖H3监督的区域。按结构有效样本独立、确定性抽样H3更利于校准一致性，但训练成本须量化。
5. **训练速度有明确同步热点。** FPS每轮.item（`evidence_memory.py:33`,37）和consensus多次.detach().cpu（404–454）在GPU上会频繁同步；memory每帧也循环FPS。先用真实batch分阶段profile定位，再向量化FPS与K=3 hypothesis；不能声称当前时间/memory模块有实际速度收益。
6. **逐bit hash失配需要确定根因。** 现有严格hash要求可发现消融失配，但同一输入/初始化仍可能被CUDA非确定算子影响。应同卡同环境重复同臂，再跨臂，对比首个不同层的输出、梯度、BN buffer与Adam state；先区分内核随机性与跨模块耦合，不随意删除hash gate，也不凭hash差异直接说存在梯度泄漏。

## 最少必要验证与实验顺序建议

1. 先修 prepass、FPS、几何轴、点索引身份与空输入合同，新增少量针对失败原因的测试；执行既有CPU合同测试与compileall。
2. 在服务器上跑实际batch：检查 B1 learned source比例、margin→support实际联动、前景memory token身份、selected索引唯一性、原始点索引novel集合，以及B3 exact action label。
3. 完成同臂重复与跨臂 B0 step1/100事务对齐；真实forward/backward与resume验证不能用本地stub合同测试替代。
4. 新scratch mini检查机制漏斗，再按项目最新确定的注册协议运行正式臂。旧checkpoint只能保留为错误版本历史证据，不初始化修复版。
5. 先验证“获取到真正的新目标点→保留→产生有益raw/bounded候选”，再校准B3；memory要在修复几何后与none/time-misaligned公平比较。

## 本次复现工件

- `reproduce_code_audit.py`：真实函数/方法AST只读加载，CPU最小反例。
- `reproduce_code_audit.json`：5项确定错误/反例和1项投票梯度正常证据。
- 未执行完整CUDA训练，也未统计上述新问题在nuScenes上的触发比例。

## 文档漂移

README当前说直接full数据预检且无mini gate；`docs/CTSEQTRACK_V26_METHOD.md` 第7节说mini后再full。`docs/CTSEQTRACK_B0_B3_METHOD.md` 第7节还写每模块独立optimizer/scaler，与当前v25/v26 unified Adam不一致。应以当前正式protocol和用户本轮计划为准，随后一次性统一这些文档，防止换实验顺序时无意改变论文证据定义。
