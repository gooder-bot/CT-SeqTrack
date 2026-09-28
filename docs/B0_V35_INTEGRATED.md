# v35 B0：合法历史条件与候选局部几何

2026-09-28。按用户确认的综合方案，在 v34 **W=1/4/4/8** 上新增一个完整版本；按用户最新要求改为四档学习率检验同一结构，新增1.5e-4，GPU依次0/0/1/1。所有正式运行从随机初始化开始，不读取旧 W-quarter 权重。旧 v33/v34/reference 的模型、配置身份与证据继续保留。本轮服务器只读，由用户上传和启动。

## 研究依据和范围

v34 八组已经完成，不能继续使用旧启动页的“尚未启动”状态。W-quarter final60 S/P 为51.464989/62.843545，late-3为51.252735/62.676513；Success 和移动守底仍未过线。正式复盘见 [八组报告](../artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)，主要问题及反证见 [研究审查](../artifacts/ct_checks/20260928-170853_b0_revision_audit/REPORT.md)。

本轮假设是两阶段定位对合法历史与候选附近真实几何的利用不充分：首测训练初始化与部署对齐，为 coarse 补显式历史条件，为 fine 补候选局部证据。整包收益只能归于综合版本，不独立归因某一部分。当前没有证据支持将全部 fine 退化归因点序分桶、IoU/loss不一致、W无效监督或辅助梯度冲突，因此保留采样、全局 token、损失、quality/history监督与 W 课程。

**严重漂移与持续失跟属于本轮明确优化目标。** 三项修改分别针对首次缺测的不合理偏移、coarse对合法历史约束利用不足、fine修坏定位后引起的后续crop偏移与误差传播；这些是待验证的机制假设，不代表已消除漂移。综合方案与结果复盘必须说明这些路径的改善或剩余问题，不能因验收不加漂移硬门而推迟到Full处理，也不要求额外堆叠一个防漂移模块。

局部读取借鉴 [PTTR §3.4](https://arxiv.org/html/2112.02857v2#S3.SS4) 和 [MBPTrack §3.4](https://arxiv.org/html/2303.05071v1#S3.SS4) 的机制；不移植其 GT proposal、CUDA骨干或独立候选头，不承诺涨分。

## 三项配合修改

### 首测与合法初框标记

真实 `frame==1` 的四分支均使用精确合法初框和 known hint；其他窗口继续共同XY/yaw扰动。`history_is_initial[B,3]` 只在有效槽、真实轨迹 frame0、未经扰动的合法来源三者成立时置真。中段 GT seed 不获得此标记，网络不读取扰动数值、分支号或当前GT。

### 同源历史描述，独立阶段编码

共享纯描述函数；每槽为归一化局部中心3、相对yaw的sin/cos2、真实时间age1、预测支持3、exists1、合法初框标记1，共11维；三槽33维。不存在槽清零，存在但没有点的历史仍保留合法几何。统计、几何、时间和来源detach。

coarse拼接33维历史与3维当前支持，以 `36→64 / LayerNorm / GELU→256` 的零出口特征增量加至原 Mini pooled256，送入原 coarse head。decoder历史编码由30改为33维，继续直接读原始 pooled256，避免把历史融合再次当成观测语义。两个阶段不共享学习权重；原 live pooled 和 coarse-corner 梯度保持。

### 候选局部真实点

q0/modes共用局部算子，历史重建query不读此增量。以现有8角点为参照，将真实点到参照的相对坐标转到候选朝向，按完整首帧LWH归一化。在半径1内每角取最近16个唯一点，用raw ID稳定打破距离并列；不足不复制，不按前景阈值删点。

逐点输入69维：live点特征64、detached相对XYZ3、预测支持1、来源类型1。两层64维MLP均使用LN/GELU；masked max与mean共128维，另拼截断前半径邻域的归一化点量及平均支持，经零初始化130→64投影注入对应角点query。点量用 `log1p(n)/log1p(1280)`；最终投影后再次按真实邻域mask，确保训练后空邻域仍严格零增量。同一点被多个角点读取不构成多份独立测量。

B0使用当前真实点与SegPointNet的64维逐点特征。Full中q0可读当前B0点及已选extension，mode只读当前B0点与自身extension成员；extension只用真实 `point_xyz` 与pre-vote特征。两类支持分别是分割概率、identity sigmoid，并有来源位区分；不输入reliability、vote坐标或物体坐标中的记忆点。新几何/选择/支持detach，点特征live；mode query detach、输出center live不变。

保留q0 direct、history/mode seed residual、共享pose/quality头、空序列原回退、唯一accepted提交及跨帧detach。新增特征路径不增加候选、KV长度、raw点预算或历史forward。B0实数参数量由3,702,998增至3,739,606，新增36,608（约0.99%）：coarse条件19,136、局部读取17,280、历史输入扩展192。真实CUDA耗时和显存待用户上传后测量，不以参数量推断运行成本。

## 诊断与可重复性

v35默认被动汇总 epoch×branch×depth×历史来源，其中depth=0为窗口首个预测端点。来源为oldest-first三槽的精确组合：0缺槽、1合法GT seed、2扰动seed、3accepted预测；`history_is_initial`独立统计，中段合法GT seed不等于初框。全量记录raw/crop/sampled目标点、预测支持、coarse/fine中心误差与观测定位/分割/BC/主quality损失的分子、有效计数及原batch贡献；quality复用已有loss标签，Full额外任务只记录实际batch标量，不伪造逐端点归因。额外IoU仅固定哈希约1/16端点并注明覆盖。记录局部截断前后点数、支持和增量幅度；同时记录raw缺测、raw有目标但crop遗漏、四帧有点但均为背景、当前采样无目标但历史有目标的频率。GT仅用于日志与监督，不改变forward、裁剪、loss权重、重置或提交。

正式逐帧记录补充带符号coarse/fine/anchor世界坐标框与首帧LWH。诊断不新增网络前向、不消耗训练RNG，每轮写独立 `training_diagnostics/epochNNN.json`；报告区分加权后的端点贡献与实际batch标量，不把归约分母不同的日志均值直接相加。关闭诊断应与开启时的预测、梯度及优化结果一致。

新身份为 `ctseqtrackv35` / `ct_seqtrack_v35` / `ct_seqtrack.joint_identity.v35`，checkpoint runtime键为 `ct_v35_runtime`。结构常量、初始化策略、种子及配方进入身份；只允许相同身份完整epoch恢复。旧版本默认值与hash不受新增字段污染。

## 正式预算、比较与停止

第一阶段仅四次：seed42、LR=2.5e-5/5e-5/1e-4/1.5e-4，GPU依次0/0/1/1；同一综合结构、20/50衰减、无warmup、scratch60、batch16、workers4、FP32、每次71,700更新，四组合计286,800。数据划分seed42、W课程、十轮课程和112 reserve不变。已有R/C/W-quarter直接复用。新增1.5e-4来自用户本次明确要求，不改变结构或其他单组参数。

总体final60/late-3四项≥固定R，且固定31条曾移动轨迹全程657预测帧的四项≥旧C，才通过原硬门。通过者按final Success、Precision、较低LR排序。没有通过者即停止四次训练。

通过后只增加胜出配方seed52和原正常SeqTrack seed52，保留原条件复验后共最多六次、430,200更新。当前只启动四组seed42，不自动进入第二阶段。seed52不改数据划分、不重新选LR；其同seed参考对照不替换固定R/C门槛。方向反转则稳定性未证实，不自动补第三seed。

严重漂移采用方案1：原硬门不变，完整报告>5m/>10m与未恢复失跟。若>10m帧数在final60或late-3高于旧W-quarter，单列风险、暂停Full待复核，不改写原硬门结果，也不直接判定综合方案失败。结合受影响轨迹、持续时长和58/59/60逐轮表现判断：小幅孤立波动不自动否决，明显持续退化必须处理并与用户讨论，不能被总体涨分掩盖。优化目标与风险复核是两个环节，统一细则见[实验协议](EXPERIMENT_PROTOCOL.md)。不要求所有切片获胜，不挑最佳epoch，不为某个漂移计数自动扩大LR、窗口或训练预算。

运行、恢复与比较见 [v35操作说明](CTSEQTRACK_V35_MINI_LAUNCH.md)。正式CUDA检查与训练均由用户执行，当前实现和工程验证不能替代新的正式成绩。
