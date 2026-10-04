# v36：测量事件驱动的连续时间跟踪闭环

本轮实现依据用户确认的综合方案。既有研究证据见 [联合设计报告](../artifacts/ct_checks/20261004_v36_b1_joint_design/REPORT.md)。工程测试只验证合同，不证明正式收益。

## 状态与时间

每帧 accepted、最近三个测量事件和身份 memory 分开保存。网络在一个端点内重编码事件，不保存跨帧神经 hidden；每步状态 detach，不增加历史点云 forward 或整轨迹 BPTT。世界框在 host 使用 float64，网络公开定位框为减去 prepare 锚后的世界轴 XYZ 和绝对 yaw。

位置 prior 从最近事件出发。有效测量边差商是区间均速；网络学习初速度修正及衰减加速度：

`d(T) = v0*T + a0*tau²*(T/tau + expm1(-T/tau))`。

速度修正与加速度头零初始化，正的全局 tau 初始化为 1 秒，短跨度用稳定展开。无运动证据时保持位置但 `motion_valid=false`；不把未知运动标为已知静止。没有永久 hidden，因此当前 loss 能监督整个最近事件编码。Z/yaw 保持最近事件。

事件编码不读取 accepted、当前点或查询 T；事件 token 21 维、运动边 50 维、投影64、CfC hidden128。只有同一 segment 的有效边参与递推。位置范围两个独立 `128→64→2` 正值头读取 detached 固定事件编码：`range=anchor_range+T*range_rate`，初始化为 0.5 米及 1 米/秒。current 范围项对 anchor_range 停止梯度。范围是经验边际误差尺度，不能宣称精确协方差或联合覆盖保证。

事件 gap、查询跨度、速度年龄、range 增长和 memory 年龄使用同一 effective 时钟；sensor 时间用于原始记录和恢复秒数。B0 原有 sensor 时间输入保持，因此工程 time controls 不能直接称为全模型纯时间因果消融。

## 获取、定位与核验

排除整个 B0 raw crop 后，prior 域先选最多384个唯一点；B0 环带只排除这批已选 raw IDs，再选384。不足768从两域未选并集按确定性覆盖借位；仍不足保留无效槽。来源分区与几何成员关系分别记录。768→256 保留每分区64 identity、48 coverage、16 hash，空槽按全池剩余 identity 排名借位。

prior 域取事件轴；物体半尺寸用 `abs(R_event.T @ R_object) @ half_size` 投影，另加夹到0.25—12/8米的 range 边距。B0 环带在原 crop 轴增加4/3米。两个域各自保留原 Z crop 余量，没有连接两中心的背景走廊。搜索 cap 不限制物理均值，也不用于运动判边。

q0 保留 v35 观察主路及合法历史条件；只读 B0 与新 B1 条件。extension/memory 不注入 q0 定位 KV。历史辅助 query 不读新增当前上下文、extension 或 memory。

B2 身份路读取 memory；几何路只使用当前 raw/local geometry 与当前 B0 特征。逐点 vote 不读取 memory，身份与 reliability 可通过 live 权重影响模式中心。固定 raw IDs 才能检验 memory→vote 隔离，不能把身份选点变化当成几何泄漏。

两个模式顺序使用同一次编码和 vote 缓存。聚类半径1米、三次稳健细化；第一簇成员和最终 seed 附近0.75米区域移除后构建第二簇。两个 seed 一次进入独立 residual pose head；最终 decoded XY 中心距离≤0.75米时第二槽失效，不补额外候选。query seed detach，输出 seed center live。

候选固定 `[q0, prior, mode0, mode1]`；四帧 B0 全空时 q0 无效，prior 仍可输出。最终框核验使用当前 B0 点及选中 extension：q0/prior 可读全部选中点，mode 只读自己的 extension 成员。memory 物体坐标、历史点、未选768池点均无当前测量权限。

核验器最终框八角点，每角半径1米取最多16唯一点，编码69维点输入和32维候选/事件元数据。几何、成员、统计与概率 detach，点特征 live。最后 `Linear(64,3)` 权重全部零，bias 为 `[0,0,logit(0.1)]`；其他层正常初始化。首步上游核验梯度为零是预期，出口更新后再验证梯度合同。

## 输出、测量与记忆的事务

output 取 S/P sigmoid 均值最高者；measurement 独立取有效 q0/mode 中框内唯一预测 FG≥1且measurement score≥0.5者，按measurement score排序，固定索引破同分。prior 永不制造测量。

最终框内预测 FG 集合以完整授权点池去重后计算，不受角点16邻点上限影响。measurement 标签要求实际GT前景至少1、支持集合纯度≥0.5、geometry_exact IoU≥0.5。GT只生成监督/被动台账，不进入 forward、排序或提交。

运动边在提交前用旧 prior/未截断 range/旧轴判断：`max(abs(R.T @ innovation)/(range+projected_half_size))<=1`。这是容差规则，没有统计保证。失败只切断边并开新 segment，合格新位置仍接纳；无运动时同 segment 后续两个事件用实际正 gap 启动，不要求相邻帧或符合静止 prior。

memory 只跟随最高分 measurement，自身框内唯一预测 FG≥3时写入；1—2点事件不刷新memory，不另选其他候选补写。初始12 FG及最近两次8 FG+4 BG容量保持。host在Adam后统一提交，accepted/measurement都用原prepare锚转世界坐标。

合法未扰动frame0有首框标记。branch0保留未扰动教师历史；branch1/2及课程期branch3、中段F重置保留共享±0.3米/±1.5°扰动；中段扰动frame0不能带合法首框标记。教师来源、branch及扰动答案不是模型特征。

## 损失与五组解释

| 项目 | 归约及外系数 |
|---|---|
| coarse/q0 | anchor-local中心三维SmoothL1均值×2，yaw×10；各有效端点均值 |
| history | 有效历史槽→有效端点；中心×0.2，yaw×1 |
| seg/BC | 唯一点→帧；current/history等权，缺组重新归一→端点；seg×0.1、BC×1；seg类权BG/FG=0.5/2 |
| 学习运动A/C/F | GT_t−GT_event，在事件轴按投影尺寸和max(1,T/0.5)归一，两轴→有效运动端点；×0.1 |
| range | 米制两轴0.9 pinball，不附加尺寸归一→anchor/current各半→prior有效端点；×0.1 |
| identity/reliability | 保留全batch实际FG/BG组等权；分别×0.2/0.1，reliability标签detach |
| vote | selected GT-FG点及XYZ→正端点；×1 |
| mode | center的seed/final各半×2；仅final yaw×10；正mode→正端点 |
| S/P | 普通soft BCE，有效候选→端点；各×0.125 |
| measurement | 普通BCE，有效非prior候选→端点；×0.25；无类别重权 |

空任务返回图连接零，不转移系数。四分支各四分之一端点曝光，不应用旧v24的分支loss权重。普通measurement BCE估计的是登记训练分布下的资格标签概率，不等于已校准概率；固定0.5阈值不调优。

A完整；C移除B2、extension和memory及禁用的mode定位参数；D仅将均值改为事件CV，仍保留事件编码和range，不能把A−D叫纯CfC消融；E为适配后的旧B1；F只改变长分支八步重置。

E保留旧accepted三框live context和单步均值；有效时起点为上一accepted，无效时走事件CV。旧mean loss包络内SL1(beta=.25)，外轴`.25*relu(.9-sign(target_unit)*pred_unit)`，两轴再有效端点均值，系数.1；不保留旧sigma/u头或loss。E的旧context也给新range头，含查询时间，故不承诺E范围非收缩；range标签必须匹配其实际origin/horizon。A−E是整体B1比较，不是纯锚点因果实验。

## 排程、随机性与诊断

前十轮W课程，第十一轮起A/C/D/E长分支从frame0连续走完整轨迹，八步仅记录计算块；F八步重置。16 lane固定因果排程，112 reserve只取短分支独立单端点；真实mini每轮19108行、1195 Adam，尾批4行。首reserve/不足整批使用B0 running BN，affine参数仍可学。

端点采样seed只含训练seed、epoch（eval固定0）、track、branch、当前帧、源帧及用途，不含window/reset/drain/arm/worker。公共模块按语义组件名独立初始化；五臂同语义张量逐位一致。允许状态改变点池，不能强制五臂读取相同裁剪。

只恢复完整epoch；run/config/source/采样计划/数据及注释时间身份均校验，保存RNG。旧产物和配置不迁移、不覆盖。新增源码改变全局source hash，旧checkpoint须使用原源码。

逐帧台账记录独立测量正确/错误接纳、漏接与排序失败、实际写入memory纯度、range覆盖/面积/年龄及获取→候选→输出链。恢复沿用原benchmark IoU<.1开始、≥.5结束，未恢复右删失。只观察提交之后的恢复；同帧恢复单列，后续新事件替代前未恢复另标，保留完整失跟段。这是关联诊断，不是事件导致恢复的因果证明。
