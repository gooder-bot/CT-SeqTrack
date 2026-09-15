# v29 partial：方法与数据契合度复核

2026-09-15；只读核对当前 `HEAD=525eb54`。生产代码、配置及 output 未修改。

本报告区分三类依据：当前源码确认的行为、9月14日已保存的合成反例、主分析转交的未完成训练汇总。合成反例不等于当前完整 nuScenes 上的实际损失或改进幅度。当前三臂尚无完成的第5轮官方验证；因此不能凭早期训练 loss 判定 B0 恢复、Full 优于 B0 或 CfC 优于 GRU。

## 1. 总体判断

B0 观测、B1 相对运动与获取、B2 新增测量、B3 有界选择更新的职责划分，适合“仍有局部可达目标证据”的稀疏跟踪任务。当前不能称为最契合数据：无效测量监督、供给范围、稀有恢复样本、共识选模和即时动作标签都有明确局限。网络宽度和 CfC/GRU 后端不是目前最有依据的第一优化项。

不需要重新修复已经落实的 v29 接线：

- 历史帧布局已在 `models/seqtrack3d.py:3822` 调用 `utils/v29_observation.py:4` 修正。
- Transformer 已读取真实测量与历史存在性，并用于 local/global/cross 注意力：`models/attn/Models.py:163`。全无效 key 返回零注意力：`models/attn/Modules.py:18`。
- 1/2点真实重复采样已经通过 `datasets/sampler.py:276` 的 v29 专用分派实现，旧2→3点丢弃断层不能再当作当前未修问题。
- 输入预测历史与 GT 历史监督已分离：`datasets/sampler.py:305`、`:1259`；物理 coarse 目标已独立：`:2370`。
- 机制训练共用状态转移，提交 accepted 框：`models/seqtrack3d.py:8401`、`:8451`；这与共享 B0 观测流隔离是有意设计，不应通过移除 detach 或把 Full 状态回灌 B0 来“修复”。

## 2. 必须先厘清的统计口径

`acquisition_supply/epoch_XX.json` 的名称不足以直接解释为获取成功率。当前定义如下。

| 字段 | 真实含义 | 源码 |
|---|---|---|
| eligible_rows | 已经进入768输入的新增点中，至少有一个目标点的行数 | `models/seqtrack3d.py:5542` |
| retained_rows | 上述行经256选择后仍至少有一个目标点 | `models/seqtrack3d.py:5543` |
| row_recall | 256保留至少一个目标 / 768已含目标；条件成功率 | `models/seqtrack3d.py:2154` |
| pool_targets | 完整原始 extension pool 的目标点数，已经排除整个 B0 raw crop | `datasets/sampler.py:1614`、`:2285` |
| sampled_targets | 最终256 selected中的目标点数 | `models/seqtrack3d.py:5522`、`:5661`、`:8524` |
| epoch point_recall | 256目标数 / 原始 extension pool目标数，跨越两级采样 | `models/seqtrack3d.py:2156` |
| step ct_acquisition_point_recall | 256目标数 / 768输入目标数 | `models/seqtrack3d.py:5551` |

因此同名 step/epoch point_recall 的分母实际不一致。Full 第2轮的转交汇总：CfC eligible/retained=10058/10058、365069/669005；GRU=9893/9893、359542/665597。前者只证明768含目标时256几乎没有完全丢失它；后者54.57%/54.02%是从原始 pool 到256的累计点保留率。不能据此说全球新增目标已经被获取，也不能把约45%的点减少全部归因于768→256选择器。

推荐后续诊断显式登记各级总量与条件分母：全球 novel → 最大合法范围可达 → 实际 support 原始 novel → 768 → 256 → 最终正确模式 → bounded → accepted。另报每级“至少一目标”的事件数及 unique 点数。target-bearing 的行率、目标点保留率和最终模式正确率应分开。

## 3. 仍成立的缺陷或不匹配

### A. 无测量槽仍训练 Seg/BC，缺失历史仍影响特征（优先级最高）

v29 仍进入 `seqtrack_reference_loss`，而非下方旧 v27 masked 分支：`models/seqtrack3d.py:5790`。`models/ct_v2/observation_reference.py:38` 的 CE、`:65` 的 BC 及`:43`的 ref loss 都不使用点/帧有效性。`datasets/sampler.py:1574` 在补零后的点上构造标签，零槽在 GT 内时成为前景，在外时成为背景；两者都是不存在的物理测量。

9月14日真实 sampler/host 合成诊断已记录：空当前帧1024无效槽全部前景，Seg与BC输出梯度非零。当前源码未改，失败方式仍然成立；空样本在本次真实训练的频率与它造成的S/P影响尚未测定。

最小修复：新版本中 CE 使用有效标签类别权重之和为分母；BC 仅按有效槽/通道归约；ref 按存在历史归约。全空批返回可反传的零测量 loss，保留有限的框/运动监督，不跳过端点、不用 GT 重置递归。保留既有三维 log_softmax→二维 NLL 确定性路径。首步只屏蔽无效槽，不同时改变重复真实点权重。

更深一层仍需独立处理：`models/seqtrack3d.py:3542` SegPointNet 与`:3559` MiniPointNet 在注意力 mask 之前已消费整序列；`models/backbone/pointnet.py:248` 全局池化与普通 BN 会混入缺失槽。`sampler.py:2197` 仅清空ID/valid，并未消除缺失帧复制点的上游影响。先做有效池化及全无效分支，再专门设计 BN 统计方案；简单置零或仅接 Transformer mask 不能证明填充影响已消除，直接全网换归一化也不是最小修复。

### B. B1 学相对运动，局部获取不保证新增证据（高优先级）

相对位移特征来自相邻框之差：`models/ct_v2/motion.py:1048`。给所有预测框加入同一个绝对偏移，这些相对运动不变。不能要求 B1 仅凭这样的输入恢复未知绝对位置；应该由观测测量纠偏，无须追加无可辨识输入的“绝对漂移头”。

v29 的 Z 包络已经修复，XY 没有随之扩大：`utils/acquisition_v29.py:44`。9月14日静止 Car 几何反例显示默认 support 可被 B0 raw crop 完全包含，此时正确 ID 差集就是空；这是几何能力边界，不是 ID 差集写错。

最小有意义对照：将有限新增带定义在实际 B0 crop 投影边界之外，或只对因果低质量/空 crop 条件提供有限备用带；保持768→256预算、同一几何用于crop/监督/可达性，禁止GT决定获取范围。不应在没有全局→可达→support分母前无条件放大搜索。

另一个数据目标风险：无 novel 目标合法行被赋最小 margin（`utils/acquisition_v29.py:132`），与有恢复需求行共同进入q=.9 pinball（`models/ct_v2/motion.py:247`）。总体正例少不证明模型必然塌缩，但条件输入难以区分时会偏小。按“可达恢复需求/无需求”分别归约并保留负例，是比单纯增大 hidden size 更直接的实验项；先报告两类的数量与梯度贡献。

### C. B2 选出目标点并不保证选对最终模式（高优先级）

当前 seed 只按单点 targetness 排序、最多3个：`models/ct_v2/evidence_memory.py:502`。三个孤立高分误检可排挤稀疏目标共识。模式分数`:546`为 targetness mass × 未加权点数比例 × 协方差项，点数被重复奖励，密集低分背景可能胜过少量高分目标。

这两种失败均由9月14日生产函数反例复现；本次尚未统计真实发生率。raw回归在 selected 含任一目标时启用（`models/seqtrack3d.py:5386`），不代表获胜模式含目标。因此接近100%的条件行保留率不能为共识正确性背书。

最小对照分别为：去掉模式排名的重复点数乘子；用局部加权支持量排序seed后沿用K=3及空间NMS。保留presence负例、原始ID去重、特征detach和点预算。不要把两个修改同时作为唯一一组试验，需区分各自收益。除raw中心误差外，记录选中模式是否含目标、目标vote质量及背景模式质量。

### D. B3适合局部即时纠偏，不能据此宣称长程恢复（中高优先级）

`utils/v27_training.py:69` 在v29只用即时S/P增益，H3确实只诊断；`models/ct_v2/action_v27.py:11` 在dt=.5秒时动作半径.75米。若初始误差大于2.75米，单步无法进入Precision的2米范围；若修正前后IoU均零，则H1 S/P可同时为零。几何上有用的恢复步可能没有正效用标签，这是目标信息边界，不是loss漏加H3。

先统计“结构合法、几何误差下降、但H1=0”的比率和限幅饱和率。若实际占比明显，再设计独立连续几何或未来恢复效用；执行分数也必须表达该目标，不能仅加辅助loss仍用原H1 q并宣称解决恢复。禁止把当前H3诊断静默重新加到同一q目标。

`utils/action_calibration_v27.py:241` 按 never 路径action mask去重不同阈值，在接受动作后状态变化的闭环中并不等价。可以先用现有权重在训练内部17场景对照阈值数值去重与少量闭环粗搜；18场景锁定诊断、官方150场景不选择阈值。无须为此重新训练全部网络。不能以目前训练行为混合策略下的动作率代替拟合后部署策略动作率。

### E. 短roll-in缓解输入偏移但不覆盖长漂移（待数据决定）

`utils/v29_rollin.py:30` 从t-4起点初始化，端点前最多3次无梯度前向。已有机制轨迹使用accepted状态，并不等于共享B0已接触同样长的预测误差。先比较teacher/roll-in/真实长轨迹的误差分布、0/1/2/3+点及有效历史数。只在证实状态覆盖不足后给少量样本更长窗口或时间相关扰动；直接把全部窗口加长会增加本来很高的训练成本。

## 4. 第2轮 B1 抽样指标的专门解释

以下来自 `validation/sampled_batch_summary.csv`，各臂第2轮均为241个已写出的机制 batch。表中是batch指标的算术平均，不是全量行级RMSE或总体覆盖率；两个Full机制递归状态不同，不能只比较跨臂数值宣称后端优劣。

| 指标 | Full-CfC | Full-GRU |
|---|---:|---:|
| learned prior物理XY位移RMSE，m | 1.225660 | 1.255552 |
| 同batch kinematic物理XY位移RMSE，m | 1.340843 | 1.372920 |
| 名义50%误差椭圆覆盖 | 91.565% | 91.067% |
| 名义80%误差椭圆覆盖 | 93.469% | 93.161% |
| 名义95%误差椭圆覆盖 | 94.626% | 94.471% |
| sigma parallel均值，m | 1.532952 | 1.571373 |
| sigma perpendicular均值，m | 1.064547 | 1.088749 |
| acquisition margin parallel均值，m | 2.105698 | 2.094400 |
| acquisition margin perpendicular均值，m | 1.000154 | 1.000163 |
| main valid比例 | 93.698% | 93.698% |

**RMSE的标签已经是物理位移。** `datasets/sampler.py:1775` 调用 `utils/candidate_utils.py:252`，最新GT历史框定义位移起点、递归最新框只定义坐标轴；因此 `motion_main_target_xy` 是当前GT与上一GT的位置差旋转到递归anchor方向，而非当前GT减去预测anchor的位置。`models/seqtrack3d.py:6076`计算XY向量范数，`:6096`为`√[Σ(valid·||mu-target||²)/Σvalid]`，没有除以XY维数2。随后CSV再对batch RMSE取均值，不能当作合并所有行后的RMSE。两个臂各自learned相对kinematic约低8.6%，可说“抽样训练批中的相对物理运动预测有所改善”，不能说“绝对漂移已经恢复”或“官方跟踪得分提升”。

**coverage不是acquisition覆盖。** `models/seqtrack3d.py:6105`将物理位移残差投影到运动parallel/perpendicular轴，以各自sigma标准化，计算二维Mahalanobis平方；50/80/95对应阈值1.386294、3.218876、5.991465。`:6136`用`main_valid`归约布尔命中率。`main_valid`来自先验可用性与有限值检查（`:5996`、`:6020`），不是crop外目标存在性；该指标不读取support/目标点是否进入搜索范围。

两臂50%椭圆都覆盖约91%：相对于主体误差，统计分布明显过宽/不够尖锐。80%也过覆盖，而95%接近标称，提示“主体很尖、尾部仍存在”的分布形状问题或样本异质性，不支持直接对所有sigma乘同一个更小系数。需要按运动速度、递归年龄、有效历史、困难位移分桶，查看标准化残差分布与NLL。这里是训练批诊断，尚不是独立验证集校准结论。

**margin与sigma是独立语义。** `models/ct_v2/motion.py:786`获取头读取detach时间context和因果获取特征，输出`min+(max-min)·sigmoid`；统计sigma不直接决定这个margin。margin日志仍按`main_valid`归约（`models/seqtrack3d.py:6088`），没有只限定“margin目标有效且有恢复需求”的行。训练margin loss则另外乘`motion_acquisition_target_valid`并做递归年龄平衡（`:6046`、`:6051`），所以日志总体均值不能替代困难需求行表现。

初始化bias=-4.6（`models/ct_v2/motion.py:693`）约对应(2.0398,1.0199)m。第2轮可说“纵向有少量扩张，横向输出收缩到几乎最小值1m；尚未展示普遍横向恢复扩张”，不能写“B1没有学习/参数被冻结”。这一实测迹象使按有/无恢复需求分解margin标签、输出和梯度成为高优先级诊断。统计sigma过覆盖与获取margin贴下界可以同时发生，并不矛盾。

## 5. 当前结果能支持的行动

先保留未完成运行供收集首个同轮次官方验证与诊断；此次分析不要求停跑、重启或改动正在运行的代码。已有checkpoint可以开展同权重observation、bounded-always和训练内部拟合策略的闭环对照。

下一轮方法修改建议顺序：有效监督 → 共识两项最小对照 → 根据分母确认获取与需求平衡 → H1死区/长递归。每项单独记录方法身份；不能作为性能等价补丁混入现有perf运行，也不能用旧checkpoint跨新方法初始化正式臂。IO冗余优化可另走等价性验证，不混入方法收益。

本轮没有重新运行9月14日合成探针或全量测试，没有服务器CUDA/原始点云；结论为当前源码复核与已保存反例解释。历史证据见 `artifacts/ct_checks/20260914_code_audit/REPORT.md` 及对应 JSON，完整partial指标由上级汇总报告负责。
