# v35 完整链候选记录

2026-10-02。仅v35的b1、b1_b2、full评测启用；v35 B0、v33/v34/reference旧逐帧格式保持。训练计算、采样、loss、梯度、RNG和accepted提交均不读取本记录。入口在`TrackingEvaluation.add_batch`，正式调用顺序为forward→commit→被动评测；没有额外网络前向。

每个非初始化预测帧在`frames.jsonl`新增`diagnostic_candidates`，schema=`ct_seqtrack.v35.candidate_records.v1`。

| 字段 | 含义与解释边界 |
|---|---|
| candidate_order | 固定q0、mode0、mode1、mode2 |
| boxes_anchor_relative | 4×4，XYZ只减anchor平移，轴仍是世界轴；yaw是绝对弧度 |
| boxes_world | XYZ加回anchor平移；不再次旋转或叠加anchor yaw |
| quality_logits / quality / valid | 四候选完整输出及decoder实际参与选择的资格 |
| geometry | 有效候选的IoU、中心/XY误差、方向及模π轴向误差；无效候选为null |
| selected_index / selected_quality | 实际选中项；必须与原帧selected字段一致 |
| current_valid / sequence_valid | 当前或四帧是否有合法测量；与GT目标点是否存在不同 |
| q0_prior_fallback | 四帧B0测量全空导致q0=prior；不等同当前raw目标点为0 |
| prior | prior框、physical/context有效性、实际获取比例u、方向、mean/sigma/运动学基项和残差范围 |
| mode_formed / modes | B2是否形成模式、最终候选资格、中心、协方差、成员槽/raw ID、成员数和GT纯度 |
| acquisition | 768池采样后有效/目标点数与分区计数；实际搜索域采样前计数可用时一并记录 |
| extension | 最多256槽中仅有效槽的紧凑台账，保留原slot、raw ID、768池索引、真实XYZ、identity、vote、reliability、分区及被动GT标签 |

`extension.capacity=256`，`slots`列出的才是真实有效测量，其余槽是padding。mode成员以原slot/raw ID映射到该表，同一真实点被多个角点读取不算多个独立测量。vote坐标仅用于离线分析，不加入q0的pre-vote输入。

**形成模式与允许选择必须分开。** B1+B2可能形成好模式，但其decoder mode_valid全部关闭；对应候选框为无效占位，geometry=null。不能将无效框当作模式定位结果。Full才实际解码并监督mode。四帧B0全空时q0固定prior，Full测量候选仍可能有效。

`target_labels/target_count/target_purity/geometry`均含GT，仅用于被动诊断，不能作为推理输入、候选选择或状态提交依据。quality目标本身不等于IoU正确概率，不能把0.5阈值失败率直接称概率校准误差。

已有帧级raw/crop/sample、最大搜索可达点数、获取目标点数、coarse/q0几何、支持和记忆写入诊断继续保留。新评测另记实际u搜索域在768池采样前的唯一点/目标点及分区计数，区分：最大范围不可达、实际搜索范围未覆盖、取点遗漏、身份选择遗漏、候选定位差、候选排名错。训练不增加这项几何扫描；合成或旧记录缺少字段时标null，不补0。

可在Full自己的同一端点比较实际选择与q0或候选最佳IoU，分析当帧选择得失；该q0沿Full历史，不替代独立B1+B2递推结果。GT选择的最优候选仅是离线诊断上限，不是正式成绩或可部署策略。
