# v28 Full 数据链与耦合独立审计（2026-09-10）

本审计只读生产代码与已完成 mini 证据；未修改模型、配置或原始 output，也未在本机声称执行 CUDA/full 数据。结论针对一次 **Full 模块、完整 nuScenes、Car、60 epoch、scratch** 诊断运行，不代表完整论文矩阵已验收。

**结论：当前模块接口支持这次完整数据训练；没有发现仅因 mini 切换 full 而必然触发的模型接口阻断。** mini Full 已实际完成各模块从头训练并证明 B0 隔离。但 B0 稳定分数、B2 获取效果和 B3 校准收益仍未验收，不能把接口可运行写成“最佳耦合”或算法已恢复。训练后必须区分 observation 验证和校准后 Full 闭环分数。

## 真实通路与证据

| 边界 | 实际实现 | 判断与限制 |
| --- | --- | --- |
| B0 观测训练 | `models/seqtrack3d.py:8856` 的 dual-stream 事务先禁用插件路由；`5785` 使用独立原 SeqTrack 整批目标；`models/ct_v2/observation_reference.py` 保留全 batch 加权 CE/moving 分母与单次 BC。 | Full 的观测优化不依赖插件输出。继承的 FeaturePointNet 排列、BN、history mask 仍是 v28 注册限制，没有在此次审计改掉。 |
| B0 特征 → B2 | `seqtrack3d.py:3501` 在一次 SegPointNet 前向导出第二层 64 维，转成 `[B,L,1024,64]`；`3821` 逐位断言 current base 点等于 B0 当前 1024 槽。`2800` 经 `B0Observation` 给出 detached current/history 特征。 | 点顺序/帧顺序一致；B2 不再读 FeaturePointNet 的伪逐点布局，不额外执行 Seg/BN。B0 的分割和 BC 监督继续训练这个共享特征源。 |
| 历史预测 → B1 | `seqtrack3d.py:7893–8024` 用按 tracklet/epoch 保存的 RecursiveTrackState、真实时间差、预测历史及输出质量构造 B1 prepass；`datasets/sampler.py:2695` 的 canonical 机制样本按完整端点读取。 | 已禁用 GT 周期重置：`27_formal_base.yaml` 的 `ct_recursive_reseed_enabled=false`。GT 历史框误差仅为训练诊断；当前 GT 不用于部署 prior 或输入尺寸。 |
| B1 → support / B2 | `seqtrack3d.py:2930` 使用实际已获取的 `AcquisitionRecord`（含 CV fallback），而非把可微 learned prior 偷换为已执行获取；`2991` 校验同坐标 support anchor。`datasets/sampler.py:2031` 强制 B1/B2 共用 query dt。 | 物理运动监督与递归 endpoint 误差分开，预期允许已有漂移导致 endpoint 偏离；B1 不是绝对目标位置 oracle。时间先在 `datasets/misc_utils.py:194/219` 转成相对秒，再转 FP32，未把绝对微秒直接压成 FP32。 |
| 原始当前点 → 新增证据 | `datasets/sampler.py:1394` 传递整个 raw B0 crop 的原始 IDs；`utils/ct_search.py:1256` 合并 endpoint/tube/corridor，按原始 ID 排除整个 base crop 并对交叠来源合并 bitmask。 | 严格是 support raw ID 减 B0 raw crop ID，不是减 B0 已采样 ID。crop 内未采到的点或被 B0 兼容分支丢掉的 1/2 点不会伪装成新增证据。 |
| B2 证据/memory | `models/ct_v2/evidence_memory.py:597` 验证 768 槽、唯一 extension ID、base 的 unique mask；`58` 用首帧大小和预测历史框从 3 帧各选 8+4 个 token。 | 最终 256 证据槽，36 padded memory 槽；无效历史和无效点无真实 token；ID=-1 为 padding；重复实际点只计一次。固定点/token预算不随完整数据规模增长。 |
| B2 → B3 | `seqtrack3d.py:1273` 对 v27/v28 实际构造 `models/ct_v2/action_v27.py::B3UtilityUpdater`；`evidence_memory.py:909` 对 v28 返回结构 availability；updater forward 把上游 inputs 全 detach。 | 活跃 B3 没有旧类的 presence 硬门。不要误读 `evidence_memory.py` 内旧 `B3SelectiveUpdater` 的 presence 条件并认定 v28 仍在使用。 |
| action 合同 | `action_v27.py:11` 的 bounded_residual_xy 被 updater、host 的 bounded_always 和监督共用；`utils/v27_training.py:18` 的 structural 规则不读 presence 阈值。 | raw、bounded、accepted 是不同输出；低 presence 结构候选仍可监督/评估。当前 action 半径没有因扩数据而变化。 |
| 优化所有权 | `seqtrack3d.py:3289` 拒绝任何启用模块 frozen parameter，构造单 Adam 的 b0/b1/b2/b3 命名组；`8713` 用 eval/no_grad 观测重新提取机制条件，随后恢复每层训练标志；`8902` 的机制事务恢复全局 RNG 并隔离 B0 BN。 | 没有冻结阶段、跨臂初始化或零学习率阶段；条件 loss 导致个别参数某步无梯度不等于冻结。B1/B2/B3 各自 loss 不反传上游，detach 是合同。 |
| 唯一递归写入 | `seqtrack3d.py:8387` host 只按 canonical slot 提交 observation 到训练状态；H3 在 `utils/v27_training.py:124` 克隆两支状态后干预，不写主状态。评测由 `utils/v27_evaluation.py` host 流程提交实际 final。 | 训练为 observation-recursive shadow learning；部署 Full 校准后才把接受的动作反馈到以后 crop。两个阶段语义明确但存在分布差异，后续必须真实闭环校准/评估。 |
| 校准缺失 | `utils/action_calibration_v27.py:330` 首先安装 never，checkpoint/config/场景/代码身份校验通过才安装策略。 | 缺失/过期策略精确回退 B0，不阻止 B1/B2/B3 学习。完整训练期间未校准验证也将是 B0 分数，不能据此评价 Full 增益。 |

## mini 实证及不能推断的结论

来自 `../20260910_v28_mini_three_arm/REPORT.md` 及该目录 full_review：

- B0/42 与 Full/42 全部 75,720 个观测 loss、末轮 B0 参数/BN/Adam 一致。B1/B2/B3 各完成 18,000 个机制事务，首个非零插件 loss 在 epoch0。证明这条组合和隔离路径已经在实际 CUDA 长跑运行。
- B0/42 final 为 45.200/47.243，B0/52 为 52.876/64.478；不能认定基线稳定恢复或仅靠随机性就解释差距。完整数据一次运行可以提供泛化/供给诊断，但不能单独确证相对 SeqTrack 的因果增益。
- 192 个 endpoint 共 22,715 个 global novel target 机会，只取得 2 帧的 4 点；4 点后续均保留。首先要查实际获取中心、边界范围与递归漂移，而非只改 relation top-k/B3 阈值。global novel 的分母含 bounded support 根本够不到的远处目标点；这个微召回本身不是原始 ID 或坐标实现错误的充分证据。
- B1 位移误差 0.212m 与递归 endpoint 误差 2.770m 可以同时成立，因为前者不包含此前递归漂移。强行把二者混成同一个误差会给获取模块错误归因。
- 未校准 Full accepted=0 是 fail-closed 预期行为；一次完整训练后仍需每个 58/59/60 checkpoint 独立内部校准和官方 val 闭环评测。

## 完整数据规模下的风险与现有测试

未发现模型尺寸/类别数在完整数据下改变当前固定张量合同。稀疏、空帧、混合历史长度和不满 16 的机制尾批已有测试；但本机没有 full 原始数据，不能排除服务器数据缺文件、资源不足、异常记录或尚未遇到的 CUDA 运算分支。

机制状态和诊断 rows 在 `seqtrack3d.py:7758` 每 epoch 清空，不是 60 epoch 无界累积；单 epoch CPU 内存仍随总端点数增加。B2 固定 768→256 及 memory36 的 GPU 单 tick 预算不随总场景数扩大；v4 在罕见机制尾部可能把多 tick 放入同一观测步，所有 tick 图在 unified loss backward 前仍存活，需由真实 full preflight/运行监控判断实际峰值。

mini 已发现 `scene_id=unknown` 导致 summary scenes=1 的日志元信息缺口，位置 `utils/v27_evaluation.py:85`。这不改变整体 S/P，但会妨碍可信的逐场景分析。完整数据长期运行之前应由主审计核对原始 dataset frame 能否提供 scene ID，并修复日志映射；不要从 unknown 推断只评价一个场景。校准工具对场景 role 有硬验证，必须使用其有真实 scene identity 的数据通路。

已有可定位回归（这里只读列举，不冒充本次执行结果）：

- `tests/test_ct_v28_b0_reference.py`：独立 oracle 整 batch loss/BC 梯度/Adam；多臂两步 B0/BN/RNG；空输入；Seg64 frame/point 对齐。
- `tests/test_ct_v28_audit_host.py`：真实 dual-stream training_step 和 Adam hooks 审计开关不改数值。
- `tests/test_ct_v28_data.py`：全数据 shuffle、原重抽、0/1/2 点兼容、raw crop 差集、历史 prior 和首 query。
- `tests/test_ct_v28_components.py`：低 presence 仍为结构候选、无效几何拒绝、export/report结构集合一致。
- `tests/test_ct_v27_host.py`：实际 host 的 acquisition/CV fallback、空支持、B2/B3 梯度所有权。
- `tests/test_ct_v27_heterogeneous_batch.py`、`test_ct_v27_mixed_slots_training.py`、`test_ct_v27_multitick_backward.py`：异构历史/空点、满/尾批、真实多 tick 反向和 B0 BN 隔离。
- `tests/test_ct_v27_sampler_coverage.py`：每个合法端点一次、长轨迹、多 tick、末尾单槽完整覆盖。
- `tests/test_ct_v27_actions.py`、`test_ct_v27_training.py`：实际效用标签、H3不写主状态、校准闭环反馈、场景身份/策略错误 fail-closed。

本次没有发现必须先改动算法才能启动单次 full Car 的证据；最合理的运行身份是“完整数据诊断首跑”，保留当前 B0/模块合同和全部原始结果。得分稳定恢复、最优获取和完整论文对照仍是之后验收事项。
