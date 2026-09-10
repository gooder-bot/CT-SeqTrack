# CT-SeqTrack 当前状态（2026-09-10）

当前任务为v29实施与三臂完整nuScenes Car/seed42/scratch60实验。使用
`29_b0_nuscenes_full.yaml`、`29_full_cfc_nuscenes_full.yaml`、`29_full_gru_nuscenes_full.yaml`。
实现说明见[合同](docs/CTSEQTRACK_V29_IMPLEMENTATION.md)，具体本地结果见
[验收](docs/CTSEQTRACK_V29_LOCAL_VALIDATION.md)，启动顺序见[服务器命令](docs/CTSEQTRACK_V29_SERVER_RUNS.md)。

- [x] 独立v29方法身份、帧布局/mask/稀疏输入、短roll-in及监督拆分。
- [x] 获取Z合同/novel可达诊断、机制accepted状态和即时B3效用接口。
- [x] 当前全量本地回归571 passed/14 skipped；compileall、diff及服务器命令语法检查通过。
- [ ] 服务器真实CUDA/同卡100步逐位一致性/epoch边界恢复。
- [ ] 三臂完整60轮、58/59/60各自评测与两Full逐checkpoint策略拟合。
- [ ] 同协议原SeqTrack对照、独立模块消融、多seed或额外数据集（另行登记）。

尚未通过分数恢复/涨分验收，不用旧481项测试或seed52高分替代v29证据。

## 历史v28结论与旧排程（当前执行以上述v29为准）

最新判定：三组60轮已完成；B0/42=45.200/47.243、B0/52=52.876/64.478，
Full/42未校准=45.200/47.243（动作0）。seed52达到历史目标但seed42仍不足，
**B0稳定恢复未通过；先补评、校准与获取诊断，不直接铺开完整数据矩阵。**
详见[三组结果分析与数据表](artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。

用户随后明确希望先做一次完整数据60轮诊断再分析。当前按 **Full/Car/seed42、scratch60轮**
准备，已有配置支持；不因mini未达标阻止这次单组诊断。完整数据资源与校准评测安排见
[接口审计和服务器命令](docs/CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。

- [x] 核查full350/150及内部17/18、自然更新数、B0—B3隔离和完整端点链路。
- [x] 修复v28真实preflight的缺省preloading字段；默认按需读取。
- [ ] 服务器执行完整数据preflight与16步Full真实事务，核对实际调度/梯度及资源。
- [ ] 一组Full Car完整数据60轮；保留final/58/59/60及全部诊断，不冻结、不跨run初始化。
- [ ] 每个Full checkpoint独立内部校准、官方val评测，并补同一权重的observation控制；
  集中分析B0后期退化、B2获取供给和B3动作。full不能沿用mini分数目标作验收线。

以[正式协议顶部](docs/EXPERIMENT_PROTOCOL.md)和[v28实施](docs/CTSEQTRACK_V28_IMPLEMENTATION.md)
为当前定义。上轮GPU1 B0/42、GPU2 B0/52、GPU3 Full/42三组mini已完成；
以下历史章节不代表最新实验状态。处理重启或训练异常先读
[CUDA故障记录](docs/CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md)。

- [x] 新增28五臂mini/full与共享实现reference配置，固定seed42/60epoch/bs16/workers12/val5。
- [x] 注册完整candidate总体shuffle、reference_batch单次BC目标、seg_second64_v1点特征，
  严格确定性与环境记录；保留train.v4、单Adam、scratch及模块所有权。
- [x] 新增v28 preflight、首轮B0/后续计划矩阵、版本化校准与工程入口约束。
- [x] 9月8日v28实施全量回归快照453 passed、12 skipped；compileall/diff通过。独立原算式、
  六配置B0两步更新及真实跨臂审计首步逐位一致，详见[本地验收](docs/CTSEQTRACK_V28_LOCAL_VALIDATION.md)。
- [x] 已登记`28_b0_seed52.yaml`，三组均scratch/60epoch/bs16/workers12/val5，
  正式输出使用`output/日期-时间-28_模块-mini_car_seedXX_60ep_bs16/`新目录。
- [x] 修复三组共同的CUDA空间NLL报错：原三维log_softmax后转二维NLL，保留整批
  加权CE目标；CE合同进入配置和resume身份。专项47 passed/1 skipped。
- [x] 修复Full的AP浮点cumsum报错：二值标签int64累计后转回浮点，两处同类路径
  一起修复。指标不参与总损失；专项15 passed/1 skipped，严格确定性保持开启。
- [x] v27五臂实际60轮训练已完成；9月7–8日报告与原始日志/逐帧CSV已复核。
  成绩只属于内部dev，Full缺校准且动作0；这不是完成官方mini_val验证。
- [ ] 服务器真实preflight、同卡B0/B0/GRU 100-step逐位对齐与Full epoch边界连续/恢复验收。
- [x] 从provenance、日志和checkpoint确认两次修复已同步，三组从epoch0完成60轮，
  各75720观测更新，保留严格确定性，无本次RuntimeError；不等同于独立重复/恢复验收。
- [x] 完整报告两seed final与闭环失跟；Full/B0 seed42全部观测loss、末轮模型与Adam
  一致，插件无冻结且各18000机制事务，未发现本次插件污染B0。
- [ ] 补官方mini_val的58/59独立评测，按固定58/59/60算late-3；不能用50/55/60。
- [ ] 定位seed42后期闭环退化；对照独立、协议匹配的SeqTrack实现，不能把共享
  `28_seqtrack_reference.yaml`标签重复当作独立算法对照，不能挑seed52宣布稳定恢复。
- [ ] 核查Full获取阶段的新目标供给：192个有机会帧、22715个crop外目标点，仅2帧4点
  进入support；后续pool/selected保留4点，优先查获取，不能先假定是B3阈值问题。
- [x] 修正评测日志的`partition=dev`与`scene_id=unknown`元信息，保留原始输出；
  本轮实际官方mini_val两场景/106轨迹/2285帧，不能将summary scenes=1当真。
- [ ] Full每个checkpoint独立校准；B0结果可解释后再扩展其余mini臂，参考实现身份单独披露。
- [ ] mini验收后再考虑完整nuScenes五类30运行矩阵；所有启用参数从头训练。

v28三组的60轮结果已读入；seed42未达到健康目标，late-3及校准Full仍缺，
不声明基线稳定恢复、稳定涨分、SOTA或因果time/memory收益。
工程短跑和CPU合同检查不能替代真实数据/CUDA验收。具体命令见[v28服务器流程](docs/CTSEQTRACK_V28_SERVER_RUNS.md)。

# 以下为 CT-SeqTrack v27 历史状态（2026-09-05）

当前实验定义以[正式协议](docs/EXPERIMENT_PROTOCOL.md)顶部及
[v27方法](docs/CTSEQTRACK_V27_METHOD.md)为准。下面v26正文保留历史状态，
不能将其中“已完成”解释为v27正式实验已经完成。

- [x] 新增v27 B2选点前局部几何、role条件、targetness-only voting、尺寸vote范围、
  mode绝对统计与134d selected-presence；新增18项CPU边界测试通过。
- [x] 新增v27五臂mini/full配置及关闭CT模块的SeqTrack外部架构参考；旧配置不覆盖。
- [x] 新增六运行/五类三十运行的可审阅矩阵工具，默认不启动训练；场景划分、原始点身份和矩阵16项测试通过。
- [x] 修复机制iterator取batch/结束检查的全局RNG隔离；新增7项隔离测试，包括实际host事务CPU壳两步B0参数/梯度/BN/Adam对齐及真实GRU/CfC公共层初始化。完整CUDA更新等价仍待服务器验证。
- [x] v27机制sampler完整覆盖不等长轨迹尾部；train.v4支持同一观测事务内有序多tick，保持B0更新预算；preflight实际遍历检查endpoint、顺序和batch数。
- [x] 完成v27数据/模型/损失/校准入口集成和本地检查：290 passed、1 skipped；compileall与diff检查通过。范围与服务器待验收项见[实施记录](docs/CTSEQTRACK_V27_IMPLEMENTATION.md)。
- [x] 二次审计修复稀疏GT重采样、分类标签dtype、epoch文件保存顺序、续训generator恢复、Full RNG extra_state加载及旧诊断标签；实际六臂网络、Full完整训练事务、五臂B0两步对齐和真实Lightning连续/恢复对照通过。详见[训练就绪记录](docs/CTSEQTRACK_V27_TRAINING_READINESS.md)与其中最新测试结果。
- [x] 按本次mini启动要求同步workers4/每5轮dev诊断，新增运行resolved_config.yaml；修复验证间隔5的last保存与延迟val-loader恢复，附GPU2/GPU3五臂[后台命令](docs/CTSEQTRACK_V27_MINI_LAUNCH.md)。
- [x] 修复服务器epoch0的异质轨迹合批KeyError：B1 margin四计数固定schema，补九类状态与真实batch16混合轨迹反传测试；见[报错验证](artifacts/ct_checks/v27_mixed_batch_fix/summary.json)。真实100-step与60轮继续待服务器验证。
- [ ] 服务器验证原始point ID、0/1/2点、所有endpoint、B1实际获取、B0更新一致性、
  所有启用参数组梯度、resume与耗时；工程checkpoint不进入正式实验。
- [ ] 从epoch0运行mini Car五臂，seed42、60epoch；报告final与58/59/60。
- [ ] 每个Full checkpoint单独做v27真实闭环策略拟合和锁定dev诊断，再评估官方mini_val。
- [ ] mini分析后启动full的Car/Pedestrian/Truck/Trailer/Bus，每类五臂独立scratch。
- [ ] 报告外部SeqTrack参考、完整分母S/P、精确几何审计、身份/获取/voting漏斗、
  多类别局部邻域与vote饱和、时间/内存成本及配对不确定性。

full使用350个train_track scene全部训练；17/18scene内部阈值拟合/诊断也参与参数训练。
官方150个val scene只用于评估。v27暂无正式训练结果，不声明涨分、SOTA或因果收益。

# 以下为 CT-SeqTrack v26 历史状态

本文件只记录状态和下一步；实验定义以
[docs/CTSEQTRACK_V26_METHOD.md](docs/CTSEQTRACK_V26_METHOD.md) 为当前协议来源；
v24/v25 仅作为只读历史证据。

## 已完成

- [x] 实现 observation-anchored bounded adaptive shell、三帧 causal backup corridor 与 B1-invalid 固定 `2m/1m` CV fallback。
- [x] 实现 deterministic 768-point novel pre-pool、三位 source bitmask、relation/FPS/stateless 256-point selection 与 K=3 robust consensus voting。
- [x] counterfactual schema 升级为 v3，修复 nuScenes wlh/local-axis 和 raw-vs-novel 支持统计，加入逐层集合不变量与 v26 报告工具。
- [x] 恢复 B3 helpful/harmful action-confidence gate，加入 consensus/covariance/inlier/margin 特征与 calibration/dev promotion artifact v2。
- [x] 新增 SeqTrack-strict、B0、B1-GRU、B1-CfC、Full-B3、Full 的 v26 full-nuScenes seed42/60-epoch scratch 配置；集成主臂固定 GRU，CfC 仅作为 B1 backend 诊断；所有启用模块从第一步参与 unified Adam。
- [x] 修复 v26 B1 acquisition-margin 事务复算遗漏、corridor 长度截断中心偏移、Full action-export 非完整 checkpoint 误载入风险。
- [x] 正式训练显式固定 `min_epochs=max_epochs=60`、`max_steps=-1`、单卡、全量 train/val batch，并保存 `last.ckpt` 与 epoch 58/59/60。
- [x] 新增零训练步 `preflight_v26_full.py`，检查完整数据根、CUDA/PointNet++/Lightning、模型构造、优化器参数组和冻结状态。

- [x] 固定 `main@001951a` 的 tracked-file、22 配置、测试和 `output/` 保护基线。
- [x] 正式 v24 配置脱离历史 21/22/23 继承链，resolved-config 逐键一致。
- [x] 删除 93 份非活动 YAML；保留 20 个正式入口、2 个 B4 入口和必要基础配置。
- [x] 删除非活动且可由 `001951a:<path>` 恢复的历史资产/文档；`output/` 指纹未变。
- [x] 删除 73 个被忽略的旧服务器输出、旧协议 manifest 和 run log（67,986,460 字节）；这些文件原本不在 Git 中。
- [x] 工具面收敛为八类正式任务，删除 73 个旧 M/TWC/CRPA/Search/Gate/replay/preflight/promotion 工具。
- [x] 删除旧 replay 专用测试并移除 preflight/promotion 门禁测试，保留当前耦合与行为合同测试。
- [x] B0--B3 所有权、candidate0、extension-only、calibration 和 scratch 合同保持不变。
- [x] 历史数值压缩为证据索引，负结果和 claim boundary 未被隐藏。
- [x] 新增 `25_b0/b1/full_minus_b3/full` 及 full-nuScenes 配置，不覆盖 v24。
- [x] B0 observation 改为无状态 shuffle/candidate/点采样，四候选损失固定为 `0.5, 1/6, 1/6, 1/6`。
- [x] 四臂统一 `ct_seqtrack.train.v2`，mechanism iterator 惰性创建并隔离 observation RNG。
- [x] v25 使用单 Adam 自动优化和 B0→B1→B2→B3 互斥参数组；重复 B0 mechanism 前向为 no-grad、BN 隔离。
- [x] checkpoint 增加 v8 runtime/RNG/B1-backend/loss/参数组/update/hash 审计和 CUDA 分阶段审计。
- [x] 修复 B1 gap2/gap4 transaction、归一化残差长尾监督、detached-mean beta-NLL 与独立 calibration/dev 校准；固定 B2 geometry 不变。
- [x] 新增参数匹配、无外部依赖的 CfC 插件与 `--b1-backend gru|cfc`，默认仍为 GRU。
- [x] 本地合同测试、全量 pytest 和 compileall 已通过；准确数量以本次交付记录为准。

## 服务器验收（待执行）

- [ ] 从 epoch 0 独立运行 v26 mini B0、B1-GRU、B1-CfC、Full-B3、Full，Car、seed42、60 epoch；不从 smoke/其他 arm checkpoint 初始化。
- [ ] mini 完成并分析后，再从 epoch 0 独立运行匹配的完整 nuScenes 五臂；SeqTrack-strict 保持为单独登记的外部参考。
- [ ] 对 final 与 late-3 的每个 Full checkpoint 分别导出互斥 calibration/dev rows、完成 promotion，并生成不可跨 checkpoint 复用的 artifact。
- [ ] 报告 schema-v3 geometry/sampling/voting/B3 漏斗、final/late-3 tracking、耗时、显存及 tracklet-paired bootstrap CI。

本机有可用 GPU，但没有发现正式 nuScenes 数据根，且当前 Python 环境缺少
`easydict`、Lightning、torchmetrics 和 nuScenes devkit。上述真实门禁完成前，
不得物理删除 `SEQTRACK3D` 中仍承载兼容行为的 dormant source branches。

当前本地机器不具备真实 nuScenes/完整 Lightning 环境，因此服务器 preflight 未运行前只能给出条件式 GO。任何工程或其他 arm checkpoint 都不能用于正式实验初始化。

## Claim guardrails

在注册实验完成前，不得声称稳定涨点、SOTA、物理时间因果贡献、memory 贡献或
分布无关风险保证。B4 默认关闭，不进入当前主结果。
