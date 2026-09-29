# CT-SeqTrack 实验协议

## 当前 v35：四组恢复与新增piecewise，共五组及条件第二seed

2026-09-29更新，覆盖此前四组总数。用户确认的 [v35综合方案](B0_V35_INTEGRATED.md) 保持：首测精确初框、合法历史条件与候选局部真实点读取作为统一版本；保留现有loss、全局pooling、W=1/4/4/8、十轮课程及112 reserve。所有正式模型从epoch0随机初始化建立运行，batch16、FP32、每轮19,108行/1,195更新，完整60轮71,700更新；相同运行的完整epoch恢复不构成新实验。

原seed42四组quarter/half/normal/scaled（初始LR=2.5e-5/5e-5/1e-4/1.5e-4，20/50轮后各乘0.1，无warmup）已完成36轮。9月29日13:33只读确认用户已暂停四组，各自完整checkpoint均为036。用户将更新同一活动项目，保持各自配置和log_dir，从036恢复、由第37轮继续，物理GPU仍0/0/1/1；原manifest和旧产物不得改写。

用户明确只追加第5组piecewise，在GPU1从epoch0随机初始化：第1–20轮5e-5、第21–50轮1e-5、第51–60轮5e-6；`lr_schedule=piecewise`、`lr_stage_values=[5e-5,1e-5,5e-6]`、`lr_milestones=[20,50]`、warmup0。完成20/50轮后的调度分别作用于第21/51轮首次更新，其他单组参数不变。原half前20轮相同，后两阶段分别为5e-6和5e-7，因此piecewise分别为其2倍和10倍。这比较的是完整调度，不能归因为单一固定LR变化，也不能从half checkpoint转换配方续训。

首阶段共五次、300epoch、358,500更新。仅原硬门通过后，锁定胜出完整配方补seed52，并运行独立SeqTrack正常1e-4、20/40衰减的seed52；数据划分始终seed42。保留这两次条件复验后，合计上限七次、420epoch、501,900更新；不自动执行后续训练。五组均失败则停止，不增加其他窗口、LR或warmup对照。已有R/C/W-quarter复用，piecewise不使用任何旧checkpoint初始化。

硬门仍为总体final60/late-3四项≥固定R、固定31条曾移动轨迹全程657预测帧四项≥旧C。五组完整后，通过者按final60 Success、Precision降序，完全相同时按初始LR升序；初始LR也同为5e-5时，预登记原half优先piecewise。旧四组工具`compare_v35_b0.py`保留；新增`compare_v35_piecewise_b0.py`要求原四组参数及`--piecewise`均完整，才作五组评选。第二seed锁定完整调度，其同seed对照不替换固定R/C门槛。原始帧重算，初始化按原口径计入总体。严重漂移采用下述方案1：原硬门结果与漂移风险复核分开记录，不要求其他切片全胜。

v35身份为ctseqtrackv35/ct_seqtrack_v35，schema=ct_seqtrack.joint_identity.v35，checkpoint runtime=ct_v35_runtime；跨版本、配方、seed恢复禁止。此次原四组迁移仅接受登记的精确before/after源码关系，旧配方计算不变；保留最初`run_manifest.json`并在`resume_manifests/`追加恢复来源，不通过改写旧manifest或关闭源码检查放行。正式诊断开启且不得改变学习。新结构只登记mini B0正式配置，Full保留工程合同。此次只新增调度，不把重复CUDA batch作为前置；服务器由用户上传、恢复和启动，代理只读。具体文件和命令见 [piecewise与四组恢复说明](CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)。

### 严重漂移方案1：本轮要优化，验收不新增数值硬门

- **优化范围明确包含漂移与持续失跟。** 综合方案须分析首次缺测、错误历史传播、coarse/fine定位及后续crop偏移等有证据的成因，并给出对应修改。不能因“不新增硬门”而只记录风险或将修复推迟到Full；可以由已有综合机制共同解决，不要求新增独立防漂移模块。
- **原总体与移动数值硬门不变。** 不新增“>10m预测帧数在final60和late-3均不得超过旧W-quarter”的通过条件，不因单项漂移计数增加而把原硬门的通过结果改为失败。不得挑选最佳epoch、改用较弱参考或事后增加漂移阈值。
- **固定报告风险证据。** 沿用既有逐帧记录，报告中心误差>5m、>10m的预测帧数、涉及轨迹和未恢复失跟，与旧W-quarter比较并保留固定R参考。final60与late-3分别报告；late-3计数取58/59/60各轮计数的均值，同时保留逐轮结果，不合并成一轮计数。预测帧不含初始化，失跟起止、持续时长和右删失沿用下方定义。
- **风险标记是复核入口，不是新增淘汰门。** >10m计数在final60或late-3增加时，标记风险、暂停Full推进并讨论；比较工具的风险/暂停字段不等于综合方案失败。结合受影响轨迹、持续时长和末轮一致性复核：小幅、孤立波动不自动否决，明显、持续的严重退化必须处理，不能被总体涨分掩盖。其他长尾或未恢复失跟的明显退化同样应报告，不能仅看>10m一项。
- **控制实验成本。** 保持综合修订和已登记训练预算；不为每个漂移疑点单独完整训练，不为压过某个计数自动扩大搜索。仅在复核结果会改变下一步决策或具体机制结论时，补必要诊断、回退或针对性归因。

本节仅澄清本轮优化目标及验收解释，不改变v35已登记结构、配方、种子和训练次数，也不表示漂移问题已经解决。实际收益需正式结果验证，旧版本协议和output/artifacts继续保留。

## v34 历史协议：八组mini单seed，S/W结构与学习率对照

八组已于9/28完成复盘，原待上传/待启动文字为历史；[正式结果](../artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)确认尚无候选通过双门槛。

2026-09-26用户最新明确批准八组，以 [B0_V34_LR_GRID.md](B0_V34_LR_GRID.md) 为当前配方协议，[B0_V34_CONTEXT.md](B0_V34_CONTEXT.md) 为结构细则。SeqTrack正常/减半，S/W各正常1e-4、原C的5e-5、推荐探索2.5e-5。参考保留20/40衰减，S/W保留20/50、窗口3/4；均无warmup、seed42 scratch60、每轮19,108行/1,195更新。旧R/C/旧v32 B0保留，新参考由用户重跑两组。总体final60/late-3四项达到已登记旧R且固定曾移动轨迹四项不低于旧C才通过，新参考不重选门槛。

v34独立模型/schema/checkpoint身份；v33/reference原配置与行为保持，不能跨版本、换窗口或LR续训。由用户上传、一次真实CUDA batch检查、GPU0跑R正常＋S三档、GPU1跑R半LR＋W三档，代理服务器只读。所有GT边界、唯一点、共享头与Full梯度合同继续适用；v34 W的1/4/4/8是明确登记的窗口例外。

下方为v33历史协议，保留用于复现；其中旧“等待启动”“三/四组”“1/3/3/8固定”不覆盖上述v34新安排。历史结果与证据不改写。

## v33 历史协议

本版比较综合 B0 修订与独立 SeqTrack，保留已有 v32 结果用于历史对照。v32 冻结源码为 Git `ddcb1a1`，v31 为 `b1d886e`；旧 checkpoint 不进入 v33。

用户最新安排为：本地完成修订，由用户上传并启动四组 mini seed42；本轮代理对服务器仅做只读操作，不自行上传、安装、启动或停止任务。此前服务器工程检查已经完成，不作为反复追加的启动前置步骤。

2026-09-25进度更新：原R/A/B/C/D已由用户启动，最近观察仍在运行，完成状态以各自完整评测证据为准。本轮追加E只在本地实施，未访问或修改服务器，上传与启动仍由用户执行。

## 四组正式配置与预算

| 组 | 配置 | 物理 GPU | 第 1–20 轮 | 第 21–40 轮 | 第 41–50 轮 | 第 51–60 轮 |
|---|---|---:|---:|---:|---:|---:|
| R | `33_seqtrack_ref_mini.yaml` | 0 | 1e-4 | 1e-5 | 1e-6 | 1e-6 |
| A | `33_b0_mini.yaml` | 0 | 1e-4 | 1e-5 | 1e-6 | 1e-6 |
| B | `33_b0_late_decay_mini.yaml` | 1 | 1e-4 | 1e-5 | 1e-5 | 1e-6 |
| C | `33_b0_half_lr_mini.yaml` | 1 | 5e-5 | 5e-6 | 5e-6 | 5e-7 |

- 数据为 nuScenes-mini Car，固定 8 场景训练、2 场景评测划分，partition seed42、frame stride1。mini 验证沿用固定评测划分，不根据验证结果挑选最佳轮次。
- 四组均 seed42、scratch60、batch16、workers4、FP32；每组仅一个设备。使用 Adam，betas=(0.5,0.999)、eps=1e-6、weight_decay=0，每 batch 一次优化。
- R/A 使用 StepLR(step_size=20,gamma=0.1)，B/C 使用 MultiStepLR(milestones=[20,50],gamma=0.1)。表中为各轮训练实际使用的学习率。
- 每轮 19,108 个名义训练行、1,195 次优化，尾批 4 行；60 轮为 1,146,480 行、71,700 次优化。B0 每个合法预测端点覆盖四个分支；R 保留原 teacher4 和无效历史重采样，名义曝光不等于实际唯一端点曝光，实际替换记录写入训练审计。
- A/B/C 的网络、初始化种子、loss、数据、分支和预算相同，仅修改注册的学习率配方。R 是独立原 SeqTrack 网络与原监督，不适用生产 B0 的 loss/递推合同。
- 每 5 轮验证；每 2 轮以及最后 58/59/60 轮保存完整 epoch checkpoint。训练后自动评测 58/59/60，每个 checkpoint 重置相同 seed 与 test loader RNG，报告 final60 与 late-3。
- 评测分母为 106 条轨迹、2,285 帧，其中初始化 106 帧、预测 2,179 帧，metric_mode=`benchmark_compat`。初始化帧每轨迹仅计一次。
- 本轮四组是单 seed 证据，不得称为多 seed 稳健性验证。nuScenes full、KITTI、CfC/GRU 与时间控制接口保留，不代表本轮运行这些实验。

已完成 v32 R 的训练与指标仍可作为复用依据；用户当前明确选择重新运行 R，最终四组比较使用本次 R。重跑 reference 使用 v33 host 身份和原 reference 网络/数据/loss，不加载 v32 checkpoint。

2026-09-25追加学习率向上探索D：`33_b0_scaled_lr_mini.yaml`，相对B全程乘1.5，1–20轮1.5e-4、21–50轮1.5e-5、51–60轮1.5e-6，物理GPU0。其余模型、loss、数据、初始化与预算相同；原R/A/B/C安排保留。D与B固定final60和late-3比较，具体命令见[追加D运行说明](CTSEQTRACK_V33_SCALED_LR_GPU0.md)。

2026-09-25追加E：`33_b0_x3_lr_warmup_mini.yaml`继承B，设置`lr=3e-4`、`lr_warmup_steps=2000`，物理GPU1。第1–2000次优化使用`3e-4 × 更新序号 / 2000`，从1.5e-7升至3e-4；之后保持峰值至第20轮完成，第21–50轮为3e-5，第51–60轮为3e-6。warmup包含在原71,700次更新内，模型、loss、数据、seed42、scratch60、batch16、workers4、FP32与Adam其余参数不变；原R/A/B/C/D配置保持。E相对B同时改变LR尺度与warmup，只能检验联合配方，不能独立归因。E固定比较final60和late-3，不追加训练预算，上传与命令见[追加E运行说明](CTSEQTRACK_V33_X3_LR_GPU1.md)。

## B0 几何、监督与递推合同

公开框中心是减去 anchor 中心后的世界轴 XYZ，yaw 是绝对世界 yaw，size 使用首帧固定尺寸。B0 几何在内部换到 anchor-local，不重复平移；B1/B2 原始物理量保留世界轴。GT 当前尺寸只用于相应监督和评分。

共享 decoder 的角点 query 为 `[corner_xyz, time, detached output_reference_xyz]`。q0 直接预测 anchor-local 最终中心；history/modes 输出 seed 残差，仍使用共享定位/质量头。q0 的 coarse 框用于角点 query；mode 输出中心保留 live 梯度，mode query 与输出参考中心输入 detach，历史框 detach。yaw 仍为 seed yaw 加角度残差，空序列回退维持既有规则。

BC 监督所有唯一真实点，包括没有 GT 前景的背景帧；current/history 两组等权，缺失组重新归一化。padding 不参与损失。history 定位先在端点内对有效历史取均值，再在端点间取均值；不因历史数量多而放大某个端点权重。coarse/main 中心系数 2、yaw 系数 10；history 中心 0.2、yaw 1；seg 总系数 0.1、BC 1、quality 0.5。日志中的已加权分项仅用于观测，不再次进入 loss_total。

唯一 raw ID、真实测量 mask、soft foreground detach、稀疏空槽不复制点等合同保持。四分支窗口 1/3/3/8，课程 10 轮；每轮 112 个预留单步窗口用于尾部安排。从首次 drain 或不足额 batch 起，B0 使用已有 running BN 统计，affine 与共享特征仍可学习。每端点仅一次 forward，不为窗口前缀增加推理或优化。

训练窗口 seed 使用共同平移与 yaw 偏差，不把人为扰动变成历史物理速度；合法历史不足通过 mask 表示，不复制不存在的历史。eval 递推只从首帧 GT 初始化，不用当前 GT 重置。接受输出后唯一状态所有者提交 detached pose，后继读取该状态；worker 不持有模型或递推状态。

当前修订不增加动静 hold 策略、GT 引导裁剪或低点数复制补齐。Full 共享接口保持，B1 历史运动边 `history_pair_valid` 的现有几何资格不改，不额外加入前后 supported gate。启用参数不得冻结，BN 统计隔离与 detach 不等于冻结参数。

## 可信状态与记忆

accepted 预测框内（scale=1、offset=0）至少 3 个唯一 predicted FG，且其前景概率均 >=0.5，定义为 strong；strong 再满足 selected_quality>=0.5，定义为 supported。strong 刷新最近强观测时刻；supported 更新可信位姿和最近支持时刻。两者不依赖 memory.update 的返回值。

框外高 FG 点不进入 memory，也不强制重标为 BG；原低 FG 背景点仍可按原记忆预算写入。每个新 supported anchor 仅能从相邻可信帧、且符合原 innovation 几何条件的速度对更新 trusted_velocity；换 anchor 而没有合格相邻对时，速度清零且 trusted_velocity_valid=False。没有新 supported 时保留原可信 anchor、速度和有效标志。

合法 GT 或共同偏移 seed 的 pose 与速度初始化保持：最后两框时间差有效即可建立初始速度，不额外要求两帧均有足够前景。seed 的 strong 则由各帧 raw 点与实际输入框决定；若没有 strong，时刻为 None，weak_age 从本窗口最后 seed 的初始化时刻算起。last_supported_time 对应合法 seed pose 初始化时刻，不把该约定解释成观测前景已充足。

## 被动诊断与结果解释

评测记录 raw/crop/sampled 目标点数及点云规模，区分空裁剪、纯背景裁剪与原始目标缺失；记录 coarse/fine 同帧几何、支持状态、速度有效性与重置原因。moving 定义为相邻 GT 的 XY 位移 >=0.15m，仅用于离线分组，不进入模型或 accepted 决策。

失跟事件由 IoU<0.1 开始、IoU>=0.5 结束，恢复帧不计入失跟长度；中间 [0.1,0.5) 帧仍属于该段。未恢复段为右删失，不能作为已经恢复的样本计算恢复均值；分别报告已恢复长度、P90、未恢复数量及持续长度。

固定按 tracklet/frame 键比较各组，raw 条件与旧 B0 裁剪条件分别呈现。裁剪是旧 B0 递推结果的内生条件，分组差异不等于随机化因果效应。不得让每个新模型按自己的裁剪筛不同样本来宣称机制收益。旧结果缺少诊断时报告覆盖率，可显式使用已留存补充证据，不静默补零或扩充真实覆盖。

A−旧 B0 检验整个综合修订；B−A 检验第二次学习率衰减时机；C−B 检验步长尺度。达标要求至少一组在 final60 和 late-3 的 Success、Precision 四项均 >= 本次 SeqTrack R。若多组达标，先比较 final60 Success，再比较 Precision；完整保留三组结果，不用各自最佳 epoch 替换注册终点。未达标也必须如实汇报，不以个别探针或 oracle 替代训练结果。

## 身份、恢复和证据保护

schema=`ct_seqtrack.joint_identity.v33`，experiment family=`ct_seqtrack_v33`，checkpoint runtime key=`ct_v33_runtime`。网络、种子、预算、优化配方、时间语义等进入 config SHA；路径和日志位置可迁移。`--init_checkpoint` 禁止，正式任务不得使用工程权重；checkpoint 仅用于相同身份的完整 epoch 边界恢复或评测。

入口在任何运行 metadata 写入前校验 checkpoint 与已有目录身份。合法同 run 恢复保留首次 resolved_config/run_manifest；独立 `--test` 使用新空目录，不覆盖原训练或 reference 证据。冻结 reference 的原网络、teacher、原始数据和评分实现有独立 source manifest；不能把修改生产 B0 说成 reference 算法发生变化。

此前服务器快照检查为 **314 passed、1 skipped**，Python3.9.19、torch2.0.1+cu118、pytorch_lightning2.0.2；真实 CUDA batch 的 forward/backward/Adam/commit 已通过且未保存 checkpoint。证据保存在 `artifacts/ct_checks/20260924-190116_v33_implementation/`。后续被动汇总增量单独记录本地验证，不声称最新源码已逐文件完成服务器复验；不重复追加相同预检，也不把检查通过写成正式四组完成。

本轮不增加 Full 独立诊断，不改写既有 output/artifacts。新工程证据使用 artifacts/ct_checks 下独立目录。当前未完成四组正式对照前，不宣称 v33 涨分、达到 SeqTrack、SOTA 或时间/记忆因果收益。实现说明见 [B0_V33_REPAIR.md](B0_V33_REPAIR.md)。
