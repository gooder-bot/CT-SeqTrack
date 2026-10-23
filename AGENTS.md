# CT-SeqTrack 工作约定

## 2026-10-23 当前：v38 clean1（覆盖下方历史结构）

- 最新用户要求：只在原 `CT-SeqTrack/` 维护和部署一套源码，不建立 `CT-SeqTrack-clean1` 目录；clean1 仅保留为执行身份。2026-10-23 22:13–22:14只读检查发现服务器原目录三组旧v38正在运行，结束前不得覆盖。服务器仍仅授权只读检查，上传与启停由用户操作。

- 唯一活动路径为`models/ct_v38/`和独立`models/ctseqtrackv38.py`；不导入v31/v36/v37。旧算法及其专属配置/工具/测试已归档到14c3b2f完整快照。参考模型独立工程入口位于`models/seqtrack_reference/`。
- 数据事务为prepare→plan_prior→supervise_prior→forward→Adam→commit；无identity memory、extension或mode字段。acquisition和q0 quality监督保留，全空q0仍参与quality BCE。
- 保留32query/1024 B0 K/V、三个B3候选、原可信写入和相邻速度规则。三份38 YAML字节保持；新登记`38_clean1_registration.json`，旧`38_registration.json`不得刷新。
- 用户确认数值边界：同seed有效初始化精确一致；FP64连续Adam、FP32同权重输出/梯度及新版自身恢复验证。compact FP32和自然dropout不承诺复现旧训练轨迹；不得为通过测试改BN/Adam数学或放宽容差。
- 旧正式运行仅用对应源码，服务器由用户操作；本轮不上传、不启停、不新增正式训练。不读取TrajTrack。output及既有artifacts受保护，新报告进独立ct_checks目录。
- 本地验证pytest、compileall和diff；真实CUDA及成对性能由用户执行。独立工具`benchmark_v38_clean1.py`支持旧/新源比较。详见[精简与恢复](docs/CTSEQTRACK_V38_CLEAN1.md)。

以下为历史约定，不将其旧路径或阶段性服务器快照解释为当前状态。

## 2026-10-22 当前：v38 强 B1＋独立 B3

- 新实现为 `models/ct_v38/`、`models/ctseqtrackv38.py`，由 `main.py` 分派。旧v37/配置/输出保留，不读取TrajTrack。用户本轮授权SSH只读检查；不上传、安装、启停任务或修改服务器文件。
- B2退出新主线；B1原quality/acquisition监督及accepted/可信几何/真实时间保留。三候选为真实q0、原prior、原fallback（trusted propagation）。位置效用、旧quality、T分开；GT仅监督或被动审计。
- 第1轮启用B3，无shadow/warmup，margin0，SPT零头，同分q0→prior→trusted propagation。仅选中真实q0可申请可信写入；下一crop跟最终accepted。位置更新不自动恢复速度，无静止门、低分hold或传播次数停止门。
- 三组为concat B1 standard、concat B1 hold、Full standard，seed42/scratch60/bs16/FP32/workers4/val1/LR1.5e-4，共180epoch/215,100更新，独立登记 `38_registration.json`。不继承旧B2开放参数或factorized hold。
- 最新设备安排为两组B1物理GPU1、Full物理GPU0，三组 `--trainer_devices 1`；带日期的独立nohup和新终端tail命令已同步至操作页。20:28只读确认环境/数据路径正常、v38尚未上传、没有CT训练进程；这不替代上传后由用户执行的一次真实CUDA Full batch检查。
- [设计](docs/CTSEQTRACK_V38_DESIGN.md)、[操作](docs/CTSEQTRACK_V38_LAUNCH.md)为新入口。fixed final60/late3、原R/C和运动保护保持。旧源码封存 `artifacts/ct_checks/v38_implementation_20261022/baseline_source_cc8458b.zip`；本地验证不代表正式收益。

## 2026-10-21 最新实施：B2 centroid mass 梯度三组

- 用户已明确批准实施。新 `v37_centroid_recipe` 仅在重心 mass 路径 detach identity×reliability，votes/精修及两头显式监督保留；共享 v36 接口默认 False，旧模型默认语义不变。结构和操作见 [B2 三组说明](docs/CTSEQTRACK_V37_B2_CENTROID_LAUNCH.md)。
- 默认三组是 `b1_b2_mass_detach`、`full_mass_detach`（主结果）、`full_mass_detach_hold_lr`；第三组保留20轮衰减、取消50轮衰减。三份YAML默认mini Car/seed42/scratch60/batch16/FP32/concat/val1、workers4、LR1.5e-4，工程options为空，180epoch/215,100更新，固定final60/late3；不混入Full去重、分解或B3改法，不自动追加训练。
- 用户最新要求当前三个B2配方支持六字段覆盖，优先级CLI > 子YAML > base YAML > 代码DEFAULTS；未传沿配置，未知参数报错。check_val_every_n_epoch/workers变更为 `runtime_override`，batch_size/epoch/lr/seed变更为 `recipe_variant`，研究变体不计原三组；结构、单卡、scratch、时间合同不放宽。新 `--lr` 与launcher参数透传见 [覆盖说明](docs/CTSEQTRACK_V37_CONFIG_OVERRIDES.md)。核验基础模板+source，完整actual配置仍进哈希及恢复检查。
- 用户也开放 `--devices`、`--cfg`、`--path`、`--log_dir`。devices为单个非负物理GPU编号，CLI > YAML devices > 现有CUDA_VISIBLE_DEVICES，初始化CUDA前应用；不注入默认devices，实际GPU进manifest但不进算法hash。trainer_devices仍只能1。launcher额外cfg限同recipe、log_dir限新/空目录，devices覆盖位置GPU；恢复/test使用main，旧运行原源码。
- 默认启动仍全部物理 GPU1（`CUDA_VISIBLE_DEVICES=1`），每进程 `--trainer_devices 1`，完整命令仍val1。此次需上传完整新源码和独立登记，默认三份YAML不改；以后六字段合法值覆盖无需改代码/登记。旧已运行继续原源码，新版同run恢复必须保持全部actual参数，例如val5启动须val5恢复。修改前版在 `artifacts/ct_checks/20261021_v37_cli_overrides/previous_source/`；17:14快照与此前val5/val1检查保留为历史。不据历史快照断言当前状态，不自动启停或恢复；旧其他正式配方不改。
- 独立登记 `37_b2_centroid_registration.json`；旧五组和B1执行登记/YAML/输出保持。旧checkpoint仍用原源码，不放宽跨源码/跨配方恢复。修改前快照在 `artifacts/ct_checks/20261021-163831_b2_centroid_implementation/baseline_source/`。
- 实施完成不代表正式涨分；三组训练与一次真实CUDA Full batch检查仍由用户操作。服务器只读、不启停任务、不读取TrajTrack。下方“本轮只读/尚未实施”属于此前审查阶段。

## 2026-10-21 当前：结果已完成部分与研究审查

- 最新状态与建议见 [研究审查](docs/CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md) 和 [待办](need_to_do.md)。原 10/09 v37 五组已完成 300 epoch / 358,500 更新；Full FP32 final60=49.9136/59.5274、late-3=50.0715/59.6327，未过原门且低于 B1。下方“尚无 v37 正式成绩”仅为历史状态。
- 10/16 新 B1 三组 batch16 已完成；concat 精确复现旧 B1，分解和 hold_lr 未通过原门。batch4 本地完整训练/验证记录到 41 轮、日志进入 42 轮，没有正式 final60/late3；不据本地快照认定服务器停止或仍在运行。四组实际 manifest 均为 CUDA_VISIBLE_DEVICES=1，与旧分卡计划不同。
- 推荐工作配方为强 concat batch16 + 原 20/50 衰减；该推荐不修改任何旧配置/运行身份、不自动停止已有任务或追加训练。旧输出、旧 artifacts、旧 checkpoint 对应源码均保持。
- 本轮授权范围为详细阅读、相关研究检索与项目文档更新；新增审计写 `artifacts/ct_checks/20261021_research_audit/`。模型改法为研究提案，不表示已经实施。服务器仍只读，不上传、安装、启停任务；不读取/参考 TrajTrack。
- 解释原五组用原登记源码/0662c21 快照，解释新 B1 用对应 09a7809 登记源码。不要用最新执行开关解释旧结果，不把同状态 oracle 当闭环成绩，不把单 seed late3 当独立重复。

## 2026-10-14 当前新增：用户批准分解投影 B1 batch4 对照

- 首先在独立工程副本实施；2026-10-14用户进一步明确要求同步到`D:/desktop/research/CT-SeqTrack`的main工作区，最新代码现直接在本项目维护。旧五份YAML、`37_registration.json`及已有输出/权重保留；迁移前源码已另存于`artifacts/ct_checks/20261014-155937_main_sync/baseline_source/`。新四组由`v37_training_recipe`绑定执行算法，登记为`37_b1_execution_registration.json`，操作唯一入口见[四组B1配置/命令](docs/CTSEQTRACK_V37_B1_EXECUTION_LAUNCH.md)。
- GPU0：分解投影batch16、原concat batch16；GPU1：分解投影batch16仅第20轮衰减、分解投影batch4保留20/50衰减。用户最新明确新增batch4配置`37_b1_dedup_factorized_bs4_mini.yaml`，当前四条命令的第四组由batch8改为batch4；除batch及配方身份外保持factorized batch16参数。原batch8、batch128配置及已有运行证据保留，不假定batch8已启动或完成，不停止、重启或迁移旧运行。全部B1、FP32、mini Car、seed42、scratch60、LR1.5e-4、每轮validation；其余原W课程/损失/采样保持。本页四组共240epoch，历史尝试和已有batch8运行成本另记；实际更新数由sampler记录，不套用原batch16预算。
- 用户明确确认“无全局拼接”是保留global信息的分解投影，不删除分支；B1不走新版prior局部读取，第二组只有concat变量，不登记空操作。五项工程去重中B1实际生效为unique_mask。
- 新配方训练/validation/late3/test/同run恢复共用执行scope，配置身份包含执行算法；不放宽旧恢复检查。分解FP32不逐位等价，不能称旧v37无损替换。最终固定final60/late3。
- 当前服务器仍只读；代理不上传、安装、启动或停止。新增batch4配方会改变源码登记；正在运行的任务保持原源码，新任务使用独立部署目录或等旧任务完成后更新原目录，不能覆盖运行中目录后继续旧任务。batch4为新scratch，不从batch8/16/128权重初始化，不启用梯度累积；旧output/artifacts受保护。

## 2026-10-09 当前：用户批准 v37 实施与五组固定实验

- 当前活动新增源码为 `models/ct_v37/`、`models/ctseqtrackv37.py`，入口仍 `main.py`；基于提速后的强 v35 B0/B1，保留物理时间、主路、W 课程、768→256 和记忆容量。v36 仅复用身份/几何分工、顺序去重等明确选定机制，不采用事件 B1、固定米制 q0 邻域或整轨迹课程。
- 设计和操作唯一新版入口：[结构/监督](docs/CTSEQTRACK_V37_DESIGN.md)、[启动/恢复/比较](docs/CTSEQTRACK_V37_LAUNCH.md)。旧 YAML、checkpoint、output、artifacts 与已训练源码证据受保护。实施前基线已封存于 `artifacts/ct_checks/20261009-153604_v37_implementation/baseline_source/`。
- 五组仅 B0/B1/B1+B2/Full FP32 和 Full 局部 BF16；mini Car、seed42、scratch60、batch16，合计 300 epoch/358,500 更新。新增参数独立 RNG，公共参数登记哈希；禁止旧权重初始化、冻结参数或自动扩实验。
- B1+B2/Full 完整训练同一候选与核验器，比较选择和提交政策；不能写成有无 B3 网络的纯消融。核验输入全 detach；输出效用不进入 previous_quality；只有最终选中候选可申请写入，prior/fallback 不写 trusted/memory。
- 正式 BF16 仅登记 v37 Full 开放；旧版仍工程专用。source/config/runtime/精度/同 run 完整 epoch 恢复严格核验。固定 final60/late-3、原 R/C 门槛，FP32 为主结果；速度须同卡同负载配对测量，不用并发训练墙钟声称加速。
- 服务器由用户操作，代理不上传、安装、启动或停止。尚无 v37 正式成绩，不宣称收益、已校准或单 seed 足以发表。下方只读分析范围为过去任务，不限制本次明确授权的实现。

## 2026-10-08 当前：v36 五组已完成，综合结果未达标

- 五组 A/C/D/E/F 均完成 scratch60 / 71,700 更新，总预算 300 epoch / 358,500 更新已用完；源码、配置、排程与末三轮原帧审计通过。正式主模型 A 未通过原总体/移动门槛，五组均低于旧同 LR 的 B0/B1/B1+B2/Full。见 [五组复盘](artifacts/ct_checks/20261008-191343_v36_five_run_review/REPORT.md)。
- A final60 S/P=38.9912/39.5449，late-3=38.8381/38.6572；F 是分数最高的对照，也不能替换预登记 A 或宣称整体提升。首测缺测、mode 去重及少量可用候选有局部进展，但 q0 精修、长期外推、测量资格和持续失跟未解决。
- 已确认 q0 局部读取由 v35 的 LWH 归一化邻域变成固定1米邻域，属于需要明确处理的额外主路语义变化，尚不能认定是全部掉分原因。下一版建议以强 v35 B0/B1 为结构起点；该建议尚未实施或登记新训练。
- 本次仅只读分析与进展文档更新，不改已训练模型/配置/权重，不覆盖旧证据，不启动额外实验。服务器仍只读。下方“尚未完成/待启动”均为历史状态，五组预算不自动扩充。

## 2026-10-04 当前：v36 测量事件闭环，五组固定预算

- 用户已授权独立 v36 实施；当前设计与逐项 loss 见 [v36 结构](docs/CTSEQTRACK_V36_DESIGN.md)，操作见 [启动/恢复](docs/CTSEQTRACK_V36_LAUNCH.md)。下方 v35 启动安排为历史，不代表当前实验计划。
- 活动入口仍为 `main.py`；v36 源码在 `models/ct_v36/` 与 `models/ctseqtrackv36.py`。accepted、测量事件与身份 memory 分开，由 host 在 Adam 后提交 detached 状态；GT 只进入监督及被动审计。
- 五臂 A/C/D/E/F 均 mini Car、seed42、scratch60、batch16、FP32、LR=1.5e-4、20/50 衰减，无 warmup。每臂 71,700 更新，总计 300 epoch/358,500 更新；不追加完整训练、LR 或 seed 搜索。
- 比较固定 final60/late-3 与原总体/移动门槛，另外报告最高验证分及同轮配对指标；强 B0/B1 为只读复用。整体改进不能拆成各部件因果收益。
- 新 schema/runtime/config/source 身份独立登记；旧 checkpoint 使用其登记源码，不关闭恢复检查。`output/` 与旧 artifacts 保持不变。服务器只读，真实 CUDA batch 检查由用户上传后执行。


## 2026-10-02 当前安排：v35 完整链、seed42、两档LR

- 用户已完成9月30日六组及一次独立SeqTrack seed52重复，共七次新运行；以[七组复盘](artifacts/ct_checks/20261002-193819_v35_seven_run_review/REPORT.md)为准。scaled B0为工作基线；seed52原固定门槛及漂移风险记录不改写。下方“尚未启动”均属历史。
- 用户现批准六次新scratch：B1/B1+B2/Full分别用1.5e-4和1e-4，全部seed42，mini Car、W、60轮、20/50衰减、无warmup。新增360epoch、430,200更新，旧同LR B0及参考结果复用；不加载旧B0权重、不冻结参数、不自动追加seed/LR。GPU0为三组scaled、GPU1为三组normal；由用户在一次Full CUDA检查及显存确认后各卡并发三组。
- 当前操作页为[完整链六组启动说明](docs/CTSEQTRACK_V35_FULL_SEED42_LAUNCH.md)，身份见`cfgs/ct_seqtrack/35_full_seed42_registration.json`。正式v35非B0限定上述seed/CfC/LR；旧B0/v34/reference配置及身份保持，旧checkpoint不新增跨源码迁移。
- 本轮仅正式臂登记和被动候选/搜索覆盖记录，不改网络、loss、采样、优化器或状态数学。当前B3是共享quality候选竞争，无外部校准artifact；Full同时开放mode监督。四帧B0全空时q0=prior的现有合同保留。
- 每预测帧完整台账见[候选记录格式](docs/CTSEQTRACK_V35_FULL_CANDIDATES.md)；GT仅监督/日志，不能进入选框、采样或accepted。新工具`check_v35_full_batch.py`只做一次工程生命周期、不保存权重；`compare_v35_full.py`只读重算六组与固定参考。
- 服务器仍严格只读；代理不上传、安装、启动或停止。用户本地提交、服务器拉取后执行命令。旧output和artifacts保护规则保持。

## 2026-09-30 当前进展（覆盖下方历史运行状态，不改实验门槛）

- 五组v35 seed42均完成60轮并下载至本地；原帧正式比较确认scaled（初始LR1.5e-4）唯一通过总体四项≥固定R、移动四项≥旧C。final60 S/P=52.3162/62.0230，late-3=53.7629/64.6080。证据见[五组复盘](artifacts/ct_checks/20260930-175409_v35_five_run_review/REPORT.md)。
- 相对旧W-quarter的>10m计数未恶化，但首测raw0及持续失跟仍有短板；原门槛通过不等于全部机制问题解决。piecewise末轮最高、late-3未通过，不可替代scaled。
- 用户随后明确批准下一轮六次新训练：scaled seed52、piecewise seed52、独立SeqTrack正常seed52、W double（2e-4）seed42、half seed52、quarter seed52。五组B0均为v35 W，不新增S或v34训练；划分seed42不变。新预算360epoch、430,200更新，覆盖此前仅补两次及七次总上限的建议，不自动扩展。六组尚未启动；由用户上传启动，服务器仍只读。见[当前六组配置与命令](docs/CTSEQTRACK_V35_W_SIX_RUN_LAUNCH.md)。
- 仅新增double配置及v35正式LR允许项；其余五份seed52配置原样复用，模型、loss、采样和优化/调度计算不改。下方旧配置注释的“仅胜出时”以及暂停/恢复、尚无final60文字为历史条件，以此次明确六组授权为准。六组均scratch，不把旧五组checkpoint迁移到新源码；不宣称跨seed稳定、W优于S或Full收益。

## 2026-09-29 当前安排（覆盖此前四组总数及下方历史安排）

- 用户已批准实现v35综合版本：真frame1四分支精确合法初框、33维同源历史描述与coarse条件、8角点×16邻点局部query证据。全部新版B0保持W=1/4/4/8，保留原loss、全局token、quality监督、共享头和Full梯度/状态合同。见 [v35结构](docs/B0_V35_INTEGRATED.md)。
- 原seed42四组LR=2.5e-5/5e-5/1e-4/1.5e-4在GPU0/0/1/1完成了36轮；9月29日13:33只读复查确认用户已暂停四组，原PID均不存在、日志记录SIGTERM，各自最新完整checkpoint均为`epoch=036.ckpt`。共同最新validation35，尚无正式final60/late-3。接下来由用户更新同一活动项目，从各自原036 checkpoint恢复并从第37轮继续，物理GPU仍0/0/1/1；代理不执行停止、上传或启动。旧output/artifacts保留，不覆盖原配置、manifest、日志或权重。
- 本次只新增第5组piecewise并由用户在GPU1从头启动：第1–20轮5e-5、第21–50轮1e-5、第51–60轮5e-6；scratch60、seed42、W及其他单组参数不变。新组使用新输出目录，不能从原half或任何旧权重初始化。之前的独立代码目录建议已由本次共享活动项目更新方案覆盖。
- 首阶段现为五次、300epoch、358,500更新；仅通过原硬门后锁定胜出完整配方，补该配方seed52及原正常SeqTrack seed52，条件总上限为七次、420epoch、501,900更新，不自动启动。此明确请求覆盖此前四次/最多六次安排。已有seed42参考、旧C/W-quarter继续复用，不重新扫S/W。新增配置为`35_b0_w_piecewise_lr_mini.yaml`及预备`35_b0_w_piecewise_lr_mini_seed52.yaml`。
- 原总体四项≥固定R、固定31条曾移动轨迹四项≥旧C的门槛保持。完整五组通过者按final60 Success、Precision、较低初始LR排序；完全同分且初始LR同为5e-5时，预登记原half优先piecewise。严重漂移采用方案1：>10m计数增加只标风险并暂停Full待复核，不改原硬门结果；小幅孤立波动不自动判失败，明显持续退化必须处理并讨论，不另设漂移数值或切片全胜门槛。五组失败或第二seed不支持稳定性即停止，不自动扩大搜索。
- **严重漂移与持续失跟是本轮明确优化目标。** 须在综合方案中说明有证据的成因与对应改法，不能因“不新增硬门”而仅统计或留到Full再修。固定报告>5m、>10m和未恢复失跟，区分原硬门结果与风险复核结论；不为压过某个计数追加无必要实验。统一细则见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。
- v34八组已经完成；9/26旧待启动文字不再代表当前进度，见 [9/28复盘](artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)。v35实现和工程检查不代表正式涨分。
- 仅本地修改活动项目；服务器仍只读，由用户暂停四组、上传、恢复及启动第5组，不得代为安装、启动或停止任务。此次仅修改调度，不改网络、loss或数据，不把重复真实CUDA batch设为前置。新增安排见 [piecewise运行说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)；[原四组命令](docs/CTSEQTRACK_V35_MINI_LAUNCH.md)仅保留用于已有运行的查询，不重新执行原scratch命令。
- 共享目录更新后的原四组恢复只接受预登记的精确before/after源码对应关系，旧配方、状态和配置身份保持。保留最初`run_manifest.json`，在`resume_manifests/`追加本次恢复来源；不得改写原记录或笼统关闭源码校验。第五组比较入口为`tools/compare_v35_piecewise_b0.py`，旧四组比较工具独立保留。
- 新版本独立schema/runtime/config身份；旧v33/v34/reference的默认值、行为、配置SHA和权重兼容保持。GT诊断不进入模型决策，旧output/artifacts不可覆盖；新增检查写独立ct_checks目录。

## 2026-09-26 当前安排（覆盖下方 v33 历史启动安排）

- 用户最新明确扩为八组mini单seed：SeqTrack正常1e-4/减半5e-5；S（1/3/3/8）和W（1/4/4/8）各1e-4/5e-5/2.5e-5。SeqTrack保留StepLR20/40，S/W保留20/50，无warmup；均seed42、scratch60、71,700更新。见 [八组协议](docs/B0_V34_LR_GRID.md) 和 [最新启动说明](docs/CTSEQTRACK_V34_MINI_LAUNCH.md)。该请求覆盖此前“两组、不重跑参考、不扩LR”。
- 活动源码继续沿用 `models/ct_v31/` 和 `models/ctseqtrackv31.py`。v34 按显式版本身份启用新结构；v33/reference 的原配置、schema、checkpoint 键和行为保持。不得只替换全局 v33 身份常量。
- 旧R/C/旧v32 B0仍为固定证据，另由用户启动新R正常/减半；GPU0跑R正常＋S三组，GPU1跑R减半＋W三组，各四个独立单卡进程。代理服务器权限只读，不得上传、安装、启动/停止任务。不得把可复用旧R写成用户无需运行本轮新R。
- 总体 final60/late-3 的 S/P 四项均达到 R，且固定 31 条曾移动轨迹全程（657 预测帧）的四项均不低于 C，才允许进入 Full 规划。未完成正式训练前不宣称涨分。
- v34 窗口4是本轮明确授权的协议变化，十轮课程、112 reserve、BN 策略及总预算不变。历史残差、loss、pooling、quality 目标与原 Full 梯度边界保持。
- 原v34实现证据为 `artifacts/ct_checks/20260926-154028_v34_implementation/`；本轮八组就绪复查与服务器只读证据为 `artifacts/ct_checks/20260926-161231_v34_eight_run_readiness/`。上传后仅一次新的v34真实CUDA batch由用户执行；旧v33服务器检查不构成v34验证。启动显式保留native allocator，避免继承旧max_split_size_mb:64。

## 当前范围与授权

- 唯一活动项目为本仓库，所有命令从本仓库根目录执行。兄弟基线只读，不修改；不读取 TrajTrack。
- 下方为 v33–v35 实验历史约定；当前状态和本轮范围以顶部 2026-10-21 记录为准，正式配方仍以各自登记及实验协议为准。v32 通过 Git `ddcb1a1`、v31 通过 `b1d886e` 复现；不改写历史。
- 用户最新要求：服务器 `lishengjie@10.109.253.86` 本轮只读；允许本地修复，由用户自行上传和启动。不得自行上传、安装、启动训练、停止进程或改写服务器文件。旧部署授权不延续到本轮。
- 本轮由用户从头运行四组 mini Car、seed42：R 独立 SeqTrack、A 综合 B0 原配方、B 综合 B0 延后第二次衰减、C 为 B 全程半学习率。物理 GPU 依次为 0/0/1/1，每组单卡。
- 2026-09-25用户追加D：综合B0的B配方全程学习率×1.5，初始1.5e-4、milestones=[20,50]，物理GPU0；原A/B/C保留。只登记新增B0配方，不改变预算或参考模型配方，见`docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md`。
- 2026-09-25用户追加E：继承B，峰值LR=3e-4、前2000次更新线性warmup（从峰值/2000升至峰值），完成20/50轮后降至3e-5/3e-6，物理GPU1；warmup计入原71,700次更新。原R/A/B/C/D不改。E相对B同时改变LR与warmup，不能独立归因，见[追加E说明](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。本轮未访问或修改服务器，由用户上传并启动。
- 已完成 v32 R 仍是可复用的历史证据；用户当前选择重跑 R。不得把“可复用”写成“本轮无需启动 R”。本轮不新增 Full 独立诊断。
- 协议以 [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) 为准；实现见 [B0_V33_REPAIR.md](docs/B0_V33_REPAIR.md)，用户启动说明见 [CTSEQTRACK_V33_MINI_LAUNCH.md](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)，工具范围见 [FORMAL_TOOLING.md](docs/FORMAL_TOOLING.md)。

## 历史 v33–v35 实验与实现不变量（v36差异以当前设计为准）

- scratch-only：所有正式运行从 epoch0 随机初始化。`--init_checkpoint` 禁止；`--checkpoint` 仅用于相同配置身份的完整 epoch 边界恢复或评测。工程检查权重不得作为正式初始化。
- 启用参数不冻结；detach 和 BN 统计隔离是耦合合同，不是参数冻结。每 batch 一次 forward/backward/Adam，随后提交 detached accepted 状态。
- B0 保持四分支；v33/v34 S 为1/3/3/8，v34 W及本轮v35为1/4/4/8。保持10轮课程、112个尾部预留窗口和既定BN策略；不靠缩小batch、截断端点、追加历史forward或冻结模块降低预算。
- 公开框使用 anchor 平移后的世界轴 XYZ 和 absolute yaw；B0 几何视图换到 anchor-local。q0 直接回归中心，history/mode 输出 seed 残差，共享最终头；mode 输出 live center 和 query detach 保持。
- 可信状态、缺测、padding、唯一 raw ID、裁剪和记忆写入按现有合同执行。GT 仅用于合法初始化、监督和被动诊断，不得用于当前预测裁剪、候选选择或 accepted 提交。
- B1 历史运动边资格保持原定义；保留 nuScenes full、KITTI、CfC/GRU、模块臂及时间控制能力。接口保留不代表对应实验已完成。
- 比较固定 final60 与 late-3（58/59/60），不得跨实验挑选不同最佳轮次；不得在四组完成前声称 v33 涨分、达到 SeqTrack、SOTA 或时间/记忆因果收益。

## 保护与验证

`output/` 和既有 `artifacts/` 的数据、日志、权重、报告、脚本及图片不得删除或覆盖。新工程产物写入 `artifacts/ct_checks/` 的独立目录；不得用 `git clean` 清理 ignored 证据。项目注释和文档使用 UTF-8 中文。

此前服务器快照验证：Python3.9.19、torch2.0.1+cu118、pytorch_lightning2.0.2 下 **314 passed、1 skipped**，一次真实 CUDA batch 的 forward/backward/Adam/commit 已通过且未保存 checkpoint。证据见 `artifacts/ct_checks/20260924-190116_v33_implementation/`。后续被动汇总增量单独记录本地验证，不声称最新源码已逐文件完成服务器复验。这些历史检查不授予本轮服务器写权限，也不是四组正式训练完成证明；不要反复堆成用户启动前置步骤。

本地代码修改后按改动范围验证：

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
```

缺 Lightning/CUDA 的本地跳过必须如实说明。旧固定 HEAD 精简门禁仅是历史工具，不作为 v33 验收。当前网络使用 PyTorch 路径，不要求旧 PointNet++ CUDA 扩展；不擅自升级服务器环境。
