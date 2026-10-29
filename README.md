# 2026-10-29：V40 已实现，待服务器运行验收

当前新主线为 **v38clean1 dense B0＋窄 FlashAttention＋B1 MTP＋Full 独立搜索保护**。四组 mini 配置：`40_b0_mini.yaml`、`40_b1_mini.yaml`、`40_b1_mtp_mini.yaml`、`40_full_mini.yaml`，统一经 `main.py --cfg cfgs/ct_seqtrack/40_..._mini.yaml` 运行。

见 [设计及解释边界](docs/CTSEQTRACK_V40_DESIGN.md) 与 [服务器验收、独立日期启动命令](docs/CTSEQTRACK_V40_LAUNCH.md)。**本机按用户要求未运行环境检查、测试、编译或训练；没有V40成绩或性能结论。** 后续只读服务器检查确认V40未部署、并发检查缺pynvml；已修复验证inference模式导致事务缓存不复用的静态缺口。用户上传后先执行六类服务器验收，工程权重丢弃，最新安排先GPU0并行B1 MTP+Full，B0/B1以后补齐。下面带日期的旧版本安排和成绩保留为历史记录。

# CT-SeqTrack

**2026-10-28：v39 工程验收通过，可以开始两组正式实验。** 新主线为 E＋B1＋M＋G＋自适应跟踪，所有模块从epoch1启用；两份 Full 仅学习率配方不同。用户最新要求新环境名`ctseqtrack3d`、OMP/MKL/OPENBLAS均为2、两组同GPU0并行，workers4和其他训练配方不变。[最新配置说明](docs/CTSEQTRACK_V39_CONFIG.md)已汇总固定参数；分别启动及新终端tail命令见 [v39操作页](docs/CTSEQTRACK_V39_LAUNCH.md)。用户本次已授权备份服务器源码并更新匹配的最新代码；服务器相关测试72项（CUDA项无跳过）、连续batch、实际main训练/验证/checkpoint及A/B同卡并行短程均已通过；配对profile已完成，正式成绩和独占速度结论仍待实验。以下v38及更早记录保留为历史。

**2026-10-26 实验结论与 profile 已整理。** 原版 v38 三组均完成60轮、各71,700更新，原工具与逐帧审计通过。无B2的Full通过原R/C总体和运动双门：final60=52.0230/63.1313、late3=53.7629/65.4792；相对旧Full改善，但尚未稳定超过强B1，严重漂移仍增加。用户指定GPU0的小规模真实训练/推理profile已完成，主要热点包含CPU数据准备、acquisition标签和掩码/归约，不应只押FlashAttention。见[结果、性能与下一步决策](docs/CTSEQTRACK_V38_RESULTS_PROFILE_20261026.md)。本次没有修改模型、启动新正式训练或部署clean1；下方“尚无成绩”等均为当时记录。

**2026-10-23 当前：v38 clean1。** 代码直接修改、部署在原 `CT-SeqTrack/`，只保留一套活动实现；clean1 是执行修订标识，不需要单独目录。 当前主线已从v31/v36/v37中抽出，B0/B1/B3独立运行；空memory、extension、旧mode计算及两个投影已删除。三份正式YAML保持，新登记为`38_clean1_registration.json`；旧checkpoint仍使用原源码。结构、数值边界、验证和历史恢复见 [本次精简说明](docs/CTSEQTRACK_V38_CLEAN1.md)。下方带日期的内容保留为历史记录，其中旧入口仅在对应冻结源码使用，不代表当前可执行工具或服务器状态。

**2026-10-22 当前主线：v38 强 concat B1＋独立 B3。** B2 扩展点、身份记忆、vote、mode退出新主线；保留完整强 B1 学习路径、accepted历史、可信位置/速度、真实时间。B3 用S/P选框、独立T决定真实q0的可信写入，从第1轮启用。见 [设计说明](docs/CTSEQTRACK_V38_DESIGN.md) 和 [检查、启动、恢复、比较](docs/CTSEQTRACK_V38_LAUNCH.md)。

三组独立scratch：B1 standard、concat B1 hold、Full standard。共同mini Car/seed42/scratch60/batch16/FP32/workers4/val1/LR1.5e-4，共180epoch/215,100更新。standard与Full在20/50轮后衰减，hold仅20轮。不由旧权重分叉，hold不替代结构对照。旧版本与产物保留，旧checkpoint使用原源码。服务器真实CUDA检查及正式训练由用户执行；当前没有v38正式成绩。下方B2及更早记录为历史，不自动重启或追加。

**上传前复核：两组B1使用物理GPU1，Full使用物理GPU0。** 最新操作页提供三条独立、带日期的nohup及新终端tail命令，验证频率仍为1。10月22日20:28 SSH只读核实环境与mini路径，服务器尚无v38、无CT训练进程；未执行上传或训练。完整记录见 [上传就绪检查](artifacts/ct_checks/20261022_v38_upload_readiness/REPORT.md)。

本地115项v38合同及真实CPU Lightning入口/恢复检查通过；全仓1351项通过，唯一源码收束期间的恢复拒绝已在冻结源码后复测通过。三组登记、保护审计和服务器待执行边界见 [实施与验证记录](artifacts/ct_checks/v38_implementation_20261022/REPORT.md)。

**2026-10-21 最新实施：B2 重心权重梯度修正。** 已新增独立三组配方：B1+B2 shadow、Full（均20/50衰减），以及Full仅20衰减；用户最新将三组统一改为每轮验证（val1），scratch60、batch16、FP32、concat及固定final60/late3保持。结构改动仅在mass路径detach identity×reliability，保留vote/精修及两头显式监督。新配置、一次CUDA检查、启动与恢复命令见 [B2 三组操作页](docs/CTSEQTRACK_V37_B2_CENTROID_LAUNCH.md)。旧其他配方/登记/输出保持，旧checkpoint继续用原源码；代理未启动服务器训练，尚无新三组成绩。下方“仅文档/尚未实施”是本轮实施之前的审查记录。

默认启动命令仍为物理GPU1、每进程单卡、workers4和每轮验证。用户最新要求三个B2配方支持CLI覆盖六项参数：验证频率/workers记为运行覆盖，batch_size/epoch/lr/seed偏离模板记为研究变体，不计默认三组。优先级CLI > 子YAML > base YAML > 代码默认，未传沿配置、未知参数报错；例子和恢复规则见 [参数覆盖说明](docs/CTSEQTRACK_V37_CONFIG_OVERRIDES.md)。需上传本次完整新源码和独立登记；默认三份YAML不改，之后六字段合法覆盖无需改代码/登记。旧已运行继续原源码，同run恢复仍要求同actual配置；17:14快照和此前检查为历史。修改前版封存于 `artifacts/ct_checks/20261021_v37_cli_overrides/previous_source/`，旧其他正式配方不改。

GPU、YAML和路径也可显式覆盖：`--devices 1` 选择物理GPU1，`--trainer_devices 1` 表示仅用一张卡；devices按CLI > YAML > 环境解析。`--cfg`、`--path`、`--log_dir`支持自选YAML、数据根和输出目录，启动器同样透传；默认三条完整命令仍有效，不开放真正多卡训练。

**2026-10-21 当前：实验结果与研究问题已同步。** 原 v37 五组均完成 scratch60；Full FP32 final60 S/P=49.9136/59.5274，late-3=50.0715/59.6327，未通过原门槛，仍低于强 B1 的 53.7133/66.2746、52.8541/65.3082。10/16 新 B1 的 concat、分解、hold_lr 三组也已完成：concat 精确复现旧 B1；分解与 hold_lr 未通过。batch4 本地完整记录到 41 轮、日志进入 42 轮，未见正式 final60；本地快照不代表服务器实时状态。

当前研究入口：[完整问题与文献审查](docs/CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md)、[最新问题](最新问题.md)、[待办](need_to_do.md)。建议保留原 concat batch16 的强 B1 配方，优先检查低运动漂移、辅助训练对主路的影响、恢复身份/几何、无证据候选接管及提交质量；评测提速先做传输/几何/记录复用。新增局部读取审计显示 67 个正 mode 全部读到 GT 点，扩大邻域缺乏优先依据。PF-RPN、LoopVL 及其他论文的可借鉴部件和限制已逐项列入报告。

合法首框保持在当前 mini/world-coordinate 协议下总体 60.5503/72.9945，高于 B1；B1 在 171 个运动端点却高于保持基线 14.06/19.06 pp，须同时研究低运动稳定性与真实运动能力，不用 GT 动静标签在线选策略。新分解运行记录了较低 allocator 峰值，但没有受控配对速度/显存收益证据。

本次仅更新文档与新增审计产物，没有修改模型/正式配置/旧输出、启动训练或操作服务器。当前入口仍为 `main.py`，v37 组合在 `models/ctseqtrackv37.py`、`models/ct_v37/`，部分核心逻辑复用 `ct_v31/ct_v36`。下方日期段是历史登记与实施过程，其“尚未运行”等表述由本节及最新报告覆盖；不按旧启动段自动重跑实验。

**2026-10-14 当前新增：分解投影 B1 batch4 对照。** 用户最新明确新增batch4，当前四条命令的第四组由batch8更新为batch4，其他参数沿用分解投影batch16：LR1.5e-4、20/50衰减、seed42、FP32、scratch60、每轮验证。当前四组为分解投影/原拼接batch16（GPU0），分解投影仅第20轮衰减batch16/分解投影batch4（GPU1）。新配置为 `37_b1_dedup_factorized_bs4_mini.yaml`；原batch8、batch128配置及已有运行证据保留，不停止或迁移旧运行。最新配置表、逐组nohup和日志见 [四组B1操作页](docs/CTSEQTRACK_V37_B1_EXECUTION_LAUNCH.md)。登记文件为 `cfgs/ct_seqtrack/37_b1_execution_registration.json`；不是改写旧五组登记。B1不执行prior局部读取，不能把该选项列为本轮有效变量。

2026-10-14 按用户要求，已将验证后的修订从独立工程副本同步到 `D:/desktop/research/CT-SeqTrack` 的 main 工作区，最新代码和配置直接在本项目中维护。本次batch4修改仅在本地，代理未操作服务器。新增配方会改变源码登记：正在运行的任务保持原源码，新任务用独立部署目录，或等旧任务完成后再更新原目录；旧checkpoint只用其原登记源码恢复/评测。旧五组YAML、登记文件和已有输出保留。分解投影保持全局信息及原参数，但FP32训练数值不同；本轮新scratch检验其实际影响，不预先宣称无损或提速。原工程去重与成对测速说明仍见 [执行实验室](docs/CT_EXECUTION_LAB.md)，真实CUDA配对速度报告尚未完成。下方各日期的“当前”是历史安排。

**2026-10-09 当前：v37 基于强 v35 B0/B1 与现有工程提速实施。** 新版隔离扩展证据与主定位，重建两个恢复候选，分别学习最终框定位效用和提交质量。B1+B2 与 Full 训练同一候选及核验系统，仅执行选择与更新政策不同。结构与监督见 [v37 设计](docs/CTSEQTRACK_V37_DESIGN.md)，五组启动、恢复和比较见 [v37 操作页](docs/CTSEQTRACK_V37_LAUNCH.md)。以下 v36/v35 的“当前”均为历史记录。

五组固定 B0、B1、B1+B2、Full FP32、Full 局部 BF16，全部 mini Car、seed42、scratch60、batch16、LR=1.5e-4、20/50 衰减及原 W 课程，共 **300 epoch、358,500 更新**。FP32 Full 是研究主结果；单 seed mini 不足以证明论文结论。真实 CUDA 生命周期和两精度实测由用户在服务器执行，本地实现不代表已涨分或已获得加速。

本地已通过新增 75 项合同检查、PL2.0.2 专项 8 项及 CPU Full batch16×1024 生命周期；B0/B1 两步 Adam 与旧版逐位相同，完整 epoch10→11 恢复通过。全仓旧测试的一次历史时间戳碰撞、复核结果及服务器验证边界见 [v37 实施报告](artifacts/ct_checks/20261009-153604_v37_implementation/REPORT.md)。

上传前复核补齐 v37 恢复事件唯一文件名、跨平台 LF 源码登记及精度边界测试；五份配置和训练参数保持不变。服务器只读确认环境与数据可用，尚未上传新版或执行 CUDA 检查。最新证据见 [上传就绪报告](artifacts/ct_checks/20261009-161150_v37_upload_readiness/REPORT.md)；[操作页](docs/CTSEQTRACK_V37_LAUNCH.md)已列出五组独立、带日期的完整 nohup 与新终端 tail 命令。

**2026-10-04 当前：v36 测量事件驱动闭环已实施，正式结果待验证。** accepted 输出、合格测量事件和身份记忆分离；事件 B1、位置范围、prior 优先双域获取、几何独立投票、最终框核验与连续长分支共同构成新版本。见 [结构/接口/监督](docs/CTSEQTRACK_V36_DESIGN.md) 与 [五组启动/恢复/比较](docs/CTSEQTRACK_V36_LAUNCH.md)。

五组 A/C/D/E/F 保持 mini Car、seed42、scratch60、batch16、FP32，总计 **300 epoch、358,500 更新**。本地合成与合同验证不替代真实 CUDA batch 检查；由用户上传后执行一次检查及正式训练。代理不写入服务器、不启动任务。旧配置、权重与产物保留；以下 v35“当前/下一轮”表述为历史记录。

本轮本地验证 **728 passed、12 skipped**；CPU Full batch16×1024、epoch10→11 合成恢复、compileall、diff 与登记身份通过。跳过项为缺 Lightning 或真实 CUDA；见 [实施与验证记录](artifacts/ct_checks/20261004-194559_v36_implementation/REPORT.md)。

上传前已使用本地封存Lightning2.0.2补充真实CPU Trainer的完整入口/评测与epoch10→11恢复，两项通过；更新了五组完整nohup和新终端tail命令，保留训练源码及配置身份。[最新就绪复核](artifacts/ct_checks/20261004-200727_v36_upload_readiness/REPORT.md)只剩用户上传后的一次真实CUDA batch检查，代理服务器只读。


**2026-10-02当前：准备v35完整链六组seed42实验。** 已完成的七组新运行及重复参考见[七组复盘](artifacts/ct_checks/20261002-193819_v35_seven_run_review/REPORT.md)。工作B0固定为v35 W scaled；本轮B1/B1+B2/Full各比较1.5e-4与1e-4，复用对应已有B0，全部scratch60，合计360epoch、430,200更新。GPU0三组scaled、GPU1三组normal。

六份正式配置、完整候选台账、一次Full真实batch工具及只读比较入口已加入；网络、loss、采样、训练与状态计算保持。最新[配置/后台启动/tail/恢复说明](docs/CTSEQTRACK_V35_FULL_SEED42_LAUNCH.md)、[候选字段](docs/CTSEQTRACK_V35_FULL_CANDIDATES.md)、[协议](docs/EXPERIMENT_PROTOCOL.md)。用户上传后执行一次Full CUDA检查，再确认每卡三组的显存；代理服务器只读，尚未启动本轮正式训练。旧B0配置和产物保留，不能将旧checkpoint作为新Full初始化。

本轮全仓测试644 passed、3 skipped，最终工具增量另行复核通过；37份旧YAML及28个旧配置身份保持，Full实际CPU batch16与完整epoch恢复通过，compileall/命令语法检查通过。见[本次交付与限制](artifacts/ct_checks/20261002-211032_v35_full_readiness/REPORT.md)，不将其称作服务器CUDA已通过。

下方9月30日及以前文字为历史记录，其“下一轮”“尚未启动”不代表当前状态。

**2026-09-30当前结果：v35五组seed42均完成60轮并已拉回本地。** 原帧固定口径重算确认，**scaled（初始LR=1.5e-4）是唯一通过总体四项≥R及固定移动四项≥C的配方**：final60 S/P=52.3162/62.0230，late-3=53.7629/64.6080。相对旧W-quarter的>10m计数未增加；首次缺测和持续失跟仍有短板。piecewise虽末轮最高，late-3未达标，不替换胜出者。详见[五组正式复盘](artifacts/ct_checks/20260930-175409_v35_five_run_review/REPORT.md)。

**当前下一轮为用户确认的六组，尚未启动。** scaled / piecewise / half / quarter各补seed52，加独立SeqTrack正常seed52，再以W seed42探索2e-4；五组B0均为v35 W，不新增S或v34训练。新增`35_b0_w_double_lr_mini.yaml`，其余五份现有配置原样复用；只增加新学习率的正式登记，模型、loss、采样和优化/调度计算不变。共360epoch、430,200更新；[六组配置、后台启动及tail命令](docs/CTSEQTRACK_V35_W_SIX_RUN_LAUNCH.md)，[正式协议](docs/EXPERIMENT_PROTOCOL.md)。该明确请求覆盖此前仅补两次建议，不宣称跨seed稳定或Full收益。下方9月29日启动文字均为历史操作记录；服务器仍由用户操作。

本次六组配置核验完成：**554 passed、3 skipped**，27份旧YAML及配置身份不变，六组实际60轮LR、命令解析、Bash语法、compileall及diff检查通过。见[本次交付记录](artifacts/ct_checks/20260930-190740_v35_w_six_run_configs/REPORT.md)。本次未访问服务器，上传后由用户启动。

2026-09-29历史piecewise调度与恢复增量已完成本地验证：**554 passed、3 skipped**，25份旧配置文件及身份SHA不变。[旧五组命令](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)及[验证报告](artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/REPORT.md)仅保留用于复现，不用于启动本次六组。

CT-SeqTrack 是面向 3D 点云单目标跟踪的研究项目。当前新增 **v35 B0 综合优化**：在 v34 W=1/4/4/8 的基础上，对齐真实首测初始化、为 coarse 提供显式合法历史条件、为 fine 提供候选局部真实点证据。活动代码仍在 `models/ct_v31/`；v33/v34/reference 原行为与权重身份继续兼容。

**2026-09-29追加第5组piecewise，覆盖此前四组总数。** 原四组LR=2.5e-5/5e-5/1e-4/1.5e-4已完成36轮；13:33只读复查确认用户已暂停四组，各自完整checkpoint均为`epoch=036.ckpt`。共同最新validation35，尚无正式final60/late-3。本次仅新增一组：第1–20轮5e-5、第21–50轮1e-5、第51–60轮5e-6；仍seed42、scratch60、W=1/4/4/8、无warmup，网络、loss、数据及其他单组参数不变。五组合计300epoch、358,500更新；仅达标后补胜出完整配方seed52与独立SeqTrack seed52，条件上限七次、420epoch、501,900更新，不自动启动。

保留固定R/C门槛和旧W-quarter风险对照。用户暂停已完成，接下来更新同一活动项目，从各自原036 checkpoint恢复四组、第37轮继续，GPU仍0/0/1/1；然后在GPU1从头启动piecewise并使用新输出目录。原四组恢复保持各自配置与运行目录，保留最初manifest，另记`resume_manifests/`；仅接受固定before/after源码关系，旧训练日志追加写入，不覆盖旧output/artifacts。新组不读取旧权重初始化。新增配置、调度解释及精确操作顺序见 [piecewise操作说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)，结构见 [v35说明](docs/B0_V35_INTEGRATED.md)。[原四组操作页](docs/CTSEQTRACK_V35_MINI_LAUNCH.md)继续用于查询已有运行；不要重新执行四组scratch命令。此次只改调度，不重复要求CUDA batch检查，代理服务器权限仍为只读。

**截至2026-09-28，v34八组已全部完成。** W-quarter final60 S/P=51.4650/62.8435，late-3=51.2527/62.6765；Precision高于R，Success及移动守底未全通过，不能晋级Full。[八组正式复盘](artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)。v35阶段状态见[9月29日只读快照](artifacts/ct_checks/20260929-130405_v35_four_run_status/server_snapshot.json)，同轮次历史对照见[逐项证据](artifacts/ct_checks/20260929-130405_v35_four_run_status/historical_comparison.json)；中途validation不替代正式验收。下方v33数字和启动文字为历史记录，不覆盖本段当前状态。

**本轮明确优化严重漂移与持续失跟。** 方案1只表示保留原总体/移动硬门、单列漂移风险，不新增漂移数值硬门，不表示仅记录或推迟修复。综合改法及其成因对应见[v35结构说明](docs/B0_V35_INTEGRATED.md)；>10m计数增加时暂停Full待复核，小幅孤立波动不自动否决方案，明显持续退化必须处理。原硬门结果与风险结论分开报告，统一口径见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。

| 模块 | 作用 |
|---|---|
| B0 | 从当前与历史真实点提取观测，生成 coarse query 并通过共享 decoder 定位 |
| B1 | 使用 CfC/GRU 建模物理时间运动先验与 context |
| B2 | 在 B0 原始裁剪之外获取 768 个点槽，选取 256 个证据点槽，结合原始身份记忆 |
| B3 | 让 q0 与三个测量模式共享定位和质量头，选择最终输出 |

v33 整合中心回归、BC 与历史监督归一化、可信状态修复及被动评测诊断。R/A/B/C/D/E 六次 mini 运行已完成并拉回本地；Full 接口保留，本轮不新增 Full 独立诊断。具体改动见 [v33 实现说明](docs/B0_V33_REPAIR.md)。

## v33 历史结果（2026-09-26）

六次运行是 1 组 SeqTrack + 5 组 B0，均 seed42、scratch60、71,700 次更新；已核验全部 58–60 轮正式逐帧结果。

| 模型 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| R：SeqTrack | 51.8239 / 61.3414 | 51.8844 / 61.9664 |
| A：原配方 | 49.1214 / 57.6499 | 47.4271 / 54.5514 |
| B：延后降档 | 47.5525 / 55.9179 | 46.6116 / 53.9143 |
| C：半学习率 | **50.4562 / 59.2560** | **49.3738 / 58.0744** |
| D：1.5 倍 LR | 49.3993 / 57.6214 | 48.1601 / 55.8435 |
| E：3 倍 LR + warmup | 44.5766 / 53.8600 | 44.0835 / 53.2586 |

**C 是已测最佳 B0，仍未达到 SeqTrack。** 后续采用 C 的 5e-5、20/50 轮衰减、无 warmup；不继续扩大 LR。C 相对旧 v32 B0 的 Success 略升、Precision 下降，不能登记综合修订全面有效。主要剩余问题是缺测时历史利用、fine 无稳定定位增益，以及完整预测历史的训练覆盖；具体建议尚未实施。完整结果、曲线与代码依据见 [六组复盘](artifacts/ct_checks/20260926_v33_six_recipe_review/REPORT.md)。

## 已完成运行的配置

以下为已完成运行的配置与物理 GPU 安排，保留用于复现：

| 组别 | 配置 | 物理 GPU | 学习率配方 |
|---|---|---:|---|
| R | `33_seqtrack_ref_mini.yaml` | 0 | 原 SeqTrack，Adam 1e-4，StepLR 每 20 轮乘 0.1 |
| A | `33_b0_mini.yaml` | 0 | 综合 B0，Adam 1e-4，同 R 的衰减 |
| B | `33_b0_late_decay_mini.yaml` | 1 | 综合 B0，Adam 1e-4，在 20/50 轮后乘 0.1 |
| C | `33_b0_half_lr_mini.yaml` | 1 | 综合 B0，Adam 5e-5，同 B 的衰减 |

四组均 scratch60、batch16、workers4、FP32，每组单卡。A/B/C 的模型、loss、数据、seed 和训练预算相同。R 使用独立原 SeqTrack 网络、teacher 数据处理与原 loss，不继承生产 B0 的修订。

本轮服务器只读；由用户上传和启动，不由代理在服务器写入或运行训练。已有 v32 SeqTrack 成绩可复用，但用户当前选择重新训练 R。四组独立后台命令、日志与恢复方式见 [运行说明](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)。

2026-09-25追加D：B配方学习率全程×1.5，初始1.5e-4，GPU0；原A/B/C保留。新配置、同步文件和后台命令见[放大学习率运行说明](docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md)。

2026-09-25追加E：继承B，峰值学习率3e-4，前2000次更新从1.5e-7线性升至峰值，20/50轮完成后降至3e-5/3e-6，GPU1，warmup计入原71,700次更新。原R/A/B/C/D保持；E相对B同时改变学习率与warmup，不能独立归因。五个同步文件、独立后台命令和新终端tail见[追加E运行说明](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。本轮未访问或修改服务器，E由用户上传启动。

## 已完成的历史结果

v32 nuScenes-mini Car、seed42，全部 scratch60、71,700 次优化；评测 106 条轨迹、2,285 帧：

| v32 模型 | final60 Success / Precision | late-3 Success / Precision |
|---|---:|---:|
| SeqTrack reference | 51.823851 / 61.341356 | 51.884391 / 61.966448 |
| B0 | 49.650985 / 59.803063 | 49.175420 / 59.365062 |
| Full-GRU | 44.699125 / 55.191466 | 45.132385 / 55.249088 |
| Full-CfC | 43.659738 / 53.991247 | 43.488330 / 53.970824 |

证据见 [四组报告](artifacts/ct_checks/20260924_v32_four_arm_final/REPORT.md) 和 [训练审查](artifacts/ct_checks/20260924_v32_four_arm_final/training/TRAINING_REVIEW.md)。这些成绩不属于 v33，不能用于宣称本次修改已经有效。

## 入口与评测

唯一入口为`main.py`；本次唯一新增scratch配方的入口示例：

```bash
python main.py --cfg cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini.yaml --path DATA_ROOT
```

原四组须使用各自checkpoint和原log_dir恢复，仅piecewise从头运行，完整命令见[新增第5组说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)。五组完整后用`tools/compare_v35_piecewise_b0.py`比较，保留旧四组工具用于原四组复核。第二seed须等第一阶段通过后再运行，seed42独立SeqTrack参考直接复用。默认每5轮验证，训练结束自动评测58/59/60并保存逐帧记录及`results.json`。比较固定final60和late-3，不挑选最佳轮次。不要传旧`--preloading`参数；当前按需读取原始点云，每worker缓存256MiB。

## 验证与边界

v35原四组版本的本地实现、旧版本兼容、诊断无干预、真实Lightning入口和epoch恢复已验证；该版本完整pytest为515 passed、3 skipped，compileall与diff检查通过，详见[9月28日就绪报告](artifacts/ct_checks/20260928-185917_v35_four_run_readiness/REPORT.md)。两项跳过需要真实CUDA，另一项只测试缺失Lightning的环境；本地已安装Lightning时不适用。新增参数36,608（约0.99%）。第5组仅新增调度，实施与验证证据单列在`artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/`，不把旧测试结果冒充本次验证。此前三组实现过程保留在[历史实施报告](artifacts/ct_checks/20260928-182545_v35_implementation/REPORT.md)。

此前服务器快照检查 **314 passed、1 skipped**，真实 CUDA batch 的 forward/backward/Adam/commit 已通过，未保存工程 checkpoint。日志与报告位于 [v33 检查目录](artifacts/ct_checks/20260924-190116_v33_implementation/)。后续被动汇总增量单独记录本地验证；这些历史检查不替代正式结果验收。六组完成状态以 2026-09-26 复盘为准，无需把同一工程检查反复作为启动步骤。

本地修改后的验证命令：

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
```

nuScenes full、KITTI、CfC/GRU 和时间控制接口保留；接口存在不等于已经完成相应实验。`output/` 与既有 `artifacts/` 受保护，不删除、不覆盖。详细约定见 [AGENTS.md](AGENTS.md)、[实验协议](docs/EXPERIMENT_PROTOCOL.md)、[工具范围](docs/FORMAL_TOOLING.md) 和 [服务器路径](docs/SERVER_PATHS.md)。
