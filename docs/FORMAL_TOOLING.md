# CT-SeqTrack 当前工具范围：V41

本轮用 `tools/train_v41_joint_pair.sh` 准备两组各自的运行目录，共用 E57 增强库和固定 E2 前缀库，再通过 `main.py` 顺序训练。结束后 `tools/compare_v41_training_arms.py --require-q0-lateral` 核验两组的联合配置并配对比较 dev 固定轮次。[两组命令](V41_JOINT_PAIR.md)

逐轮更新前缀库的独立配方仍用 `tools/train_v41_joint.sh`；`tools/compare_v41_joint_search.py` 用于与历史 uniform 的固定 test 结果比较。[单组运行说明](V41_JOINT_TRAINING.md)

## 2026-11-21 训练历史对照

用 `tools/build_v41_history_augmentation.py` 建 E57 库，`main.py` 从头训练两臂并生成公共 E2 库。`tools/audit_v41_training_gt.py` 做训练集 GT 分叉，`tools/compare_v41_training_arms.py` 核验两组预算和固定轮次。可用 `tools/build_v41_fixed_replay.py` 提前从已保存的 E2 构建同一库。[完整命令](V41_HISTORY_TRAINING.md)

## 2026-11-21 q0 横向中心单次训练

用 `tools/build_v41_q0_replay.py` 离线生成 train 前态，`main.py` 开训，`tools/compare_v41_q0_lateral.py` 对照固定三轮历史 uniform test。[q0 单训命令与口径](V41_Q0_LATERAL_REPLAY.md)

## 2026-11-21 联合搜索

`tools/compare_v41_joint_search.py --joint-run RUN_DIR` 逐帧配对联合组与历史 uniform 的固定三轮 test。可附加同一运行 E60 的单轨延迟记录。[命令与口径](V41_JOINT_SEARCH.md)

## 2026-11-18 当前工具

| 要核查的内容 | 入口及产物 |
|---|---|
| v3 的 q0 轴权 A/B | 两个 Full YAML 均通过 `main.py` 启动，各自记录实际配置、数据与源码。固定 dev 58/59/60 齐备后，`tools/compare_v41_q0.py A_RUN B_RUN` 核对运行身份和预算，逐帧配对计算 q0 及最终框指标，报告六项复核判据。[命令与指标](CTSEQTRACK_V41_RUN_WORKFLOW.md) |
| v3 工程验证 | `tools/check_v41_server.py` 支持 A、B 两份配置的真实主辅更新。`tools/benchmark_v41_training.py` 在实际训练入口按窗口计时，可单独检查 v2 与 v3 公共训练路径。吞吐需注明源码、设备、课程、工作量及计时范围。[测速配置与记录](V41_TRAINING_PERF.md) |
| v1 的 KITTI 固定 test | `tools/compare_ct_v41.py` 读取 v1 Full 与适配 SeqTrack 的 58/59/60 轮产物，以 test 0019–0020 的原帧和初始化标志配对。两组均有正式输出。[已存比较器结果](../artifacts/ct_checks/20261117_kitti_final_research_review/official_comparison.json) |

## 历史工具 · 2026-11-16 exec v2

当时的 v2 运行流程规定：`main.py` 使用具名 CLI + YAML，`tools/launch_ct_v41.py --arm full` 按实际配置打印命令，不强加旧 epoch/batch/LR；仍不执行训练。无 `--set`、参数别名、新调度器或预登记门。默认每轮 test/完整保存，日志和结果路径见新运行页。本轮仅静态实施，服务器运行验收待执行。

下表中的 baseline 固定测试和 `tools/compare_ct_v41.py` 保留 **v1 final60/late3** 用途，不是任意 v2 新配方的比较门；旧 checkpoint 使用原 run 对应源码。

整理日期：**2026-11-08**。现行实验规则见 [实验协议](EXPERIMENT_PROTOCOL.md)，状态及证据索引见 [当前状态](PROJECT_STATUS.md)。本节按当前源码说明工具职责，不表示本次执行了工具或补齐运行验收。下方旧工具面只服务其对应源码/实验，不将v24或V40登记要求套到V41。

## V41 工具职责（含 v1 固定两组与 v3 q0 对照）

| 工具 | 职责与边界 |
|---|---|
| `main.py --cfg cfgs/ct_seqtrack/41_full_kitti.yaml` | 分派 `models/ct_v41/entry.py`，运行V41 scratch训练、同run完整epoch续训或评测；当前 CLI 及路径见 [V41 v3 运行流程](CTSEQTRACK_V41_RUN_WORKFLOW.md)。 |
| `tools/launch_ct_v41.py` | **只打印**两组独立nohup、日期目录、PID和tail命令，不创建目录、不执行训练。`--arm full`默认GPU0，`--arm seqtrack`默认GPU1；`--gpu`仅可覆盖单组。实际执行打印命令才会启动。 |
| `tools/compare_v41_q0.py A_RUN B_RUN` | 核对同预算、同初始化、数值环境和固定 dev 逐帧结果。输出三轴 q0 MAE、最终框 S/P、长轨表现和独立 seed 复核判据。 |
| `tools/check_v41_server.py [--cfg FULL_YAML]` | 默认使用 A 配置。可指定 B 配置执行真实 KITTI Full 主辅 forward、梯度隔离、BN/mode/RNG、FA2、同次 Adam 和两流 commit 检查。报告记录所选策略和逐步 q0 中心损失。每个主 batch 插入一个辅助 batch，检查产物写入新的工程目录。 |
| `tools/audit_v41_kitti.py` | 检查Car输入metadata与文件清单，识别四个已知缺失bin并保留空观测政策；报告写新的ct_checks目录。不运行模型，不证明点框几何或跟踪正确。 |
| `tools/verify_ct_v41.py verify [--resolved]` | **可选**静态语法和实际源码身份读取；`--resolved`另读取当前Full配置。只支持verify，无register动作，不依赖旧登记，不是训练/smoke前置，也不替代运行验收。 |
| 原始SeqTrack工程的 `tools/evaluate_kitti_v41_baseline.py` | 对该run固定58/59/60权重独立评测test，生成 `fixed-test/fixed_comparison.json`。属于已适配的服务器baseline工程，不是本地CT工具；不要用CT专有CLI调用baseline入口。 |
| `tools/compare_ct_v41.py` | 读取Full运行与baseline固定test目录，核对约定配方、Full60轮及辅助流完成、固定权重轮次/同run目录、共同数据身份/原帧集合和初始化计分，重算final60/late3。默认stdout；`--output`只接受ct_checks下不存在的新JSON文件。 |

v1 比较器 `tools/compare_ct_v41.py` 只接受0019/0020测试帧，开发集汇总不能替代输入。它检查baseline三个评测记录的实际源码一致性，并报告两组源码身份；**不依赖预登记，也不等于重新执行模型、重建全部几何或证明模块因果贡献**。Full同run源码/配置/数据/主辅采样/RNG的恢复身份由运行入口和checkpoint恢复逻辑核验，不能笼统写成比较器重新执行了这些检查。

## V41 运行身份与 11 月 8 日操作记录

V41由 `models/ct_v41/provenance.py` 在运行时自动记录实际Python源码等身份，CRLF/LF归一化。`41_registration.json`与外部baseline receipt保留历史但不作为CT启动、smoke或比较依赖；不刷新历史登记，不引入原始字节上传门。取消预登记没有取消同run完整主辅epoch恢复、有限数值与输入完整性检查。

11 月 8 日为本地文档同步，只做文本/链接/diff静态核对，**不运行环境检查、pytest、compileall、模型或CUDA smoke**，不访问或修改服务器、不启停训练。`output/`、旧artifacts、源码、正式YAML和冻结基线保持。文档中的启动命令是复现入口，不是新增运行授权。

11 月 8 日整理时读取的运行快照是 **2026-11-04 10:46:53（北京时间）**：V41已完成9轮并进入第10轮，baseline60轮完成，该审计未找到固定test结果；这不是11月8日实时状态。见 [KITTI审计](../artifacts/ct_checks/20261104_kitti_readonly_audit/REPORT.md)。因此不再把旧“未上传/未运行”当当前事实，也不能据训练启动反推所有专项验收已通过。

V40四组60轮已完成并有 [保存结果复算](../artifacts/ct_checks/20261102_v40_research_audit/results/REPORT.md)，但当次未运行完整 `compare_ct_v40.py`。以下V40工具与六阶段要求保持其历史用途，不要求对V41重做原两波并发或登记流程；已有运行是否缺某项证据须按实际报告单独说明。

---

# 历史工具面：2026-10-29 V40 补充

V40入口为 `main.py` + 四份 `cfgs/ct_seqtrack/40_*_mini.yaml`。新增：

- `tools/verify_ct_v40.py register`：仅标准库静态写入新登记，不导入模型、不执行环境检查。`verify` 供上传后核对；已有运行不得刷新登记掩盖差异。
- `tools/check_v40_server.py`：上传后显式执行 contracts/kernel/lifecycle/memory/performance/parallel 六阶段，失败明确记录，不降级通过。
- `tools/launch_ct_v40.py`：默认只打印两波GPU0命令计划；用户显式 `--execute` 且六阶段证据匹配时才启动四组从零训练。
- `tools/compare_ct_v40.py`：四组final60/late3、完整sampler/audit、候选/可信/搜索台账及固定R/C证据核验。

本机本轮不执行测试、编译、CUDA或环境检查；服务器操作由用户完成。工程输出仅进新建`artifacts/ct_checks/`目录。详见 [V40操作](CTSEQTRACK_V40_LAUNCH.md)。以下旧工具面保留原实验用途。

# CT-SeqTrack 工具范围

## 2026-10-28 v39工具

- `main.py` 单入口；`register_v39.py` 管理两组源码与配方登记，不匹配时列出 changed/missing/unexpected 文件。真实batch与profile在CUDA工作前共用这项只读校验，不自动刷新登记。
- `check_v39_full_batch.py` 验证真实CUDA生命周期、FA2调用、FP32主参数、几何双精度抽样误差和Adam后唯一提交；合成CPU模式不替代CUDA验收。
- `benchmark_v39.py` 分离plain延迟和trace，比较两种执行配方并记录递推工作量差异；不把BF16下不同crop当作同张量因果加速。
- 当前用户指定同GPU0并行，使用[v39操作页](CTSEQTRACK_V39_LAUNCH.md)两条独立nohup命令；`launch_v39_sequential.sh`仅保留为可选串行工具，不用于本轮并行启动。
- `compare_v39.py` 只比较完整final60/late3，并核对两组source/初始化；两档和阈值曲线完整保留。详见[v39操作页](CTSEQTRACK_V39_LAUNCH.md)。

## 2026-10-23 当前v38 clean1工具

- 当前入口`main.py`；五个v38工具为`launch_v38.sh`、`resume_v38.sh`、`check_v38_full_batch.py`、`compare_v38.py`、`register_v38.py`。登记只写新的`38_clean1_registration.json`，原登记不刷新。
- 新增`benchmark_v38_clean1.py`用于旧/新源码独立进程工程对照；默认20预热/100测量/3交替，真实性能仍需用户同卡执行。
- 当前比较器独立持有R/C指纹和核验函数，不导入历史比较脚本。旧v31–v37及B2专用工具只在对应冻结源码使用。
- 数值边界与恢复方法见[clean1说明](CTSEQTRACK_V38_CLEAN1.md)。以下工具列表按原日期保留，不能直接在当前精简树执行。

## 2026-10-22 v38 工具面

- `main.py` 分派 `ctseqtrackv38`；独立新模型和host，旧入口保留。
- `tools/launch_v38.sh`、`tools/resume_v38.sh`：登记三组scratch或同run完整epoch恢复，严格核验配置/source/run身份。
- `tools/check_v38_full_batch.py`：一次真实CUDA Full batch16×1024生命周期，工程报告新建于artifacts，不保存checkpoint。
- `tools/compare_v38.py`：只读核验三组预算/原帧/配置/source，重算final60/late3及原R/C，报告Full−standard、hold−standard及可信/运动风险。

具体见 [v38操作页](CTSEQTRACK_V38_LAUNCH.md)。下方工具为历史版本，不能混用新三组身份。

## 当前 v37 五组（2026-10-09）

- `main.py --cfg cfgs/ct_seqtrack/37_*_mini.yaml` 分派独立 v37；`tools/launch_v37.sh` 每次只启动指定的一组。五组配方与恢复命令见 [v37 操作页](CTSEQTRACK_V37_LAUNCH.md)，登记见 `37_registration.json`。
- `tools/check_v37_full_batch.py` 提供真实 Full 生命周期检查；`tools/check_execution_performance.py --mode all --candidate_profile optimized_bf16 --baseline_profile optimized_fp32 --inference` 检查两精度差异并配对测训练/闭环推理吞吐。20 步预热、100 步计时、3 次轮换顺序重复；保留真实 Trainer 日志和诊断。工程权重不用于正式初始化。
- `tools/compare_v37.py` 只读验证五组身份、完整预算和 final60/late-3 原帧；比较当前 B1 与旧强 B0/B1，原 R/C 门槛和漂移风险保留。FP32 Full 为预登记主结果，BF16 为配方比较。
- 工具报告仅写 `artifacts/ct_checks/` 新位置；不覆盖历史、关闭身份检查或自动追加训练。服务器命令由用户执行。以下工具仍仅服务其历史版本。

## 当前 v36 五组（2026-10-04）

- `main.py --cfg cfgs/ct_seqtrack/36_*_mini.yaml` 分派独立 v36 入口，A/C/D/E/F 五组；旧入口保留。准确命令见 [v36 启动/恢复](CTSEQTRACK_V36_LAUNCH.md)。
- `tools/launch_v36.sh ARM GPU` 每次启动用户指定的一组；新目录、独立 log/pid，不循环扩充实验。代理不在服务器执行。
- `tools/check_v36_full_batch.py` 仅一次 A 真实 batch16×1024 FP32 forward/backward/Adam/commit，含核验记录、模块梯度与显存；报告写 `artifacts/ct_checks/` 新文件，不保存工程权重。CPU 合成验证不能替代该 CUDA 检查。
- `tools/compare_v36.py` 只读核对五臂、强 scaled B0/B1 与原 R/C/W 来源，固定 final60/late-3、预算、门槛、漂移及机制诊断；最高验证分另外报告。登记见 `36_registration.json`，不存在未完成实验的正式收益结论。
- 新比较输出仍限 `artifacts/ct_checks/` 新文件；退出码 0 表示 A 通过原硬门、1 表示有效完整证据下 A 未通过、2 表示证据无效。对照得分和漂移风险另列，不能将返回 0 解释为各模块贡献均成立。
- 以下旧工具只服务对应历史版本，不能关闭旧身份检查用于新 v36。


## 当前 v35 完整链六组（2026-10-02）

- 正式入口仍是`main.py`。六配置为`35_{b1,b1_b2,full}_w_{normal,scaled}_lr_mini.yaml`，全部seed42/scratch60；`tools/launch_v35_full_seed42.sh LABEL GPU`每次只启动对应一组，日期新目录、nohup、独立log/pid。当前[完整命令](CTSEQTRACK_V35_FULL_SEED42_LAUNCH.md)取代旧B0六组启动安排。
- `tools/check_v35_full_batch.py`：一次真实Full batch16前向/反向/Adam/commit、完整候选JSON、显存/耗时；输出独立ct_checks报告，不保存权重。服务器由用户执行一次，本地不能伪称CUDA已通过。
- `tools/compare_v35_full.py`：六组完整58/59/60及60轮预算验证、精确登记source/config、旧B0同LR复用来源、固定R/C/W帧SHA；原帧重算，分别报告同LR链式增量、最强scaled比较、候选选择与恢复诊断、原门槛和漂移风险。
- 比较默认stdout JSON，`--output`只允许ct_checks新文件；0=至少一个Full通过原门，1=完整证据无Full通过，2=证据无效。风险及相对最强B0增量另列，返回0不等于所有研究结论已成立。
- 候选记录只在v35非B0评测扩展，字段见[格式](CTSEQTRACK_V35_FULL_CANDIDATES.md)。源码登记`35_full_seed42_registration.json`不授权旧checkpoint迁移；当前各组正常epoch恢复保留自身身份和source。
- 旧比较/检查工具继续保留服务旧实验，不将新Full塞入旧B0比较器。下面工具面为历史记录。

## 当前 v35 工具（2026-09-29：四组恢复与新增piecewise）

- 唯一正式入口保持`main.py`。原`35_b0_w_{quarter,half,normal,scaled}_lr_mini.yaml`四组已由用户暂停，各自完整checkpoint均为036；更新同一活动项目后使用原cfg、原log_dir和各自036 checkpoint恢复，GPU0/0/1/1，第37轮继续。原`run_manifest.json`保持，`resume_manifests/`追加迁移来源，旧日志使用追加写入。
- 新增`35_b0_w_piecewise_lr_mini.yaml`在GPU1 scratch60：1–20轮5e-5、21–50轮1e-5、51–60轮5e-6；seed42、无warmup、W及其他单组参数不变。`35_b0_w_piecewise_lr_mini_seed52.yaml`仅作预备。首阶段五次、358,500更新；达标后只补胜出完整配方seed52及独立SeqTrack seed52，条件上限七次、501,900更新，不自动启动。
- `tools/check_v35_batch.py`仍保留一次真实CUDA batch16的forward/backward/Adam/commit工程检查能力，不保存checkpoint；本次只改调度，不把重复运行它设为恢复四组或启动第5组的前置。
- `tools/compare_v35_b0.py`独立保留，用于原四组完整58/59/60结果比较，`--quarter --half --normal --scaled`均必填；不将piecewise混入旧四组入口。
- 新增`tools/compare_v35_piecewise_b0.py`：保留旧入口的固定对照参数，再要求`--quarter --half --normal --scaled --piecewise`五组完整。以原逐帧SHA锁定R/C/旧B0/W-quarter，原硬门与漂移风险分离；排序为final60 S/P、较低初始LR，仍同分同初始LR时原half优先piecewise。旧四组和新配方的源码差异仅接受本次登记的精确before/after文件SHA；按原manifest和追加resume来源核验，不能泛化取消源码检查。
- 两个比较入口均重算原始帧，保留>5m/>10m、raw0–2和raw可见coarse/fine等解释性统计。可附加锁定完整配方的新版seed52及原正常SeqTrack seed52，分别报告固定R/C与同seed差值，并核对原reference协议；不得重新选择学习率调度。
- 比较默认stdout JSON；`--output`只允许ct_checks下的新文件。返回0表示原硬门有通过者，1表示完整证据未达标，2表示证据无效；漂移风险和第二seed稳定性另行报告，不能把返回0直接写成Full可启动。
- 结构与完整CLI见 [v35说明](B0_V35_INTEGRATED.md) 和 [piecewise与恢复操作页](CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)。本次实施及验证证据位于`artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/`。旧工具和配置继续用于重现，禁止覆盖旧实验。

## v34 历史工具

- 唯一训练/评测入口仍为 `main.py`，两份配置为 `34_b0_context_mini.yaml`、`34_b0_context_w4_mini.yaml`；用户GPU0/1单卡从头启动。
- `tools/check_v34_batch.py`：一次真实batch16的forward/backward/Adam/commit，不保存checkpoint；按v34记录身份。只在本次结构修改后由用户检查一次。
- `tools/compare_v34_b0.py --reference R_RUN --baseline C_RUN --old-b0 OLD32_RUN --s S_RUN --w W_RUN`：只读重算与验收，默认stdout JSON；`--output`仅允许ct_checks内新文件。返回0通过、1未达标、2证据无效。
- `tools/compare_v34_lr_grid.py`：本轮八组只读比较；单独指定旧R为`--registered-reference`，本轮新R正常/减半与S/W六组另外传入。固定旧R/C硬门，所有候选固定final60排序，保留late-3及同LR的W−S。
- 具体命令见 [v34最新八组操作说明](CTSEQTRACK_V34_MINI_LAUNCH.md)，配方见 [八组协议](B0_V34_LR_GRID.md)。本轮不新增Full训练；八组LR是用户明确授权的扩展，原33配置与原34半LR配置保持。

## v33 历史工具

唯一训练和闭环评测入口是 `main.py`。本轮由用户上传本地代码并启动 R/A/B/C 四组，物理 GPU 为 0/0/1/1；服务器操作对代理仅开放只读。运行命令见 [v33 运行说明](CTSEQTRACK_V33_MINI_LAUNCH.md)，实验不变量见 [协议](EXPERIMENT_PROTOCOL.md)。

## 正式入口

```bash
python main.py --cfg cfgs/ct_seqtrack/33_seqtrack_ref_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_late_decay_mini.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml --path DATA_ROOT
```

默认 scratch60、batch16、workers4、FP32、每 5 轮验证，结束后自动评测 58/59/60。不传 `--checkpoint` 或 `--init_checkpoint` 开始新正式实验；后者始终被禁止。

CLI 支持 `--cfg`、`--path`、`--tag`、`--log_dir`、`--checkpoint`、`--test`、`--seed`、`--batch_size`、`--epoch`、`--workers`、`--check_val_every_n_epoch`、`--trainer_devices`、`--accelerator`、时间控制及工程检查参数。正式预算由配置校验固定，能解析参数不代表允许改变正式预算。不要传旧 `--preloading`、`--proposal_mode`、`--gpus` 或 `--precision`；这些不是当前入口参数。物理 GPU 用环境变量 `CUDA_VISIBLE_DEVICES` 指定，每组 `trainer_devices=1`。

`--no_late3` 会跳过训练后的全部自动评测，不只是改变汇总方式；本轮四组不要使用它。

不指定 `--log_dir` 时，目录为 `output/YYYYMMDD-HHMMSS-33_ARM-TAG`，R 的 ARM 为 `seqtrack_ref`，其余为 `b0`。独立 `--test` 默认追加 `-test`。建议后台命令显式给每组不同的新目录，便于定位日志和 PID。

## 恢复与独立评测

同 run 恢复：保持原配置和原 `--log_dir`，仅添加该 run 完整 epoch 边界的 `--checkpoint RUN/formal_checkpoints/epoch=NNN.ckpt`。不得换学习率配方后复用 checkpoint。

```bash
python main.py --cfg cfgs/ct_seqtrack/33_b0_mini.yaml --path DATA_ROOT --checkpoint RUN/formal_checkpoints/epoch=060.ckpt --test --log_dir NEW_EVAL_DIR
```

独立评测要求新空目录，不能把原训练或 reference 目录作为评测输出目录。入口在写 metadata 前验证已有 manifest 与 checkpoint 身份；合法恢复保留首次 `resolved_config.yaml` 和 `run_manifest.json`。v32 checkpoint 不兼容 v33，旧模型必须使用其冻结源码评测。

## 已有检查与可选工程工具

此前服务器快照留存 **314 passed、1 skipped** 和真实 CUDA batch 通过记录，证据目录为 `artifacts/ct_checks/20260924-190116_v33_implementation/`。后续被动汇总增量的本地验证单独记录，不将旧记录视为最新源码逐文件服务器复验；不要求用户在每次正式启动前重复同一检查。

`tools/check_v33_batch.py` 仅用于代码或环境改变后的必要定位：执行一次真实 batch 的 forward/backward/Adam/commit，写独立报告，不保存 checkpoint，不进入正式初始化。

`tools/launch_v33_b0.py` 当前只启动 A/B/C 三个 B0，并要求读取已有 reference 与旧 B0。它属于“复用 R”的三组启动路径，不能当作本次 R/A/B/C 四组独立命令的替代。

## 结果比较

```bash
python tools/compare_v33_b0.py --reference R_RUN --old-b0 OLD_B0_RUN --a A_RUN --b B_RUN --c C_RUN
```

工具只读 JSON/JSONL，重新积分 final60 与 late-3；reference 可为已完成 v32 R 或本次重跑的 v33 R，本轮使用新 R。返回码 0 表示至少一组在 final60/late-3 的 Success/Precision 四项均不低于 reference，1 表示完整证据未达标，2 表示证据无效或不完整。

诊断固定使用旧 B0 的帧键和裁剪分组，报告 raw/crop/sampled、移动与缺测交叉、同帧 coarse/fine、失跟与右删失；可缺失的旧诊断必须报告覆盖率，不能静默填零。该工具不训练、不推理、不替代正式实验。`compare_v32_baselines.py` 保留为 v32 结果审查工具。

本地修改按需运行 pytest、compileall 和 `git diff --check`。保护 `output/` 与既有 `artifacts/`；新增结果或检查产物使用独立目录。旧 31/32 工具和协议按 Git 历史复现，不恢复到当前活跃入口。
