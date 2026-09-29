# CT-SeqTrack 当前状态（2026-09-29）

## v35 原四组暂停恢复与新增第5组piecewise（当前）

本节覆盖此前首阶段四次/条件最多六次安排。用户已暂停原四组，接下来更新同一活动项目，再从各自完整`epoch=036.ckpt`恢复四组（GPU0/0/1/1）并从第37轮继续，在GPU1从头启动piecewise；代理只读服务器，不执行停止、上传或启动。首阶段五次、300epoch、358,500更新；仅通过后保留两次条件seed52复验，总上限七次、420epoch、501,900更新，不自动追加。

- [x] v34八组已完成并复盘；W-quarter最接近总体目标，但未通过总体/移动双门槛，见 [八组报告](artifacts/ct_checks/20260928_v34_eight_run_review/REPORT.md)。下方旧待启动条目均为历史。
- [x] 用户已批准统一v35：首测对齐、coarse合法历史条件、fine候选局部真实点；保留W课程、loss、采样和共享头。详见 [结构说明](docs/B0_V35_INTEGRATED.md)。
- [x] 明确漂移方案1：严重漂移与持续失跟属于本轮优化范围，不能仅统计或留到Full处理；保留原总体/移动硬门，不新增漂移数值硬门。规则见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。
- [x] 结构说明已对应首次缺测、历史参照不足、fine错误细化及后续crop偏移给出综合改法；没有把漂移留到Full再处理或额外堆模块。
- [ ] 正式结果产生后核对上述机制的实际效果与剩余问题；实现完成不等于已解决漂移。
- [x] 完成本地实现、旧版本兼容、诊断不干预、完整epoch恢复与pytest/compileall/diff检查。最新四组版本全套515 passed、3 skipped（两项需要真实CUDA，一项缺失Lightning分支在已安装环境不适用），无失败；见[就绪报告](artifacts/ct_checks/20260928-185917_v35_four_run_readiness/REPORT.md)。此前CLI/测试路径问题的修复过程保留在[历史实施报告](artifacts/ct_checks/20260928-182545_v35_implementation/REPORT.md)，操作见[运行说明](docs/CTSEQTRACK_V35_MINI_LAUNCH.md)。
- [x] 原四组seed42已在GPU0/0/1/1运行，9月29日只读状态为第36/37轮、共同最新validation35；[服务器快照](artifacts/ct_checks/20260929-130405_v35_four_run_status/server_snapshot.json)与[同轮历史对照](artifacts/ct_checks/20260929-130405_v35_four_run_status/historical_comparison.json)已留存。尚无正式final60/late-3，中途分数不作验收。
- [x] 用户明确追加piecewise：第1–20轮5e-5、第21–50轮1e-5、第51–60轮5e-6，其他单组参数保持。相对原half前20轮相同，之后为2倍和10倍；这是完整调度对照，不是一个固定LR倍数。新配置为`35_b0_w_piecewise_lr_mini.yaml`及seed52预备配置。
- [x] 完成本次调度、旧四组精确源码迁移恢复及五组比较验证：全仓554 passed、3 skipped，compileall、diff及五条Bash命令检查通过；25份旧配置文件及身份SHA不变。证据见[本次报告](artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/REPORT.md)。本地更新由用户提交GitHub，服务器拉取后恢复；原四组工具和配置身份保留，此次不重复要求真实CUDA batch检查。
- [x] 9月29日13:33只读确认用户已停四组，原PID均不存在、日志有SIGTERM，各自完整checkpoint均为036；停止后的证据见`artifacts/ct_checks/20260929-131823_v35_piecewise_schedule/process_snapshot_2.json`。
- [ ] 用户上传本次源码后保持原配置、原log_dir及各自036 checkpoint恢复，从第37轮继续；最初`run_manifest.json`保留，新增`resume_manifests/`记录迁移来源，旧日志仅追加。随后仅新增piecewise scratch60，物理GPU1。操作见[第5组与恢复说明](docs/CTSEQTRACK_V35_PIECEWISE_LAUNCH.md)，不重新执行[旧四组scratch命令](docs/CTSEQTRACK_V35_MINI_LAUNCH.md)。
- [ ] 五组全部完成后使用`tools/compare_v35_piecewise_b0.py`与固定R/C原门槛比较；报告final60/late-3的>5m、>10m与未恢复失跟，区分原硬门结果和风险复核结论。排序依次为final60 S/P、较低初始LR；仍同分同初始LR时原half优先piecewise。
- [ ] >10m计数增加时暂停Full待复核，结合轨迹、持续时长及逐轮表现判断；小幅孤立波动不自动判失败，明显持续退化须处理并讨论，不为某个漂移计数自动增加训练。
- [ ] 仅通过后锁定胜出完整配方补seed52和独立SeqTrack seed52；保留两次条件复验后总上限七次、420epoch、501,900更新，当前仅登记五组seed42，不自动执行后续。
- [ ] 新版尚无正式成绩；通过及稳定性证据充分后再讨论Full。

## 历史：v34 八组安排（已于9月28日完成，以本页顶部状态为准）

- [x] 新增SeqTrack半LR与S/W正常、quarter档；保留原R、S-half/W-half配置身份，其他参数不变。第三档2.5e-5依据见 [八组协议](docs/B0_V34_LR_GRID.md)。
- [x] 最新启动页已给GPU0 R正常＋S三档、GPU1 R半LR＋W三档，独立nohup、带日期目录和新终端tail；比较工具支持全部八组，固定旧R/C晋级门。
- [x] 服务器只读核对：原环境和mini数据有效，旧六组完整结束无日志异常，本项目没有运行进程；服务器尚无34配置，用户需上传本轮代码。
- [x] 本轮复用已有只读Lightning2.0.2完成全套440 passed、3 skipped（两项CUDA、一项缺Lightning条件不适用）；新增S/W实际Trainer恢复通过，详见 [八组就绪报告](artifacts/ct_checks/20260926-161231_v34_eight_run_readiness/REPORT.md)。
- [ ] 上传后仅执行一次v34真实CUDA batch，再按 [最新启动说明](docs/CTSEQTRACK_V34_MINI_LAUNCH.md) 启动八组；代理不写服务器、不启动训练。
- [ ] 八组固定final60/late-3比较，保留弱观测和移动守底；原结构主对照仍为S-half−旧C，同LR内W−S评估课程。

## 历史：v34 实施记录

- [x] 用户批准共同 query context：历史局部几何/存在性/预测支持，与 live coarse pooled256＋当前支持分别编码，只供 q0/modes 使用；共享头、loss、Full梯度边界保持。
- [x] 登记 S=1/3/3/8、W=1/4/4/8，均 C 配方5e-5、20/50降档、无warmup，seed42 scratch60；复用现有 R/C/旧B0。版本身份与旧 v33 分离。
- [x] 完成本轮本地结构、兼容、恢复、预算与比较工具验证：408 passed、8项Lightning/CUDA环境跳过；真实网络CPU epoch恢复逐位一致。以 [实施报告](artifacts/ct_checks/20260926-154028_v34_implementation/REPORT.md) 为准，真实CUDA检查仍待用户执行。
- [ ] 用户同步后执行一次 v34 真实 CUDA batch 检查，再在 GPU0/GPU1 分别启动 S/W；代理不上传、不访问服务器或启动训练。见 [v34运行说明](docs/CTSEQTRACK_V34_MINI_LAUNCH.md)。
- [ ] S/W 全部完成后固定比较 final60/late-3；总体四项达到 R，同时31条曾移动轨迹四项至少不低于C，才进入 Full 规划。
- [ ] 若两组均未达标，先检查新历史/语义路径实际依赖，再登记一个后续改动，不继续扩大LR/模块网格；Full质量语义与支持阈值风险留待后续。

## v33 六次运行已完成：当前决策覆盖下方历史启动状态

- [x] R/A/B/C/D/E 已全部完成并拉回本地：1 组 SeqTrack + 5 组 B0，每组60轮、71,700次更新，58–60轮正式评测帧键与分数重算通过。
- [x] 完成训练曲线/实际学习率、逐帧缺测/运动/定位/失跟、代码读取路径审查，见[六组复盘](artifacts/ct_checks/20260926_v33_six_recipe_review/REPORT.md)。本次未访问服务器、未改训练源码或正式配置。
- [x] 已测最佳 B0 为 C：final50.456237/59.256017，late-3为49.373815/58.074398；固定5e-5、20/50轮降档、无warmup作为后续优化基准，不继续放大LR。
- [ ] **追平条件未达成**：C对SeqTrack的final S/P仍差1.367615/2.085339个百分点，late-3差2.510576/3.892050；相对旧B0仅Success微升、Precision下降。
- [ ] 下一版实现建议：q0显式历史几何/支持输入与coarse语义上下文、保留共享head并改善fine定位、窗口1/4/4/8提高完整预测历史覆盖；尚未实施，不宣称收益。
- [ ] 下一版完成必要工程验证后，仅新增一组C配方seed42 scratch60，复用现有R/C对照；固定final60与late-3，同时检查移动退步、首测缺测和失跟长度。
- [ ] 综合收益明确后再补随机种子及必要单项消融；暂不开展Full独立诊断。

下方2026-09-25及更早内容是历史记录；“仍在运行”“等待启动E”等均已由本节完成状态覆盖。

## 2026-09-25 启动与实现记录（历史）

2026-09-25追加：用户希望适度放大学习率，已配置D为B配方全程×1.5（初始1.5e-4、20/50轮衰减），GPU0、其他参数不变。原R/A/B/C保留；新增正式配置与入口登记已通过20项相关本地测试，由用户上传启动，见[追加D说明](docs/CTSEQTRACK_V33_SCALED_LR_GPU0.md)。

当前进度：原R/A/B/C/D已由用户启动，最近观察仍在运行；本轮未访问或修改服务器，不推断五组已完成。用户追加E为B的峰值学习率×3与2000步线性warmup联合配方，GPU1，预算仍71,700次更新；E与B的差异不能独立归因为LR或warmup。新配置、五个同步文件及命令见[追加E说明](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。

## v33 综合 B0（当前，覆盖下方历史安排）

- [x] v32 四组已完成60轮与58–60评测，B0 final49.650985/59.803063，SeqTrack51.823851/61.341356；见[9/24审查](artifacts/ct_checks/20260924_v32_four_arm_final/REPORT.md)。
- [x] 用户批准一次整合中心参数化、BC监督、端点归约、可信状态，并以相同综合B0比较原配方/延后衰减/半学习率。
- [x] v33模型、正式配置与恢复身份、逐帧评测及只读比较工具已实现；用户最新选择重跑独立SeqTrack，组成R/A/B/C四组。
- [x] 本地首次契约测试285 passed、7 skipped；此前最终服务器原环境314 passed、1 skipped，真实CUDA batch16前向/反向/Adam/commit通过，见[v33说明](docs/B0_V33_REPAIR.md)。
- [x] 最新本地配置文档与四条后台命令已统一为GPU0/0/1/1；修复中文编码损坏，保留历史报错修复。此次服务器仅只读检查。
- [x] 修正SeqTrack缺失目标点诊断被汇总为0的问题，改为None并记录覆盖数；本地相关测试40 passed、2 skipped，训练与评分数学不变。
- [x] 用户已启动原R/A/B/C及追加D，最近观察仍在运行；原四组[独立命令与tail](docs/CTSEQTRACK_V33_MINI_LAUNCH.md)作为运行记录保留。
- [x] E配置、按更新warmup与按epoch衰减、恢复与学习率日志已接入；本地343 passed、3 skipped，compileall/diff通过，原R/A/B/C/D配置SHA保持。
- [ ] 用户同步E的五个运行文件后，从头启动GPU1独立scratch60；[启动与tail命令](docs/CTSEQTRACK_V33_X3_LR_GPU1.md)。
- [ ] 四组完成后比较final60/late-3双指标、缺测/移动交叉分组、coarse→fine和失跟长度；目标B0四项均达到SeqTrack。
- [ ] 综合结果明确后才安排必要单项消融与Full迁移；本轮不开展Full独立诊断。

下方内容均为历史记录；其中“v32尚无成绩”和旧部署/启动安排以本节为准。当前本轮服务器只读，由用户上传启动。

## v32 B0 修复（覆盖下方历史“下一轮”安排）

用户已批准只修 B0、递推训练与必要连接，保留 B1/B2/B3 机制。活动代码原位升级 v32，独立 SeqTrack 保留原 teacher4 与有效历史重采样。实现和检查记录见 [v32 修复说明](docs/B0_V32_REPAIR.md)。

- [x] B0局部观测、前景梯度隔离、真实点token、历史支持定位、损失归约与递推/尾批BN已实现。
- [x] 独立SeqTrack、v32配置/身份、采样审计和四次实验只读验收工具已接入。
- [x] 最新本地249 passed、3 skipped，包含四组模型真实Lightning合成入口与中断恢复；compileall/diff通过。
- [ ] 授权服务器环境中的真实批次前向、短训练与 CUDA 检查。
- [ ] 按用户最新安排先跑 SeqTrack/B0/Full-GRU/Full-CfC，各 seed42，物理 GPU 0/0/1/1，scratch60；[后台命令与 tail](docs/CTSEQTRACK_V32_MINI_LAUNCH.md)。
- [ ] 本轮检查 seed42 B0 final60 S/P 分别不低于 SeqTrack 超过 2 个百分点，报告两个 Full 对 B0 的差异及四组 late-3。
- [ ] 后续补齐 B0/reference seed52，才判定原计划的双 seed 条件；现有双 seed 比较工具不接收本轮四臂替代输入。
- [ ] B0 达标后再处理模式、质量与状态写入问题；本轮 Full 只观察现有机制的联合结果。

下方为 v31 结果与清理的历史状态；v32 尚无正式成绩。

## 最新实验结论（2026-09-21，覆盖历史待上传/待训练状态）

- [x] 服务器与本地正式版本64ad056；v31 B0/Full-CfC/Full-GRU三组60轮、58–60 late-3均完成。三组均71,911次Adam、1,146,480训练端点曝光；评估106轨迹/2285帧，JSONL重算一致。
- [x] 三组final S/P分别23.959519/27.834792、22.491247/29.352297、22.839169/31.844639；三组确实不同，无旧标定回退。
- [x] [完整结果、SeqTrack比较及优化方向](artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)：真实权重/数据只读CPU探针完成，记录当前帧近全FG、BN影响、模式重复、记忆不纯和物理历史污染。
- [ ] **性能条件未达成**：两个Full的final Success低于同版B0，B0也显著低于v30历史参照；不能标为mini晋级通过。
- [ ] 落实报告中的统一修订：保护分割概率语义但保留共享特征联合学习；局部观测坐标与有效点token；尾批BN/端点归约/连贯扰动和长递推；模式去重、统一质量与身份支持、候选一致的记忆及可信速度写入。
- [ ] 下一轮先固定新版B0/Full-GRU；CfC保留同接口时间机制对照，不用扩大点数/增加模型候选代替根因修复。
- [ ] 修复后报告final60和late-3、当前帧身份质量、目标模式、写入纯度、持续恢复及成本；mini达标后推进full/KITTI和模块/真实时间消融。

本轮仅读取服务器，未上传/改服务器文件，未改训练模型或配置、未启动训练；新增本地报告与状态文档。已有三组工程运行成功，不等于原计划中的独立100步检查被单独执行。

## 2026-09-22 工作树收敛

本轮仅保留 v31 活动源码与文档，不实施上述算法修订。旧版配置、工具与说明由 [历史索引](docs/HISTORY_EVIDENCE_INDEX.md) 的 Git 对象复现，既有 output/artifacts 原样保护。本轮清理检查以 artifacts/ct_checks/20260922_v31_slimming/ 中的统一审计为准，不沿用历史测试数量宣称通过。
