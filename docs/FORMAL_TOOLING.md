# CT-SeqTrack 工具范围

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
