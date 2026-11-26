# 四槽 uniform 与局部三维中心票

新配置沿用旧 uniform 的 B0、B1、MTP、双 ROI 和状态提交。前三槽为 q0、物理时间先验和可信状态传播，由旧 B3 的 S/P 效用决定胜者。第四槽用车体模板和当前实际采到的点生成一个三维中心，沿用 q0 航向。门比较它与本帧前三槽胜者的预期积分差。差值严格为正且第四槽有效时才切换，平分沿用前三槽胜者。

模板优先取合法首框内的唯一点。若首框不足三点，则以空模板推理。连续两次可信本地写入、且第二次满足原测速条件后，用第二次获选框内的唯一前景支持点在该框车体坐标系建一次模板，下一步开始使用。模板只属于当前递推分支。训练窗口的历史重置和旧前态库均不能凭空补出模板支持点。

最终胜者决定下一帧主裁剪。q0 或局部票各自通过最终框的 T ≥ 0.5，并有至少三个唯一预测前景点，才能更新可信位置。速度更新继续要求相邻可信测量及旧运动一致性条件。门或模板匹配分都没有额外的身份否决权。新配方的主路、整轨辅助及评估均不实例化 R、q0 横向修正、远搜或跨扫描确认。

## 训练与恢复

普通受扰历史保留原共同 XY、航向和同供体连续 XYZ 误差差分，再将供体末框带符号 Z 加给整段。中途重置读取同一训练轨被替换片段之前的真实可信测量记录。几何扰动影响点云裁剪与观测差分，最近可信时间与测速时间来自真实提交。已有困难/正常配对以完整真实前态替换 branch 0 的部分单步槽。精修回归的资格来自实际参与精修的输入点，质量和写入标签则依据最终框。

学习门使用训练轨整轨辅助流的自然样本。每轮在线累积 18 维特征与真实逐帧 S/P 积分差的充分统计量，最后一次主辅更新后拟合标准化岭回归，惩罚 20，截距不惩罚。首轮拟合前由前三槽输出。评估和下一轮递推使用同一组拟合系数，系数、均值与尺度随模型权重保存。主路的重复扰动视图不用于校准。B0/B1、投票头及 B3 通过各自原定主辅流从头训练。

在项目根目录运行。三个训练输入分别是连续残差库、E10/E60 配对前态和 E60 全训练集因果账本。本机已生成三份文件。E60 账本包含 441 条训练轨的 19,081 个预测前态，逐行核对了时间戳。WSL 对 12 条训练轨各取一个中途端点，覆盖四个普通训练分支，worker 和主进程构造的 36 个批字段逐项相同。[账本及数据路径校验](../artifacts/ct_checks/20261126_uniform_local_prefix/validated.json)

```bash
DATA=/home/lishengjie/data/cxtrack/training
BANKS=artifacts/ct_checks/20261121_v41_local_speed_gpu0
PREFIX=artifacts/ct_checks/20261126_uniform_local_prefix/uniform_e60.sqlite
```

`DATA` 指 KITTI 的 `training` 目录，本机 WSL 可设为 `../cxtrack3d/cxtrack3d-main/data/KITTI_tracking/training`。复用以上输入可直接运行下方训练命令。若要重建 E60 账本，在包含冻结运行和对应提交的仓库执行以下命令。

```bash
DONOR=output/20261118-v41-q0-dev-gpu0-180229-k1yozphm/uniform
mkdir -p artifacts/ct_checks
BUILD=$(mktemp -d "$PWD/artifacts/ct_checks/uniform-prefix-XXXXXXXX")
mkdir "$BUILD/donor"
git archive b819e76 main.py requirement.txt models datasets utils cfgs | tar -x -C "$BUILD/donor"
PREFIX="$BUILD/ledger.sqlite"
python tools/build_v41_local_prefix.py --donor-source "$BUILD/donor" \
  --donor-run "$DONOR" --path "$DATA" --output "$PREFIX"
```

若需重建另外两份输入，使用同一冻结源码、运行和数据根目录运行以下命令。

```bash
BANKS="$BUILD"
python tools/build_v41_history_augmentation.py --donor-source "$BUILD/donor" \
  --donor-run "$DONOR" --path "$DATA" --output "$BANKS/history_e57.json"
python tools/build_v41_q0_replay.py --donor-source "$BUILD/donor" \
  --donor-run "$DONOR" --path "$DATA" --output "$BANKS/q0_prefixes.json"
```

正式配方为 KITTI Car、seed 42、batch 16、60 轮，主路 1/4/4/8 窗口和原整轨辅助预算。选择一张可用 GPU，在项目训练环境执行。运行日志会记录实际源码、输入库指纹、门的样本数和拟合状态。

```bash
CFG=cfgs/ct_seqtrack/41_uniform_local_vote_kitti.yaml
mkdir -p output
RUN=$(mktemp -d "$PWD/output/v41-uniform-local-vote-XXXXXXXX")
ARGS=(--cfg "$CFG" --path "$DATA" --log_dir "$RUN"
      --v41_history_bank "$BANKS/history_e57.json"
      --v41_q0_replay_manifest "$BANKS/q0_prefixes.json"
      --v41_local_prefix_bank "$PREFIX")
python main.py "${ARGS[@]}"
```

同一运行从完整 epoch 边界恢复时，复用以上参数和运行目录，追加 `--checkpoint "$RUN/formal_checkpoints/epoch=NNN.ckpt"`。独立评估只需要配置、KITTI 路径、该运行的 checkpoint 和新的评估目录。

## 固定轮次验收

```bash
python tools/compare_v41_local_vote.py --run "$RUN" > "$RUN/local_vote_comparison.json"
```

工具读取 E58、E59、E60 的 test 完整逐帧记录，与归档 uniform 的同轮相同物理帧配对。基线回执默认位于 `artifacts/ct_checks/20261121_q0_final_review/formal/test_verification.json`，移机时可用 `--baseline-receipts` 指定。S/P 计入各轨合法首框。局部漏搜指整扫描中至少三个真实目标回波，而主备窗并集里一个也没有。报告区分避免漏搜、新增漏搜和误差超过 10 m 的预测帧。

完整集合、去掉 0020/122、去掉最大净减少漏搜轨分别重算。E60 单独选择最大轨，末三轮按三轮合计选定同一轨。两种排除方式选中同一轨时复用同一份统计。原冻结实验的 186 个净减少漏搜中，185 个来自 0020/122；该结果是本工具复算的工程参照。新模型的正式成绩由上述单次从零训练产生。

## 六候选诊断与四候选工程验证

对旧六候选 E60 的 6,304 个预测步，本地重放中远巡触发 2,541 次，进入选框 0 次。q0 横向修正有 3,328 次非零，最大幅度低于 1 mm。局部票有 66 次选中当帧唯一 IoU ≥ 0.5 的框，也有 238 次写入事后未满足定位或支持点质量条件。一次误写会影响后续的裁剪与测速。这些结果支持保留局部测量、分别验证选框和写入的改法。各模块对正式六槽降分的独立贡献仍未量化。[六候选分支诊断](../artifacts/ct_checks/20261126_v41_branch_budget_review/REPORT.md)

下述整集 GPU 回放和两轮训练核对在补回稀疏首框模板逻辑前完成。使用旧 uniform E60 权重、原局部票供体和归档学习门，在 WSL RTX 4060 Laptop GPU 上从合法首框分别递推。完整 dev 共 18 轨、1,336 个预测步，旧 uniform 的 S/P 为 75.7829 / 89.2578，当时的四槽实现为 75.8087 / 89.2504，共 7 次选择局部票。强制关门时，1,336 步的输入点 ID、q0、前三槽 S/P/T、胜者、可信写入和速度全部逐值一致。开门时发生首次分叉的 4 条轨均从正收益的第四槽接管开始。[配对 dev](../artifacts/ct_checks/20261126_uniform_local_validation/dev.json) · [逐帧前缀核验](../artifacts/ct_checks/20261126_uniform_local_validation/prefix_parity.json)

同次 WSL GPU test 使用相同冻结权重和训练集拟合的门，两臂各从合法首框独立递推。6,304 个预测步逐帧核对了实际主窗和主备窗并集内的标注点数。局部票在 13 条轨迹上共获选 106 次，99 次写可信位置，89 次更新速度。持续恢复指 IoU < 0.1 进入丢失后，在连续三个相邻扫描上恢复到 IoU ≥ 0.5。分母包含轨迹结束时仍未恢复的事件。[补模板修正前的完整 test 与逐轨敏感性](../artifacts/ct_checks/20261126_uniform_local_validation/host_test.json)

| 修正前冻结回放，KITTI Car test | 旧 uniform S / P | 局部票 S / P | 漏搜（旧 → 新） | >10 m（旧 → 新） | 持续恢复（旧 → 新） |
|---|---:|---:|---:|---:|---:|
| 全部 120 轨 | 68.7920 / 80.6281 | 70.7277 / 83.1631 | 516 → 330 | 516 → 337 | 5/12 → 5/11 |
| 去掉最大收益轨 0020/122，119 轨 | 70.8283 / 83.0588 | 70.7911 / 83.0511 | 331 → 330 | 336 → 337 | 5/11 → 4/10 |

剩余 119 轨 ΔS/P 为 −0.0373 / −0.0077 个百分点。按另一个配对指标，基线误差 >10 m 且局部版误差 ≤2 m 连续达到三次相邻扫描的片段有 1 个，排除该车后有 0 个。

以下为旧冻结学习门的 CPU 工程归档，沿用其原版宿主、批次和注意力实现。S/P 差用百分点。[旧冻结 E60 逐轨敏感性](../artifacts/ct_checks/20261126_uniform_local_validation/archived_e60.json)

| 轨迹范围 | ΔSuccess / ΔPrecision | 漏搜（旧 → 学习门） | 误差 >10 m（旧 → 学习门） | 持续恢复（旧 → 学习门） |
|---|---:|---:|---:|---:|
| 全部 120 轨 | +1.8369 / +2.3653 | 516 → 330 | 516 → 337 | 5/12 → 4/11 |
| 去掉最大收益轨 0020/122 后的 119 轨 | −0.0332 / −0.0053 | 331 → 330 | 336 → 337 | 5/11 → 3/9 |

冻结 CPU 回放的旧宿主 E60 为 68.7695 / 80.6149。服务器正式 uniform 同一 checkpoint 的归档为 68.7897 / 80.6227。上表在同一 CPU 回放内做差；正式验收工具使用对应轮次的服务器归档。

同版单轨 B=1、104 对物理预测的端到端延迟，旧 uniform 的 P50/P95 为 65.01 / 131.14 ms，四槽局部票为 75.50 / 178.21 ms，逐帧配对差的中位数为 +9.26 ms。[WSL 配对计时及逐帧样本](../artifacts/ct_checks/20261126_uniform_local_validation/profile.json)

WSL RTX 4060 的两轮训练冒烟使用 8 条真实短训练轨、2 条 dev 轨和完整训练数据身份。每轮有 144 个主端点、9 个自然辅助端点、9 对困难/正常复放、36 次 Adam 更新，并开启完整诊断。从 E1 保存后恢复至 E2，与连续运行到 E2 的模型张量、Adam 状态、门系数、学习率、采样计划和 26 个 dev 预测的指标一致。[两轮恢复核对](../artifacts/ct_checks/20261126_uniform_local_training/resume.json)

另在 WSL CPU 上直接启用成熟的 1/4/4/8 窗口，完整执行 60 次普通重置、84 个续步和 20 个自然复放端点，完成 41 次 Adam 更新及该轮学习门拟合。[成熟窗口核对](../artifacts/ct_checks/20261126_uniform_local_training/mature.json)

同机 B=4、1,024 点、workers=0，按旧/新/新/旧顺序测量首轮训练。每次舍去前 4 步后计入 128 个主端点和 9 个辅助端点。含数据准备、模型前后向、优化和递推提交，按两次合计用时，旧 uniform 为 16.44 主端点/秒，补模板修正前的四槽配方为 13.40 主端点/秒。[B4 训练计时](../artifacts/ct_checks/20261126_uniform_local_training/profile.json)

## 稀疏模板修正与当前训练速度

仅回放 KITTI dev 轨 0018/3 的四个预测步。扫描 55、56 均从空模板推理，q0 连续通过可信写入和测速。用扫描 56 的 7 个唯一前景支持点与该步获选框补建模板，扫描 57 首次使用，58 沿用。按原始点 ID 从 Velodyne 点云独立复算，模板坐标最大差为 1.54 × 10⁻⁷ m。四步的选框、写入和测速均与旧归档一致。[四步原始记录](../artifacts/ct_checks/20261126_uniform_local_brief_profile/template_bootstrap.json)

WSL RTX 4060 Laptop 8 GB、B16、每个加载器 4 个 worker、FA2。在真实 KITTI 主辅采样计划上提前启用成熟的 1/4/4/8 窗口。修正后预热 6 步，再计 16 次 Adam 更新，含 256 个主端点和 16 个辅助端点。同步墙钟 6.52 秒，吞吐 39.29 主端点/秒，显存峰值分配 2.66 GiB、预留 3.85 GiB。按每轮 76,324 个主端点外推，约 32.4 分钟/轮、60 轮纯训练约 32 小时。另以已有 B8 冻结回放的完整 test 用时 232 秒作为每轮评估的粗略参考，60 次合计约 3.9 小时。正式运行的 B16 评估、前十轮课程、末三轮完整诊断、读写及持续负载均会影响总时长；以训练日志中实际各轮的 `Time train/eval/save` 更新估计。[B16 短计时](../artifacts/ct_checks/20261126_uniform_local_brief_profile/REPORT.md)
