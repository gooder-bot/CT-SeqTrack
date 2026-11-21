# V41 q0 横向中心与困难前态单次训练

本配方使用 KITTI Car、uniform、seed 42、batch 16，从随机初始化训练 60 轮。每轮用 test 0019–0020 评估，第 58、59、60 轮写完整逐帧诊断。比较对象是已有 v3 uniform 的固定三轮记录。历史 test 已参与问题分析，结果衡量这套改动在已研究场景中的表现。[独立配方](../cfgs/ct_seqtrack/41_full_kitti_q0_lateral_replay.yaml) · [历史参照与数据](../artifacts/ct_checks/20261121_q0_final_review/REPORT.md)

## 从历史权重生成训练前态

将当前 V41 工作树同步到训练服务器后，在项目根目录运行以下命令。使用原 uniform 的运行目录与匹配的 v3 源码，供体推理读取 train 0000–0016 原始扫描。若已有与运行清单一致的冻结源码，可直接将 `V3` 指向该目录，省去解包步骤。以下路径对应已归档的 KITTI 实验。

```bash
set -euo pipefail
PY=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
DATA=/home/lishengjie/data/cxtrack/training
DONOR=output/20261118-v41-q0-dev-gpu0-180229-k1yozphm/uniform
CHECK=$(mktemp -d artifacts/ct_checks/q0-lateral-XXXXXXXX)
V3="$CHECK/uniform-v3"
mkdir "$V3"
git archive b819e760ac4c255820b47f1e8459872701c1e748 \
  main.py requirement.txt models datasets utils cfgs | tar -x -C "$V3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/lishengjie/study/lcyu/nuscenes-devkit/python-sdk${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/build_v41_q0_replay.py --donor-source "$V3" --donor-run "$DONOR" \
  --path "$DATA" --output "$CHECK/prefixes.json"
```

脚本核对 E10、E60 权重、供体源码、训练轨迹与数据身份。两份权重分别从合法首框运行整轨，记录预测当前帧之前的已接受历史框、时间、可信位置、速度及测量资格。已归档 E60 文件记录的保存时磁盘源码与 v3 启动源码不同。供体工具只对已核对指纹的该文件按既有 v3 回放口径读模型权重，并把两种身份记入清单。训练恢复仍使用原运行身份。[已归档回放与指纹](../artifacts/ct_checks/20261121_q0_final_review/formal/verification.json)

困难端点要求实际采到至少 10 个标注框内唯一点，且粗框 IoU < 0.5、车宽方向偏差 > 30 cm、垂直中心偏差 > 15 cm、矩形轴向角误差 > 10° 四项中至少一项成立。相邻命中合为事件。正常对照的距离差至多为 5 m 或困难帧距离的 25%（取较大值），点数比例为 0.5–2，同车优先。缺少合格对照时保留原训练槽。生成命令会输出匹配与未匹配事件数，完整 JSON 保存供体指纹及入选前态。

## 启动一组正式训练

在同一终端继续运行。若重新开终端，先将 `CHECK` 指向上一节实际生成的清单目录。选择 GPU 时沿用服务器分配的 `CUDA_VISIBLE_DEVICES`。

```bash
CFG=cfgs/ct_seqtrack/41_full_kitti_q0_lateral_replay.yaml
RUN=$(mktemp -d output/v41-q0-lateral-XXXXXXXX)
"$PY" -u main.py --cfg "$CFG" --path "$DATA" --log_dir "$RUN" \
  --v41_q0_replay_manifest "$CHECK/prefixes.json"
```

主训练每轮每个入选事件最多使用一对端点。困难与正常合计占 branch 0 单步槽的比例不超过一半，占全部主端点不超过 12.5%。恢复前态后重新裁取原始当前帧。其余窗口沿当前模型输出递推。每轮 `training_audits/epoch=NNN.json` 记录清单 SHA256、两份供体指纹和实际使用数量。续训使用原运行目录、同内容清单和完整轮次 checkpoint。

横向头根据 B0 当前唯一点的 XYZ、点特征、前景概率、BoxCloud 和 q0 的实例上下文估计到中心的有符号位移。沿原 q0 预测车宽轴，取前景概率 ≥ 0.5 的点投票的概率加权中位数。少于 3 个唯一点或 q0 无效时位移为零。原 q0 继续学习等权中心与航向。逐点位移和采用比例的新增损失各取系数 1，纯背景投票监督采用比例趋零。采用比例末层初始 bias 为 −3。第 1–10 轮训练两项新损失，第 11 轮起将修正后的 q0 送入 B3，单步上限为首框车宽。B3 从最终框重新取支持点、评分并提交。

## 复核固定轮次

当前四槽推理停用旧 R，历史 v3 uniform 的推理启用 R。比较器在两侧分别记录 `inference_recovery_enabled`。与历史记录的分差包含推理策略差异，衡量横向头的独立贡献时，两边须采用相同的 R 策略。

```bash
"$PY" tools/compare_v41_q0_lateral.py "$RUN" > "$CHECK/comparison.json"
```

比较工具需要历史 uniform 的 `formal/test_verification.json`、同目录的 `dev_fixed_comparison.json`（核对 60 轮实际训练预算）和 `test_frames/uniform_E{58,59,60}.jsonl.gz`。归档位于其他目录时，传 `--baseline-receipts /path/to/formal/test_verification.json`。完整评估逐帧记录包含原 q0、最终 q0、横向建议、实际位移、采用比例和唯一点数。完整训练诊断还记录全部实际回放端点的事件、配对角色、供体轮次与修正量；普通端点沿用原来的固定抽样。

按物理帧配对，分别报告 E60 与 E58–60 算术均值。关注末三轮整体 Success、Precision 达到历史基线，共同可见困难帧的横向 MAE 降低，以及中心误差 > 10 m 帧数没有增加。同时检查逐轨和等轨权 Success、新增持续失跟、纵向与高度 MAE、轴向角误差 P95。新增持续失跟与单车集中增益同时出现时，单列受影响车辆并否决采用。历史与本次的执行身份、数据和训练预算随报告列出；一次训练可判断联合方案的实测表现，两个修改的独立贡献仍待受控对照。
