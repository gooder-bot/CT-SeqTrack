# V41 单视角历史扰动对照

本组与 11 月 22 日启动的 `four_candidates` 比较。同一个物理端点在每轮普通主训练中只处理一次，按独立单步重置。合法首个预测端点用原始首框。其他普通端点按 1/4 干净、3/4 受扰的比例选择训练历史，匹配四候选组中一份干净、三份受扰的分布。受扰样本复用现有 E57 密度条件连续残差和共同末框平移、航向扰动，并据此重裁实际点云。这里的 CGP 指借鉴 PointRePar 的条件扰动思路，本组实际使用项目已有的经验残差库。[单视角配置](../cfgs/ct_seqtrack/41_joint_single_view_cgp.yaml) · [四候选比较组](../cfgs/ct_seqtrack/41_joint_four_candidates.yaml) · [11 月 22 日比较组参数快照](../artifacts/ct_checks/20261122_gpu1_joint_parameters/four_candidates.yaml)

| 项目 | 四候选组 | 单视角组 |
|---|---:|---:|
| 每个物理端点每轮的普通主路样本 | 4 | 1 |
| KITTI Car 每轮主端点 | 76,324 | 19,081 |
| 每轮 Adam 更新（batch 16） | 4,771 | 1,193 |
| E3 起固定困难前态的每轮上限 | 512（占主端点 0.67%） | 512（占主端点 2.68%） |

两组均使用 KITTI Car、seed 42、60 轮、batch 16、4 个数据加载进程、Adam、初始学习率 0.00015，第 20 和 40 轮后各乘 0.2。三帧历史加当前帧、六个在线候选、B1/MTP、横向纠偏、远搜和状态提交沿用联合模型。辅助整轨队列仍按物理端点数的 0.25 分配预算，损失权重为 0.25。训练使用场景 0000–0016，每轮按首框初始化、在 test 的 0019–0020 自主跟踪，并保存 checkpoint。E58–60 保留完整逐帧记录。[联合协议](V41_JOINT_PAIR.md)

每个普通端点用 seed、轮次、轨迹及端点键，从原四候选的抽样编号中等概率选择一个。首个预测端点的四种选择都使用真实首框，编号只影响超额点的抽样。后续端点选中编号 0 使用干净历史，选中 1–3 复用对应扰动和取点随机键。E57 找不到匹配供体时继续使用原有的末框扰动。同一训练可重现，每轮实际抽中的数量可略有波动。

E2 权重取自**本组自己的训练**。E3 起按原规则回放本组已保存的自然前态，完整快照占用单步槽，该端点直接使用快照。两组可共用同一 E57 输入库。E2 前态的来源模型、实际个数及内容会随各自训练改变。

## 启动与恢复

在上传的新代码副本根目录执行。以下命令使用 11 月 22 日归档中记录的原项目位置、E57 文件和物理 GPU 1。可在已分配的另一张同型号空闲卡上调整 `GPU_ID`。独立实验目录容纳本组的 E2 数据库及训练运行。

```bash
REFERENCE_PROJECT=/home/lishengjie/study/lcyu/CT-SeqTrack
export HISTORY_BANK="$REFERENCE_PROJECT/artifacts/ct_checks/v41_history_pair/history_e57.json"
export V41_CONFIG=cfgs/ct_seqtrack/41_joint_single_view_cgp.yaml
GPU_ID=1
mkdir -p output
SINGLE_DIR=$(mktemp -d "$PWD/output/v41-single-view-cgp-XXXXXXXX")
echo "$SINGLE_DIR"
bash tools/train_v41_joint.sh --devices "$GPU_ID" \
    --v41_fixed_replay_bank "$SINGLE_DIR/natural_e2.sqlite" --log_dir "$SINGLE_DIR/single_view_cgp"
```

从已完成轮次恢复时，将 `SINGLE_DIR` 指向上次打印的目录，使用同一份源码、E57 文件、E2 路径及运行目录，传入最近一个完整 checkpoint。以下例子从 E12 恢复。

```bash
bash tools/train_v41_joint.sh --devices "$GPU_ID" \
    --v41_fixed_replay_bank "$SINGLE_DIR/natural_e2.sqlite" --log_dir "$SINGLE_DIR/single_view_cgp" \
    --checkpoint "$SINGLE_DIR/single_view_cgp/formal_checkpoints/epoch=012.ckpt"
```

## 看训练过程与得分

每个运行目录的 `metrics.csv` 和 `epochs/epoch=NNN.json` 保存逐轮训练和评估记录，`training_audits/epoch=NNN.json` 保存实际主端点、更新数、历史来源和回放次数。新组的 `ordinary_reset_seeds` 分别列出首个预测端点、普通干净端点和普通受扰端点，可以直接用后两项核对实际扰动比例。两组各完成至少一轮后，即可从新代码副本比较共同完成的轮次。两组都到 E60 时同时输出固定 E60 和 E58–60 平均。变量 `REFERENCE_PAIR` 使用 11 月 22 日参数快照中已登记的原路径，`SINGLE_DIR` 使用本组打印的路径。

```bash
REFERENCE_PAIR="$REFERENCE_PROJECT/output/20261122-v41-joint-test-gpu1-000647-viH8qjCz"
python tools/compare_v41_single_view.py \
    --candidate-run "$REFERENCE_PAIR/four_candidates" \
    --single-run "$SINGLE_DIR/single_view_cgp" > "$SINGLE_DIR/comparison.json"
```

固定比较 E58–60 的平均 Success/Precision 和 E60 的 Success/Precision，同时看逐轮训练耗时、优化步数、损失和得分曲线。训练与 test 耗时分列。这一对照可以回答同样训练 60 轮时，每端点只取一个视角会怎样影响总耗时和最终得分。同一轮的学习率相同，单视角到达该轮时累计更新更少，困难回放和辅助流相对主训练的占比也较高。主队列末段仍预留 112 个单步窗口，此时 B0 使用已有的 BN 统计，这些批次占整轮的比例也会变化。test 全程用于监测，结果按上述固定轮次解释。
