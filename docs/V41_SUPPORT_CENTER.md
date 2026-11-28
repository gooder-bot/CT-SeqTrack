# V41 四候选前景支持中心修正

KITTI Car 的跟踪器在 q0 做完原有解码后，利用已采到的当前点校正位置，再交给 B3 选框和写入。第四槽仍提供独立的局部投票框。正式比较使用 11 月 27 日四候选训练的同轮结果，新增一组 scratch 训练。[单组配置](../cfgs/ct_seqtrack/41_support_center_kitti.yaml)

## 正式训练

在服务器主项目拉取本版代码后，从下面的目录启动一组训练。示例使用物理 GPU 1，每轮评估 KITTI test。日志和权重写进 `output/` 下本次实验的目录。

```bash
project=/home/lishengjie/study/lcyu/CT-SeqTrack
cd "$project" && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/41_support_center_kitti.yaml --devices 1 \
  --eval_split test --check_val_every_n_epoch 1 \
  --log_dir "$project/output/$(date +%Y%m%d)-v41-support-center-kitti-lr1e-4-step20-gamma01-gpu1" \
  --path /home/lishengjie/data/cxtrack/training \
  --v41_history_bank "$project/artifacts/ct_checks/v41-test-pair-ready-odnbYGul/artifacts/ct_checks/v41_history_pair/history_e57.json" \
  --v41_q0_replay_manifest "$project/artifacts/ct_checks/20261128_ideas_review/q0_prefixes.json" \
  --v41_local_prefix_bank "$project/artifacts/ct_checks/20261128_ideas_review/uniform_e60.sqlite"
```

训练使用 seed42、B16、workers4、60 轮和 1/4/4/8 递推窗口。初始学习率为 `1e-4`，每完成 20 轮乘 0.1。每轮评估完整 test，E58–60 输出完整诊断。恢复时将上面的 `--log_dir` 指向原运行目录，并传入 `--checkpoint "$RUN/formal_checkpoints/epoch=NNN.ckpt"`。独立评估载入同源码同配置的 checkpoint，使用新的输出目录。

## 单帧计算与训练边界

- 坐标以原始 q0 车体系表示，除以首框 L/W/H。取三轴绝对值均不超过 0.8 的当前点，按原始点 ID 去重。记预测前景概率至少 0.5 的个数为 n，至少 0.85 的个数为高置信点数。
- 高置信点至少 10 个时，42 项特征描述可见点分布、粗框、运动先验、历史位置和视线。三个回归输出以米计，分别限幅到 0.65、0.38、0.30，再乘 0.5 × n/(n+6)，将偏移旋回并加到 q0 中心。局部票以这个 q0 为种子生成一次。
- 其余情况先由原 q0 生成局部票。如果当前观测和局部票均有效，且两中心三维距离 ≤1 m，将 q0 中心朝局部票移动中心差的 30%。航向沿用原 q0。B3 对最终四框各自取支持点、打分并提交。无梯度评估将局部框和前三槽胜者的形状匹配放在同一批里，每个框独立计算框内成员。
- 每轮从现有训练集自然整轨辅助流收集原始 q0 的特征。n≥3 的样本参与拟合，目标为标注中心与原始中心在原 q0 轴上的三维修正量。实际采到目标点的样本权重为 1，其余为 0.25。按原型使用标准差下限 0.04、标准化截断 ±4、岭惩罚 0.018 和 8 次 Huber 迭代，Huber 转折点为 0.18 m。原 q0 的直接位置监督作用于原始输出，B3 的 S/P/T、收益门和 q0 的质量监督都使用最终几何。
- 第一轮沿用原 q0。第 k 轮验证和该轮保存权重都使用本轮已生效的系数；本轮辅助流新拟合的系数从第 k+1 轮起生效。两组系数随 checkpoint 保存。标注只在拟合或计分时读取。summary 与 full 逐帧记录中的 `support_center` 给出实际分支、前景和高置信点数、修正前后的锚点相对中心及投票到种子的距离。局部票无效时该距离为 null。

## 工程验收

在 A40 上用 16 条真实 KITTI 短训练轨跑两轮，每轮 468 个主路样本、29 个辅助样本、30 次 Adam 更新。第一轮拟合 19 行，第二轮启用这组系数并拟合 15 行。从第一轮 checkpoint 恢复后，与连续训练逐轮比较，442 个模型状态张量、1,041 个 Adam 状态张量、学习率调度和评估逐帧记录全部相同。第二轮共 28 个预测步，其中 26 步实际启用稀疏融合。[训练与恢复](../artifacts/ct_checks/20261128_v41_support_upgrade-b80z4Mth/final/smoke.json) · [逐帧对账](../artifacts/ct_checks/20261128_v41_support_upgrade-b80z4Mth/final/smoke_rows.json)

在 A40 上用最终源码重放完整 KITTI Car test。跟踪器输入只保留在线字段。6,304 个预测步的选槽、可信写入、IoU、中心误差和逐帧 S/P 与冻结原型逐值一致。其中 99 步保留原 q0，5,558 步走密集修正，647 步走稀疏融合。[逐帧核对结果](../artifacts/ct_checks/20261128_v41_support_upgrade-b80z4Mth/final/switch.json)

在 A40 上用 4 条 dev 轨迹做单目标计时，每条预热 8 步、计时 24 步。同一步复制相同跟踪前态，交替执行三个版本，时间覆盖组批、前向与状态提交。旧版、仅形状合批和完整新版的 P50 分别为 70.65、69.10、71.32 ms，P95 分别为 124.13、118.89、119.37 ms。相对旧版，合批和完整新版的配对中位差分别为 −1.90 和 +0.20 ms。三个版本的单步增量显存峰值均为 24.44 MiB。[96 组成对计时](../artifacts/ct_checks/20261128_v41_support_upgrade-b80z4Mth/final/bench_b1.json)

在同一张 A40 上做满 B16 复测，预热 8 批，计时 32 批、共 512 个端点。与旧版相比，仅形状合批每批的配对中位耗时减少 4.46 ms，完整新版增加 2.13 ms。三个版本的单步增量峰值均为 504.01 MiB。此次旧版与新版都出现秒级长尾，P95 分别为 1,821.74 和 2,370.64 ms，实测吞吐为 45.29 和 31.39 端点/秒。此次测速支持减少形状计算开销，尚未显示完整新版有稳定的吞吐收益。[满 B16 计时](../artifacts/ct_checks/20261128_v41_support_upgrade-b80z4Mth/final/bench_b16_test.json)

在 RTX 4060 上对同一批输入运行合并坐标变换前后的中心修正。1,020 个样本的 42 项特征、修正框和分支逐位一致。每种批量计时 160 对，B1 和 B16 的配对中位耗时分别减少 0.92 ms 和 0.94 ms；B16 P95 从 4.40 升至 5.16 ms。该计时只覆盖中心修正。[原始计时与输入检查](../artifacts/ct_checks/support_center_wsl_latency/merged_timing.json)

## 固定结果比较

将 `RUN` 设为新训练目录。在上述服务器源码目录执行下列命令。

```bash
/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python tools/compare_v41_support_center.py \
  --run "$RUN" --baseline "$project/output/20261127-v41-uniform-local-vote-kitti-lr1e-4-step20-gamma01"
```

旧正式台账的 E60 为 **70.4429 / 82.8483，336 个 >10 m 预测帧**；E58–60 均值为 **70.7046 / 82.9692，平均 332.67 个 >10 m 预测帧**。比较器按物理帧配对，两序列共 120 轨、6,304 个预测步，S/P 计入 120 个合法首框。报告逐轨变化、两场景、去掉最大 Success 净贡献轨，以及分别去掉 0020/122 和 0020/123 的结果。

冻结 E42 的探索性原型从 70.9075 / 83.2951 到 71.2469 / 83.4009，>10 m 预测帧从 329 到 286。其中 42 个减少的严重漂移帧来自 0020/123；去掉该轨，S/P 差为 +0.23 / −0.09 个百分点。“单做半强度中心修正”的既有对照为 71.2368 / 83.3554、328 帧。该权重与路线曾用 test 选择。新正式训练的分数和论文结论以固定轮次的逐轨比较为据。[冻结原型与逐轨敏感性](../artifacts/ct_checks/20261128_v41_b0_route_pilot/REPORT.md)
