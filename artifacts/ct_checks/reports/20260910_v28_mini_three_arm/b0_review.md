# v28 两组 B0 已完成运行的只读复核（2026-09-10）

两组均正常完成 60 轮，严格确定性的 CUDA 训练已跑通；**尚不能宣布 B0 已稳定恢复到预定健康水平**。seed52 达到历史参考，事先固定的 seed42 则同时未达到 Success 和 Precision 的恢复门槛。后续可以补齐既有 checkpoint 的评测及同协议 SeqTrack 对照，不宜据此直接放行完整 nuScenes 的大规模论文实验。

## 最终结果与恢复目标

| 运行 | final 60 S | final 60 P | 相对历史 S=50.986 | 相对历史 P=59.962 | S≥49.986 且 P≥58.962 |
|---|---:|---:|---:|---:|---|
| B0 seed42 | 45.200219 | 47.242888 | -5.785781 | -12.719112 | 未通过，两项均不足 |
| B0 seed52 | 52.876368 | 64.478118 | +1.890368 | +4.516118 | 通过历史数值目标 |
| 两 seed 算术均值（仅描述） | 49.038293 | 55.860503 | -1.947707 | -4.101497 | 不能代替固定 seed42 验收 |

seed52 比 seed42 高 **7.676149 / 17.235230 个百分点**。这证明本轮不同 seed 存在较大的结果差异，不能证明 v25/v26 历史差距全部由 CUDA 非确定性引起。历史参考存在输入尺寸等协议差别，因此上述差值只能用作恢复目标判断；严格“相对 SeqTrack 不低超过 1 分”仍需同协议、同评价器 reference。

来源：两目录 `output/20260909-003318-28_b0-mini_car_seed{42,52}_60ep_bs16/lightning_logs/version_0/dev_diagnostics/epoch_60_summary.json` 的 `metrics.S/P`，已与同目录 TensorBoard 的 `success/dev`、`precision/dev` 核对，误差小于 0.00001。完整精度数字、每轮曲线、运行身份及 checkpoint 元信息保存在 [b0_review.json](b0_review.json)。

## 协议与训练完成证据

- 两 run 的 provenance 均为 scratch，初始化与恢复 checkpoint 路径为空；类别 Car，mini_train 全 8 场景，274 tracklets / 5051 frames。
- 四候选展开为 20204 行，batch16、drop_last，尾部丢弃 12 行，实际 1262 步/轮；60 轮为 75720 次 B0 更新。
- 两 run 的 58/59/60 checkpoint 均存在，对应零起算 `epoch=57/58/59`、`global_step=73196/74458/75720`，`ct_epoch_boundary_complete=true`。
- final checkpoint 的每个有 Adam 状态的参数，其优化步数均为 75720；参数组仅 `b0`，`active_frozen_parameters=[]`。
- Adam `betas=(0.5,0.999)`、`eps=1e-6`、weight decay=0、foreach=false、fused=false。StepLR 20 轮衰减 0.1；final 保存时已经执行第 60 轮结束后的调度，保存 lr=1e-7，并不表示第 41—60 轮以 1e-7 训练。
- 两日志均出现 `Trainer.fit stopped: max_epochs=60 reached`；未发现 Traceback、RuntimeError 或 CUDA error。
- 记录的服务器为 A40、PyTorch 2.0.1+cu118、Lightning 2.0.2、CUDA 11.8、cuDNN 8700、驱动 535.247.01。strict deterministic=true、warn_only=false、TF32=false、cuDNN benchmark=false、CUBLAS `:4096:8`。
- git commit 为 `749bc13a0143fcdedba5cee465e2273c639e3b17`；dirty_tracked 来自已删除历史 artifact，provenance 未列出训练源码修改。不能把 dirty 标记本身解释为代码不一致。

**诊断命名存在遗留问题：** `dev_diagnostics`、TensorBoard `success/dev` 和 CSV `partition=dev` 实际来自本轮官方 mini_val 验证。provenance 明确是 `scene-0103`、`scene-0916`，106 tracklets / 2285 frames；CSV 的 scene_id 均为 unknown 导致 summary `scenes=1`，这不是只评了一个场景。不能将其与 v27 单个内部 dev 场景的分数直接比较。

## 验证曲线

| 轮次 | seed42 S | seed42 P | seed52 S | seed52 P |
|---:|---:|---:|---:|---:|
| 5 | 29.658 | 31.852 | 35.045 | 33.918 |
| 10 | 39.270 | 45.909 | 35.539 | 64.158 |
| 15 | 26.005 | 24.056 | 40.822 | 57.701 |
| 20 | 49.533 | 65.557 | 38.872 | 42.405 |
| 25 | 52.043 | 60.449 | 51.988 | 66.696 |
| 30 | 44.170 | 45.606 | 47.653 | 60.301 |
| 35 | 41.863 | 43.724 | 53.522 | 65.502 |
| 40 | 41.925 | 41.009 | 54.034 | 65.927 |
| 45 | 44.428 | 46.133 | 55.174 | 66.815 |
| 50 | 45.209 | 47.007 | 53.625 | 64.798 |
| 55 | 45.496 | 47.248 | 53.434 | 64.274 |
| 60 | 45.200 | 47.243 | 52.876 | 64.478 |

seed42 在第 25 轮达到过健康数值，随后第 30—60 轮持续偏低。第 50/55/60 轮都约 45/47，因此不能将问题描述为仅最后一轮的偶发测量。第 25 轮不能替代预先固定的 final，也不能用 seed52 替代 seed42 选择有利结果。

**58/59/60 checkpoint 已保存，但已同步数据中只有每 5 轮的验证结果。没有 58、59 的独立评测，late-3 尚不能计算。** 不得以 50/55/60 均值冒充 late-3。

## 损失与失跟诊断

两 run 的 total、seg、BC、motion classification 共四项损失，各 75720 个记录全部有限。最后一轮 total loss 平均为 0.225643（seed42）和 0.226455（seed52），训练损失非常接近；seed42 从 25 轮的 0.277646 降到最后一轮 0.225643，验证闭环性能却变差。现有证据不支持“训练 loss 数值爆炸”这一解释，下一步需要区分训练泛化、闭环误差累积与输入/数值路径，而不能仅凭 loss 下降宣称恢复。

| final 闭环诊断 | seed42 | seed52 |
|---|---:|---:|
| 预测帧（不含初始化） | 2179 | 2179 |
| 至少一次 IoU<0.1 的 tracklets | 38/106 | 29/106 |
| 至少一次 IoU=0 的 tracklets | 25/106 | 21/106 |
| 当前 raw crop=0 的帧 | 178（8.17%） | 167（7.66%） |
| 当前 raw crop≤2 的帧 | 312（14.32%） | 308（14.13%） |
| 空 crop 最长连续观测长度 | 33 帧 | 16 帧 |
| 整段无有效观测帧 | 0 | 0 |

两者空 crop 比例接近，不能简单将 17 分 Precision 差解释为空 crop 数量变化。闭环各自形成的 `>50` 原始点桶中，seed42 为 S/P=38.012/38.127（997 帧），seed52 为 47.789/62.411（986 帧）；点多也不能保证没有漂移。各 run 的 crop 和桶归属由各自预测决定，此比较不是匹配帧因果分解。

空 crop 期间发生失跟的段有 12（seed42）和 22（seed52）；在记录定义的后续窗口内，两者均没有观测到这些段重新达到 IoU≥0.5。该恢复率是含右删失的观测下界，不能解释成永不恢复；所有空 crop 段的其他正常返回不等于失跟恢复。完整定义和删失数量见 JSON `recovery`。

## 本轮判断

1. CUDA 两次启动阻断问题已在这些完整训练中通过真实服务器运行；这与 B0 分数恢复是两项不同验收。
2. seed52 已证明当前实现具有达到健康历史分数的能力，但 seed42 尚未达标，整体仍有明显的 seed 敏感性与后期闭环退化。
3. 可以继续补 final/late-3 评测、同协议 SeqTrack reference 及定位性工程实验；目前证据不足以确认“基线稳定恢复”，不宜据此开展大量完整数据、多类别模块长跑。
4. 本复核未运行新的训练、未改动任何 output 文件、未使用 TrajTrack；checkpoint 只在 CPU 读元信息，临时 EasyDict 映射仅用于反序列化配置容器，不进行模型前向或重算得分。
