# v28 三组 mini：协议与历史参考可比性审计

日期：2026-09-10。只读核查三组 `output/20260909-003318-28_*` 的 provenance、resolved config、epoch60 endpoint/summary，以及生产训练/评价入口。本记录不改训练代码、历史输出或冻结参考仓库。

## 结论

本次评价确实是官方 mini_val，不能因目录、TensorBoard tag 或 CSV 的 `dev` 字样误认成 v27 内部 dev。B0 seed52 的第60轮达到历史健康区间；B0 seed42 明显未达到此前分别登记的 S/P 恢复目标。两次不同 seed 的结果不足以宣告基线稳定恢复，也不能证明历史 v25/v26 的掉分全部由随机性造成。

Full seed42 当前评价未装入校准策略，2285 个端点全部记录 `missing_calibration_artifact`，实际动作数为0，其 S/P 等于 B0 seed42。因此这组成绩是 Full 模型的 observation fallback 成绩，不能解读为校准后 Full 没有增益。

## 三组实际协议

| 项目 | provenance / resolved config 的一致记录 |
| --- | --- |
| 源码 | `749bc13a0143fcdedba5cee465e2273c639e3b17`，包含两次 CUDA 修复；dirty 状态列举的是 artifacts 删除与未跟踪产物，未见已跟踪生产源码修改 |
| 参数训练 | `scene-0061,0553,0655,0757,0796,1077,1094,1100` 共8个 mini_train 场景、274条轨迹、5051帧 |
| 评价 | `scene-0103, scene-0916`；dataset 实际 `protocol_role=test`、`split=mini_val`、106条轨迹、2285帧 |
| 评价分母 | epoch60 CSV 每组2285行，106个初始化首帧、2179个预测帧；未因空crop剔除端点 |
| 预算 | 5051×4=20204候选样本；batch16、自然drop_last尾部12行；1262步/轮、60轮登记75720次更新 |
| 随机与输入 | B0/42、B0/52、Full/42；workers12、preloading；每帧1024槽、3帧历史；网络使用初始化首帧尺寸 |
| 目标 | `reference_batch` 整批原目标、单次BC、`seqtrack_original_slots_v1`；完整四候选自然shuffle，不按view分别归约 |
| 优化与数值 | Adam 1e-4，StepLR(20,0.1)，FP32、strict deterministic、warn_only=false、TF32/benchmark关闭、foreach/fused=false、CUBLAS工作区 `:4096:8` |
| 实际环境 | Python3.9.19、PyTorch2.0.1+cu118、Lightning2.0.2、CUDA11.8、cuDNN8700、NVIDIA A40、驱动535.247.01；GPU分别1/2/3 |
| 初始化 | `checkpoint_path`、`init_checkpoint_path` 均null，scratch_only |
| Full机制训练 | 同8场景，4777预测端点/轮，268条含预测帧轨迹，300事务/轮；B0仍为1262更新/轮 |

更新完成情况应与独立 checkpoint 审计交叉确认；上表明确区分 provenance 登记的训练计划和逐帧已经落盘的评价数量。

## 遗留日志命名与场景元信息缺口

[main.py](../../../../main.py:839) 对 v28 已选择官方 val；[base_model.py](../../../../models/base_model.py:2078) 仍只检查 `ct_enable_v27` 决定指标标签，故写入 `success/dev`、`precision/dev`、`metrics/dev`、`dev_diagnostics/` 和 CSV `partition=dev`。三组实际 dataset provenance 对应官方2场景，不能用这些旧名字改变数据集结论。

三组 CSV 的 `scene_id` 均为 `unknown`，所以 summary 中 `scenes=1` 只是 unknown 的去重计数，**不是实际只有1个评价场景**。现有端点可以支持整体、按轨迹和稀疏桶分析，但缺少可靠场景映射，不能直接制作逐场景统计或场景 bootstrap。后续修正导出元信息时保留原输出，生成新评测目录及映射。注册的 calibration/dev 场景与参数训练重叠；官方 mini_val 与参数训练场景不重叠，不要把 manifest 顶层 `parameter_training_overlap=true` 误加到官方 val。

## 健康参考的边界

历史20260528 SeqTrack 第60轮为 S=50.98577880859375、P=59.96171188354492；同为8场景 mini_train、官方 mini_val、batch16、seed42、workers12、60轮、75720更新。这比 v27 内部 dev 对照更接近，但不是完全匹配的算法比较：历史实现读取当前GT尺寸，本次使用首帧尺寸；历史评价代码、采样随机流、CUDA归约/确定性路径和运行环境也并非同一受控记录。历史值可用作预先登记的健康量级与恢复目标，不能直接宣称严格公平涨分。

| 实验 | epoch60 S | epoch60 P | 对历史 ΔS | 对历史 ΔP | 分别不低于历史1点 |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 seed42 | 45.200219 | 47.242888 | -5.785560 | -12.718824 | 未通过 |
| B0 seed52 | 52.876368 | 64.478118 | +1.890589 | +4.516406 | 达到点估计目标 |
| Full seed42，未校准 | 45.200219 | 47.242888 | -5.785560 | -12.718824 | 实际为observation fallback |

这里 S/P 为 endpoint summary 的百分数，主报告另与 TensorBoard 聚合核对微小舍入差异。登记下限精确为49.98577880859375和58.96171188354492。不可选seed52替代seed42的失败，也不可用两个seed均值代替逐项验收。

来源：[历史原始TensorBoard抽取表](../20260907_v27_mini_five_arm/historical_seqtrack_baselines.csv)、[v27报告的可比范围](../20260907_v27_mini_five_arm/REPORT.md)、[版本审计](../20260907_v27_mini_five_arm/B0_VERSION_DECISION.md)、[当前协议](../../../../docs/EXPERIMENT_PROTOCOL.md)。历史20260801/0825原SeqTrack也只有31.684/31.337和27.997/26.484，说明旧系统有低分运行证据；不能因本次存在seed差异就反推旧差异的全部原因。

## 最少补证据与后续实验安排

1. 三组均只落盘每5轮评价，epoch58/59有checkpoint而没有评价。补评已有58/59 checkpoint，并与现有60同评价器汇总算术late-3；不使用50/55/60、不挑best、不追加训练轮数。补评使用原run配置、同一代码与数据身份，新目录保存结果。
2. Full分别在58/59/60 checkpoint上拟合各自的内部calibration策略并锁定内部dev，再做官方mini_val完整闭环；同一Full checkpoint的 never/bounded/selective 对照可以辨认动作贡献。官方val不能拟合阈值；内部两场景已用于参数训练，必须披露重叠。
3. 严格独立 SeqTrack 对照需要明确的上游实现匹配包装：同场景、样本和更新预算、首帧尺寸、同评价器、同CUDA数值修复及环境。当前 [28_seqtrack_reference.yaml](../../../../cfgs/28_seqtrack_reference.yaml) 直接继承28 B0，解析后只差 `experiment_name` 与 `ct_reference_baseline`；代码中该标签只用于身份/记录，没有独立前向。再跑这份配置只能作为共享实现重复性检查，不能冒充独立SeqTrack算法对照。
4. 可以继续已有checkpoint的补评、Full校准和有针对性的工程/机制诊断。seed42未达恢复门槛、late-3及Full闭环证据尚缺，当前不应判定可以直接铺开五类别完整nuScenes大规模论文长跑。若另登记独立matched SeqTrack，固定final/late-3且分别比较S/P；同时不能由两次匹配低分宣称健康恢复。

本报告未在CPU上重跑CUDA训练或nuScenes评测，也没有对未知scene_id自行补造映射。
