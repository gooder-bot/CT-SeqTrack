# CT-SeqTrack v32 实验协议

本协议定义 B0 修复后的新实验；v31 算法、配置与协议从 Git `b1d886e` 复现。四份 `31_*` YAML 原样保留，当前入口拒绝旧身份。物理模块路径保留，不代表沿用旧权重 schema。

## 登记实验与验收

2026-09-22 用户更新本轮排程：四组 mini 单 seed42。除模型及指定物理卡外，其余正式设置保持原值。

| 模型 | 配置 | 种子 | 物理 GPU |
|---|---|---|---|
| 独立 SeqTrack | `32_seqtrack_ref_mini.yaml` | 42 | 0 |
| 修复 B0 | `32_b0_mini.yaml` | 42 | 0 |
| Full-GRU | `32_full_gru_mini.yaml` | 42 | 1 |
| Full-CfC | `32_full_cfc_mini.yaml` | 42 | 1 |

本轮在 seed42 的 final60 检查 B0 的 Success、Precision 是否分别不低于独立 SeqTrack 超过 2 个百分点；并报告两个 Full 相对 B0 的差异及全部四组的 58/59/60 late-3。不挑最佳轮次或种子，历史高分不视为本次可重复基线。这些实验检验整体修复包，不归因每项改动。

原计划的 reference/B0 seed52 两次复验留待后续；只有两 seed 对照均齐全才可用 `compare_v32_baselines.py` 判定原双 seed 条件。当前四组不能替代它。B1/B2/B3 的机制、监督和写入规则保持，仅适配 B0/共享 decoder 内部局部几何；本轮允许先观察两个 Full 的整体结果。

## 共同预算与合法输入

- nuScenes mini Car，8 train / 2 val 场景；初帧尺寸为模型输入，当前 GT 尺寸只用于监督和评测。
- scratch60、batch16、workers4、FP32、单卡；Adam lr=1e-4、betas=(.5,.999)、eps=1e-6、weight_decay=0；StepLR20轮乘.1。
- 每5轮验证、每2轮保存完整epoch，58/59/60全部保存并自动评测。
- 每轮19,108 nominal行、1,195次Adam（最后4行），共1,146,480行、71,700次更新。普通mean loss与标准Adam；日志按实际行数归约，不声称端点优化影响完全等权。
- 评测完整连续递推，106轨迹/2,285帧，含2,179预测帧和106初始化帧，沿用benchmark_compat。
- 每worker原始云缓存256MiB；无旧preloading。工程权重不用于正式初始化。

## B0递推与观测合同

- 四分支课程最终1/3/3/8，10轮完成；每个非首帧端点每branch恰好一次。主进程唯一提交accepted状态，跨帧detach。
- 非canonical仅初始化共同局部XY±.3m、yaw±1.5°；XY转世界平移加到全部种子框，不旋转轨迹中心。物理GT、点云与首帧记忆不变。扰动/预测hint=.2/.8，已知0/1，当前.5。
- 预留112单步窗口（mini每branch28个）补齐尾部。首次消费reserve起及所有partial批，B0 BN使用running统计，affine/input仍有梯度；小工程数据按实际可用窗口预留。
- B0/共享decoder内部anchor-local几何，公共框仍是相对anchor中心的世界轴XYZ及绝对yaw。B1/B2物理几何、B2/B3监督不变。
- 前景soft gate停止梯度；coarse不读质量摘要。current/history CE各50%，缺组归一化；BC按帧/端点归约。质量头目标与梯度保留。
- 每帧最多128有效点token，不复制稀疏点；sequence_valid允许历史支持q0，current_valid仍只表示当前测量。

## 独立SeqTrack

网络、原loss、teacher4与预处理来自冻结SeqTrack最小依赖源，SHA与适配清单随包保存；不使用production B0或新loss。保留argmax前景、moving硬门、原fine/帧布局、重复1024点、scale1.25标签/hint、伪时间、newest-first及独立历史扰动。

保留历史有效点不足时重采样，总行数与更新预算固定，保存nominal→actual曝光审计。合法非首帧预算、训练及推理网络bbox_size统一首帧尺寸、共同数据/评分、运行兼容和小尾批BN为显式适配；原teacher标签与目标尺寸保留。每个正式checkpoint评测前重置共同seed与test loader generator，保证自动评测与独立test同起点，仍使用原历史采样算法。两模型不声称逐端点或训练状态分布完全配对。

## 身份、恢复与状态

模型schema为ct_seqtrack.joint_identity.v32，family为ct_seqtrack_v32。模型、seed、训练设置、扰动/调度与时间语义进入配置摘要；参考源码/原协议也进入对照身份。`--init_checkpoint`禁止；`--checkpoint`仅供同身份完整epoch恢复或评测，不跨版本/臂/工程身份。

记录resolved config、源码、数据/采样manifest、RNG、训练覆盖与评测帧；恢复重建相同调度，不支持任意batch恢复。路径和日志位置不改变权重身份。联合模型保留full/KITTI、四模块臂、CfC/GRU和true/fixed/shuffled；SeqTrack保留伪时间，不套B1控制。

v32尚无正式成绩，CPU/合成检查不替代真实nuScenes/CUDA和四次scratch60。实现见[修复说明](B0_V32_REPAIR.md)，命令见[工具面](FORMAL_TOOLING.md)。
