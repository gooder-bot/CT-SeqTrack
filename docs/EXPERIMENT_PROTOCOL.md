# CT-SeqTrack v31 实验协议

2026-09-22。本协议只定义当前 v31；旧协议通过 [历史索引](HISTORY_EVIDENCE_INDEX.md) 读取。此次工作树收敛保持现有算法和四份 v31 YAML，不把后续修订建议混入清理。

## 模型与训练

- 入口为 `main.py`，使用 `net_model: ctseqtrackv31` 和 `experiment_family: ct_seqtrack_v31`；未知或旧 formal 字段必须报错。
- B0 为观测分支；B1 为真实时间物理先验与获取/context；B2 从 B0 crop 外获取新增证据并形成三个模式；B3 直接选择主观测与三模式的共享解码结果。
- 同帧特征联合学习，每个训练 batch 一次 Adam；窗口内每个端点仅提交一次 accepted 状态。sigma、离散几何、GT 与跨帧状态 detach 保留，不做跨帧 BPTT。
- 跨臂 B0 允许因联合训练而不同；不要求旧共享 B0 逐位相同，不调用旧策略拟合或标定回退。
- 四条独立状态分支使每个非首帧训练端点共四次曝光。课程内短/长窗口最终为3/8；窗口起点使用合法历史初始化，窗口内无 GT 重置。当前分布差与改进建议见 [最新问题](../最新问题.md)。

## 运行身份和固定预算

三份已登记 mini 配置为 `31_b0_mini.yaml`、`31_full_cfc_mini.yaml`、`31_full_gru_mini.yaml`，共用独立 `31_formal_base.yaml`。每个臂从 epoch0 随机初始化，所有启用参数可训练。

| 项目 | 正式设置 |
|---|---|
| 数据/类别/种子 | nuScenes mini / Car / seed42 |
| 训练预算 | 60轮、batch16、workers4、FP32、每进程单卡 |
| Adam | lr=1e-4、betas=(0.5,0.999)、eps=1e-6、weight_decay=0 |
| 学习率 | StepLR每20轮乘0.1 |
| 观测/证据 | 三帧历史+当前、每帧1024槽、唯一真实点有效；新增768→256、三模式 |
| 验证/保存 | 每5轮验证；每2轮保存epoch边界，58/59/60全部保留 |
| 汇报 | final60与58–60 late-3；禁止按各实验最佳轮次挑选 |

`--init_checkpoint` 禁止；`--checkpoint` 仅用于相同身份的完整 epoch 边界恢复或评测。不跨臂、不跨版本、不用工程 checkpoint 正式初始化。配置摘要包含模型、训练与时间语义；文件路径、日志目录不决定权重身份，shuffled manifest 内容进入摘要。

## 保留的数据与控制接口

- `v31_arm` 支持 `b0/b1/b1_b2/full`；时间后端支持 `cfc/gru`。注册的正式三臂只是当前已执行子集。
- nuScenes 支持 `v1.0-mini`、`v1.0-trainval`，使用 `global` 坐标。mini 为8训练/2验证场景；full 使用既有 train_track350/val150 划分。
- KITTI 使用 `dataset: kitti_mf`、`version: kitti_tracking`、`ct_coordinate_mode: sensor_relative`，保留既有序列、标定、时间与原始点身份接口。
- `dynamics_time_mode` 支持 `true/fixed/shuffled`；shuffled 必须指定合法 `dynamics_time_manifest`。这些控制只改变登记的时间输入，物理监督与数据身份保持明确。
- 数据按需读原始帧，每 worker 256MiB 有界缓存；不使用旧 `--preloading`。GT 只用于训练目标与诊断，不进入推理候选选择；目标尺寸使用首帧合法输入。

仅上述 mini 三臂已有正式结果。full/KITTI、模块递进与真实时间消融尚未完成，不将接口保留解释成实验完成。

## 结果与研究主张

[2026-09-21正式结果](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md) 已覆盖“尚未训练”状态：三组完成60轮、58–60独立评测，各71,911次Adam、1,146,480端点曝光；官方mini_val为106轨迹/2285帧，JSONL重算一致。

final S/P 为 B0 23.959519/27.834792、CfC 22.491247/29.352297、GRU 22.839169/31.844639。两个 Full 的 final Success 低于同版 B0；v30历史参照为40.473742/47.840262。当前性能验收未通过，late-3和Precision单升不能替代预定final60双指标要求。

后续先修当前帧身份支持、观测表示、训练分布、模式与状态写入，再做新版 B0/Full-GRU 固定预算 mini；CfC保留同接口机制对照。mini达标后推进full/KITTI和模块/时间消融。不以单seed、CPU探针、BN交换、构造反例或同状态oracle主张稳定涨分、SOTA或时间/记忆因果收益。
