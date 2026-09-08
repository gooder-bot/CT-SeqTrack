# v27 五臂 mini：训练完整性与 B0 可比性审计

审计日期：2026-09-07。只读检查 `output/20260905-173106/109/114/118/121-27_*` 对应五个完整目录；没有更改模型、配置或原实验结果。可复核数据：[checkpoint_fairness.json](checkpoint_fairness.json)，采集脚本：[audit_checkpoints.py](audit_checkpoints.py)。

## 结论

五个实验均已正常训练完 60 epoch，late-3 checkpoint 完整，所有最终模型张量和 Adam 状态有限，启用的模块都实际获得了更新。现有结果不能支持“B1-GRU 比 CfC 跟踪更好”：五臂原本应共享的 B0 已经在第一次 Adam 更新后分叉，最终 B0 参数、BN 统计和 observation 跟踪结果不再相同。

目前的证据将分叉位置收窄到**首次 observation forward 之后、首次 optimizer 更新及其数值计算路径**。没有证据将第一次差异归因于 B1/B2/B3 的学习耦合：第一次机制流发生在 global step 4，而 B0 首次参数和 Adam 差异已经发生在更新 1。CUDA 非确定性数值运算是优先验证项，但当前结果只保存 hash、没有首次逐层梯度张量，不能宣布已定位具体 CUDA 算子。

## 1. 完成与 checkpoint 状态

| 运行 | B0 更新 | B1 更新 | B2 更新 | B3 更新 | 完整停止标记 | epoch 60 边界 |
|---|---:|---:|---:|---:|---|---|
| B0 | 63,420 | — | — | — | 有 | 完整 |
| B1-CfC | 63,420 | 15,060 | — | — | 有 | 完整 |
| B1-GRU | 63,420 | 15,060 | — | — | 有 | 完整 |
| B1+B2 | 63,420 | 15,060 | 15,060 | — | 有 | 完整 |
| Full | 63,420 | 15,060 | 15,060 | 15,060 | 有 | 完整 |

- 每个最终文件 `formal_checkpoints/epoch=060.ckpt` 的内部 `epoch=59`、`global_step=63420`、`ct_epoch_boundary_complete=true`。内部 epoch 从 0 开始，这与训练 60 轮一致。
- 全部 `train.log` 有 ``Trainer.fit stopped: max_epochs=60 reached``，未找到 Traceback、RuntimeError 或 CUDA out-of-memory 标记。
- 五臂的 epoch 58/59/60 均可在 CPU 只读反序列化，内部 epoch 为 57/58/59，global step 为 61,306/62,363/63,420，均带完整 epoch 边界标记。
- 全部最终 `state_dict` 浮点张量及 Adam 一、二阶矩均通过有限性检查；不是“进程退出但没有完整训练 checkpoint”。这不等于在本地重新执行了一次实际 nuScenes 模型恢复/评测。
- 所有 B0 的 236 个参数张量都有 Adam 状态，逐参数 step 都为 63,420。CfC 的 26 个、GRU 的 20 个、B3 的 12 个参数张量都有更新，逐参数 step 均为 15,060。
- B2 的 66 个参数张量都有 Adam 状态；B1+B2 的逐参数 step 范围为 14,733–15,060，Full 为 14,702–15,060。部分监督依赖有效目标证据，不能把这些条件梯度的计数差异认定为冻结；没有启用参数从未更新。
- `active_frozen_parameters=[]`，初始化与恢复 provenance 均没有传入 checkpoint。所有运行保持同一个 Adam、启用的命名参数组、60 次 StepLR；第 60 轮保存时学习率已从 `1e-4` 经 20/40/60 的衰减到 `1e-7`（该值是 epoch 边界之后的状态，不表示整个最后 20 轮用了 `1e-7`）。

## 2. 数据集合与预算

五个运行 provenance 一致：

- 参数训练：6 个场景 `scene-1077, scene-1094, scene-1100, scene-0757, scene-0655, scene-0553`；228 个 tracklet、4,231 帧。
- 训练期内部 dev：`scene-0061`；12 个 tracklet、321 帧。这里的分数不是官方两场景 `mini_val` 分数。
- calibration：`scene-0796`。正式 mini test manifest：`scene-0103, scene-0916`。
- 四个插件臂 mechanism 使用相同的 226 个有非首帧的 tracklet、4,003 个 prediction endpoint，selection SHA256 为 `08e7ed2737145c2722fce987f0a1fa539fde252181a8b20ff0fa78feefcff194`。两个只有首帧的 tracklet 不存在预测更新，不属于遗漏训练轨迹。
- Observation 每轮 1,057 个更新；mechanism 每轮 251 个更新且每轮遍历一次。四插件臂 mechanism 的集合、步数、时间模式完全一致。
- seed=42、batch=16、workers=4、60 epoch、每 5 轮 dev、Adam、LR=1e-4、StepLR(20,0.1)、true time 一致。
- 所有运行 provenance 记录相同 commit `8b8b8d958954e3a8ccc61655ec3ec934804e9e73`。`dirty_tracked=true` 的唯一列出的 tracked 改动是删除旧诊断压缩文件 `artifacts/ct_checks/b2_diag_v2_export_20260827_140008.tar.gz`，没有列出源码/config 修改。不能把 dirty 标记本身当成五臂运行了不同源码的证据。

`resolved_config.yaml` 中虽然有 `val_split: mini_val`，实际构造传入 `protocol_role='dev'`，dataset 按角色解析为 scene-0061。解释分数应以实际 manifest/provenance 为准。[main.py](../../../../main.py#L827)

## 3. B0 分叉：已排除什么、尚缺什么

### 3.1 确认的相同部分

1. 五臂 B0 初始参数 SHA256 均为 `798a8def3e825b91dc8f75aee70895bdf67e54d691da7bdc0beded8801e355be`。
2. 首 100 次完整 observation batch 指纹逐项完全一致。指纹包括张量 dtype、形状、字节内容及其他字典字段；不是只比 tracklet/frame 索引。[指纹实现](../../../../models/seqtrack3d.py#L7792)
3. 首个 forward 的 4 个 B0 view loss 五臂完全相同：`15.5456991196 / 19.2395896912 / 18.5371723175 / 18.9520416260`，加权值 `17.2276515961`。所有臂至少前 6 次 view0–3 的记录逐 float32 相同。
4. 最终 checkpoint 的 Python、NumPy、Torch CPU、Torch CUDA RNG 状态五臂完全相同。这仅证明保存时状态一致；不能替代每一步前向的 RNG 快照。
5. B0 所有 28 个 BatchNorm 的 `num_batches_tracked` 均为 63,420。没有观察到插件臂额外增加 B0 BN 更新次数。
6. B0 每个参数的 Adam 更新次数、学习率和 optimizer 超参数一致。

### 3.2 确认的不同部分

第一次 optimizer update 后，5 个 B0 参数 hash 和 Adam 状态 hash 已经全部不同。四个插件臂相对 B0 的首次可见 view0 loss 差异如下（step 从 0 开始）：

| 臂 | 首次 view0 loss 不同的 step | B0 loss | 对应臂 loss | 绝对差 |
|---|---:|---:|---:|---:|
| B1-CfC | 7 | 15.34894085 | 15.34918022 | 0.00023937 |
| B1-GRU | 13 | 16.35947990 | 16.36114311 | 0.00166321 |
| B1+B2 | 6 | 19.83074570 | 19.82852554 | 0.00222015 |
| Full | 7 | 15.34894085 | 15.34994698 | 0.00100613 |

最终 B0 的 236 个参数张量全部与独立 B0 臂不同；整体相对 L2 距离为 0.2093 / 0.2087 / 0.2155 / 0.2092。56 个 BN mean/variance 张量也全部不同，BN 计数则完全一致。不能将这解释为“只是最终 hash 不同但模型实际相同”。

插件 mechanism 首次实际 forward 日志出现在 global step 4，第一步参数差异发生得更早。当前 `training_step` 也明确将 observation transaction 的 B1/B2/B3 路由全部关闭，然后才构造 mechanism，并在机制前后保存/恢复全局 RNG；机制 B0 采用临时完整 eval/no_grad。[训练事务](../../../../models/seqtrack3d.py#L8814)、[机制 observation](../../../../models/seqtrack3d.py#L8657)

### 3.3 优先验证的原因

现象符合 GPU 反向归约/Adam 数值差异先产生微小参数扰动，随后递归和非线性训练放大的模式。当前入口设置 float32 matmul precision 为 `high`，没有请求 deterministic Trainer/算法；Adam `foreach=None`。这提供了需要检查的运算路径，但**不是已经定位的具体错误**。[入口数值配置](../../../../main.py#L63)、[Adam 构造](../../../../models/seqtrack3d.py#L3324)

由于第一次机制流之前已分叉，不能优先归咎于 CfC/GRU 模块损失、B2 memory 或 H3 shadow。当前可排除“不同随机初始化”“首 100 次 observation 输入不同”“B0 多了 optimizer 更新”“机制额外累积 B0 BN 计数”作为首次差异的直接解释。

### 3.4 下一次应怎样定位

先做小规模诊断，不直接再跑五臂 60 轮：在同一张 GPU 上**顺序**跑两次 B0 和一次 B1-GRU，各 100 step，从同 seed scratch 开始。保存 step 0/1/2/5/10/100 的输入、B0 forward 输出、BN、逐参数梯度、Adam 一二阶矩及更新后权重的实际张量或分层数值摘要。若同臂重复也在第一步分叉，优先查 CUDA 运算的确定性与 TF32/foreach；若同臂一致而跨臂不同，再查内存布局、优化器路径、设备执行次序及恢复 RNG 边界。改变 deterministic/TF32 设置是诊断条件，需要单独标记，不能把新设置下的分数混进现有主表。

不需要冻结 B0，也不需要从某个 B0 checkpoint 分阶段训练插件。针对已有 checkpoint，先补每个 B2/Full checkpoint 自身的 observation-only 与正式动作策略成对评测，测量同一个 B0 下插件实际改变了什么。

## 4. 日志可解释性问题

`loss_loss_b0_transaction`、`loss_loss_total` 等使用 observation 与 mechanism 共用的 `add_scalars('loss', ...)` 通道；有机制的 global step 会出现两条同 step 数值。后一条中的 B0 loss 只是机制样本上计算出来的监测值，最终反向选择的是启用插件的 transaction loss；不能将后一条当成实际 B0 optimizer 的训练损失。

例如 step 4：五臂 observation 的 B0 loss 都为 `13.9290618896`，CfC/GRU 的同一步随后又写入 `15.8659267426`，B1+B2/Full 写入 `12.6625680923`。如果按“同 step 最后一条”或整文件平均聚合，会制造虚假的 B0 训练损失差异。此次 B0 初期对齐审计使用只属于 observation 的 `loss_loss_b0_view0..3`，规避该问题。[共用 logger](../../../../models/seqtrack3d.py#L9012)

建议后续把标签改为 `observation/loss_*` 与 `mechanism/loss_*`，并显式记录每个 stream 的 row denominator、update index 和 epoch。该改动属于诊断清晰度，不需要改变已有训练算法。

## 5. 与 SeqTrack 比较的边界

这五个运行本身没有 SeqTrack reference 的第六个同步匹配实验。是否能与历史 SeqTrack 直接比较，需要同时核对历史 reference 是否也是 6 个 train 场景、321 帧 dev、同评价器/首帧尺寸、同更新预算。如果历史 SeqTrack 指标来自官方 mini_val 或训练了全部 8 个 mini_train 场景，不能将分数相减称为模型增益。

当前 B1-only 臂正式输出 observation；Full 的 resolved config 中 `ct_action_calibration_path=null`。训练结束只证明权重训练完成，不能证明 Full 已经过完整阈值拟合、闭环锁定与官方 mini_val 正式评测。对此应结合主报告对评测导出与 calibration artifact 的盘点判断。
