# v29 perf 完整 nuScenes 三臂：未完成训练快照

本报告只读 2026-09-11 02:30 启动的三组 `output/` 原始 train.log、run_provenance、TensorBoard 和 acquisition_supply JSON；没有加载 checkpoint，没有修改训练代码、配置或 output。这里描述本地已同步快照，不能据此确认服务器当前进程仍在运行。

## 进度与可比范围

| 臂 | 完整轮数 | 最新进度（日志 epoch 从 0 开始） | 完整 checkpoint |
|---|---:|---|---|
| B0 | 4 | Epoch 4，11708/50687 | 2、4 |
| Full-CfC | 2 | Epoch 2，40067/50687 | 2 |
| Full-GRU | 2 | Epoch 2，41576/50687 | 2 |

三臂共同完成窗口为 epoch 1—2。不能把 B0 第 4 轮与 Full 第 2 轮当作同训练预算的性能对照。快照中没有达到第 5 轮正式验证的记录，不能据此判断基线恢复或 Full 涨分。sanity/初始化验证另看验证分析，不能当作训练完成后的验证。

每个完整 epoch 的全量 `ct_epoch_core_loss` 累加器确认：三臂各 50,687 次 observation transaction、810,992 行；两个 Full 各 11,956 次 mechanism transaction、191,288 行，和 provenance 的 endpoint 数一致。机制末尾有小批次，因此不能拿 11,956×16 当作真实行数。未在三个 train.log 中发现主异常或 `max_epochs=60` 完成标记；已提取的 scalar 均为有限数，但这不等于证明未记录的所有中间张量均正常。

日志显示 B0 的完整第 1/2/3/4 轮分别 18:35:16、18:29:07、49:30:58、16:13:21；CfC 第 1/2 轮 27:14:30、57:58:10；GRU 27:08:19、56:38:47。三臂均出现明显长时段波动，不能直接将单次 epoch 时间差解释成算法速度差，也不能用这里的运行时间替代 ABBA 性能测量。

## B0 在学习，但 loss 下降不能作为分数恢复证据

下表是所有 observation transaction 的行数加权 epoch 均值，不是每 50 步抽样平均。

| 指标 | e1 | e2 | e3（仅 B0） | e4（仅 B0） |
|---|---:|---:|---:|---:|
| 总损失 | 2.309548 | 1.550123 | 1.407329 | 1.355100 |
| 最终输出 center（aux，未乘权重） | 0.393515 | 0.288125 | 0.264590 | 0.257341 |
| physical coarse center | 0.109165 | 0.078244 | 0.071683 | 0.069616 |
| physical motion center | 0.237970 | 0.163310 | 0.146930 | 0.140608 |
| history ref center | 0.200125 | 0.140196 | 0.129673 | 0.125843 |
| BC | 0.644558 | 0.385693 | 0.346056 | 0.330209 |

e1/e2 的全部 observation epoch loss 和四项 accuracy 在三臂中相同。对每步序列中已记录的 30 项非计时 observation scalar，B0 对 CfC 有 2836 个共同 step（0—141449），对 GRU 有 2866 个（0—142949），差值均为 0。这只支持日志精度的一致性，本次没有逐位复验模型权重、BN、梯度或 Adam。

BC 并非总 loss 最大项。e2 的 weighted contribution：最终 center 37.17%、BC 24.88%、physical motion center 21.07%、coarse center 10.10%；e4 分别为 37.98%、24.37%、20.75%、10.27%。各分项先在 FP32 累加再乘权重，分项求和和总 loss 累加器存在约 1e-6 的舍入差，不属于损失双计证据。不能由这些权重贡献推断各分支梯度大小，也不能由下降趋势证明 coarse 目标修改的因果收益。

动/静类日志 e2 为 0.577878/0.976234，e4 为 0.599296/0.978396。`models/seqtrack3d.py:150` 构造二类 multiclass `Accuracy(average='none')`；`:9133` 开始每个 observation batch 计算返回标量，再以 batch size 写入全量 epoch 均值。这是逐批的逐类 accuracy（multiclass 下为召回式归约）的平均，不能称为汇总全 epoch TP/FN 后的全局 recall；没有动态/静态支持数，不能据此倒推静态占比或精确动态漏检率。类别差距值得按运动强度和支持数补诊断。

e2 observation 抽样 1015 个 batch：平均 11.9635/16 行为 roll-in、4.0365/16 行为 teacher；历史存在性平均 0.892898；当前 crop 点数的 batch 均值再平均为 332.834。现有日志没有每行 0/1/2/3 点桶、all-empty、有效 slot 数或无效监督发生数。不能把 batch 点数均值当成“没有空帧”，不能由历史存在性均值量化 9 月 14 日无效槽监督反例的实际发生率。

## Full 的参数损失与证据供给

| 全量机制 loss | CfC e1 | CfC e2 | GRU e1 | GRU e2 |
|---|---:|---:|---:|---:|
| B1 transaction | 0.534760 | 0.382092 | 0.528755 | 0.376062 |
| B2 transaction | 1.031008 | 0.743124 | 1.068168 | 0.710899 |
| B3 transaction（已乘权重） | 0.024238 | 0.024605 | 0.024270 | 0.024153 |
| 实际 mechanism 总 loss | 1.590011 | 1.149822 | 1.621199 | 1.111112 |

B1/B2 在降低当前监督损失；B3 即时目标未呈明显下降。各机制 loss 非零和 learning-rate 日志只能说明目标被计算，不能替代参数梯度所有权/更新计数的 checkpoint 核验。

机制目录中 `loss_b0_transaction` e2 为 8.877826/9.018718，它是递归机制输入上的 B0 诊断损失，没有加入这里的实际 mechanism 总 loss；不能将它误判成 B0 观测训练 loss 暴涨，或推断插件梯度已回流 B0。机制最终 center 的诊断损失也明显高于 observation 流，这支持继续检查两类状态分布差异，不能把非配对且历史监督构造不同的两组均值当作同一误差量直接相除解释因果。

e2 全量供给 JSON：CfC 的 768 pool 已含目标行 10058、256 仍保留目标行 10058；GRU 为 9893/9893。相比 191288 个机制端点，最终选点中出现新增目标的行仅约 5.258%/5.172%。低占比本身不能证明 acquisition 出错：许多帧没有恢复需求，必须联合 B0 漂移、crop 内目标和最大合法范围可达性分析。

**同名 point_recall 的分母不同：** epoch acquisition_supply JSON 用 `selected256 / raw extension pool`，e2 CfC 为 365069/669005=0.545689，GRU 为 359542/665597=0.540180；训练 loss 字典中的 `ct_acquisition_point_recall` 则是 `selected256 / prepool768`（`models/seqtrack3d.py:5551`）。`row_recall` 分母是 **768 已有目标** 的行，100%只表示这些行在256仍至少留一点，不能说 support/acquisition 召回100%。该口径需要在之后诊断导出中更明确命名；本次未修改生产代码。

`targetness_balance` 是跨 epoch 的累计类计数，`populations.candidate0` 是单 epoch 供给；例如 CfC e2 positive_points=582691=217622+365069，不能与单轮 pool_targets 混用。

e2 的 241 个机制抽样 batch 显示 CfC/GRU 的平均 parallel margin 为 2.10570/2.09440，perpendicular margin 为 1.000154/1.000163，后者几乎贴下限。结合 9 月 14 日实际几何反例，这支持优先检查低需求样本对 margin 目标的压制及范围相对 B0 crop 的定义。仅凭 margin 均值无法证明所有低点数样本都需要扩张。

## 动作多数不利，分类开始学到区分；仍不是闭环评测

| e2 sampled 合法候选诊断 | CfC | GRU |
|---|---:|---:|
| 样本数 | 665 | 682 |
| helpful | 85（12.78%） | 94（13.78%） |
| harmful | 437（65.71%） | 428（62.76%） |
| bounded 即时平均效用增益 | -0.152838 | -0.145070 |
| presence AP | 0.709707 | 0.705221 |
| helpful AP | 0.371866 | 0.400595 |

增益用 0—1 量纲。以上是训练期间变化权重、混合行为访问状态上的结构合法候选抽样，不是实际 accepted 策略的官方分数，不能换算成 Full 在官方 val 下降了 15 分。e1→e2 presence/help AP 提升，说明分辨能力有学习信号；候选平均效用仍负，说明不能只调 B3 阈值而忽略证据和候选质量。不能据此选择 CfC/GRU 胜者，二者递归状态已经不同。

H3 每臂每轮 scheduled=5977；e1/e2 的 sampled=557/619、not_sampled=5420/5358完全一致，符合稳定抽样。CfC valid=199/205、GRU=206/207，其余采中事件因合法性/有效性条件不执行。e2 H3 有效事件平均 S/P 增益分别 CfC -0.161585/-0.128293，GRU -0.151208/-0.117633；它们是一次干预后 B0 回退续推诊断，不能替代完整策略长期效果。

## 可复核产物

- `extract_training.py`：重新读取原始 run，输出 `training_summary.json` 与压缩 `all_scalars.csv.gz`，不读 checkpoint。
- `summarize_training.py`：从 summary 和 resolved config 生成四张精简表。
- `epoch_metrics.csv`：全量 epoch loss/accuracy 与显式 sampled 诊断，带原始来源。
- `b0_weighted_loss_contributions.csv`：逐项权重和占比。
- `matched_observation_scalars.csv`：30项标量的共同 step 一致性，仅日志层面。
- `sampled_diagnostics_first_two_epochs.csv`：共同完整 epoch 的采样 batch 均值，不能当作全量逐帧分布。
