# v30 首次 mini 报错与 B0 显存分析（2026-09-15 晚）

**后续预算更新**：用户接受相对旧B0约+5GB，当前默认`ct_b0_masked_bn_recompute:false`，
直接计算有效BN，保留重复mask优化与native分配器；true才启用下文的局部重算方案。
下文44.33%是上一轮低显存重算路径的CPU数据，不是当前默认直算的数据。
最新选择、文献与102/2定向回归见[显存预算说明](CTSEQTRACK_V30_MEMORY_TRADEOFF.md)。

结论：两个 Full 的直接故障是 B3 探索分支把 bool 交给 CUDA 排序；B0 显存由真实激活增加与缓存池膨胀共同构成。
本地已经修复排序、减少掩码激活保存和重复拷贝，并更新启动分配器设置。训练参数与方法目标不变。
此次只读了服务器已有日志、TensorBoard事件和进程状态，没有上传、启动、停止或修改服务器任务。

## Full 的退出原因

两组均在第4步进入 `choose_mode_action(..., policy='explore')`，运行
`torch.argsort(~valid, stable=True)` 后报 `Sort currently does not support bool dtype on CUDA`。
这是本次新动作探索路径的类型兼容缺陷；与CfC/GRU的动态模型选择无关。
PyTorch 2.0.1 CUDA排序源码明确拒绝bool，见[官方源码](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/Sort.cpp#L71-L74)。

`utils/v30_policy.py` 将排序键转为int64，仍按合法动作优先、固定原槽顺序稳定排列。
不改变均匀探索、轨迹/帧哈希、动作集合或梯度。新增回归对全部64种六槽合法掩码和7个seed执行实际探索，
CPU测试主动拒绝bool排序，因此能在没有CUDA时防止本次代码退回。
原来的CPU行为测试只检验哈希，没有实际触发CUDA排序；先前868项通过不等于此分支已获CUDA验证。

`tqdm.__del__` 是上述退出后的清理异常。B0服务器日志到第90步收到SIGTERM，未记录OOM；
不能由SIGTERM判断是谁或为何终止。只读检查时三组用户提供的PID均已不在。

## B0 的35GB来自哪里

用户报告单进程约35GB，以前同参数约5GB。读取本次B0事件文件
`output/20260915-181956-30_b0-mini_car_seed42_60ep_bs16/lightning_logs/version_0/events.out.tfevents.1789467645.tesla-a40-2.3917896.0`，
其中CUDA指标只有一个已落盘采样：TensorBoard `step=49`，即第50次更新。
历史对照来自 `artifacts/ct_checks/reports/20260915_v29_partial/training/training_summary.json` 的B0：
v29 perf、nuScenes-full、batch16、FP32，共4289个抽样。数据规模不同但每步点数及B0网络宽度一致，
可对照单步显存；不能把不同数据规模下的训练速度直接比较。

| 指标 | 历史v29 B0 | 本次v30 B0第50步 |
|---|---:|---:|
| 前向实际allocated | 3931.9–3936.1 MiB | 6969.0 MiB |
| 本步实际peak allocated | 3979.6–3983.9 MiB | 7059.3 MiB |
| 步末实际allocated | 123.0–124.5 MiB | 121.7 MiB |
| 步末reserved | 7942 MiB | 34672 MiB |
| 本步peak reserved | 7942 MiB | 34716 MiB |

也就是：实际峰值约3.89→6.89GiB；池保留约7.76→33.86GiB。35GB不是35GB都在保存活跃计算图。
该采样反向后allocated回到约0.119GiB，没有出现大计算图跨步常驻的迹象；一个采样不能证明所有时刻都无泄漏。
PyTorch缓存池中未用块也计入nvidia-smi进程占用，见[官方内存说明](https://docs.pytorch.org/docs/2.0/notes/cuda.html#memory-management)。

已定位两个代码层面来源：

1. 部分有效BN用Python张量运算展开方差、中心化和affine。一个无效槽便使整层离开原生BN路径；
   `[16,1024,4096]` 的一个FP32张量就是256MiB，多层反向保存多个这样的副本。
   `masked_sequence` 原先还在Conv/ReLU前后反复where，生成本可共享的激活拷贝。
2. roll-in活跃行数变化使GPU分配形状变化，v30长窗口又增加了形状访问次数。
   旧启动值`max_split_size_mb:64`禁止拆分大于64MiB的缓存块，与大量变化的大激活尺寸不匹配，
   是缓存复用变差的合理解释。现有事件不含分配器快照，不能量化其中每个因素对33.9GiB的独立贡献。

长窗口明确在eval/no_grad下逐波执行，端点只有一次带梯度前向；没有保存八步梯度图。
原始窗口和每worker256MiB点云LRU在CPU，不能解释成256MiB/worker显存。

## 本地修复与数值保持

- `utils/masked_observation.py`：部分有效BN只保存输入、mask和affine参数；反向逐层重建原公式，
  使用原autograd及归约顺序。running statistics只在前向更新一次；全有效BN和少于2有效元素分支不变。
  删除Conv→BN及ReLU周围的重复mask，保留入口和必要输出隔离，避免无效NaN污染卷积权重梯度。
- `utils/v30_policy.py`：探索稳定排序使用int64键。
- `tools/run_ct_v30_server.py`与[mini启动文档](CTSEQTRACK_V30_MINI_LAUNCH.md)：环境值改为
  `PYTORCH_CUDA_ALLOC_CONF=backend:native`，显式覆盖shell中残留的64MiB限制。
  官方默认允许拆分所有块；max_split_size_mb是有相应OOM/碎片证据后才调节的选项，
  见[分配器参数](https://docs.pytorch.org/docs/2.0/notes/cuda.html#environment-variables)。

没有减少batch16、1024点、历史3帧、roll-in长度或候选数，没有改FP32、strict、Adam和学习率。
没有每步empty_cache。局部BN重算增加少量计算，实际速度与CUDA峰值仍需修复版实跑确认；
它不支持二阶导数，当前正式一阶Adam不受影响。

独立CPU测量按底层storage去重，真实宽度三PointNet、batch2、每帧1024点：

| 保存的反向张量 | 修复前 | 修复后 |
|---|---:|---:|
| 全有效 | 419.300 MiB | 400.796 MiB |
| 一个历史帧无效 | 719.813 MiB | 400.731 MiB |

部分无效保存量下降44.33%，并接近全有效保存量。这是CPU张量保存量，不能写成“服务器显存已经下降44%”
或承诺恢复到5GB。已验证原BN与修复BN的输出/梯度/状态；真实host及三臂一致性回归结果在
[本次验证记录](../artifacts/ct_checks/20260915_v30_cuda_memory_fix/REPORT.md)。

## 后续运行

同步以上两个运行时utils文件和命令生成器，并使用mini专页的新命令；三臂共用修复后的B0。
模型YAML和batch/epoch/workers/seed保持原样。此次失败运行保留日志，新日期目录重新scratch运行，
不把不完整的第0轮结果作为正式checkpoint继续混用。
本轮没有把重新实跑设成额外审批门，也没有在用户服务器主动执行训练。
