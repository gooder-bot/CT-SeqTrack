# v35/v36 执行提速与配对验证

2026-10-09 更新：v37 已复用本页执行优化，且仅登记的 `37_full_bf16_mini.yaml` 开放正式局部 BF16。下方“BF16仅工程检查”仍适用于 v35/v36；不将这一例外扩大到其他旧版本或 v37 B0/B1/B1+B2。v37 五组命令和一次 CUDA 生命周期、短 BF16 检查见 [v37 操作页](CTSEQTRACK_V37_LAUNCH.md)，同卡速度测量独立于正式五组进行，不能用并发墙钟推断加速。

本轮仅优化执行实现，不改变网络、参数、采样、loss、课程、Adam、递推更新或科研指标。训练端的全部诊断字段、记录频率、候选台账和评测归约保留。尚无 A40 实测速度或 BF16 最终得分结论。

## 执行模式

`main.py` 的 v35/v36 入口新增 `--execution_profile`：

| 值 | 用途 |
| --- | --- |
| `optimized_fp32` | 新源码默认。BN 有效计数复用、batch 规范化复用、事务内原帧数据复用、诊断合并传输；仍为严格 FP32，TF32 关闭。 |
| `legacy` | 原计算与获取路径，供配对检查或速度回退。不是旧 checkpoint 的来源豁免。 |
| `optimized_bf16` | 上述优化加特征层局部 BF16；必须显式选择，且仅允许 `ct_engineering_check`。 |

旧 v33/v34/reference 不启用这些优化，也不接受新模式参数。没有更改旧 YAML 或正式登记。缺省与显式 `optimized_fp32` 的配置摘要一致；另外两个模式有独立配置身份。

BN 复用的是整数 count，不是统计值或网络特征；均值、方差、running buffers、尾批/drain 规则不变。canonical 缓存按张量版本校验原始/规范化输出别名，extension 后加入时独立更新。CPU 原帧缓存只活在 prepare→acquire→commit 内，整云 GT 统计按需计算，不增加历史帧扫描。所有缓存均不进入 checkpoint、memory 或候选 JSON。

BF16 仅用于 PointNet 特征层、token 投影和时序 attention 特征计算。BN、最终输出头、B1、B2 几何、loss、门控和状态保持原 FP32/FP64；Trainer、参数和 Adam 状态仍为 FP32。没有启用全局半精度、FP16 scaler、TF32、compile 或新 CUDA 算子。不能将该模式标成“全 FP32”，也不能把短检查通过解释为末轮不掉分。

## 旧实验与新运行

新 manifest/checkpoint 增加执行模式和源码摘要；评测及完整 epoch 恢复必须匹配。旧 v35/v36 checkpoint 缺少新执行身份，使用其登记的原源码复现。没有绕过旧 config/source/run/RNG 校验，也没有跨源码正式迁移。

本轮修改前的源码来自 Git `aec7fd6fa41b08fd8d2bdfe7061da176b023f31f`，生产源码摘要为 `00731c9d1053332b6e24e10db07177b4187c2c78509c95dbcdd728db52c393bb`。本地保存了独立快照及24份原配置摘要：

`artifacts/ct_checks/20261009-142559_execution_optimization/baseline_manifest.json`

该快照与 v36 原运行源码一致；更早的 v35 checkpoint 仍以其自身 manifest 为准。不要拿当前快照身份代替所有历史版本，也不要覆盖旧输出目录。

新正式运行使用原命令即可默认进入优化 FP32。工程测速完成前，可显式用 `--execution_profile legacy` 对照；已有运行不能在恢复中切换模式。BF16 本轮不登记新的正式实验，不自动追加 epoch、seed 或 LR 搜索。

## A40 上一次配对检查

在原服务器 Python3.9 / torch2.0.1+cu118 / Lightning2.0.2 环境、项目根目录运行。使用正确数据根 `/home/lishengjie/data/nuscenes-mini`，不要传 nuscenes Python 包目录；其余环境路径见 `SERVER_PATHS.md`。示例选择 GPU0，不会停止或调整服务器其他进程。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python tools/check_execution_performance.py --cfg cfgs/ct_seqtrack/35_b1_w_scaled_lr_mini.yaml --path /home/lishengjie/data/nuscenes-mini --mode all
python tools/check_execution_performance.py --cfg cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml --path /home/lishengjie/data/nuscenes-mini --mode all
python tools/check_execution_performance.py --cfg cfgs/ct_seqtrack/36_a_full_mini.yaml --path /home/lishengjie/data/nuscenes-mini --mode all
```

每次自动创建新的 `artifacts/ct_checks/` 子目录，不写工程 checkpoint。不要同时启动这三个测速命令；工具内部轮换 legacy/optimized 执行顺序。报告会保存 GPU 进程快照，其他作业的负载变化仍可能影响结果。

- `check`：默认分别在零基 epoch 0、10 的采样课程下检查各8步，并做短程连续评测。比较数据、候选、所有 loss/gradient、BN、Adam、RNG、状态、原诊断和指标；记录实际递推深度及尾批/drain 覆盖。它使用原 host 的数值事务，不将手动检查冒充真实 Lightning logger 验证。
- `benchmark`：真实 Lightning Trainer，保留 CSV/TensorBoard、原诊断和日志频率。默认20步预热＋100步计时，重复3次；同步只在窗口边界，另运行短 profiler。报告配对输入顺序、初始化、日志一致性、吞吐、step分布及显存。
- `all`：先严格 FP32 等价检查，再测速。若等价检查失败，不继续测速。

测速是 epoch0 初始课程的短窗口，不代表全部60轮的平均耗时；成熟递推课程的数值正确性由 `check` 的 epoch10 阶段覆盖。CPU enqueue 时间不是 GPU 延迟，报告分别标注 CUDA event 与窗口时间。profiler 的 instrumented 耗时不加入吞吐比较，各嵌套阶段时间不能简单相加。

## 可选 BF16

先检查数值和递推分歧：

```bash
python tools/check_execution_performance.py --cfg cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml --path /home/lishengjie/data/nuscenes-mini --mode bf16
```

也可一次完成 BF16 检查及独立吞吐测量：

```bash
python tools/check_execution_performance.py --cfg cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml --path /home/lishengjie/data/nuscenes-mini --mode all --candidate_profile optimized_bf16
```

报告单列训练/连续评测中的位置与角度偏差、候选和测量选择变化、资格阈值翻转、memory/递推状态差异、RNG及Adam步数。BF16 不要求与 FP32 输出逐位相同；有限值与事务检查通过只能标记工程检查完成，最终得分仍未验证。

为避免只检查随机初始化，也支持 `check`/`bf16` 只读加载已有同模型、同臂权重：同时传 `--weights 原run内的checkpoint --run_manifest 原run/run_manifest.json`。工具校验原 config/schema/source（v36含run_id），严格加载模型参数，重新创建工程优化器和状态。它不调用正式 resume，不保存工程权重；带历史权重时不允许进入 benchmark/all。

## 验收与限制

FP32 必须先通过逐位数值/完整日志一致性，再看端到端速度。若某卡的结果回退，使用 `legacy` 对照并依据 trace 撤回无收益项，不能通过关闭诊断取得速度。短测速不会自动提升 BF16 或启动正式实验。

本地机器只有 torch2.3.0+cpu，没有 Lightning、nuScenes 和 CUDA。全仓回归829项通过、17项因缺依赖/设备跳过，CPU 测试和旧源码字节对照已通过；真实 Lightning logger/完整epoch恢复、CUDA 数值、A40 吞吐及 BF16 得分需要服务器检查。详细结果见[本轮独立工程报告](../artifacts/ct_checks/20261009-142559_execution_optimization/REPORT.md)。
