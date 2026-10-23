# v38 clean1：结构精简与复现边界

2026-10-23。本次只整理当前 B0/B1/B3 的实现和执行路径，不新增正式实验，也不改变三组训练配方。源码基准为 `14c3b2fd33be9f9b7aca10e28d5f8fbbd0b9986e`。

实施清单、最终测试、数值差异和待测项见[交付报告](CTSEQTRACK_V38_CLEANUP_REPORT.md)。

## 当前结构

当前代码直接维护在原 `CT-SeqTrack/`，服务器也使用原目录更新，不建立 clean1 源码目录。`clean1` 仅用于区分执行修订和 checkpoint 身份；历史实现按需从 Git 恢复。原目录更新前须确认旧任务已经结束，具体流程见[操作说明](CTSEQTRACK_V38_LAUNCH.md)。

`main.py → models/ctseqtrackv38.py → models/ct_v38/` 是当前主线。v38 自己持有观测、CfC/GRU prior、decoder、损失、W sampler、数据事务、状态提交、运行身份和评测；不再导入 v31/v36/v37。独立 SeqTrack reference 位于 `models/seqtrack_reference/`。

数据事务为 `prepare → plan_prior → supervise_prior → forward → backward/Adam → commit`。`supervise_prior` 只准备原 acquisition 监督和被动诊断，不获取额外点。输出位置、可信位置和可信速度的规则保持。

| 项目 | 冻结 v38 | clean1 |
|---|---|---|
| 当前路径的 identity memory | 36个空原始槽位及特征占位 | 无字段、无状态对象 |
| extension | 768个空原始槽位、256个特征槽位 | 无点池、无搬运、无投影 |
| decoder query | 56 | 32：3历史框和q0各8角点 |
| decoder K/V（正式输入） | 1316，末292项被屏蔽 | 1024，全部来自B0 |
| q0局部读取 | 4候选，其中3个无效mode | 1个q0 |
| 删除的参数 | 两个64×64投影及bias | 共8,320个 |
| 最终选择 | q0 / prior / trusted propagation | 相同 |

保留原69维局部输入、LWH邻域、16点上限及 `log(1281)` 统计尺度。保留 acquisition head/loss、q0 quality、四帧全空时的 quality BCE、运动sigma、历史上下文、真实时间和可信几何。删除mode/identity/vote/reliability损失和恒零B2日志；`last_supported_quality`仍是有效诊断。

两个退役投影不再作为参数或buffer存在。构造时的临时随机抽样只补偿旧Linear默认初始化及Xavier抽样顺序，保持有效参数的同seed初始化；不参与前向或优化。

## 数值验收约定

用户确认保留紧凑路径，并接受 FP32 归约形状变化带来的训练轨迹差异。这个边界不限于dropout：即使关闭dropout，微小矩阵舍入差异也可能经BN的接近零梯度及Adam的 `eps=1e-6` 放大。因此，不宣称旧、新FP32连续更新逐位等价。

验收分开检查：

- seed42/52保留参数、buffer及构造结束CPU RNG逐值一致；Full不改变公共B1初始化。
- 非零输出头下，FP64连续三次Adam比较输出、有效损失、梯度、BN和optimizer moments；共同dropout mask的对照只存在于测试。
- FP32在每步相同权重下比较输出、损失和梯度。前向容差仍为 `rtol=1e-5/atol=1e-6`，梯度为 `rtol=1e-4/atol=1e-5`，没有放宽原数值容差。
- 新实现自然dropout自身复现、CPU Lightning合成数据完整epoch10→11恢复、独立评测，以及两条缺测/重获短轨迹的accepted/crop/trusted/速度递推。

最初FP32连续Adam未通过的证据保留于工程目录，不将其写成通过。CPU合成检查不能证明真实CUDA吞吐、正式得分或长期跟踪效果保持。

## 身份与历史恢复

三份 `38_*_mini.yaml` 字节保持；默认解析注入 `v38_execution_revision: clean1`。新登记为 `38_clean1_registration.json`，模型schema为 `ct_seqtrack.selective_update.v38_clean1`，checkpoint运行键为 `ct_v38_clean1_runtime`。这些身份由当前identity模块统一导出。

原 `38_registration.json` 保持原字节，不能刷新为新源码。旧checkpoint必须配原登记源码；当前入口拒绝跨身份恢复，不提供正式训练权重转换或宽松加载。

旧算法、配置、工具和测试的完整源树保存在本轮新目录的 `baseline_source_14c3b2f.zip`，并附提交号、SHA256及退役文件清单。Git也可恢复：

```bash
git show 14c3b2f:models/ct_v31/decoder.py
git archive --format=zip --output=../ctseqtrack-v38-original-14c3b2f.zip 14c3b2f
```

历史代码无需常驻本地或服务器。确需复核历史时，按 Git 提交恢复匹配环境和配置，或仅在工程对照期间临时解压；当前活动源码始终只有一版，不将旧实现作为 import 依赖。更早的源码身份以对应run manifest为准，不能认为所有旧checkpoint都属于14c3b2f。

GRU、真实/固定/打乱时间、KITTI/full数据接口、独立参考、分解投影、执行精度及可复用调度器仍保留。它们不因此成为新增正式配方。只依赖退役B2/v36/v37算法的工程组合器随旧源码归档。

## 工具与服务器待测

当前五个v38入口保持，比较器与检查器使用独立公共helper；旧版本专属工具由冻结源码运行。新增 `tools/benchmark_v38_clean1.py` 可在独立子进程中加载两份源码，默认对B1和Full各做模型、完整事务两项，20步预热、100步计时、三次交替重复，报告中位/P90及allocated/reserved峰值。

```bash
python tools/check_v38_full_batch.py --cfg cfgs/ct_seqtrack/38_full_standard_mini.yaml --path DATA_ROOT --device cuda
python tools/benchmark_v38_clean1.py --baseline-source /path/to/frozen/14c3b2f --path DATA_ROOT --device cuda --gpu 0
```

性能工具不保存checkpoint；完整事务测量包括raw读取、prepare、监督、forward/loss、被动诊断、backward/Adam和commit，不包括Lightning文件日志IO。CPU `--synthetic`仅用于工具检查。正式性能比较需同卡、无并发干扰及相同配置。

`output/`和既有`artifacts/`保持。缓存清理只针对活动源码缓存及确认仍为空的指定目录，不删除分析环境或外部链接。本轮没有上传服务器、启动/停止任务或运行新增正式训练。
