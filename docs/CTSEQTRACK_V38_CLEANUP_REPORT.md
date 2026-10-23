# v38 clean1 结构精简交付报告

**当前进展：** 按后续要求，源码和部署均直接使用原 `CT-SeqTrack/`，不建立 clean1 目录。异常 prior 导致部分状态提交的边界问题已修复，最终登记已更新；最新测试与服务器只读核验见[原目录复核报告](../artifacts/ct_checks/20261023_v38_inplace_review/REPORT.md)。下文是首轮精简记录，其源码指纹及测试计数保留为当时证据。

2026-10-23。当前主线已完成独立化和无 B2 执行路径重构。冻结对照为 `14c3b2fd33be9f9b7aca10e28d5f8fbbd0b9986e`，工程验证按用户确认后的数值边界验收；未新增正式训练，未操作服务器。

## 交付内容

当前依赖为 `main.py → models/ctseqtrackv38.py → models/ct_v38/`，数据集和公共工具来自 `datasets/`、`utils/`。生产代码、当前工具和测试 helper 不再导入 v31/v36/v37。独立参考模型位于 `models/seqtrack_reference/`。详细结构与操作边界见[clean1 说明](CTSEQTRACK_V38_CLEAN1.md)。

| 原实现职责 | 当前归属 / 处理 |
|---|---|
| v31/v36/v37 中仍有效的 observation、prior、decoder、几何及损失 | 迁入 `models/ct_v38/`，不保留版本包装或 legacy 包 |
| 继承旧宿主的 v38 训练、状态和评测生命周期 | 独立 `models/ctseqtrackv38.py`、v38 data/runtime/entry |
| 空 memory、extension、evidence、mode 及恒零分支 | 从数据事务、网络、输出、状态、损失和日志中移除 |
| 旧比较脚本、检查脚本和测试入口中的公共逻辑 | `tools/_tracking_audit.py`、`tools/_batch_check.py`、`tests/helpers/` |
| 旧 B1 配方对照 | 带来源提交、内容指纹的固定 fixture |
| 参考模型借用的旧宿主、入口、身份和数据事务 | 独立 reference config/entry/identity/host/runtime |
| A 类旧算法及专属配套 | 61 个模型文件、72 个配置/登记、25 个工具、84 个测试，共 242 个文件移出活动树 |
| F 类缓存和空目录 | 原生 PowerShell 清理 20 个缓存目录、78 个空目录；共 392 个缓存文件、7,771,229 字节；除明确授权的缓存目录外，只删除空目录 |

decoder query 从 56 缩为 32，正式输入 K/V 从 1316 缩为 1024；q0 局部读取从四槽缩为一个候选。只移除两个退役投影的 8,320 个参数，保留有效参数名称、形状和初始化。初始化补偿使用临时普通张量，不注册废弃参数或 buffer。

保留 acquisition head/loss、全空样本的 q0 quality 监督、三个历史框、LWH 邻域/16 点/69 维局部输入及 `log(1281)`、三候选决策、accepted/trusted/相邻可信速度和真实时间规则。GRU、其他数据集、时间对照、独立参考、分解投影、执行精度、现役缓存和调度器继续保留。工程时间清单按数据角色解析并按文件内容入身份；部分训练与自动 late-3 的冲突在入口报错。

## 冻结和保护证据

主要工程证据及索引位于 [20261023_v38_clean1](../artifacts/ct_checks/20261023_v38_clean1/)；锁定源码的 Lightning 生命周期证据另见下文。

- 完整 Git 快照：`baseline_source_14c3b2f.zip`，SHA256 `9e445f4a404b4852ebd393c0958c226cabec24aa38d7af0e2472f42b5d681ee5`。
- 提交号、原文件清单和 1,490 个快照文件的逐文件 SHA256：`baseline_commit.txt`、`baseline_tracked_inventory.txt`、`baseline_source_inventory_sha256.json`。
- 删除清单及原工作区内容指纹：`retired_files.json`、`retired_worktree_hashes.json`；变更索引：`change_inventory.json`。
- 缓存删除脚本和逐项记录：`cleanup_caches.ps1`、`cache_cleanup.json`。仅遍历明确指定的仓库目录，检查绝对路径并跳过链接；不进入 output、artifacts 或分析环境。
- 原有 197,119 个受保护文件全部存在，大小和修改时间保持；三份当前 YAML 与原登记的 SHA256 全部保持。见 `protection_final.json`。这项保护核对采用文件清单/大小/时间，未将其表述为全量输出内容哈希复核。

## 新旧身份

| 项目 | 原 v38 | 当前 clean1 |
|---|---|---|
| 执行修订 | 原冻结源码 | `clean1` |
| 模型 schema | `ct_seqtrack.selective_update.v38` | `ct_seqtrack.selective_update.v38_clean1` |
| checkpoint runtime key | `ct_v38_runtime` | `ct_v38_clean1_runtime` |
| 登记 | `38_registration.json`，原字节保留 | `38_clean1_registration.json` |

最终登记源码 SHA256 为 `4429fbfa624c217122d78e6ce0b37102ccc46837350a5cab5802fad593843f09`。三个正式臂的 YAML 文件名和训练内容不变，默认解析注入 clean1 身份。源码身份统一核验模型、运行、台账及比较；新入口严格拒绝旧身份 checkpoint 和宽松加载。

旧权重只在冻结源码的工程对照进程中映射，不提供正式训练初始化转换。历史恢复方法见[clean1 历史恢复说明](CTSEQTRACK_V38_CLEAN1.md#身份与历史恢复)：按旧 run manifest 选择 Git 提交恢复，完整快照也可用于临时工程对照，无需常驻第二套源码。不能把所有历史 checkpoint 都视为来自 14c3b2f。后续按用户要求已改为原 `CT-SeqTrack` 目录更新，clean1 只保留为执行修订，不要求第二套源码。

## 最终验证结果

| 验证 | 实际结果 |
|---|---|
| 最终源码完整活动 pytest | **323 passed、6 skipped、1 xfailed**，295.04 秒；`pytest_final_locked.txt` |
| seed42/52，B1/Full 初始化与构造结束 CPU RNG | 保留参数及 buffer 逐值一致；仅两投影共 8,320 参数被移除 |
| 非零输出头，eval 与连续三次 FP64 Adam | B1/Full 的输出、有效损失、梯度、BN、优化器状态均通过原容差 |
| 共同有效位置 dropout mask | FP64 前向、反向和一步更新对照通过；生产代码没有占位 mask |
| FP32 同权重对照 | 每步输出、损失和梯度通过原容差；不将重新对齐权重的检查写成累积 Adam 等价 |
| 新实现自然 dropout | 同 seed 短跑状态逐值复现 |
| 五步短轨迹 B1/Full | acquisition 标签、accepted、下一 crop、trusted、速度/时间和有效评测诊断一致 |
| 离散与梯度合同 | 全空/稀疏/重复/仅历史点、历史不足、无效速度、别名、平局、去重、T=0.5、支持点2/3、非相邻重获、GT 不入决策、B3 梯度隔离等活动测试通过 |
| 最终锁定源码 CPU Lightning 2.0.2 | **2 passed**：实际入口保存后独立评测、完整 epoch10→11 恢复；另 **2 passed**：checkpoint marker/保存排程；`pytest_lightning_final_locked.txt` |
| 配方与入口 | 三组配方/LR/身份、旧身份拒绝、工程时间和部分训练边界通过 |
| 保留 CLI / Shell / 编译 / diff | 帮助命令、两份 Shell 的 `bash -n`、compileall、`git diff --check` 全部通过；`command_checks_final.json` |
| 配对 benchmark 工具 | 最终 Full 旧/新源码 × 模型/完整事务四进程通过；初始化、输入、请求顺序和测量期间源码不变检查通过；`benchmark_final_locked/report.json` |

默认环境跳过项包括 Lightning 不可用和三项 CUDA 检查。上述隔离 Lightning 补跑覆盖了相应缺失项；CUDA 三项仍未运行。预期失败保留的是原分解投影 FP32/Feature/train/empty_row 梯度敏感案例，未放宽原容差，不属于本次紧凑路径的验收替代。默认测试另有一个第三方 dateutil 弃用警告。

锁定源码下的生命周期证据另存于新目录：`artifacts/ct_checks/v38_lightning_4ttqhsd2/{entry,standalone}` 和 `v38_lightning_7m34ilqv/{uninterrupted,resumed}`。前者完整跑一轮 12 行、3 次 Adam，独立评测结果及逐帧文件一致；后者连续和恢复运行都到 epoch11/global_step123，参数、BN buffer、Adam、scheduler、请求顺序及 LR 精确一致。启动、结束和 checkpoint 中的源码指纹均匹配最终登记。这是 **CPU 合成数据** 验证，不是 nuScenes CUDA 实验。

独立参考的实际一轮训练→checkpoint→独立评测另已通过，见 `data_checks/reference_complete_1h3ql9g7/verified.json`。其随后增加的整批提交前 shape/finite 校验由最终活动测试覆盖；不把此前工程 checkpoint 当作最终源码可恢复的正式权重。

## 数值差异的完整记录

用户明确接受：保持紧凑路径，用 FP64 连续 Adam、FP32 同权重输出/梯度、新实现自身恢复复现验收，并记录旧/新 FP32 累积训练轨迹差异。没有改 BN、Adam 或放宽容差。

最终对照使用前向 `rtol=1e-5, atol=1e-6`，梯度/更新 `rtol=1e-4, atol=1e-5`。主要实测最大绝对差如下；通过条件是绝对与相对项组合，不能只与 atol 比较：

| 检查 | 输出/损失 | 梯度 | 参数/BN |
|---|---:|---:|---:|
| FP64 三步 Adam，B1/Full | 1.38e-12 | 8.95e-10 | 3.96e-12 |
| FP64 共同 dropout mask，Full | 5.56e-16 | 4.40e-14 | 9.33e-13 |
| FP32 每步同权重，B1/Full | 1.20e-7 | 2.68e-5 | 不作为累积更新等价检查 |

原 FP32 独立累积 Adam 对照确实失败：初次更新的参数最大绝对差约 1.73e-4，三步后部分参数/BN 差异约 1.82e-3。空槽移除改变矩阵舍入路径，小梯度经 Adam 可放大；这不能被解释成 FP32 训练轨迹相同。原失败工件保留于 `equivalence_5dim1c5h/`，最终对照在 `equivalence_48qh_28q/`，汇总为 `numerical_results_final.json`。本轮也不承诺自然 dropout 的旧/新随机轨迹相同。

## 仍由服务器完成的验收

当前只确认结构、已测数值合同与 CPU 生命周期。尚未完成真实 nuScenes CUDA Full batch16 检查，以及旧/新 B1、Full 的同卡配对性能测量。工具已准备好，默认 20 步预热、100 步计时、三次交替顺序，分开报告模型和完整事务耗时及 allocated/reserved 峰值；完整事务不包含 Lightning 文件日志 IO。

CPU synthetic benchmark 只验证工具过程，报告明确 `performance_evidence=false`。无真实同卡实测前，不宣称加速、显存收益或正式训练效果保持。未上传服务器、启停任务、创建正式训练或将工程权重用于正式初始化；本轮改动尚未提交 Git。
