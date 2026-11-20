# V41 KITTI Full 配对训练测速

## 2026-11-20 B0 算子和输入校验组合验收

当前 v4 源码 `e7e0ee6b` 在 GPU 1 上消除了原版 `e0d3c397` 回填路径的 14 次流同步。主、辅网络各自只检查一次相同框尺寸和一次相同帧时间。四对连续训练窗口的主端点吞吐从 44.36 增至 48.23/s（+8.73%），峰值已分配显存减少 525.3 MiB。固定批梯度、BN 和成熟模型 32 步更新均通过对照，两套 E38 权重在完整 KITTI Car dev 上各 1,336 个预测框逐位相同。[组合测量口径及原始记录](../artifacts/ct_checks/20261120_v41_ac_integration/REPORT.md)

## 2026-11-20 主辅训练通路核对

源码 `e0d3c397` 修复了独立历史描述接口对旧式字节掩码的年龄校验。真实主辅训练、固定 E38 梯度与完整 dev 回放均与上版 v4 一致。对四个正常训练步的追踪进一步确认，每次 B0 的 7 次稠密回填在当前 A40 确定性路径上触发 14 次流同步。R 种子的邻居搜索也与部分最终候选重合。[完整通路、两版对照及逐算子记录](../artifacts/ct_checks/20261120_v41_flow_audit/REPORT.md)

## 2026-11-20 B0 三分支索引复核

源码 `08cb0bc4` 的 Seg、Mini、Feature 共用同一份有效点索引，Feature 按帧分组并单独取有效 token。相对上次源码 `477e87d6`，清理未使用的坐标、重复的帧存在性归约和历史时间校验的压紧操作后，真实训练每次 B0 前向的 `aten::nonzero` 由 3 次降至 2 次。共享 A40 上的 64 对固定批次测量中，B0 前向加反向的均值为 77.66 → 77.61 ms。CPU 和 CUDA 各 96 组对照以及完整 dev 逐帧核验通过。[本次复核、计时及两版源码](../artifacts/ct_checks/20261120_v41_b0_indices/REPORT.md)

## 2026-11-20 v4 二次复核

以上一轮已验收 v4 为基线，GPU 1 上每个课程做四对交换顺序的真实训练短窗，每窗预热 8 步、计时 96 步。首轮主端点吞吐从 **46.63 增至 50.41/s（+8.11%）**，E38 采样计划从 **40.67 增至 44.48/s（+9.38%）**。每窗包含 1,536 个主端点和自然插入的辅助工作。两轮的逐步损失、选框和接受框都与上一轮 v4 相同。此次改动包括共享 B0 有效点索引、缩短首行 IPC 填充尾部、B1 与完整诊断去重，以及批量有限梯度检查。[训练窗口和 CUDA profile](../artifacts/ct_checks/20261120_v41_v4_recheck/REPORT.md)

## 2026-11-20 exec v4 验收

新工作树中，`tools/benchmark_v41_training.py` 默认比较 exec v3 与 v4。`--reference-root` 应指向 `b819e76` 对应的独立 v3 工作树。可用 `--equivalence only` 在真实主辅训练中复查请求、共同随机初值、首行输入与首次轨迹差异。该模式报告 `trajectory_compared`，保留新执行布局引起的梯度和递推差异。历史 v2→v3 对照可在新版工具中显式传 `--revision-pair v2-v3`，并分别指定两份旧源码目录。

```bash
python tools/benchmark_v41_training.py --reference-root ../CT-SeqTrack-v41-v3 \
  --source-root . --path /home/lishengjie/data/cxtrack/training \
  --devices 1 --workers 4 --repeats 2 --equivalence after
```

GPU 1 双顺序复测，E1 主端点吞吐由 21.10 增至 44.35/s，E38 采样计划由 21.13 增至 40.46/s。独立的同卡首轮原生配方对照中，SeqTrack 与 v4 分别为 82.25 和 45.89/s。固定 E38 权重回放完整 dev，v3/v4 的 S/P 分别为 67.1898/85.1994 与 67.1123/85.1699，1,336 个预测中有 40 个改变候选。[分层实测和固定权重复核](../artifacts/ct_checks/20261120_v41_v4_acceptance/REPORT.md)

## 11 月 18 日 v3 正式训练窗口

KITTI Car 两组 q0 对照在同一张物理 A40 GPU 0 并发，源码 `b819e76`，各为 batch16、每个 loader 4 个 worker。比较首轮第 500 至 1,200 主步的日志，`uniform` 和 `inverse_lwh` 分别处理 **21.76 和 21.98 个主端点/秒**。历史 v1 对应区间为 5.06，历史 SeqTrack3D KITTI 适配基线对应区间为 69.57。历史与当前运行处于不同时段，设备负载、执行配方和辅助工作量影响吞吐。这个窗口在首轮训练内，整轮还需计入 dev 与保存，成熟课程的工作量也会改变。[正式运行复盘与计时边界](../artifacts/ct_checks/20261118_v41_live_speed_1814/REPORT.md) · [运行清单及原始时序](../artifacts/ct_checks/20261118_v41_live_speed_1814/measurements.json)

## 配对测速方法

配对工具可在一张独占 GPU 上依次运行旧、新两个源码目录。每版从相同随机初值和 RNG 状态出发，运行自己的 Lightning 入口、真实 KITTI loader、原始主辅插入计划、反向、Adam、闭环状态提交和步级日志。默认每版一次，预热 20 个主步，统计接下来的 200 个主步。两个窗口都处于第一个 epoch 内。

## 2026-11-18 已测结果

### GPU 1、每个 DataLoader 一个 worker 的复测

服务器为共享 A40，PyTorch 2.0.1+cu118、Lightning 2.0.2、FA2 2.5.8。CT 用默认 `41_full_kitti.yaml`、`uniform` 和源码 `9ffc95b1`。所有训练进程在独立目录 `~/study/lcyu/v41-worker1-gsdi9XNl` 顺序运行。计时使用真实 KITTI、Lightning、反向、Adam 和在线提交，窗口两端同步 CUDA。两次重复分别先跑两侧的其中一侧，再交换顺序。主、辅各自的数据加载器均核实有一个存活的 worker。

| 对照与工作量 | 第一对主端点/秒 | 交换顺序后主端点/秒 | 两遍累计端点数除以累计墙钟 |
|---|---:|---:|---:|
| CT 改前 / 当前，第十轮采样计划、随机初始化、B8，每遍 4 步预热 + 20 步计时 | 17.47 / 17.09 | 16.01 / 16.96 | 16.71 / 17.03 |
| 当前 CT / 冻结 SeqTrack，首轮原生配方、B16，每遍 8 步预热 + 24 步计时 | 17.38 / 16.56 | 22.12 / 32.90 | 19.47 / 22.03 |

成熟计划的每次计时包括 160 个主端点、8 个自然辅助端点，其中 110 个主端点延续已提交的窗口。相同请求下，主路点云视图调用由 687 降到 468，两遍的主路 CPU prepare 总时间下降 10.1%。每遍 21 次主辅候选监督均命中 CPU GT 引用。CUDA 峰值已分配显存从 3812.70 降到 3549.19 MiB。整步吞吐合计增加 1.9%，单对为 −2.2% 和 +6.0%。该辅助批处于轨迹起点。[成熟课程逐步对照](../artifacts/ct_checks/20261118_v41_worker1/mature/comparison.json)

两版的 24 个更新、25 个主辅批逐步比对了采样点 ID、候选选择、提交框、exact IoU 和损失。最后一步的 308 个参数、308 个梯度、924 个 Adam 状态和 78 个 BN 缓冲逐元素相同。另用同一张卡、B16 的合成重叠邻域，q0 gather 反向阶段的新增峰值分配从 46.00 降到 15.00 MiB。前向逐元素一致，梯度最大绝对差 9.54e−7，两种实现各自的重复运行梯度逐元素相同。[q0 CUDA 独立复核](../artifacts/ct_checks/20261118_v41_worker1/corner_gpu.json)

首轮跨项目对照每侧合计 768 个主端点。CT 在同一计时内另处理 64 个辅助端点，总墙钟为 39.45 秒，SeqTrack 为 34.86 秒。相同主端点数下 CT 多用 13.2% 时间。两组的快慢方向相反，SeqTrack 两次重跑的吞吐相差近一倍。共享卡监测记录到同期其他进程活动。这些数据覆盖指定训练窗口，未计入验证或存档。[跨项目数据](../artifacts/ct_checks/20261118_v41_worker1/pairs/b16/summary.json) · [共享负载](../artifacts/ct_checks/20261118_v41_worker1/load_summary.json)

### 三项清理后的 V41 Full

本轮在同一训练目标、骨干和数值设置下比较 V41 v3 源码 `00b109e4` 与 `8db18bed`。训练数据为 KITTI Car，服务器环境为 A40、PyTorch 2.0.1+cu118、Lightning 2.0.2、FA2。测试使用独立目录 `~/study/lcyu/ct-opt-qeSYlmtn`，物理 GPU 1，batch 2，workers 0。每个进程的 PyTorch 显存上限为整卡的 15%。

| 本轮改动 | 可直接量到的变化 | 耦合检查 |
|---|---|---|
| 同窗历史 ROI | 每次提交最多保留两份下一步仍需使用的未抽样点池。两轮合成 B16 回放中，成熟主窗裁剪调用均由 512 降到 288，辅助长轨由 592 降到 320，单步窗均为 256。 | 复用所需的物理帧、裁剪框和视图一致。重新按端点 seed 采样、换锚和生成标签，提交失败不发布新池。 |
| 四候选监督 | 每端点只准备一次公共 GT 几何，GT 和尺寸保留 CPU 引用。动态预测回读一次，监督标签合并传回一次。B16 四候选各异的主、辅监督 CPU 中位数分别由 25.44 降至 16.67 ms、27.15 降至 17.01 ms。 | exact 与 compat 分别计算高度，exact 保留双精度，q0 回退遵循原尺寸下限。监督专用的 GT 引用对在线选择不可见。 |
| B0 掩码与 q0 取点 | 正常 B0 前向省下 18 次重复清零。B16、1024 点、64 通道、八角各取 16 个邻居时，CPU profiler 记录的特征 gather 反向分配由 32 降至 4 MiB。 | 每个子网入口和全局特征拼接后仍清零，masked BN 统计保留。CUDA 回归覆盖不同角选择同一原始点时的梯度累加。 |

真实多步回放从相同随机初始化出发，在两条轨迹上执行第十轮课程的 8 帧长窗。前两步预热，后四个主步计时，包含取数、前后向、Adam 和状态提交。改前、改后的主端点/秒在先旧后新的一组为 7.01、6.45，反序一组为 6.80、6.90，合并为 **6.90、6.67**。四次启动前共享卡利用率为 70%–88%。两组测量中的峰值已分配显存均由 **942.31 降至 880.31 MiB**。六步的采样点 ID、损失、exact IoU、选择和提交框逐项相同。计时后另插入两次主辅联合更新，308 个参数的梯度、78 个 BN 缓冲的逐项哈希、两路选择与提交框均一致。此窗口使用随机初始化，无 Lightning 日志、验证、保存或自然辅助调度。[交叉计时和回放对照](../artifacts/ct_checks/20261118_v41_mature_speed/paired_result.json) · [历史池 CPU A/B](../artifacts/ct_checks/v41_history_pool_20261118/cpu_ab.json) · [几何 CPU A/B](../artifacts/ct_checks/20261118_v41_candidate_geometry/results.json)

另走两项目真实 Lightning 训练入口，使用各自采样器，以首轮 1 步预热、4 步计时交叉比较。CT 先跑一组为 **4.84 对 8.27** 主端点/秒，SeqTrack 先跑一组为 **5.20 对 12.25**。合并为 **CT 5.02，SeqTrack 9.87**。两次 CT 窗口都未发生辅助插入。该短测覆盖两个项目实际执行的不同训练配方。正式 batch 16、完整课程和整轮用时仍待独立测量。[首次配对](../artifacts/ct_checks/20261118_v41_mature_speed/comparison-1.json) · [反序配对](../artifacts/ct_checks/20261118_v41_mature_speed/comparison-2.json)

GPU 数字对应归档源码 `8db18bed`。随后工作区加入独立的 q0 中心损失加权配置。在源码 `16de1830` 上复测默认 `uniform` 配方，三个 Adam 步、两次辅助插入所比较的 5,226 项 CPU 张量逐元素相同。新增 `inverse_lwh` 配方的训练速度与精度需单独测量。[默认配方闭环对照](../artifacts/ct_checks/20261118_v41_mature_speed/cpu_concurrent_default.json)

### 本轮清理前的共享 A40 对照

在服务器物理 GPU 1 依次运行两个项目的训练入口。两侧的 KITTI Car 训练场景、输入标注哈希和随机种子相同，各自使用原采样器及目标函数。本次工程配置为 batch 2、workers 0、单 CPU 线程、每进程 PyTorch 显存上限 15%。每次从头启动，预热 1 步、计时 4 步，两端同步 CUDA。窗口包含真实取数、前向、反向、Adam 和 CT 状态提交。四次窗口均在第一轮，CT 的窗口未遇自然辅助插入。服务器独立目录为 `~/study/lcyu/ct-speed-check-naDW7GC9`。

| 运行顺序 | CT 主端点/秒 | SeqTrack 主端点/秒 | CT / SeqTrack 吞吐 |
|---|---:|---:|---:|
| SeqTrack 先、CT 后 | 6.37 | 4.50 | 1.42 |
| CT 先、SeqTrack 后 | 4.50 | 12.03 | 0.37 |
| 合并各自的端点数与墙钟秒数 | 5.27 | 6.55 | 0.81 |

每个系统累计计时 8 个 Adam 步，各处理 16 个主样本（第二次重复相同种子）。CT 与 SeqTrack 的每轮采样顺序不同。合并时，同等样本量 CT 耗时约为 SeqTrack 的 1.24 倍。顺序交换后快慢反转，共享设备、CPU 调度、文件缓存和不同取样均限制归因。两次 CT 的 allocator 峰值分配约 961 MiB，SeqTrack 约 599 MiB。这组结果覆盖当前 CUDA 主路，正式 batch 16、自然主辅混合、多步课程和整轮墙钟仍待测。[第一对原始数据](../artifacts/ct_checks/20261118_v41_seqtrack_short/comparison.json) · [反序数据](../artifacts/ct_checks/20261118_v41_seqtrack_short/comparison-reverse.json) · [测速脚本](../artifacts/ct_checks/20261118_v41_seqtrack_short/short_window.py)

### 数据准备与数值回归

另用四条各四帧的 TinySource 轨迹，在独立导出的 v2 和当前 v3 上运行实际 Lightning 2.0.2 `Trainer.fit`、Adam 与生产一致性探针。两侧均完成 8 个主步，在索引 0、3、7 对照；后两处是正常排程的辅助插入，索引 7 实际读到前次已提交的辅助状态。1,261 项浮点张量通过设定容差，157 项离散张量精确相等。该检查使用本机 CPU math 注意力。[真实 Lightning hook 复核](../artifacts/ct_checks/20261118_v41_local_lightning_probe/report.json)

KITTI Car 首个普通主批有 16 个端点。两侧使用相同请求和 72 个预热的 bin，36 个共有输入及监督张量的 dtype、形状和内容 SHA256 全部一致。单进程、单线程回放的 worker 消息从 206.51 MB 减至 21.35 MB，GT10 命中时其中的整云数组从 175.31 MB 减至 0。主进程 prepare 从 577.41 ms 减至 93.38 ms。该批没有原云回退，这次没有计量实际多进程队列等待、GPU 或整轮墙钟。[服务器逐项记录](../artifacts/ct_checks/20261118_v41_cpu_firstbatch_1423/summary.json)

本机用合成轨迹和 math 注意力进行 v2/v3 三个 Adam 步的 CPU 对照，其中两个步含辅助批。补修主机 prior 调用、点池化与 M 时间检查后重新执行，公共初值和修正后的旧目标对齐，候选、梯度、BN、Adam 与提交状态等 5,228 项张量逐元素相等。最终 S/P 仍待验证。[本轮模型回放记录](../artifacts/ct_checks/v41_speed_review_20261118_145342/equivalence.json)

补修后，本机使用 Lightning 2.0.2 运行 V41、物化缓存与点云缓存相关 CPU 测试，167 项通过。上轮实现快照的全库结果为 753 项通过、9 项跳过、1 项预期失败。

## 18 项问题的复核

| 原编号 | 当前默认训练路径 | 仍影响解释的部分 |
|---|---|---|
| 1、2、3、18 | 普通训练的 worker 交付 GT10 小视图与原云定位信息。物理帧缓存提供只读数组，标注视图区分目标，worker 跨轮保留。主进程仅在搜索越出视图时补取整云。 | 首次生成 GT10 仍要读、扫原云，整轨辅助与评估使用 raw。前一帧 accepted 决定下一帧 ROI，跨步预制预测裁剪会改变轨迹。缓存命中与回退率需按数据流分别统计。 |
| 4、5 | 旧 acquisition 头、标签及 0.05 损失已移除。首帧只需初始化时可取元数据，主辅与评估按消费者生成监督。 | B1 少了一项训练梯度。当前点的前景支持仍决定 R、可信写入与下一步搜索，删除它会改变模型。 |
| 6、13 | 复用候选平面交、可共享的 q0 几何、最终支持集合与当前点的 raw ID 顺序。R 的种子读取与修正后最终框读取各用自己的框。 | 稳定距离排序仍在。替换成 top-k 时需覆盖等距、重复 ID 和邻域边界，验证实际选择点。 |
| 7、8 | B0 的池化省去整块特征的一次 gather，masked BN 使用同批有效点与 token 计数，减少回读。 | 固定槽上的宽 Conv/Linear 和手写掩码 BN 仍是 GPU 候选瓶颈。历史有效点槽 42.15%，不能由填充比例直接推算 GPU 速度。 |
| 9、10、11、14 | 保留严格 FP32、确定性与原 Adam，以便对齐训练。合并部分数值检查，合并 prior 与观测的公共输入准备；FA2 保留 varlen 封装。 | Adam 逐张量更新、梯度有限性验证、FA2 逐层打包及精度转换仍执行。需要时间线区分 GPU 忙、CPU 提交不足和同步等待。 |
| 12、16、17 | 默认 summary 训练，逐帧详细候选审计按配置启用。评估 GT 框与前景标签留在 CPU。CSV 采用追加写，默认每 50 步记录。 | 仍有逐步分项损失的 Lightning 累计、验证 S/P、简表与存档。历史 v1 第 39–60 轮的验证均值为 3.33 分钟，占 fit 约 1.39%。 |
| 15 | 整轨辅助的插入时机、B3/R 反传、状态提交均保留。上游在隔离的模式与 RNG 下运行。 | 历史平均多 9.66% 的 forward 批次。主指标为每个墙钟秒提交的主端点数，分母包含这些辅助工作。 |

历史 v1 在第 39–60 轮的平均 fit 为 239.64 分钟，适配版 SeqTrack 进度条为 13.43 分钟。两者计时边界和资源条件不同。v2/v3 的 20+200 步用于测工程改动。判断与 SeqTrack 的差距还应比较同卡、同 KITTI split 和 batch、各自完整训练配方下的主端点/秒及每轮墙钟，分别列出训练、验证与保存。首轮的主窗口均为单步，比较持续跟踪阶段还需覆盖课程转入多步之后的轮次。[历史计时与分母](../artifacts/ct_checks/20261118_v41_speed_reaudit/timing.json)

相同历史框在相邻端点的重复裁剪是首轮短窗看不到的开销。按正式主采样器，在第 10 和 60 轮，223,696 个有效历史槽中有 87,783 次再次请求同一窗口中同一帧的历史裁剪，第 1 轮此项为 0。可复用的是该历史框的有序未抽样点池，仍须按端点重新采样、换锚和编码。该计数对应 39.2% 的历史裁剪请求。帧、视图和 ROI 一致时，新版跳过这部分几何扫描。[历史裁剪复算](../artifacts/ct_checks/v41_train_path_counts_20261118_153019/history_crops.json)

默认 summary 训练仍按步生成标量指标并做梯度检查。本地一次 CPU math 合成主批观察到 36 次日志接口调用，Adam 写入前须检查 308 个有梯度的参数张量，共 3,890,507 个梯度元素。日志次数与 CUDA 同步次数属于不同指标。[本地调用计数](../artifacts/ct_checks/v41_train_path_counts_20261118_153019/counts.json)

## 服务器命令

2026-11-18 14:43 只读核对时，服务器常用项目目录仍是 `5c92b1c`、执行版本 v1，且该 Git 对象库没有 `25345a5`。执行下面的命令需要已部署到独立目录的 v3 源码和 v2 对照。可在拥有该提交的工作区用 `git archive -o v41-v2.tar.gz 25345a5 main.py requirement.txt cfgs datasets models utils` 导出对照，解压到独立目录后传给 `--reference-root`。工具分别核对原版 v2 和候选 v3 的执行标识，登记实际源码 SHA，支持无 Git 元数据的导出目录。

若运行环境已有含两个版本及新工具的仓库，可用以下命令建立旧版工作树。把数据路径和物理卡号改成实际值。

```bash
git worktree add --detach ../CT-SeqTrack-v41-v2 25345a5
python tools/benchmark_v41_training.py \
  --reference-root ../CT-SeqTrack-v41-v2 --source-root . \
  --path /path/to/KITTI --devices 0 --trace-mode torch --equivalence after
```

同一工具在每个独立子进程优先导入指定 checkout 的模型、配置和数据代码，并以该 checkout 为当前工作目录。旧树无需复制脚本或打补丁。采用同一 Python 环境、数据盘及已有的线程/allocator 设置。默认使用各自的 `cfgs/ct_seqtrack/41_full_kitti.yaml`、相同 workers、summary 诊断。实际路径、git HEAD 和源码内容 SHA 均写入结果。

两版的工程入口都要求运行日志位于各自源码树的 `artifacts/ct_checks`。独立参考目录需要允许创建该目录及一次性日志子目录。工具先在该树内运行，再把 manifest、解析后的配置和日志原样归档到配对结果的 `reference-01/run/`，随后清理该次临时日志目录。manifest 命令行和解析后的配置保留了原始运行路径。候选使用另一独立源码目录时按同样方式归档。

服务器已安装 Python 3.9.19、Torch 2.0.1+cu118、Lightning 2.0.2 和 FA2 2.5.8。核对其 Lightning 源码可见，训练循环依次取批、自动优化、调用模型的提交 hook、更新步级指标并处理 logger 保存，然后工具结算该步。共享 A40 的小批训练已跑通 V41 v3 的正式 FA2 配置。

省略 `--trace-mode torch` 只跑 20+200。带上后，同一进程在 200 步统计完成后等待下一次真实辅助插入，默认追踪 4 步，可以使用 `--trace-steps` 调整。使用 Nsight Systems 时可选 `--trace-mode nvtx` 并让采集器跟踪子进程；该模式提供 NVTX 范围和主进程流量计数，GPU 时间轴由外部采集器生成。偶发抖动需要估计时可显式使用 `--repeats 3`，每一对交替运行顺序。一对结果只能给出这次测量的差值。

## 可比性

工程子进程只在窗口边界调用 CUDA 同步，吞吐窗口中每步记录主机时钟。两边实际提交的主、辅助端点数、步数、aux 插入索引、采样计划和训练源元数据都进行配对校验。两套计划摘要包含端点的轨迹、窗口、分支、帧、drain 状态与 batch 边界。完整 sampler 仍决定辅助预算，工具不截短 sampler 或强插辅助批。

原版完成随机初始化后，工具临时保存 tracker 参数、缓冲及 Python/NumPy/Torch CPU/CUDA RNG。两版在创建优化器前对齐公共张量和构造后的 RNG。共有键检查模块类型、参数或缓冲身份、可训练性、形状、数据类型，容许旧 `prior.acquisition_head.*` 在新版退休。输出记录公共张量内容 SHA 和共同初值包 SHA。每版继续训练自己的目标函数，优化产生的参数和接受状态可以分化。工具结束时回收跨进程传递的临时随机初值，不产生训练 checkpoint。入口默认的逐轮校验、评估和 checkpoint 协议仍用于整轮正式运行。

## 真实数据一致性

`--equivalence after` 在两侧吞吐都已写盘后，以单独短程工程进程从同一初值重建两版。只想检查数值时，使用相同命令并将该选项改为 `--equivalence only`，会直接跳过 20+200。短程检查一直用 `旧主损失 − 0.05 × loss_acquisition` 反传并执行真正的 Adam，辅助损失与插入时机沿用原配方。测速进程仍各自训练其原始目标。

自动比对第一主步和前两次自然插入辅助批的主步。首轮普通主窗长度为 1，第二个辅助批才提供真正消费上次已提交状态的机会。工具核对每个辅助请求的轨迹键、下一帧索引、实际历史输入和提交状态，记录 `continued_auxiliary_endpoints`；短数据若排不到两次辅助插入或未遇到存活前驱，输出 `not_verified`。对照项还包含主/辅各分项和实际组合损失、Adam 更新前的公共参数梯度与连接性、BN 统计与更新计数、采样点及 raw ID、四候选与选择、R 点集和可信写入门。浮点比较使用 `rtol=1e-4, atol=1e-5`，离散张量、掩码和决定要求完全相同。首次不一致会报步号和字段路径，并保留已生成的吞吐结果。通过时 `equivalence.json` 记录步号和比较张量数；大体积参考梯度只存在于这次比较的临时目录。

## 读结果

新目录默认在 `artifacts/ct_checks/v41-perf-*/`。根目录的 `comparison.json` 给出新版主端点/秒除以旧版主端点/秒，每一侧的 `measurement.json` 及 `run/run_manifest.json` 保留原始数字、实际模型和执行配方。

| 指标 | 分母与范围 |
|---|---|
| `main_endpoints_per_second` | 200 步真实提交的主端点数 / 完整墙钟秒数，分母含正常辅助开销。主比较采用这个指标。 |
| `total_endpoints_per_second`、`steps_per_second` | 主辅端点总数或 Adam 更新步数 / 同一墙钟秒数。 |
| `host_step_interval_p50_ms/p95_ms`、`final_device_drain_ms` | 相邻主机步完成时刻的间隔、窗口末等待 GPU 的时长。每步含排队、实际等待和日志，单步 GPU 延迟请读取 trace。 |
| `host_step_intervals_by_role` | 分别报告仅主流和含辅助插入的主步数、主机间隔 p50/p95。分组没有样本时分位数为空。 |
| `main_process_cpu_seconds`、`child_cpu_seconds` | 主进程 CPU 和 Linux 下两端均在场的同一组子进程 CPU；并行 CPU 秒数可以超过墙钟秒数。子进程不可观测或集合变动时记为空。 |
| `peak_gpu_allocated_bytes/reserved_bytes` | 吞吐窗口起点重置峰值后的 PyTorch allocator 高水位。 |
| `attribution_window.trace_diagnostics.stages` | 短 trace 中真实 loader 收取、准备、H2D、网络、loss、反向、Adam 更新和 commit 的 CPU 含子调用时间、归属 CUDA 时间和调用数。范围可嵌套，时间不可相加。打开同目录 `trace.json` 可看 memcpy、同步和 CPU/GPU 空档。 |
| `device_events` | torch profiler 实际采到的 GPU 事件数、按时间轴合并重叠事件后的忙碌毫秒，以及识别到的 H2D/D2H 拷贝次数。各阶段发生的异步活动可跨域重叠。缺少 Kineto 设备时间线时记为空。 |
| `parent_loader_traffic` | 分主辅角色统计短 trace 实际交付的批数、整云/GT10 NumPy 数组逻辑字节（同一个批内按对象去重）和 worker 结果队列实际收到的序列化载荷字节。队列读包括预取乱序返回，对象统计按交给消费者的批次；二者可跨窗口边界。进程外共享张量区、管道头与存储设备读流量另计。单进程 loader 没有 worker 队列字节字段。 |
| `parent_view_calls` | 主进程实际请求 GT10 时命中或回退 raw 的次数。直接选择 raw 的辅助输入不会记作 GT10 回退。新版父进程惰性整云获取、实际文件读取在适用时也有独立范围。worker 文件读取与缓存内部阶段未单独采样。 |

前缀吞吐反映第一轮这一段实际工作量。首轮建表、跨 epoch worker 复用、完整验证和落盘耗时应读取完整 epoch 的对应记录。将两版联合改动的整体差值归因到单项时，使用调用次数、字节和时间线定位；该对结果不能直接分解出独立贡献。

## 生命周期 smoke

```bash
python tools/check_v41_server.py --path /path/to/KITTI --devices 0 --steps 2 --diagnostics-level summary
python tools/check_v41_server.py --path /path/to/KITTI --devices 0 --steps 2 --diagnostics-level full
```

smoke 强制每个主批都有辅助批，检查两份状态提交、FA2、梯度与 RNG 隔离。summary 模式遵循训练路径跳过逐轨辅助诊断，full 模式另验其记录。smoke 中的同步和梯度核查耗时只用于生命周期排错。
