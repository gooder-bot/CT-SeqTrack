# CT-SeqTrack 当前工作提醒（2026-09-17）

## 活动版本更新：v31（2026-09-19）

用户最新授权：服务器 `lishengjie@10.109.253.86` **仅只读**；可修改本地，不得自行上传/安装/启动训练/停止进程。项目路径 `/home/lishengjie/study/lcyu/CT-SeqTrack`。
`main.py` 根据31配置分流到 `models/ct_v31/entry.py`，独立host为 `models/ctseqtrackv31.py`。本次三臂是31_b0_mini、31_full_cfc_mini、31_full_gru_mini；物理GPU分别1/0/0，单进程一张卡。
用户已批准同帧联合训练，跨臂B0不再要求逐位相同；sigma/离散几何/GT/跨帧detach保留。Full直接四假设质量选择，不调用旧校准。v31需继承自己的base，不混用旧formal字段、工具或checkpoint。
正式预算60轮/batch16/workers4/seed42/FP32，Adam(.5,.999)/eps1e-6/lr1e-4，StepLR20轮×.1，每5轮验证，final60及late3=58–60自动评测；不使用旧preloading。
当前只有本地测试与服务器环境只读核验，真实GPU训练/显存/涨分仍未验证。详见 `docs/CTSEQTRACK_V31_MINI_LAUNCH.md`。下面旧v30状态不能覆盖本节；历史output和旧版本复现入口保留。

## 最新：四组同分与B0根因已经实际前向定位

先读[根因报告](artifacts/ct_checks/reports/20260917_v30_root_cause/REPORT.md)。四组独立进程，但确定性B0初始化/观测/更新隔离相同，未校准Full提交B0所以同分；当前为单Adam不相交参数组，不是四个手动optimizer。
只读服务器提取实际v30/v28 track14/15首预测输入，本地实际epoch60权重前向：v30轨迹14真实XY静止，raw X=5.619m、p=.989、coarse X=5.557m、final误差6.293m；仅置零raw后0.0587m。不是decoder独立造成主要跳跃，也不是该帧低p补偿；soft门退化另有真实loss反例。
v30有效槽[1024,0,0,0]；v28同物理1点按≤2规则清零成全空，触发保持anchor。仅给v30恢复旧全空条件仍救不了该单点状态；应联合看真实点保留与弱观测motion约束。只保留1有效槽与1024重复槽eval结果一致，不能说重复点直接放大位移。两轨迹54帧贡献v28到v30净Success下降31.11%，仅为结果分解。
本轮仅新增诊断/文档，未改模型配置、未启动服务器评测训练。优先新版本一致的全行物理motion与分类门职责，再单独研究roll-in/BC、residual；现有Full先补校准后评测。下方首次稀疏点推导结论由本节真实两版本前向覆盖。

## 最新：v30四组已训练完成，先读实际结果

见[9月17日四组结果](artifacts/ct_checks/reports/20260917_v30_mini_four_arm/REPORT.md)。
B0 false/true、Full-CfC/GRU均完成60轮/75720次B0更新；12次验证同为共享B0，final S/P=40.473742/47.840262。
Full每个插件18000次更新，但未做策略拟合，验证动作0；不能把该同分视为模块无效或用来选CfC/GRU。
四组B0最终参数/BN/Adam及观测loss一致，真实CUDA重算等价已验证；主配置仍false，true约省1.84GiB张量峰值。
58/59已保存未评测，late-3仍缺；下一步先补各Full checkpoint独立策略拟合与官方mini_val评测。
本地output中的195151 GRU目录误拷了旧v29 full内容，真正GRU已获用户授权只读从服务器取回，位于报告目录server_gru/；不覆盖原output。
服务器本轮仅只读文件/元数据/4幅LiDAR，无写入或任务操作。轨迹14/15首预测实测历史crop各1点、当前0点，推导槽mask=[1024,0,0,0]，并非全窗口无效。
`all_empty_frames`统计的是当前整云空，不能拿它证明B0四帧窗口有效性。重点诊断单点弱观测跳跃、coarse物理位移与anchor纠偏、roll-in状态。
本轮未改模型或配置，未声称分数恢复。下方9月15日“未实跑”等为历史状态，以本节覆盖。

## 当前v30与最新mini启动安排

后续用户预算更新：可接受比旧B0增加约5GB，优先涨分与迭代速度。现在
`ct_b0_masked_bn_recompute:false`为三臂共用默认（直接计算）；true是可选低显存重算，
不应再将下方“现在部分有效BN重算”作为默认。保留掩码/副本优化、bool排序修复和native分配器。
依据见`docs/CTSEQTRACK_V30_MEMORY_TRADEOFF.md`；本轮102 passed/2 skipped，未连接服务器。

9月15日晚实际报错更新：Full第4步CUDA bool argsort已改int64；B0第50步实际峰值7059MiB、
reserved34716MiB，旧展开masked BN与重复mask增加激活。现在部分有效BN局部重算原公式，
running仅前向更新一次；新命令用`PYTORCH_CUDA_ALLOC_CONF=backend:native`解除旧64MiB分割限制。
见`docs/CTSEQTRACK_V30_CUDA_MEMORY_FIX.md`。真实首次运行失败不能被此前CPU通过覆盖。
此次服务器只读日志/TB/进程，未同步或重启；B0日志末尾是SIGTERM而非OOM。

当前以用户v30方案、`docs/EXPERIMENT_PROTOCOL.md`顶部及`need_to_do.md`为准。
本地实现已完成，完整回归868 passed/15 skipped；真实CUDA、正式训练与分数仍待实测。
最新用户要求轻量核对后给四任务独立后台命令，不增加全套哈希/长检查/工程报告门：
GPU0 Full-GRU、GPU2 Full-CfC，GPU1并行B0有效BN反向重算false/true；mini Car、seed42、scratch60、batch16、workers4、val5。
Full均保持`ct_b0_masked_bn_recompute:false`，主三臂比较用B0 false，true为额外执行方式对照。
B0使用`30_b0_mini_bn_recompute_false.yaml`与`30_b0_mini_bn_recompute_true.yaml`；
Full仍用`30_full_gru_mini.yaml`、`30_full_cfc_mini.yaml`；不加`--preloading`，
默认每worker256MiB缓存。`CUDA_VISIBLE_DEVICES`选择物理卡，每个进程`trainer_devices=1`。
输出显式`output/YYYYMMDD-HHMMSS-30_模块-mini_car_seed42_60ep_bs16/`，包含`train.log`和`train.pid`。
命令见`docs/CTSEQTRACK_V30_MINI_BN_AB_LAUNCH.md`。两份B0仅名称/重算开关不同，配置合同52 passed。
仅索取命令不等于授权自动修改或启动服务器任务。
至少一个Full final60 S/P同时高于同版本B0后进入full/KITTI，late-3只报告，无额外seed前置门。
下方v29及更早“当前”“最新”均为历史记录，不覆盖本节与用户最新指令。

## 历史v29记录

2026-09-11：用户要求加速v29 B0/Full-CfC/Full-GRU，允许昂贵纯诊断抽样，但训练输入、
损失/更新、BN/RNG、证据与递归耦合不变。仅新`*_nuscenes_full_perf.yaml`启用性能路径；
旧配置保留。短测速/逐位对照及新日期output命令见[性能记录](docs/CTSEQTRACK_V29_PERFORMANCE.md)。
不运行全套哈希门禁；本地CPU通过不能写成服务器CUDA或加速比已验证。
日志不能整体减频：`self.log(on_epoch=True)`和训练统计每步更新，仅直接TB序列化抽样，
尾批在`on_train_epoch_end`统一flush去重。H3是v29 H1-only目标下的诊断抽样，明确披露覆盖变化，
记录scheduled/executed/valid/not_sampled；稳定摘要包含事件和版本。保留NLL2D、int64 AP cumsum、
固定槽max及strict deterministic。普通100步不代表H3已覆盖，另有真实合法事件微基准，
测速采用同卡ABBA及三卡并行；不启用persistent_workers。

## 最新：v29已获实施授权

最新启动偏好：仅轻量核对，不重做全套哈希/长测试，不强制`--assert-passed`后才提供命令。
最新重启要求：停止当前v29三臂，统一改为workers4，从新日期output目录scratch重启。
先同步`utils/online_contract.py`和`cfgs/ct_seqtrack/29_formal_base.yaml`；v29允许workers4或12，
4为新默认，不能沿用旧workers12硬校验。保留旧日志/checkpoint，不跨workers续训。
三臂在物理GPU1/2/3分别后台运行，完整nuScenes、每两轮保存checkpoint，额外保留059；验证每5轮。
这不代表服务器CUDA/数值恢复工程检查已完成。命令使用最新v29代码，不加preloading。

当前以[正式协议顶部](docs/EXPERIMENT_PROTOCOL.md)的v29三臂为准：B0、Full-CfC、Full-GRU，
完整nuScenes Car/seed42/scratch60。旧文中的“单组Full诊断”“mini后才可运行full”排程已替代。
实施/限制见[合同](docs/CTSEQTRACK_V29_IMPLEMENTATION.md)、[审计](docs/CTSEQTRACK_V29_CHANGE_AUDIT.md)，
本地与未完成CUDA验收见[验证](docs/CTSEQTRACK_V29_LOCAL_VALIDATION.md)，命令见
[服务器运行](docs/CTSEQTRACK_V29_SERVER_RUNS.md)。允许额外no-grad短递归，但不增加Adam更新或冻结模块。
新29配置继承v28数值修复，不继承原baseline逐位等价声明；必须按ct_enable_v29先分派语义身份。
不要读TrajTrack、改冻结参考、覆写历史output；不根据本地CPU通过宣称CUDA或分数恢复。

以下保留v28历史证据与报错提醒，运行安排以上述v29为准。

本文件补充上级目录 AGENTS.md 中的历史 v24 说明。当前以用户最新要求、
[正式协议](docs/EXPERIMENT_PROTOCOL.md) 顶部和 [当前状态](need_to_do.md) 为准。
开始处理 v28 训练、报错或重启命令前，先读
[CUDA 故障记录](docs/CTSEQTRACK_V28_CUDA_TROUBLESHOOTING.md) 和
[服务器命令](docs/CTSEQTRACK_V28_SERVER_RUNS.md)。

## 最新实验结论（先读，勿再沿用“尚未训练”状态）

三组已完成60轮/75720观测更新，结果已同步本地。先读
[9月10日结果分析](artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)：
B0/42 final S/P=45.200/47.243，B0/52=52.876/64.478；前者未达恢复目标。
Full/42与B0/42观测loss、末轮权重/BN/Adam一致，但B3未校准、动作0，不能解释为
校准后Full无增益。插件已训练、无冻结；B2新增目标供给仅4/22715，仍需获取诊断。
58/59 checkpoint存在但未评，不能用50/55/60替late-3。当前允许补评/校准/定位，
不凭seed52较高分宣布稳定恢复或直接扩完整数据矩阵。
`dev_diagnostics`只是遗留名称，本轮实际官方mini_val两场景/106轨迹/2285帧；
`scene_id=unknown`使summary记scenes=1，不能将其解释为只评1场景。

## 已确认的运行安排

- 最新指令是单组Full/Car/seed42完整nuScenes、scratch60轮后集中分析。见
  [完整数据诊断](docs/CTSEQTRACK_V28_FULL_DIAGNOSTIC.md)。不能再以旧mini得分门槛
  拒绝这次诊断或要求重新批准；仍不能宣称基线已恢复、耦合最优或全部矩阵就绪。
  用`28_full_nuscenes_full.yaml`，batch16/workers12/val间隔5；首次不加`--preloading`，
  当前全量点云缓存按轨迹重复、观测/机制/val各持对象，full可能大量占RAM。
  训练结束逐58/59/60独立内部校准和官方val评测，使用run的`resolved_config.yaml`。
- GPU1：`28_b0.yaml` / seed42；GPU2：`28_b0_seed52.yaml` / seed52；
  GPU3：`28_full.yaml` / seed42。三组均 mini Car、scratch、60 epoch、batch16、
  workers12、每5轮官方 mini_val 验证。用户已授权同时运行 Full，不能再将历史
  “首轮只 B0”的排程当作禁止提供 Full 命令的理由。
- 从 `main.py` 进入。各臂所有启用模块从头训练，不冻结、不跨臂 checkpoint 初始化。
  v28 B0 使用 `reference_batch` 整批原目标及单次 BC，不套用上级历史逐 view 权重。
- 正式运行目录：`output/YYYYMMDD-HHMMSS-28_模块-mini_car_seedXX_60ep_bs16/`，
  显式传 `--log_dir`，内部记录 `train.log` 和 `train.pid`。已有结果不覆盖或删除；
  工程/诊断临时文件仍写 `artifacts/ct_checks/`。三组重启前确认旧任务已经结束。

## 不要重复引入的 CUDA 问题

- 三组曾一起报 `nll_loss2d_forward_out_cuda_template`：三维分割 CE 的 CUDA
  mean 原子归约被 strict 拒绝。保留原三维 `log_softmax(dim=1)`，仅将其结果整理为
  二维 NLL；保持全 batch 的类别权重分母。入口为
  `models/ct_v2/observation_reference.py::seqtrack_segmentation_cross_entropy`，
  合同为 `ct_b0_ce_contract=class_axis_logsoftmax_flat_nll_v1`。
- 随后只有 Full 报 `cumsum_cuda_kernel`：B2 relation AP 的浮点二值累计被 strict
  拒绝。这是日志指标，不参与训练目标。`models/seqtrack3d.py` 两处 AP 均使用
  `cumsum(..., dtype=torch.int64)` 后转回浮点；不得恢复为浮点累计。
- 保留已有 `utils/deterministic_pooling.py` 的固定槽池化及首个最大值梯度语义。
  不使用 `warn_only=True`、关闭 deterministic 或换 seed 来掩盖以上错误。
- `tqdm.__del__` 是这两次主异常后的清理错误，先定位前面的首个 RuntimeError。
  本地 CPU 测试不能验证 CUDA 算子支持，也不能证明历史低分是随机性造成的。

## 证据与检查范围

9月10日完整数据接口审计还修复了两个问题：独立`preflight_ct_v28.py`缺少main的
`preloading=False`默认值；验证/测试导出缺真实scene/tracklet及partition/重叠信息。
后者只在导出层修复，不能为改日志去修改前向sequence的tracklet_key或递归采样key。
`dev_diagnostics`与`precision/dev`旧名称保留，实际划分看修复后的行/provenance。
修复后本地检查以[验收记录](docs/CTSEQTRACK_V28_LOCAL_VALIDATION.md)最新节为准。

完整回归 453 passed / 12 skipped 是9月8日历史快照；CE 修复专项为47/1，
AP 修复专项为15/1，不能相加或改称最新全量/CUDA通过。9月10日已核实修复后真实
CUDA训练完成，跨臂B0一致；独立同卡重复/epoch边界恢复和S/P稳定恢复仍未全通过。
纯文档或仅索取命令时不重跑训练测试；
需要验证代码修改时按故障记录选择相关检查，并明确未运行项目。

仅修改本仓库，不参考 TrajTrack；不改冻结基线、历史配置或历史输出。
