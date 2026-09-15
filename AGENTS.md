# CT-SeqTrack 当前工作提醒（2026-09-15）

## 当前v30与最新mini启动安排

当前以用户v30方案、`docs/EXPERIMENT_PROTOCOL.md`顶部及`need_to_do.md`为准。
本地实现已完成，完整回归868 passed/15 skipped；真实CUDA、正式训练与分数仍待实测。
最新用户要求轻量核对后给三组独立后台命令，不增加全套哈希/长检查/工程报告门：
GPU0 B0、GPU1 Full-GRU、GPU2 Full-CfC；mini Car、seed42、scratch60、batch16、workers4、val5。
用`30_b0_mini.yaml`、`30_full_gru_mini.yaml`、`30_full_cfc_mini.yaml`；不加`--preloading`，
默认每worker256MiB缓存。`CUDA_VISIBLE_DEVICES`选择物理卡，每个进程`trainer_devices=1`。
输出显式`output/YYYYMMDD-HHMMSS-30_模块-mini_car_seed42_60ep_bs16/`，包含`train.log`和`train.pid`。
命令见`docs/CTSEQTRACK_V30_MINI_LAUNCH.md`。仅索取命令不等于授权自动修改或启动服务器任务。
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
