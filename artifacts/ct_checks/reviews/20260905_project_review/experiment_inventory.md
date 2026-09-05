# CT-SeqTrack 实验与 Git 历史独立核验（2026-09-05）

本次只读原始 `output/`、Git 对象、协议与配置，没有改写实验输出或用户已有的 20260905 报告。当前代码为 `b445ecd04fcdc41474c29d829b2626f3780f759d`。先阅读既有报告定位，再从 TensorBoard event、checkpoint 和候选 CSV 独立复算；本目录的 `recompute_experiment_evidence.py` 可重跑 v26 核验，输出 `recomputed_experiment_evidence.json`。历史 TensorBoard 复算值保存在 `recomputed_history_metrics.json`。

## 1. 最新实验实际完成程度

五臂运行目录均为 `output/20260903-2301-26_{arm}-..._20260903_225946`，具体完整路径见上述 JSON 的 `run` 字段。五组 provenance 均记录同一 commit、seed42、Car、batch16、`init_checkpoint_path=null`、`checkpoint_path=null`，是从头训练；mini_train 为 274 tracklets / 5051 frames，mini_val 为 106 / 2285，数据选择 SHA 一致。dirty_tracked 记录仅删除旧诊断 tar，另有未跟踪日志/诊断目录；没有记录源代码被修改，不能只凭 dirty 标记否定运行。

| 臂 | 最后完整验证/保存 epoch | Success | Precision | 最后验证 FPS | B0 / 插件更新次数 |
|---|---:|---:|---:|---:|---:|
| B0 | 60 | 26.902624 | 25.600657 | 7.915045 | 75720 / — |
| B1-GRU | 60 | 51.972652 | 61.952961 | 2.001249 | 75720 / 12780 |
| B1-CfC | 60 | 28.795403 | 28.017506 | 1.948862 | 75720 / 12780 |
| Full−B3 | 8 | 28.211159 | 29.428886 | 1.240399 | 10096 / 1704 |
| Full | 60 | 48.801968 | 54.888409 | 1.797342 | 75720 / 12780 |

表中 S/P 来自每组 `lightning_logs/version_0/metrics_mini_val_{success,precision}` 的原始 scalar，更新次数来自 checkpoint 的 `ct_module_audit.update_steps`。Full 的 B1/B2/B3 各为 12780 次，所有启用模块都有非零有限梯度记录，`active_frozen_parameters=[]`。

四个完成臂均实际保存 `formal_checkpoints/epoch=058.ckpt`、`059`、`060`。训练期间隔 2 epoch 验证，所以已有最后三个验证点是 **56/58/60，不能称为正式 late-3**；59 必须单独评估。Full−B3 只有最后 epoch8 checkpoint/4 次完整验证，主 event 已有 epoch9（零基，即第10 epoch）的训练数据，但本地没有原始崩溃控制台日志；“第10 epoch validation 因 FPS 唯一性报错”是既有报告的定位，应由代码复现或服务器日志进一步确认，不能仅由验证序列缺失证明。

FPS 是当前带诊断流程的训练期验证日志，未记录可核验的单卡型号/UUID、并发负载、单独延迟与同步边界，不应直接用作论文部署速度比较。五组 event 主机名都含 `tesla-a40-2`，这不足以确认使用相同物理 GPU 或相同 GPU 型号。

## 2. 最重要的归因问题：同一输入，B0 第一步已经不同

checkpoint 的 B0 initial hash 全为 `798a8def...`，前 100 个 `ct_observation_batch_fingerprints` 逐项完全一致，初始 Adam hash 一致；第一次更新后参数和 Adam hash 都分歧：

| 臂 | B0 step1 hash 前缀 | B0 step100 hash 前缀 |
|---|---|---|
| B0 | 126580d7 | 4231eff7 |
| B1-GRU | 6b901e2f | ab98b21c |
| B1-CfC | 726e7516 | dc9467bf |
| Full−B3 | b875c2ab | 8e6910b0 |
| Full | 2d036f20 | d94bb0c3 |

因此 B1-GRU 相对 B0 的 +25.07 Success、Full 相对 B0 的 +21.90 不能归因给模块。B1-only 部署本来就是 observation，Full 未校准也回到 observation，分数主要呈现不同 B0 训练轨迹的结果。**hash 不一致只证明非逐位一致，不证明梯度泄漏，也不能单独衡量数值差异的大小**：CUDA 非确定性、算子路径、dropout/RNG 和优化器实现都应区分。建议在同一 GPU 顺序执行相同输入的 step1/100，记录输出、loss、逐层梯度及 Adam 数值误差和 hash，先建立单臂重复运行噪声基线，再比较跨臂隔离；不要直接靠调 seed 寻找高分轨迹。

## 3. 同一 checkpoint 内的机制证据

Full e60 原始候选 CSV 为 `output/20260903-2301-26_full-ct26_full_mini_car_seed42_60ep_bs16_20260903_225946/lightning_logs/version_0/candidate_diagnostics/epoch_60.csv`；Full−B3 对应目录的 `epoch_08.csv`。CSV 第一行是字段名、其后每行是 endpoint；以下为全文件聚合，不是挑例。

| 指标 | Full e60 | Full−B3 e8（未完成） |
|---|---:|---:|
| CSV endpoints / 应有非首帧 endpoints | 1928 / 2179 | 1952 / 2179 |
| 缺失 endpoints | 251（11.52%） | 227（10.42%） |
| active_prior_source = b1 | 0 | 0 |
| fallback_cv / base_only | 1846 / 82 | 1870 / 82 |
| 结构 available | 249 | 851 |
| globally observable need 中有 novel target | 106 / 1676 = 6.32% | 143 / 1702 = 8.40% |
| B0 raw 完全漏目标时 novel 补获 | 1 / 123 = 0.81% | 28 / 759 = 3.69% |
| 有目标 prepool 的 selection 行召回 | 100% | 100% |
| selection 目标点召回 | 97.38% | 73.52% |
| available 上 raw 搜索平均 center gain | −0.52558 m | −0.13314 m |
| available 上 raw 搜索平均 IoU gain | −0.15642 | −0.05607 |
| available 中中心误差恶化 >0.1m 比例 | 60.24% | 56.76% |
| search_valid >0 | 146 | 0 |
| B3 已校准 / 实际 action | 0 / 0 | 0 / 0 |

这说明现有链路首先没有稳定取得新增目标证据；已进入 prepool 的目标行多数被保留，当前最明显的瓶颈在支持域供给及背景候选伤害，不能只继续加采样头或 voting 容量。6.32% / 8.40% 未达到方法协议 15% 门槛，且当前 online B1 source 全未启用，必须先修复接口再评价设计本身。所有 CSV 的 `recursive_age_valid` 均为 0，无法可靠给出递归年龄分层结果。

Full 的 `proposal_inference_mode=observation`，逐行 `final_error==observation_error`。Full−B3 配置确实是 `raw_search`，但 presence_probability 最大仅 0.385678、`search_valid` 全为 0，所以部署候选全部回退 observation；不能将 e8 总分解释成 B2 已实际起作用，也不能误判为运行模式填错。

主 TensorBoard 还有机制诊断 Success：Full e60 observation=49.977833、raw_search=48.046799；Full−B3 e8 为26.433790 /24.050634。这两个诊断指标与总 tracking Success 分母/路径不同，**不能混算成总性能差**。它们及 CSV 的同 checkpoint 比较均呈现 raw candidate 的负信号；正式部署收益仍需完整 endpoint、正确输出和配对评估。

三个 counterfactual（fixed_2_1/adaptive_local/adaptive_dual_support）的 `support_raw_target_count` 与 `support_novel_target_count` 在两个文件中逐行完全相同。这是必须追查在线集合/key 一致性的诊断信号，不应直接将 counterfactual 表作为支持域改进证据；浮点变换根因需代码审计确定。

缺失的 251/227 行不能当成“无害或无需恢复”；总 tracking 可以继续统计这些帧，而候选机制表已失去该部分分母。应完整输出 early-return/fallback 原因和该帧可观测性后再做覆盖率、风险、CI。这里不能仅由 CSV 缺行断定所有缺失都等于“当前 crop 为空”。

### B1 的有效行统计

| 臂 | 有效行 | Learned RMSE / CV RMSE（m） | 50% / 95% coverage |
|---|---:|---:|---:|
| B1-GRU e60 | 1844 | 6.314978 / 6.302485 | 91.54% / 93.76% |
| B1-CfC e60 | 1831 | 12.491975 / 12.758745 | 43.04% / 60.29% |
| Full e60 | 1846 | 6.455751 / 6.446156 | 84.56% / 92.47% |

这些由同一 CSV 中 `b1_valid>0` 且误差有限的行复算。GRU 在自己的轨迹上未优于 CV；CfC 对自己的较差轨迹略优于 CV，但绝对误差更大且严重欠覆盖。两后端的上游轨迹及有效行不同，不能把跨臂绝对 RMSE 或总 tracking S/P 当成后端晋升依据。需共同 endpoint/预测历史的 backend 对照，加独立 calibration/dev promotion、tracklet-paired CI；本次没有重复引用既有报告中硬编码的 bootstrap 区间为新计算结果。

## 4. Git 发展方向及历史数值

| Commit / 日期 | 主要变化 |
|---|---|
| 001951a /08-22 | 固定 candidate 协议、瘦身冻结基线；不是当前 HEAD |
| 2bdacd8 /08-22 | 恢复四候选预算及双流训练语义 |
| 62e1f90 /08-24 | Safe-SeqTrack v25：stateless observation、统一自动 Adam、惰性 mechanism iterator/RNG 隔离、v8审计 |
| b8222bb /08-25 | B1 normalized residual、tail supervision、beta-NLL、gap2/gap4、CfC 插件及held-out校准 |
| e9a2d6d /08-26；d384282 /08-26 | B2 acquisition funnel 诊断 |
| 5225ff0 /08-28 | v26 adaptive bounded shell、backup corridor、768→256 relation/coverage/exploration、robust consensus、B3 artifact v2 |
| ad70d36 /09-03 | 修复在线验证 novel support 误判 |
| b445ecd /09-03 | 注册当前五个 mini scratch arms；本轮实验对应版本 |

原始 TensorBoard 的 B0 final S/P 轨迹为：

| 历史轮次 | Success / Precision |
|---|---:|
| 07-25 d86990c SeqTrack 背景参考 | 53.359955 /64.381836 |
| 08-13 历史 SeqTrack | 51.001095 /60.892784 |
| 08-22 v24 B0 | 31.414663 /31.102844 |
| 08-24 v25 B0 | 50.690372 /59.280090 |
| 08-25 v25 B0（val5） | 29.869804 /30.398251 |
| 08-26 v25 B0（diag_v2） | 54.826038 /66.340263 |
| 09-03 v26 B0 | 26.902624 /25.600657 |

这是反复出现的 baseline 不稳定/不可归因问题，不是模块版本逐渐提升的证据。v25 08-24 四臂 final 分别为 B0 50.690/59.280、B1 33.443/35.138、Full−B3 47.852/54.570、Full52.553/61.194；08-26 B1-GRU 又为30.734/29.626、CfC35.318/37.697。不能从每轮挑高分拼一张消融表。

历史 d86990c 的 `models/base_model.py` 真实使用 `bbox_size=this_frame['3d_bbox'].wlh`，当前安全口径固定首帧尺寸；采样 seed/recursive-state 也不同，所以53.36只是背景参考，不能直接要求当前复制该数值或声称净增益。另一方面，08-24、08-25、08-26和当前 v26 已都记录 `observation_safe_bbox_size=true`，所以近期 54.83→26.90 **不能再统一归咎于尺寸协议修复**。08-26 与 v26 同时变化了 workers12→4、preloading false→true、验证间隔1→2、schema v2→v3等，需要隔离验证具体因果。

历史 cadence5 的最后三个观测是50/55/60，cadence1 才是58/59/60；本报告只将各轮 final 用作描述性历史，不混用这些不同窗口。

## 5. 下一步优先事项与完成边界

1. 在修复后的同 GPU 可复现实验里验证共享 B0 数值轨迹，区分纯 CUDA 非确定性与真实随机流/更新差异；补充分辨所需的 GPU UUID、库/CUDA/cuDNN/PointNet++构建信息、确定性开关和实际命令。
2. 修复 B1 online acquisition 接口、退化 FPS、early-return 恢复通路和诊断集合一致性，并让每个 endpoint 写明支持缺失原因；做真实 batch/100-step/resume 验收。
3. 方法/训练核心路径改变后，注册新轮次，从 epoch0 训练匹配 mini arms；不能把当前 Full−B3 checkpoint续训成修复版正式结果。保留本轮作为失败/工程证据。
4. 先以同 checkpoint 的固定CV/learned、extension-only oracle与raw gain拆开“取不到点、点被丢弃、点在但预测差”；修复后仍取不到点才讨论更长时域/重检测方案。
5. Full final及58/59/60分别导出互斥 calibration/dev，独立绑定 artifact；报告promoted/未通过状态与实际action数，再计算完整tracklet配对CI。当前没有已部署的校准Full正结果。
6. 单独补 SeqTrack reference。`cfgs/26_seqtrack_strict_nuscenes_full.yaml`继承的配置第22行仍是`candidate_trajectory_mode:shared_se2`，而CT B0为independent/stateless/候选加权，因此它应列为明确协议的外部参考，CT B0才是模块消融对照；命名strict不能替代训练/评估协议核对。
7. 有效 mini 结论后再进入完整nuScenes、额外seed、真实时间true/fixed/shuffled、memory real/empty/time-misaligned；目前不存在这些v26证据，不能提出涨点、跨seed稳定性、物理时间因果性、memory收益或SOTA声明。

文档还需同步状态：`need_to_do.md:38`把mini五臂全部留待执行，实际是四臂完成、一个未完成且整体未通过机制/归因验证；`README.md:107`仍写“No mini run”，与`docs/CTSEQTRACK_V26_METHOD.md:81`的先mini顺序冲突。应在修订时表达“训练完成”与“方法通过验收”的区别，不把失败实验改成未发生，也不把已实现模块改成已证实创新。
