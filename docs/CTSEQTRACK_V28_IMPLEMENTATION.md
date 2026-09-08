# v28 共享观测底座与实施范围（2026-09-08）

v28 先恢复可审计的 SeqTrack 观测训练与推理合同，首轮正式实验只运行 B0。当前没有新的 v28 60 轮结果，不承诺恢复某次历史高分；B1/B2/B3 的收益仍须后续匹配实验验证。v25 高低分的参数首步已分叉，但现有证据尚未定位具体算子，更未证明历史总分差全部来自随机性。

## 观测与模块合同

- 使用 `ct_enable_v28=true` 与 `ct_enable_v27=true`，保留 `safe_seqtrack_auto_v1`、单 Adam、`ct_seqtrack.train.v4` 外层事务。v28 观测身份为 `seqtrack_reference_compatible_v1`。
- B0 采用 `seqtrack_original_slots_v1` 的原候选总体：对完整 dataset 索引无状态随机排列，batch 不强求四类候选等比例；保留四种 candidate 的数据总体，损失 `reference_batch` 直接在整 batch 上归约。兼容字段 `ct_b0_candidate_weights` 不参与 v28 目标，不能把它解释为逐 view 0.25 加权。
- BC 只加入一次；原始观测槽和预测历史 prior、空输入及重采样按 v28 明确合同处理。新增原/实际索引与拒绝重抽原因用于复核。B2 原始测量 ID、唯一性与 extension-only 合同独立保留。
- B2 的 base 逐点特征来自 `seg_second64_v1`，避免把旧 FeaturePointNet 帧/通道错排分支冒充真实点特征。B0主网络的进一步结构改造、ST、soft segmentation、更多记忆不属于本次最小修复。
- 结构可用性独立于 learned presence；B1+B2 的 bounded-always、B3、逐帧导出与校准共享合法候选。presence 保持连续学习特征，不重新引入固定0.5门。
- B0是递归状态唯一写入者；B1-only正式输出 observation。训练递归使用 observation；B1/B2/B3 在各自第一个合法机制 tick 获得监督，所有启用参数从epoch0参与训练，无冻结、无旧checkpoint初始化。
- v28 reference 使用 `ctseqtrack + ct_variant=b0` 的同一隔离/采样/目标实现，通过 `ct_reference_baseline=true` 命名。它与28 B0没有网络或训练数学差异，是共享实现的架构参考，不能包装成独立算法消融或未经修复的原仓复现。

## 确定性与身份

v28 显式 FP32；matmul precision=highest，CUDA与cuDNN TF32均关闭；cuDNN benchmark关闭、deterministic开启；Adam foreach/fused均为false；`CUBLAS_WORKSPACE_CONFIG=:4096:8` 在CUDA初始化之前设置；`torch.use_deterministic_algorithms(True, warn_only=False)`。不支持的操作必须修复或报错，不能自动降级。

B0 AdaptiveMaxPool1d 的CUDA backward存在严格确定性阻断，固定1024槽到128槽及全局池化使用保留首个最大值梯度语义的确定性实现；随机输入、并列最大值和全零边界须分别验证。此兼容修复不等于已证明该算子导致历史降分。

实际 Python/PyTorch/Lightning/CUDA/cuDNN、GPU标识与驱动、数值flags写入 `run_provenance.json` 的resolved config与工程/正式checkpoint。数值选择和观测身份绑定resume/config；实际机器信息用于同run恢复核对，不进入跨机器推理校准的配置hash。校准另绑定checkpoint、完整源码内容、场景manifest、指标与动作定义，v28产物schema为 `ct_seqtrack.action_calibration.v28`。

## 数据与正式运行

mini：全部8个mini_train场景参与参数训练，官方2个mini_val只用于固定轮次评测。calibration/dev各1个场景仍由稳定seed42选择，属于训练内部子集，明确 `parameter_training_overlap=true`，不能写成held-out泛化校准。full：350个train_track全部训练，内部17/18拟合/诊断，官方150个val评测。

正式固定Car、seed42、60epoch、batch16、workers12、每5轮官方验证。保留58/59/60checkpoint，最终报告final60与late-3，补评58/59而非拿50/55/60替代。官方val不用于挑后端、超参或best epoch。

mini/Car 的真实完整候选总体必须自然产生每轮1262个更新、60轮共75720个更新。`ct_v28_expected_mini_car_updates_per_epoch=1262` 进入配置与resume身份；main/preflight核对真实 `len(dataset)//16` 与sampler长度。不匹配时停止排查数据/类别/采样来源，不重复或补齐样本凑数。full配置显式置null，按实际总体计算步数。

正式健康参考的Success/Precision为 **50.9857788 / 59.9617119**；恢复目标分别为 **S ≥49.986、P ≥58.962**（各项允许约1个百分点退化）。按固定final60与late-3完整报告，不能挑best epoch。另行做相同协议、相同固定轮次的matched reference比较，S与P差距各自不得超过1个百分点。两臂同时低分即使彼此接近也不能记为恢复；历史健康参考是验收目标，不是本次已复现的结果。

首轮只启动28 B0；100-step数值与真实epoch-boundary恢复验收通过、B0结果可解释后，再考虑五臂和reference以及完整五类。新配置已经注册，不代表对应实验已经运行。

## 验收状态

- 最终全量回归453 passed、12 skipped，编译与diff检查通过；数值证据和跳过范围见 [本地验收](CTSEQTRACK_V28_LOCAL_VALIDATION.md)。
- 服务器真实数据preflight、严格CUDA数值一致性、Full连续/恢复、28 B0正式60轮与late-3仍待执行。
- Windows本机缺nuScenes数据/CUDA训练环境，不能把CPU合同测试记成服务器验收。

可执行流程见 [服务器运行说明](CTSEQTRACK_V28_SERVER_RUNS.md)。旧v27五臂已完成60轮但只有内部dev、Full未校准，准确结果仍保存在 [9月7日审计](../artifacts/ct_checks/reports/20260907_v27_mini_five_arm/REPORT.md)。

各项修改的必要性见 [修改审计](CTSEQTRACK_V28_CHANGE_AUDIT.md)；评测会附带 [固定定义的恢复诊断](CTSEQTRACK_V28_RECOVERY_REPORTING.md)。
