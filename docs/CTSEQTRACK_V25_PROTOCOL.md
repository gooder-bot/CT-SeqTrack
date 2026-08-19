# CT-SeqTrack v25 数据耦合协议与服务器实验手册

## 1. 当前状态

v25 已完成代码实现与本地单元测试，但尚未完成真实 nuScenes 训练。因此当前不能声称涨点、稳定增益、SOTA，或物理时间的因果收益。历史配置和结果已由恢复标签与外部 Git bundle 归档；v25 的所有数值必须由新 scene split 下的从头训练重新产生。

## 2. 固定训练契约

四臂 `B0 / B1 / Full-B3 / Full` 统一使用：

- 随机初始化，命令行不再提供 `--init_checkpoint`；
- rollout horizons `[1,2,4,8]`；
- `ct_training_reanchor_policy=periodic_past_gt`；
- `ct_b0_rng_protocol=post_observation_shift_v1`；
- 相同 batch、采样顺序、优化器和 scheduler；
- 所有启用模块从 epoch 0 更新，不冻结 producer；
- canonical state 只写 B0 observation，c1/c2 不写状态，B3 只做 shadow 学习；
- 推理永不使用 GT 重锚。

论文中应称该训练方式为 **mixed-horizon past-GT re-anchored rollout training**。重锚和 RNG 日程是优化/数据协议，不是创新点。

## 3. 数据隔离与证据语义

`scene_v2` 对 scene 哈希排序后分配 `70% / 15% / 7.5% / 7.5%`：

- `train`
- `dev`
- `calibration_select`
- `calibration_audit`

mini 的 8 个训练 scene 固定得到 `5/1/1/1`。运行 provenance 会保存完整 scene manifest；scene 不得跨分区，物理 frame token 不得跨 scene。bootstrap 的统计单位是 scene，不是 tracklet 或 frame。

B0 segmentation 始终沿用 `bb_scale=1.25`。B2 presence、targetness、vote/raw eligibility 和 acquisition counts 使用独立的 `ct_b2_target_bb_scale`，主配置为 `1.0`，`1.25` 仅作对照。固定点采样保留原始索引和 unique-valid mask：1–2 个点可供 B0 使用，但不能伪装成 B2 的结构有效证据。

## 4. 配置入口

mini 四臂：

```text
cfgs/ct_seqtrack/25_b0.yaml
cfgs/ct_seqtrack/25_b1.yaml
cfgs/ct_seqtrack/25_full_minus_b3.yaml
cfgs/ct_seqtrack/25_full.yaml
```

正式 full-data 四臂：

```text
cfgs/ct_seqtrack/25_b0_full.yaml
cfgs/ct_seqtrack/25_b1_full.yaml
cfgs/ct_seqtrack/25_full_minus_b3_full.yaml
cfgs/ct_seqtrack/25_full_full.yaml
```

关键对照：

```text
25_full_minus_b3_uniform{,_full}.yaml
25_full_minus_b3_b2_scale125{,_full}.yaml
25_full_time_fixed{,_full}.yaml
25_full_time_shuffled{,_full}.yaml
```

B1 GRU/CfC 单 seed mini 筛选：

```text
cfgs/ct_seqtrack/b1_gru_mini_seed42.yaml
cfgs/ct_seqtrack/b1_cfc_mini_seed42.yaml
```

两份配置都继承 `25_b1.yaml`，只允许时序 backend 和实验名不同。训练都从
epoch 0 随机初始化；CfC checkpoint 不得初始化 GRU，反之亦然。

seed 通过 `--seed 42/43/44` 覆盖；每个 seed 都必须独立从头训练。shuffled 时间控制还必须传入该数据 split 的离线 permutation manifest。

## 5. 服务器执行顺序

先做四臂 2-batch smoke，Full 再至少跑 20 batches 检查 H3；再做四臂 5 epoch kill test；方向成立后才做 seed42 的 60 epoch mini 趋势实验。完整 nuScenes 最终运行 seeds `42/43/44`。

训练示例：

```bash
python main.py \
  --cfg cfgs/ct_seqtrack/25_b0.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --seed 42 \
  --limit_train_batches 2
```

### B1 GRU/CfC 筛选

先分别执行有限 smoke；5 epoch 筛查是独立短实验，之后的 60 epoch 正式实验
必须重新从 epoch 0 开始：

```bash
DATA_ROOT=/home/lishengjie/data/nuscenes-mini

python main.py --cfg cfgs/ct_seqtrack/b1_gru_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --epoch 1 \
  --limit_train_batches 2 --limit_val_batches 2 --tag smoke
python main.py --cfg cfgs/ct_seqtrack/b1_cfc_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --epoch 1 \
  --limit_train_batches 2 --limit_val_batches 2 --tag smoke

python main.py --cfg cfgs/ct_seqtrack/b1_gru_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --epoch 5 --tag kill5
python main.py --cfg cfgs/ct_seqtrack/b1_cfc_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --epoch 5 --tag kill5

python main.py --cfg cfgs/ct_seqtrack/b1_gru_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --tag formal60
python main.py --cfg cfgs/ct_seqtrack/b1_cfc_mini_seed42.yaml \
  --path "$DATA_ROOT" --seed 42 --tag formal60
```

分别用两条正式运行的 final/last checkpoint 在同一 `mini_val` 上测试后，执行：

```bash
python tools/compare_ct_module_audits.py GRU_LAST.ckpt CFC_LAST.ckpt --modules b0

python tools/compare_b1_backbones.py \
  --gru-proposals GRU/proposal_diagnostics/proposal_endpoints.csv \
  --cfc-proposals CFC/proposal_diagnostics/proposal_endpoints.csv \
  --gru-tracking GRU/proposal_diagnostics/tracking_endpoints.csv \
  --cfc-tracking CFC/proposal_diagnostics/tracking_endpoints.csv \
  --output artifacts/b1_cfc_vs_gru_seed42.json
```

比较工具要求逐 scene/tracklet/frame 身份完全一致，报告 B1 mean/CV RMSE、NLL、
二维 coverage/ECE、时间间隔/稀疏度/recursive-age 分层，以及 B2 support 的目标
覆盖、目标点数量、unique extension 数量、support volume 和证据效率。只有全部
promotion gates 与独立 B0 hash 审计同时通过，CfC 才进入后续 Full 筛选。

B2/Full 启动前必须先生成 checkpoint-free preflight：

```bash
python tools/export_ct_acquisition_preflight_rows.py \
  --config cfgs/ct_seqtrack/25_full_minus_b3.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --output artifacts/v25_preflight_rows.jsonl \
  --data-manifest-output artifacts/v25_preflight_data_manifest.json

python tools/preflight_ct_acquisition.py \
  --rows artifacts/v25_preflight_rows.jsonl \
  --data-manifest artifacts/v25_preflight_data_manifest.json \
  --config cfgs/ct_seqtrack/25_full_minus_b3.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --output artifacts/v25_preflight.json
```

正式 preflight 不得使用 `--max-batches`。它逐 tracklet、逐 horizon、逐帧导出，预期行数为：

```text
sum(tracklet_frames - 1) * 4 horizons * 3 candidates
```

状态只写 fixed-CV 预测；当前 GT 仅在 support/crop 固定后生成标签。任何 dropped row、覆盖不足、manifest/SHA 不一致都会拒绝 B2 启动。

## 6. B3 两阶段校准

先用 Full checkpoint 在 `calibration_select` scene 上导出 observation 连续状态的候选诊断；直接使用测试输出的 CSV：

```bash
python main.py \
  --cfg cfgs/ct_seqtrack/25_full_full.yaml \
  --test --checkpoint checkpoints/full.ckpt \
  --test_split train_track \
  --ct-eval-partition calibration_select
```

第一阶段只用这些 selection scene 选择阈值：

```bash
python tools/calibrate_ct_actions.py \
  --selection-rows <selection日志目录>/proposal_diagnostics/proposal_endpoints.csv \
  --checkpoint checkpoints/full.ckpt \
  --config cfgs/ct_seqtrack/25_full_full.yaml \
  --selection-scene-manifest-sha256 <manifest中calibration_select的content_sha256> \
  --output artifacts/action_threshold_selection.json
```

然后用 provisional artifact 在 `calibration_audit` 上运行真正连续的 `proposal_inference_mode=selective`。该临时 artifact 只能和 `--ct-eval-partition calibration_audit` 一起使用：

```bash
python main.py \
  --cfg cfgs/ct_seqtrack/25_full_full.yaml \
  --test --checkpoint checkpoints/full.ckpt \
  --test_split train_track \
  --ct-eval-partition calibration_audit \
  --proposal-mode selective \
  --ct_action_threshold_selection_path artifacts/action_threshold_selection.json \
  --ct_calibration_select_scene_manifest_sha256 <calibration_select content_sha256>
```

最后只用该 selective 连续闭环的 audit CSV 做安全审计：

```bash
python tools/calibrate_ct_actions.py \
  --audit-rows <audit日志目录>/proposal_diagnostics/proposal_endpoints.csv \
  --selection-artifact artifacts/action_threshold_selection.json \
  --checkpoint checkpoints/full.ckpt \
  --config cfgs/ct_seqtrack/25_full_full.yaml \
  --selection-scene-manifest-sha256 <calibration_select content_sha256> \
  --audit-scene-manifest-sha256 <calibration_audit content_sha256> \
  --output artifacts/action_calibration_audit.json
```

最终 `passed` 只由 audit scene 的 action coverage、harmful-rate 上界、center gain 下界和 IoU gain 下界决定。selection 数据不进入安全结论。
若直接传 `--selection-scene-manifest/--audit-scene-manifest` 文件，校准工具会校验完整 manifest，并自动提取相应分区的 `content_sha256`；artifact 还记录实际 scene key，audit 会显式拒绝任何 scene 重叠。

正式 `val` 比较复用同一个 Full checkpoint，分别运行 `--proposal-mode observation`、`raw_search` 和 `selective`。其中 selective 必须绑定最终 audit artifact 及两份分区 SHA：

```bash
python main.py \
  --cfg cfgs/ct_seqtrack/25_full_full.yaml \
  --test --checkpoint checkpoints/full.ckpt \
  --proposal-mode selective \
  --ct_action_calibration_path artifacts/action_calibration_audit.json \
  --ct_calibration_select_scene_manifest_sha256 <calibration_select content_sha256> \
  --ct_calibration_audit_scene_manifest_sha256 <calibration_audit content_sha256>
```

## 7. 诊断与论文验收

每个 checkpoint 的 `ct_module_audit.b0_hash_timeline` 记录 initialization、step1、step100 和每个 epoch-end。四臂比较：

```bash
python tools/compare_ct_module_audits.py \
  b0.ckpt b1.ckpt full_minus_b3.ckpt full.ckpt \
  --modules b0
```

逐帧导出包含 gap 分布、boundary/outside 满足率、support recall/volume、截断率和跨 epoch selector migration。主指标使用 scene-paired bootstrap：

```bash
python tools/analyze_ct_v25_tracking.py \
  --baseline b0/proposal_diagnostics/tracking_endpoints.csv \
  --method full/proposal_diagnostics/tracking_endpoints.csv \
  --output artifacts/full_vs_b0_scene_bootstrap.json
```

论文主张门槛保持为：Full 至少在 2/3 seeds 同时优于 B0；三 seed 平均 Success 和 Precision 都提升；Success scene-paired bootstrap 的 95% 下界大于 0；并具备 `B1 → support recall → B2 raw gain → B3 selective gain` 的机制证据链。

真实实验与训练在服务器执行；本地仅做代码实现、静态检查、单元测试和必要的轻量 smoke，不持续检查本地不存在的数据集或训练环境。
