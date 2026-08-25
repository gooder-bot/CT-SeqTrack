# CT-SeqTrack 正式工具面

当前工具只覆盖八类正式任务。报告工具只读取实验产物并写入用户指定的新路径，不作为训练门禁，也不生成跨实验初始化 checkpoint。

| 类别 | 工具 | 职责 |
|---|---|---|
| B1 报告 | `report_ct_b1.py`、`export_b1_calibration.py`、`calibrate_b1_uncertainty.py` | learned mean 与 CV、NLL、coverage、support 和校准残差 |
| B2 acquisition 报告 | `report_ct_b2.py` | supply、retention、presence、raw gain、oracle headroom、harm |
| B3 校准 | `calibrate_ct_actions.py` | 在独立 calibration tracklets 上生成绑定 checkpoint/config 的阈值 |
| risk--coverage | `report_ct_risk_coverage.py` | action coverage、harmful rate、gain 和风险曲线 |
| memory 对照 | `report_ct_memory.py` | real/empty/time-misaligned 配对报告，不阻断训练 |
| 共享参数审计 | `compare_ct_module_audits.py` | 比较 matched-scratch 共享前缀参数 hash |
| 运行检查 | `verify_ct_slimming.py`、`check_candidate_shared_se2.py`、`check_forward_batch.py`、`check_time_batch.py`、`check_train_steps.py` | 配置/保护区、candidate 几何、真实 batch、时间链路和有限训练步 |
| 点云和预测框 | `visualize_pointcloud_sample.py`、`visualize_model_predictions.py` | 检查非空点云、有限框、逐帧输出和同序列模型对照 |

所有接收 `--cfg` 的保留工具都通过 `utils.config.load_yaml_config` 解析 `_base_`，与 `main.py` 的 resolved-config 语义一致。

`check_train_steps.py` 同时识别只读 v24 和 Safe-SeqTrack v25。v25 manifest 会记录 runtime protocol、单优化器拓扑、observation RNG 和精确的 B0 candidate 权重。用 `--steps 100` 分别运行四臂后，checkpoint 中的 `ct_b0_prefix_hashes` 与 `ct_b0_optimizer_state_hashes` 必须在 initial、step1、step100 一致；每个启用参数组还必须出现过有限非零梯度且无 frozen 参数。`ct_cuda_stage_audit` 会保存 batch transfer、forward、loss、backward、step 的 allocated/reserved/peak。一次短跑仍不能替代 epoch-boundary resume 等价证明。

服务器真实 batch forward 与 B1 后端 disposable smoke 示例：

```bash
python tools/check_forward_batch.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --require-full-history --b1-backend gru
python tools/check_forward_batch.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --require-full-history --b1-backend cfc
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 8 --b1-backend gru --artifact-dir artifacts/ct_checks/v25_b1_gru_smoke
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 8 --b1-backend cfc --artifact-dir artifacts/ct_checks/v25_b1_cfc_smoke
```

正式比较前必须再做 matched-seed 100-step 公平性审计；三个目录都必须是新目录：

```bash
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b0.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --artifact-dir artifacts/ct_checks/v25_b0_100
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --b1-backend gru --artifact-dir artifacts/ct_checks/v25_b1_gru_100
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --seed 42 --b1-backend cfc --artifact-dir artifacts/ct_checks/v25_b1_cfc_100
python tools/compare_ct_module_audits.py B0_100_LAST_CKPT B1_GRU_100_LAST_CKPT B1_CFC_100_LAST_CKPT --modules b0
```

三个 checkpoint 的 B0 参数和 Adam `step/exp_avg/exp_avg_sq` hash 必须在
initial、step1、step100 全部一致；任一点不一致就停止正式比较。实际 checkpoint
路径读取对应目录的 `check_manifest.json`；smoke checkpoint 只用于审计，之后删除，
不能初始化正式实验。胜出后端确定后，Full-B3/Full 也按同一方式从 epoch 0
另跑 100-step，并与 matched B0/B1 审计，不能迁移上述 checkpoint。

旧 M/TWC/CRPA/Search/Gate/replay、旧报告生成器、preflight 和 promotion 脚本不属于当前工具闭包。需要审计原实现时，从 `001951a:<path>` 只读恢复。

本地 `output/` 是保护区。工具默认临时产物写入 `artifacts/ct_checks/`；服务器正式实验使用 `main.py` 自动生成的新 `output/YYYYMMDD-HHMM-*` 目录。

## B1-GRU/CfC 与独立校准命令

两个后端复用同一份 `25_b1.yaml`，都从 epoch 0 训练；不要把 B1-only 权重
迁移到 Full-B3：

```bash
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --tag b1_gru_seed42 --b1-backend gru
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --tag b1_cfc_seed42 --b1-backend cfc
```

每个 scratch checkpoint 分别导出 calibration 与 dev tracklets。两份 artifact
必须来自相同 checkpoint、resolved config、seed 和 split：

```bash
python tools/export_b1_calibration.py --config cfgs/ct_seqtrack/25_b1.yaml --checkpoint B1_LAST_CKPT --path /home/lishengjie/data/nuscenes-mini --b1-backend BACKEND --partition calibration --output artifacts/ct_checks/b1_calibration_fit.npz
python tools/export_b1_calibration.py --config cfgs/ct_seqtrack/25_b1.yaml --checkpoint B1_LAST_CKPT --path /home/lishengjie/data/nuscenes-mini --b1-backend BACKEND --partition dev --output artifacts/ct_checks/b1_calibration_dev.npz
python tools/calibrate_b1_uncertainty.py --fit-input artifacts/ct_checks/b1_calibration_fit.npz --eval-input artifacts/ct_checks/b1_calibration_dev.npz --output artifacts/ct_checks/b1_calibration.json --checkpoint B1_LAST_CKPT --output-checkpoint artifacts/ct_checks/B1_EVAL_ONLY.ckpt
```

`B1_EVAL_ONLY.ckpt` 只能用于评估，不能续训。GRU/CfC 的逐 endpoint 配对报告
以 CfC 为 candidate、GRU 为 reference：

```bash
python tools/report_ct_b1.py --rows CFC_ROWS.jsonl --reference-rows GRU_ROWS.jsonl --output artifacts/ct_checks/b1_cfc_vs_gru.json
```

只有 `candidate-minus-reference RMSE` 的 tracklet paired-bootstrap 95% CI 上界
小于 0，且 NLL 与 coverage ECE 均不劣于 GRU，CfC 才可晋升。

真实序列可视化比较两个后端时，不新建重复 YAML；为每个 checkpoint 显式绑定
其训练后端：

```bash
python tools/visualize_model_predictions.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path /home/lishengjie/data/nuscenes-mini --model GRU cfgs/ct_seqtrack/25_b1.yaml B1_GRU_LAST_CKPT --model CFC cfgs/ct_seqtrack/25_b1.yaml B1_CFC_LAST_CKPT --model-backend GRU gru --model-backend CFC cfc --output-dir artifacts/ct_checks/b1_backend_boxes
```
