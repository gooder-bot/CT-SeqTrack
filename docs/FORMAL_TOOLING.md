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

`check_train_steps.py` 同时识别只读 v24 和 Safe-SeqTrack v25。v25 manifest 会记录 runtime protocol、单优化器拓扑、observation RNG 和精确的 B0 candidate 权重。用 `--steps 100` 分别运行四臂后，checkpoint 中的 `ct_b0_prefix_hashes` 必须在 initial、step1、step100 一致；`ct_cuda_stage_audit` 会保存 batch transfer、forward、loss、backward、step 的 allocated/reserved/peak。一次短跑仍不能替代 epoch-boundary resume 等价证明。

服务器 disposable smoke 示例：

```bash
python tools/check_train_steps.py --cfg cfgs/ct_seqtrack/25_b0.yaml --path /home/lishengjie/data/nuscenes-mini --steps 100 --artifact-dir artifacts/ct_checks/v25_b0_100
python tools/compare_ct_module_audits.py artifacts/ct_checks/v25_b0_100/checkpoints/last.ckpt artifacts/ct_checks/v25_b1_100/checkpoints/last.ckpt artifacts/ct_checks/v25_full_minus_b3_100/checkpoints/last.ckpt artifacts/ct_checks/v25_full_100/checkpoints/last.ckpt --modules b0
```

实际 checkpoint 路径以 Lightning 在对应 artifact 目录中的输出为准；smoke checkpoint 只用于审计，之后删除，不能初始化正式实验。

旧 M/TWC/CRPA/Search/Gate/replay、旧报告生成器、preflight 和 promotion 脚本不属于当前工具闭包。需要审计原实现时，从 `001951a:<path>` 只读恢复。

本地 `output/` 是保护区。工具默认临时产物写入 `artifacts/ct_checks/`；服务器正式实验使用 `main.py` 自动生成的新 `output/YYYYMMDD-HHMM-*` 目录。
