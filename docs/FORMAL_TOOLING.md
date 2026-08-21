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

`check_train_steps.py` 保留有限步、损失/梯度和工程 checkpoint 检查能力，但不能把一次短跑解释为完整的 epoch-boundary resume 等价证明。用户本轮跳过服务器 smoke，这些项目保持“未执行”。

旧 M/TWC/CRPA/Search/Gate/replay、旧报告生成器、preflight 和 promotion 脚本不属于当前工具闭包。需要审计原实现时，从 `001951a:<path>` 只读恢复。

本地 `output/` 是保护区。工具默认临时产物写入 `artifacts/ct_checks/`；服务器正式实验使用 `main.py` 自动生成的新 `output/YYYYMMDD-HHMM-*` 目录。
