# P0-B2 Recursive Predicted-History Crop Reachability — 2026-07-17

本报告汇总同一个 standard A1-order checkpoint 在 standard、gap1124、burst-drop 三协议上的递归预测历史诊断。每条 tracklet 只运行一条正常 baseline A1 轨迹；四种 anchor 在同一 endpoint 上被动统计 crop reachability，不改变 baseline 预测历史。因此结果回答“已有 A1 预测历史能否支持 GT-free trajectory recenter”，不是 active recenter 的跟踪性能。

## 1. 完整性与可比性

| protocol | endpoints | reference exact match | missing / unexpected | checkpoint SHA256 |
| --- | ---: | --- | ---: | --- |
| standard | 4246 | true | 0 / 0 | `a2fbffb...f24a82` |
| gap1124 | 2127 | true | 0 / 0 | `a2fbffb...f24a82` |
| burst-drop | 2098 | true | 0 / 0 | `a2fbffb...f24a82` |

三组使用同一 checkpoint，并与上一轮 oracle CSV 的 `(tracklet_id, frame_index, frame_token)` 完全一致。日志无 traceback，summary/CSV 均完整。

## 2. 四种 anchor 的总体结果

| protocol | previous-GT recall | previous-A1 recall | A1-pred-history CV recall | CV - previous-A1 | GT-history CV recall | CV / previous-A1 points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 85.41% | 69.69% | 72.61% | +2.91 pp | 99.95% | 0.99x |
| gap1124 | 76.78% | 63.73% | 66.38% | +2.65 pp | 98.96% | 0.98x |
| burst-drop | 77.72% | 63.24% | 66.27% | +3.03 pp | 99.05% | 0.91x |

| protocol | previous-A1 center outside | pred-history CV center outside | GT-history CV center outside |
| --- | ---: | ---: | ---: |
| standard | 31.51% | 28.69% | 0.12% |
| gap1124 | 36.39% | 34.56% | 1.18% |
| burst-drop | 36.51% | 34.41% | 0.91% |

Pred-history CV 的方向总体偏正：standard/gap/burst 分别净减少 120/39/44 个 center-outside endpoint，完全丢失目标点的可见 endpoint 减少 48/29/42 个；只有 12/6/4 个原本 center-inside 的 endpoint 被 CV 推到外部。但总体 recall 只提高 2.65–3.03 个百分点，远小于 GT-history oracle 的上限。

## 3. 预注册 Go/No-Go 判断

预注册要求：

- gap1124 与 burst-drop 总体 recall 均至少提高 5 pp；
- 两者 `>4 m` 位移桶均至少提高 10 pp；
- 平均点数不超过 1.25x；
- standard 不下降超过 1 pp。

实际结果：

| protocol, displacement >4 m | previous-A1 recall | pred-history CV recall | delta |
| --- | ---: | ---: | ---: |
| standard | 10.06% | 19.31% | +9.25 pp |
| gap1124 | 1.68% | 10.13% | +8.45 pp |
| burst-drop | 1.81% | 11.76% | +9.96 pp |

点数和 standard 安全条件通过，但总体 +5 pp 与强协议 `>4 m` +10 pp 均未通过。因此结论是：

> **No-Go for always-on raw predicted-history CV recenter.**

不能把当前 `a1_pred_history_cv` 直接接成唯一 search anchor，也不应据此启动正式训练。

## 4. 根因：预测历史呈现可靠/失控两种状态

| protocol | previous prediction error P50 / P75 / P95 | current prediction error P95 | empty fallback |
| --- | ---: | ---: | ---: |
| standard | 0.82 / 10.05 / 78.33 m | 84.10 m | 915 / 4246 |
| gap1124 | 0.67 / 15.22 / 89.25 m | 101.26 m | 425 / 2127 |
| burst-drop | 0.67 / 16.12 / 94.84 m | 104.14 m | 430 / 2098 |

| protocol | endpoints with previous error <=4 m | previous-A1 recall | pred-CV recall | endpoints with previous error >4 m | previous-A1 recall | pred-CV recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 3047 | 94.77% | 98.59% | 1199 | 0.37% | 0.80% |
| gap1124 | 1419 | 93.56% | 97.34% | 708 | 0.93% | 1.21% |
| burst-drop | 1385 | 94.76% | 98.64% | 713 | 0.28% | 1.61% |

这揭示了明确的状态边界：

- 当预测历史仍可靠时，真实时间 CV 可以把 reachability 推到约 97%–99%；
- 当上一预测已经偏离超过 4 m，previous-A1 与 pred-history CV 几乎同时失效；
- CV 对错误历史做外推只能传播漂移，不能凭空恢复绝对位置；
- GT-history CV 与 pred-history CV 的巨大差距主要来自 state reliability，而不是 crop 尺寸或点数。

`previous_prediction_error <=4 m` 只是离线 GT 分桶，推理时不可使用。它证明 reliability gate 值得研究，但不能直接成为方法阈值。

## 5. 方法路线调整

当前证据不支持“用 predicted CV 替换 previous prediction anchor”，但支持一个更窄、更合理的方向：

```text
previous-pred observation anchor ─┐
                                  ├─ test-time reliability / agreement ─> observation refinement
clipped CV/Kalman trajectory anchor ┘
```

trajectory proposal 的职责应是**在漂移发生前提供第二个预防性搜索假设**，不是在已经跟丢后单独完成全局恢复。下一步首先要回答：能否只用测试时可得信号，识别“历史仍可靠”和“即将失控”的边界。

优先记录和验证的信号：

- model proposal confidence / best-box score；
- predicted foreground count 与 mean foreground score；
- previous crop point count、empty fallback；
- `||c_cv - c_prev||`、predicted speed、CV shift；
- local observation proposal 与 trajectory proposal 的 agreement。

如果这些信号不能预测下一步 crop failure，就停止 gate/dual-anchor 主线，不增加 Mamba、ODE、occupancy memory 或更大 trajectory encoder。

## 6. 下一步顺序

1. 扩展递归诊断，记录上述 test-time reliability signals，并报告其预测 `previous_prediction_error >4 m` / next-step crop failure 的 AUROC、AUPRC 与 calibration；GT 只用于离线标签。
2. 只有可靠性信号有效时，做无训练的 active dual-anchor inference：previous-pred crop 与 clipped-CV/Kalman crop 各 forward 一次，用推理时可得 confidence/agreement 选择或保守融合。
3. active 版本必须固定同一 A1 checkpoint、相同 endpoints；先验证能否减少首次失控和连续失败，而不是只看平均 oracle recall。
4. active 机制通过后，才实现正式 trajectory guidance，并进入 `true/fixed/shuffled-dt` 因果矩阵。
5. 末端 bounded residual、TWC、复杂 memory/SSM 模块继续后置。

## 7. 数据来源

- `output/diagnostics/recursive_crop_reachability/standard_a1_recursive/`
- `output/diagnostics/recursive_crop_reachability/gap1124_a1_recursive/`
- `output/diagnostics/recursive_crop_reachability/burst_drop_a1_recursive/`
- `logs/diagnostics/p0b2_*_a1_recursive.log`

