# CT-SeqTrack 实验结果简要总结

更新时间：2026-07-16

这份文件只保留实验主线，不展开所有 epoch 数据。完整表格和曲线见 `compare_results/`。

## 0. 当前总判断

### 2026-07-16 最新证据与结果口径

代码审查发现，旧 active-TWC sampler 在 candidate 1/2/3 下分别为 A/B 两路采样最近历史框扰动，导致两路 current search crop 和局部坐标系不同；旧检查比较归一化后天然接近零的 `ref_boxs[:, 0]`，没有发现这个问题。因此：

- 旧 A1+TWC 的 precision-positive 信号暂时撤回，不能归因给 TWC。
- 旧 A2+TWC 的退化也暂时撤回，不能据此判断 TWC 与 dynamics 冲突。
- 两路各自的 supervised loss 仍有效，但跨 view TWC loss 不是干净的一致性约束。
- 共享 candidate offset、`coordinate_anchor` 和 point-sampling seed 的修复已经完成；修复后的 A1/A2 seed42 训练均已完成，anchor gap max 与 current XYZ gap max 都为 0。
- `A2-residual-dyn` 已完成工程实现和纯逻辑 smoke test，但尚无性能结果。

修复后的 seed42 结果显示：A1+corrected-TWC 相对旧配置对齐 baseline 的 final 为 `+1.49 Success / +5.03 Precision`，late mean 为 `+0.99 / +2.67`；A2+corrected-TWC 的 final 为 `-0.93 / -2.07`。前者是值得复现的单 seed 正信号，后者不支持把 TWC 接入当前 A2 主线。由于 baseline 来自旧 run、没有 git commit，二者仍只是配置级参考，不能视为严格同提交因果结果。

HTV 六组 seed42 筛选也已完成：旧 feature-concat `A2-order-dyn` 在 random20 上相对 A1 final 为 `+9.09 / +14.23`，但在 gap1124 为 `-4.01 / -9.55`、burst-drop 为 `-7.45 / -14.40`。这不支持“时间间隔越不规则，旧 A2 越有效”，而支持继续验证 observation-first bounded residual、candidate 运动监督和 crop 可达性。

TrajTrack aligned seed42 run 虽得到 64.94 / 79.07，但当前本地 evaluator 使用当前帧 GT overlap 触发 refinement，并用 GT overlap 选择 proposal。该数值只能作为 oracle-assisted 实现诊断，不能作为对 SeqTrack3D 或 CT-SeqTrack 的公平在线增益。

目前结果不支持继续把真实时间直接塞进 SeqTrack3D 主干时间 token。更稳的方向是：

```text
主干保留 SeqTrack3D 的 order-time 语义；
真实 delta_t 主要作为保守 residual dynamics prior；
用固定 manifest 的 variable-rate / HTV 因果矩阵验证 residual；
corrected TWC 先独立复现 A1，gate 暂缓；TrajTrack 先修正为 GT-free evaluator。
```

旧 60ep seed42 汇总里，`A2-order-dyn` 的 final success 基本追平 SeqTrack baseline，final precision 高于 baseline，因此它曾是最清楚的真实时间正向信号。但 2026-07-08 整理的 5 次复核改变了当前判断力度：`A2-order-dyn` seed43 崩坏到 23.64 / 23.77，seed44 只有 46.90 / 52.62；`A2-order-dyn+TWC w0.01` 仍只有 22.88 / 24.27；`A3-conf-res` best-e14 复测只有 28.06 / 37.70，没有复现旧 62.04 / 76.30 高点。现在的主线应从“证明 A2 已稳定有效”改为“先建立 variable-rate 问题设置，再围绕 residual dynamics 做 seed 稳定性、checkpoint 对齐和机制诊断”。

### 0.1 当前能说和不能说

当前能说：

- 真实时间方向没有被否定；失败主要来自不合适的注入方式。
- SeqTrack3D 主干对 order-time token 语义敏感，直接替换为 raw real-time token 不稳定。
- `A2-order-dyn` 是旧 60ep seed42 下最强正向信号，说明真实 `delta_t/current_delta_t` 作为 dynamics prior 有潜力。
- 标准 fixed-step 整体 final 不是当前最有把握的涨点场景；variable-rate、long-gap、sparse / re-appearance 子集更适合证明真实时间的价值。
- `A2-order-dyn-cand1` 在当前 60 epoch 协议下明显退化，不支持简单去掉非 0 candidate。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本持平，final precision 小幅更高，说明小权重 displacement loss 不伤主线，但也不是主要解释。
- 旧 active `A1-order+TWC` 曾观察到 final precision +3.24，但该 run 的 nonzero candidate 坐标不共享，不能作为 TWC 正向证据。
- 旧 active `A2-order-dyn+TWC` 及 `twc_weight=0.01` 曾明显退化，但同样受坐标污染，不能作为 TWC 与 dynamics 冲突的证据。
- `A3-order-gate-safe` 比旧 P5 full 安全，但相对 A2-order-dyn final success / precision 仍下降 -2.64 / -8.45。
- `A3-order-conf-res-gate` 旧汇总 best checkpoint 很高（success 62.04 / precision 76.30），但最新 best-e14 复测只有 28.06 / 37.70，暂时不能把旧 best 当作确认收益。
- corrected A1+TWC seed42 在坐标修复后形成 +1.49/+5.03 的配置级正信号，且两路 anchor/current XYZ gap 均为 0。
- HTV 六组说明旧 feature-concat dynamics 的效果依赖 protocol：random20 为正，gap1124/burst-drop 为负。
- TrajTrack 论文的“历史轨迹 proposal + local/global proposal agreement”值得借鉴，但当前本地 GT-assisted evaluator 的高分不能进入公平主表。

当前不能说：

- 不能说完整 CT-SeqTrack full model 已经稳定超过 SeqTrack3D。
- 不能说 corrected-TWC 已稳定有效；当前只有 A1 seed42，且 baseline 不是同代码提交的严格配对。
- 不能说 gate 已经无效，因为 gate-safe 比旧 P5 full 安全，conf-res 又出现很高 best；但也不能说 gate 已经稳定有效。
- 不能按 `A3-order-conf-res-gate` 旧 best 下正向结论，因为最新复测未复现。
- 不能说 candidate noise 已被彻底排除，因为 `cand1` 只有 `num_candidates=4` 实验约 1/4 的 optimizer step，且还缺少 candidate 分桶日志。
- 不能说 displacement 监督已经是必要模块；目前它只是一个小幅、温和的正向/不伤信号。
- 不能只靠普通 fixed-step benchmark 讲论文成功；如果没有 variable-rate / HTV 协议和分桶收益，真实时间贡献会显得证据不足。
- 不能把 TrajTrack 本地 64.94 / 79.07 写成公平结果，也不能把其与 SeqTrack3D 的算术差值写成方法增益。

## 1. 第一轮：Baseline vs P5 full

比较：

```text
SeqTrack baseline
vs
CT-SeqTrack P5 full
```

P5 full 同时打开：

```text
real timestamp + DynamicsEncoder + ObservabilityGate
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
P5 full final:          success 31.19, precision 31.89
```

说明：

- P5 full 明显退化，尤其 final precision 掉得很重。
- 但这不能说明 timestamp-native 思路失败，因为这一轮把 real time、dynamics、gate 一次性混在一起。
- P5 full 的 best precision 曾接近 baseline，说明模型不是完全学不到定位，问题更像后期递归跟踪不稳定。

下一步由此产生：

```text
必须拆开 A1 / A2 / A3 消融，先定位退化来自真实时间主链路、dynamics，还是 gate。
```

## 2. 第二轮：A1-raw 和 A2 Dynamics

比较：

```text
SeqTrack baseline
vs
A1 CT-base: real timestamp, no dynamics, no gate
vs
A2 Dynamics: real timestamp + DynamicsEncoder, no gate
```

关键结果：

```text
A1-raw final: success 28.28, precision 27.43
A2 raw-dyn final: success 45.27, precision 58.83
```

说明：

- `A1-raw` 大幅退化，说明直接把真实秒数输入主干时间通道非常危险。
- `A2 raw-dyn` 相比 `A1-raw` 明显恢复，尤其 precision 接近 baseline final。
- 这支持一个判断：dynamics 分支本身有作用，但 raw real-time 主干路径有明显问题。

下一步由此产生：

```text
先诊断 A1 的时间输入形式，确认是不是 raw 秒数尺度或注入位置导致崩坏。
```

## 3. 第三轮：A1 时间编码诊断

比较：

```text
A1-raw
A1-pseudo
A1-MLP
A1-Fourier
SeqTrack baseline
```

关键结果：

```text
A1-pseudo final:  success 48.34, precision 52.25
A1-MLP final:     success 27.44, precision 26.28
A1-Fourier final: success 30.72, precision 29.82
```

说明：

- `A1-pseudo` 明显接近 baseline，说明 CT 代码主链路不是完全坏掉。
- `A1-MLP` 和 `A1-Fourier` 没有救回 real-time A1。
- 问题不只是“raw 秒数太大”，也不是简单换一个 scalar-preserving time encoding 就能解决。

下一步由此产生：

```text
不要继续在 MLP/Fourier 上堆复杂编码；
需要检查真实时间注入主干的位置和语义。
```

## 4. 第四轮：scaled real time

比较：

```text
SeqTrack baseline
vs
A1-scaled
vs
A2-scaled-dyn
```

关键结果：

```text
A1-scaled final:     success 31.33, precision 31.22
A2-scaled-dyn final: success 29.41, precision 31.51
```

说明：

- 把真实时间缩放回接近伪时间数值范围，并没有修复 A1。
- `A2-scaled-dyn` 也没有带来稳定收益。
- 这进一步说明：问题不只是时间数值尺度，而是主干分支对时间 token 的语义很敏感。

下一步由此产生：

```text
主干不再强行使用 real-time token；
改为恢复 SeqTrack3D 原本的 order-time 语义。
```

## 5. 第五轮：order-time 主干恢复

比较：

```text
SeqTrack baseline
A1-pseudo
A1-order
A2-order-dyn
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
A1-order final:          success 51.23, precision 57.86
A2-order-dyn final:      success 50.96, precision 63.31
```

说明：

- `A1-order` 基本修复了 A1-raw / A1-scaled 的崩坏，说明主干确实需要保留 order-time token 语义。
- `A2-order-dyn` 在 final success 上基本追平 baseline，在 final precision 上超过 baseline。
- 这给当前论文主线提供了最清楚的正向证据：真实时间更适合进入 dynamics prior，而不是直接替换主干时间 token。

当前结论：

```text
A2-order-dyn 是当前最值得作为主线继续推进的配置。
```

## 6. 第六轮：cand1 / displacement 诊断

比较：

```text
SeqTrack baseline
A2-order-dyn
A2-order-dyn-cand1
A2-order-dyn-disp
```

关键结果：

```text
SeqTrack baseline final:    success 50.99, precision 59.96
A2-order-dyn final:         success 50.96, precision 63.31
A2-order-dyn-cand1 final:   success 26.68, precision 24.50
A2-order-dyn-disp final:    success 50.54, precision 63.85
```

best 结果：

```text
A2-order-dyn-cand1 best: success 41.99 epoch 10, precision 54.62 epoch 5
A2-order-dyn-disp best:  success 52.44 epoch 10, precision 64.81 epoch 40
```

重要注意：

```text
cand1 的 num_candidates=1 会把每个 epoch 的训练 batch 数减少约 4 倍。
因此 cand1 60 epoch 的 final_step 只有 18899，
而 A2-order-dyn / disp 60 epoch 的 final_step 是 75719。
所以 cand1 与 A2-order-dyn 不是严格 optimizer-step 对齐。
```

结果解读：

- `A2-order-dyn-cand1` 在前 5-10 epoch 还有可用信号，但随后 success 和 precision 都明显塌到 20 多分。当前 60 epoch 协议下，它不支持“直接移除非 0 candidate 可以清理 dynamics”的假设。
- cand1 的退化至少有两个可能来源：一是移除 multi-candidate 后训练鲁棒性下降；二是 optimizer step 只有原来的约 1/4，学习率按 epoch 衰减导致有效训练不足。若要严格判断 candidate noise，后续需要 `cand1-240ep` 或等 step 版本。
- `A2-order-dyn-disp` 与 `A2-order-dyn` 基本同水平：final success 低 0.42，final precision 高 0.53；best success 高 0.90，best precision 高 1.24。
- 这说明 `dynamics_displacement_weight=0.01` 没有破坏主线，并且对 precision 有一点温和正向信号；但幅度还不足以把 displacement loss 作为核心贡献。

当前决策：

```text
保留 multi-candidate 训练。
displacement loss 可以作为小权重稳定项或诊断项继续观察，
但当前主线贡献仍应放在 order-time main branch + real-time DynamicsEncoder。
```

图表和完整表格：

```text
compare_results/cand1_disp_dynamics_curves.png
compare_results/cand1_disp_dynamics_success_curve.png
compare_results/cand1_disp_dynamics_precision_curve.png
compare_results/cand1_disp_dynamics_best_final_summary.png
compare_results/cand1_disp_dynamics_metrics_points.csv
compare_results/cand1_disp_dynamics_metrics_summary.csv
compare_results/cand1_disp_dynamics_comparison.md
```

## 7. 第七轮：旧 order+TWC 诊断（inactive）

比较：

```text
SeqTrack baseline
A1-order
A1-order+TWC
A2-order-dyn
A2-order-dyn+TWC
```

关键结果：

```text
A1-order final:          success 51.23, precision 57.86
A1-order+TWC final:      success 45.61, precision 50.77
A2-order-dyn final:      success 50.96, precision 63.31
A2-order-dyn+TWC final:  success 38.27, precision 38.85
```

关键诊断：

```text
两组旧 order+TWC 的 loss_twc / twc_valid_ratio / twc_center_gap / twc_angle_gap
全程为 0。也就是说，这一轮训练没有真正施加 TWC consistency 项。

旧结果也使用 num_candidates=1，60 epoch final_step 只有 18899，
不是 active-TWC，也不是严格 optimizer-step 对齐结果。
```

结果解读：

- 旧 `A1-order+TWC` 和 `A2-order-dyn+TWC` 的下降是真实观察，但不能据此判断 TWC 机制本身有害，因为 TWC loss 没有激活。
- 这轮的价值是暴露了 TWC validity / paired-view 训练协议问题。
- validity 修复后已经重跑，正式 active-TWC 结论见下一节。

## 8. 第八轮：旧 active TWC / gate-safe / conf-res（TWC 归因已失效）

> 2026-07-11 更正：本节 TWC run 的 validity mask 确实激活，但 nonzero candidate 下 A/B 不共享坐标系。指标和 loss 日志只作历史记录，不能再用来判断 TWC 的收益、损害或它与 dynamics 的兼容性；gate 结果不受这个 sampler 问题影响。

比较：

```text
SeqTrack baseline
A1-order
A2-order-dyn
A1-order+TWC
A2-order-dyn+TWC
A3-order-gate-safe
A3-order-conf-res-gate
```

关键结果：

```text
SeqTrack baseline final:      success 50.99, precision 59.96
A1-order final:               success 51.23, precision 57.86
A2-order-dyn final:           success 50.96, precision 63.31
A1-order+TWC final:           success 51.16, precision 61.10
A2-order-dyn+TWC final:       success 28.23, precision 32.04
A3-order-gate-safe final:     success 48.32, precision 54.87
A3-order-conf-res-gate final: success 31.17, precision 30.92
```

best 结果：

```text
A1-order+TWC best:           success 53.16, precision 63.35
A2-order-dyn+TWC best:       success 45.24, precision 57.43
A3-order-gate-safe best:     success 50.99, precision 60.17
A3-order-conf-res-gate best: success 62.04, precision 76.30
```

关键诊断：

```text
A1-order+TWC:
  twc_valid_ratio mean 0.750, tail1000 0.753
  loss_twc tail1000 0.0081
  vs A1-order final success -0.07, precision +3.24

A2-order-dyn+TWC:
  twc_valid_ratio mean 0.750, tail1000 0.750
  loss_twc tail1000 0.0077
  vs A2-order-dyn final success -22.73, precision -31.28

A3-order-gate-safe:
  obs_alpha_dyn_mean mean 0.127, tail1000 0.116
  vs A2-order-dyn final success -2.64, precision -8.45

A3-order-conf-res-gate:
  obs_alpha_dyn_raw_mean mean 0.493
  obs_alpha_dyn_clamped_mean mean 0.181
  obs_dyn_residual_norm mean 0.0315
  best-final gap: success 30.86, precision 45.38
```

结果解读：

- 当时只修复了 validity mask，没有修复 nonzero candidate 的共享坐标；因此 A1 的 precision-positive 和 A2 的后期崩坏都不能归因给 TWC。
- 后续必须先用 corrected sampler 复验 candidate 1/2/3、dataset length 和 `coordinate_anchor`，再讨论权重、warmup 或与 dynamics 的组合。
- `A3-order-gate-safe` 相比旧 P5 full 安全很多，但仍低于 A2-order-dyn。它说明强融合问题被缓解，却还没有证明 gate 能带来最终收益。
- `A3-order-conf-res-gate` 出现非常高的旧 best checkpoint，但 last checkpoint 崩坏。后续 2026-07-08 的 best-e14 复测只有 28.06 / 37.70，旧 best 暂时不能作为确认收益。

当时决策：

```text
当时暂以 A2-order-dyn 作为主线配置。
当时把 TWC 视为 A1 上的 precision-positive 候选；2026-07-11 坐标审计后该判断已撤回。
gate-safe / conf-res 暂时都不能作为最终收益模块；conf-res 先复测 best checkpoint。
```

注意：2026-07-08 五次复核后，`A2-order-dyn` 已从“当前主配置”降级为“最值得诊断的真实时间使用方式”。后续优先级转向 variable-rate 协议和 residual dynamics。

图表和完整表格：

```text
compare_results/twc_gate_ablation_curves.png
compare_results/twc_gate_ablation_success_curve.png
compare_results/twc_gate_ablation_precision_curve.png
compare_results/twc_gate_ablation_best_final_summary.png
compare_results/twc_gate_ablation_twc_diagnostics.png
compare_results/twc_gate_ablation_gate_diagnostics.png
compare_results/twc_gate_ablation_metrics_points.csv
compare_results/twc_gate_ablation_metrics_summary.csv
compare_results/twc_gate_ablation_diagnostics_summary.csv
compare_results/twc_gate_ablation_comparison.md
```

## 9. 第九轮：2026-07-08 五次稳定性复核

比较：

```text
A3-conf-res best-e14 retest
A2-order-dyn seed43
A2-order-dyn seed44
A2-order-dyn+TWC w0.01 seed42
A3-conf-res rerun seed42
```

关键结果：

```text
A3-conf-res best-e14 retest:        success 28.06, precision 37.70
A2-order-dyn seed43 final:          success 23.64, precision 23.77
A2-order-dyn seed44 final:          success 46.90, precision 52.62
A2-order-dyn+TWC w0.01 final:       success 22.88, precision 24.27
A3-conf-res rerun seed42 final:     success 32.11, precision 31.87
```

关键诊断：

```text
A2-order-dyn+TWC w0.01:
  loss_twc tail1000 mean 0.0083
  twc_valid_ratio tail1000 mean 0.7541
  twc_center_gap tail1000 mean 0.1748
  twc_angle_gap tail1000 mean 0.0103

A3-conf-res rerun:
  obs_alpha_dyn_mean tail1000 mean 0.4988
  obs_alpha_dyn_clamped_mean tail1000 mean 0.1810
  obs_dyn_residual_norm tail1000 mean 0.0314
```

说明：

- `A3-conf-res best-e14 retest` 没有复现旧汇总里的 62.04 / 76.30，高 best 暂时应视为未确认信号。
- `A2-order-dyn` seed43 / seed44 差异很大，说明旧 seed42 的 precision-positive 结果不能直接当成稳定结论。
- `A2-order-dyn+TWC w0.01` 当时只有 validity mask 正常；后续坐标审计证明它仍受 nonzero candidate 坐标污染，不能据此归因权重或 TWC 机制。
- `A3-conf-res rerun` 仍低，gate/conf-res 现在应先转向评测路径、alpha/residual 行为和困难分桶诊断。

归档文件：

```text
compare_results/reports/latest_5runs_comparison.md
compare_results/data/latest_5runs_metrics_points.csv
compare_results/data/latest_5runs_metrics_summary.csv
compare_results/data/latest_5runs_diagnostics_summary.csv
compare_results/data/latest_5runs_hparams_summary.csv
compare_results/figures/line_charts/latest_5runs_success_curve.svg
compare_results/figures/line_charts/latest_5runs_precision_curve.svg
compare_results/figures/bar_charts/latest_5runs_best_final_summary.svg
compare_results/figures/diagnostics/latest_5runs_diagnostics_tail_mean.svg
```

## 10. 2026-07-16：corrected-TWC、HTV 与 TrajTrack 参考

### 10.1 Corrected-TWC seed42

| family | baseline final | corrected-TWC final | final delta | late-mean delta |
| --- | --- | --- | --- | --- |
| A1 | 51.23 / 57.86 | 52.72 / 62.89 | +1.49 / +5.03 | +0.99 / +2.67 |
| A2 | 50.96 / 63.31 | 50.04 / 61.25 | -0.93 / -2.07 | -1.33 / -2.53 |

两组 corrected run 均完成 60 epoch、75720 optimizer steps，TWC anchor gap max 和 current XYZ gap max 都为 0。A1 值得补 seed43/44；A2 不建议继续组合 TWC。旧 baseline 没有 commit 记录，因此下一轮必须在同一提交上重跑 paired baseline。

完整报告：`compare_results/reports/corrected_twc_seed42_comparison.md`。

### 10.2 HTV 六组 seed42

| protocol | A2-A1 Success final | A2-A1 Precision final | 解释 |
| --- | ---: | ---: | --- |
| gap1124 | -4.01 | -9.55 | epoch10 曾有高点，后期明显崩落 |
| burst-drop | -7.45 | -14.40 | 强不规则条件下稳定低于 A1 |
| random20 | +9.09 | +14.23 | 温和随机丢帧下 final/late mean 都为正 |

六组都是确定性 seed42 配对，但没有冻结 manifest，且只在 mini_val 上评估。它们适合筛选假设，不适合统计性结论。最重要的机制问题不是继续扩大 feature concat，而是检查：nonzero candidate 是否制造伪速度、目标是否离开 search crop、dynamics proposal 何时优于 observation proposal。

完整报告：`compare_results/reports/htv_6runs_comparison.md`。

### 10.3 TrajTrack evaluator 边界

aligned seed42 运行预算与 plain SeqTrack3D 基本一致，但 evaluator 并不等价：TrajTrack 当前 `pre_w_refine()` 读取当前帧 GT，先用 GT overlap 判断是否 refinement，再用 GT overlap 从 proposals 中选最大者。因此 64.94 / 79.07 是 GT-assisted 参考，不是公平在线结果。

下一步应固定 epoch60 checkpoint，分别评测：

1. `pre_wo_refine()`：GT-free local proposal baseline。
2. paper-aligned GT-free refinement：只用 local/global proposal IoU、预测置信度、点数等推理时可得量。
3. 当前 `pre_w_refine()`：只保留为 oracle upper-bound 诊断。

完整报告：`compare_results/reports/trajtrack_gt_assisted_vs_plain_seqtrack_reference.md`。

## 11. 当前各实验共同说明了什么

可以支持的结论：

- 真实时间方向没有被否定，失败主要来自不合适的注入方式。
- SeqTrack3D 主干对原始 order-time token 很敏感，直接替换为 real-time token 会破坏已学到的时间/顺序语义。
- DynamicsEncoder 仍是当前最有潜力的真实时间使用方式，但 feature-concat 在强 gap/burst 下失败；下一轮应验证 observation-first residual，而不是把旧 A2 当主方法。
- 当前 `cand1` 结果不支持简单移除非 0 candidate；multi-candidate 训练暂时应保留。
- 小权重 displacement 辅助监督不伤主线，并给 precision 带来温和正向信号，但不是主要收益来源。
- 旧 TWC 只有 validity mask 生效，坐标共享仍有缺陷；旧 A1 正向和 A2 负向信号均已撤回。
- corrected-TWC 的共享 offset、`coordinate_anchor` fail-fast 和 optimizer-step 对齐已实现；A1 seed42 为正、A2 为负，但仍缺同提交 baseline 与多 seed。
- P5 full 旧结果不能作为最终 gate 结论；gate-safe 比旧 P5 full 安全，但仍低于 A2-order-dyn。
- conf-res 旧 best checkpoint 未被最新 best-e14 复测确认；当前不能按旧 best 写正向收益。
- corrected-TWC 如果继续，应先只在 `A1-order` 上做最小重跑；gate 仍只做诊断，不与 residual 同时启用。
- TrajTrack 当前高分含 GT oracle；它只能提示 trajectory proposal 的潜力，不能证明公平收益。

还不能说明的事情：

- 还不能说完整 CT-SeqTrack 已经稳定超过 SeqTrack3D。
- 还不能说 TWC 已稳定有效或能与 dynamics 组合；当前只有 A1 seed42 正信号，而 A2 seed42 为负。
- 还不能说 gate 有效；gate-safe final 不够好，conf-res best 复测未确认，但仍可做困难样本诊断。
- 还不能彻底解释非 0 candidate 是否污染 dynamics，因为 cand1 没有与原 A2 做 optimizer-step 对齐，也缺少 candidate 分桶日志。
- 还不能说 displacement loss 是必要模块，因为当前只是小幅、不决定性的正向信号。

## 12. 接下来应该做什么

当前优先顺序：

```text
1. 固定 TrajTrack epoch60 checkpoint，先完成 `pre_wo_refine()` 和 paper-aligned GT-free refinement，消除外部参考中的 evaluator 混杂。
2. 在服务器完成 bounded residual 的 standard/gap1124/burst-drop 真实 batch / forward / loss / 2-step 验收。
3. 冻结 manifest，对 residual 做 true-dt / fixed-dt / shuffled-dt、seed42/43/44 的同容量因果对照。
4. corrected A1+TWC 补 seed43/44，并用同一代码提交重跑 paired A1 baseline；不继续 A2+TWC。
5. 补 candidate0/nonzero、target-in-crop、delta_t/sparse/displacement 和 observation-vs-dynamics proposal 分桶。
6. 只有 residual 的 true-dt 因果证据成立后，才升级为轻量 bbox-only time-conditioned trajectory proposal + GT-free proposal agreement。
```

可选复核：

```text
A2-order-dyn-cand1-240ep
```

作用是让 `num_candidates=1` 的 cand1 与 `num_candidates=4` 的 A2-order-dyn 做 optimizer-step 对齐。只有这个版本也退化，才能更干净地说明移除非 0 candidate 本身有问题。

### 12.1 A1-order+TWC

目的：

```text
先在没有 dynamics 的 order-time 主干上检查 TWC 是否有效。
```

看什么：

- 旧 run 虽然 `twc_valid_ratio > 0` 且 final precision +3.24，但坐标共享缺陷使该增益无法归因。
- 先用 corrected sampler、candidate 1/2/3 和相同 optimizer steps 重跑原权重；不要先做权重网格。
- 同时直接报告同一 endpoint 下不同采样路径的预测方差是否下降。

### 12.2 A2-order-dyn+TWC

目的：

```text
检查真实时间 dynamics prior 和 TWC 是否互补。
```

看什么：

- 旧 active / w0.01 退化都受坐标污染，不能判断 TWC 与 dynamics 是否互补。
- corrected A1+TWC 没有形成稳定证据前，不重跑 A2+TWC，也不与 gate/residual 混合。

### 12.2b A2-residual-dyn

目的：

```text
检查真实时间 dynamics prior 是否能以更保守的 residual correction 形式稳定发挥作用。
```

当前实现：

- 主干保持 `A1-order`，不要把 raw real-time token 放回主干。
- `DynamicsEncoder` 继续预测 `velocity_pred / dynamics_displacement_pred`。
- 最终中心预测采用小幅 residual：`center = obs_center + scale * alpha * clamp(dyn_disp)`。
- 已支持 `scale / max_norm / max_alpha / warmup / long_gap_only / sparse_only`，gate 近零初始化且受 `dynamics_valid` 约束。
- 先验收默认 `scale=0.1, max_norm=1.0, max_alpha=0.2, warmup=5`；默认配置没有稳定信号前不做大网格。
- 与当前 `A2-order-dyn` feature-concat 版本做同 seed 对照。

判断标准：

- 如果普通 final 只持平，但 long-gap / sparse bin 稳定提升，可以作为更强论文证据。
- 如果 residual 仍 seed collapse，问题更可能在 dynamics 监督质量、candidate history 或真实 `delta_t` 信号不足。

### 12.3 A3-order-gate-safe / conf-res

目的：

```text
在干净的 order-time 主干上重新测试保守 gate。
```

看什么：

- gate-safe 比旧 P5 full 稳，但 final 仍低于 A2-order-dyn。
- conf-res best-e14 复测没有复现旧 best；先核对旧汇总路径，再回到更保守的 alpha / residual 约束。
- 分桶分析仍有价值，但它现在用于解释 gate 行为，而不是证明 gate 已经有效。

## 13. 后续需要补的诊断

建议增加 dynamics 诊断日志：

```text
candidate_id
candidate0_loss_velocity
candidate_nonzero_loss_velocity
velocity_label_norm
velocity_pred_norm
dynamics_displacement_norm
dynamics_valid_ratio
```

作用：

- 直接判断非 0 candidate 是否让 dynamics 监督变脏。
- 判断 velocity prediction 是否爆炸、塌缩或量级不匹配。
- 给 cand1 / disp 的结果提供机制解释，而不是只看 success 和 precision。

## 14. 当前论文叙事建议

暂时不要写：

```text
CT-SeqTrack full model outperforms SeqTrack3D.
```

更稳的写法：

```text
We find that directly replacing SeqTrack3D's order-time tokens with raw
timestamps destabilizes the main branch. A more stable design is to preserve
the order-time semantics in the SeqTrack3D backbone while injecting real
delta_t through a conservative timestamp-conditioned residual dynamics prior,
especially under variable-rate and long-gap tracking.
```

中文主线：

```text
真实 timestamp 改变历史状态的物理含义，但它不应该粗暴替换 SeqTrack3D
主干里的顺序 token。当前最稳的 CT-SeqTrack 路线是：主干保留 order-time，
先用 variable-rate / long-gap / sparse 协议把问题设置做清楚，再让真实 delta_t
进入保守 residual dynamics prior。A2 的 seed 稳定性、TWC 与 dynamics 的组合、
以及 gate / conf-res 的 checkpoint 可复现性都还没有解决。
```
