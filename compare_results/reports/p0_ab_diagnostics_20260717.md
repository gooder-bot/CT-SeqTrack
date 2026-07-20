# P0-A / P0-B Diagnostics — 2026-07-17

本报告汇总服务器回传的 search-crop reachability 与 bounded-residual 诊断。它们都是 `mini_train` 上的机制检查，不是跟踪性能结果，也不能替代冻结协议后的 `true/fixed/shuffled-dt` 对照。

## 1. 回传完整性

| 项目 | 当前状态 | 样本量 | 结论边界 |
| --- | --- | ---: | --- |
| P0-B standard smoke | 完成 | 220 endpoints | 只验证脚本 |
| P0-B standard full-history | 完成 | 4246 endpoints | 可判断 standard 的 oracle crop 可达性 |
| P0-B gap1124 full-history | 完成 | 2127 endpoints | 可判断 gap-pattern 的 oracle crop 可达性 |
| P0-B burst-drop full-history | 完成 | 2098 endpoints | 可判断 burst-drop 的 oracle crop 可达性 |
| P0-A standard warmup | 完成 | 2 batches / 32 samples | 可验证 warmup 行为 |
| P0-A standard active | 部分完成 | 64 batches / 1024 samples | 可诊断默认量级；不是完整训练 split |
| P0-A gap1124 / burst-drop | 未完成 | 0 | 不能做强 gap residual 结论 |
| P0-A 2-step optimizer smoke | 未完成 | 0 | 尚未验证参数更新后的两步稳定性 |

因此，P0-B 的三协议 oracle 诊断已经完整；P0-A 已足以否定“当前默认 residual 有可见作用”，但还不足以校准正式 bound。

## 2. P0-B：三协议 crop reachability

所有模式都使用当前帧 GT 统计目标点。`base/expanded` 以前一帧 GT 框为 anchor，`cv_recenter` 用 GT 历史做常速度中心外推。因此这些结果是乐观的 oracle 可达性诊断，不是在线跟踪结果。

### 2.1 协议强度与样本可比性

| protocol | endpoints | visible endpoints | current `delta_t` mean / P50 / P95 | displacement mean / P50 / P95 / max |
| --- | ---: | ---: | ---: | ---: |
| standard | 4246 | 2996 (70.56%) | 0.497 / 0.500 / 0.549 s | 1.745 / 0.080 / 7.157 / 15.593 m |
| gap1124 | 2127 | 1503 (70.66%) | 0.987 / 0.950 / 2.001 s | 3.433 / 0.137 / 16.892 / 36.615 m |
| burst-drop | 2098 | 1478 (70.45%) | 1.001 / 0.500 / 2.001 s | 3.478 / 0.137 / 18.613 / 37.668 m |

强协议把平均时间间隔约翻倍，并把位移 P95 从 7.16 m 推到 16.89/18.61 m。三协议的目标可见比例都约为 70.5%，所以后续 recall 差异不是由“当前 GT 框本身无点”的比例变化解释。需要注意，三个协议的 endpoint 数量和 current frame 并不完全相同；这里是同一 `mini_train` 来源上的协议级机制比较，不是逐 endpoint 配对的因果实验。

### 2.2 三种 crop mode 的总体结果

| protocol / crop mode | center outside | visible endpoint retains any target point | mean target-point recall | crop points mean / P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| standard / base | 15.97% | 90.72% | 85.41% | 285 / 16 / 1864 |
| standard / 2x expanded | 0.14% | 99.93% | 99.57% | 1622 / 123 / 14864 |
| standard / GT-history CV | 0.12% | 99.97% | 99.95% | 290 / 17 / 1867 |
| gap1124 / base | 25.20% | 80.44% | 76.78% | 312 / 15 / 1952 |
| gap1124 / 2x expanded | 12.27% | 90.35% | 89.08% | 1650 / 119 / 15463 |
| gap1124 / GT-history CV | 1.18% | 99.33% | 98.96% | 286 / 16 / 1851 |
| burst-drop / base | 23.83% | 81.19% | 77.72% | 331 / 16 / 1993 |
| burst-drop / 2x expanded | 12.30% | 88.84% | 87.65% | 1719 / 110 / 15718 |
| burst-drop / GT-history CV | 0.91% | 99.46% | 99.05% | 290 / 17 / 1836 |

对可见目标，base crop 在 standard/gap/burst 中分别完全丢失 278/294/278 个 endpoint；GT-history CV 后只剩 1/10/8 个。2x expanded 的平均点数分别是 base 的 5.68/5.29/5.19 倍，但在强协议下仍有约 12.3% 中心越界，说明固定扩大搜索区既昂贵又不足。CV recenter 的平均点数是 base 的 1.02/0.92/0.88 倍，强信号来自移动中心而不是吞入更多背景。

### 2.3 按真实位移分桶

| protocol, displacement >4 m | endpoints | base outside / recall | 2x expanded outside / recall | GT-history CV outside / recall |
| --- | ---: | ---: | ---: | ---: |
| standard | 960 | 70.63% / 45.20% | 0.63% / 98.11% | 0.31% / 99.94% |
| gap1124 | 617 | 86.87% / 23.10% | 42.30% / 62.09% | 3.89% / 96.40% |
| burst-drop | 592 | 84.46% / 26.85% | 43.58% / 56.53% | 3.04% / 96.69% |

对 `>4 m` 的真正困难样本，强协议下 base recall 只剩 23%–27%，而 2x expanded 也只能达到 57%–62%。CV oracle 仍保持约 96.4%–96.7%，这比总体均值更能说明时间条件轨迹外推的潜在价值。反过来，单看 `delta_t` 不是充分条件：慢速目标即使时间间隔较大仍可能留在 crop 内，因此正式方法应建模 `velocity × delta_t`/轨迹状态，而不是只按 `delta_t` 放大 crop 或调 gate。

base center-outside 失败分别覆盖 standard 的 75/260、gap1124 的 106/243、burst-drop 的 105/243 条 tracklet；强协议中 top-10 失败 tracklet 只解释 29.10%/32.40% 的越界样本，问题比 standard 更分散，不是少数异常序列造成。

### 2.4 P0-B 判断

即使使用前一帧 GT 框这一乐观 anchor，三协议的高速位移样本仍会在模型 forward 前离开 base crop。后续 P0-B2 已用同一 A1 checkpoint 完成 predicted-history 诊断，确认预测误差会累积并形成灾难性长尾。因此：

- crop 前可达性已被证明是主要瓶颈之一；
- 强 gap/burst 已量化确认 crop 前瓶颈，并使固定 2x expanded 也明显失效；
- 直接放大 crop 不是首选，因为平均背景点开销约为 5 倍且困难样本 recall 仍低；
- GT-history CV recenter 给出了强 oracle 上限，但 predicted-history CV 总体只提高 2.65–3.03 pp，未通过预注册门槛；
- always-on raw CV recenter 已 No-Go；后续只研究测试时可靠性控制的 dual-anchor 预防机制。详见 `p0b2_recursive_crop_reachability_20260717.md`。

## 3. P0-A：bounded residual 量级与梯度

### 3.1 warmup

standard warmup 运行加载了 A1-order checkpoint，320 个 observation key 匹配，新增 dynamics/gate 的 14 个 key 缺失符合预期。2 个 batch 中：

- applied ratio = 0；
- applied residual = 0；
- gate gradient = 0；
- observation head 输入维度 = 256，未拼接 `z_dyn`；
- loss/backward finite。

warmup 行为通过。

### 3.2 active residual（64 batches）

| metric | P50 | P75 | P95 | max / note |
| --- | ---: | ---: | ---: | --- |
| observation error norm | 0.213 m | 0.577 m | 3.838 m | max 7.519 m |
| target motion norm | 0.052 m | 0.215 m | 3.401 m | max 6.155 m |
| raw dynamics norm | 0.036 m | 0.038 m | 0.041 m | untrained branch |
| alpha | 2.0e-5 | 2.0e-5 | 2.0e-5 | gate bias -9.2102 |
| applied residual norm | 7.25e-8 m | 7.54e-8 m | 8.25e-8 m | max 9.03e-8 m |
| gate gradient norm | 4.00e-10 | 1.26e-8 | 3.74e-8 | 31/64 batches exactly 0 |
| encoder gradient norm | 2.30e-3 | 6.15e-3 | 3.94e-2 | finite |

默认理论上限是 0.02 m，但 observation error 的 P50/P75/P95 分别约为上限的 10.6/28.9/191.9 倍。更关键的是，近零 gate 使实际 P50 correction 只有 `7.25e-8 m`，约为 observation P50 error 的 `3.4e-7`；它在数值上等同于没有修改预测。

日志中的 `applied_ratio=1` 不能解释为 residual 有效，因为实现只用 `norm > 1e-8` 判定 applied，而实际 correction 仍只有约 `1e-7 m`。`clamp_ratio=0` 也只说明未训练 dynamics 输出约 3–4 cm，不能证明 1 m clamp 合理。

### 3.3 candidate 分桶

| candidate | n | observation error P50 | P75 | P95 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 255 | 0.193 m | 0.581 m | 3.389 m |
| 1 | 257 | 0.217 m | 0.673 m | 3.796 m |
| 2 | 255 | 0.215 m | 0.561 m | 4.295 m |
| 3 | 257 | 0.216 m | 0.526 m | 3.950 m |

candidate0 的中位误差略低，但四组都有很大长尾；现有 64-batch 证据不支持把问题简单归因于 nonzero candidate，也不支持据此删除 candidate augmentation。

### 3.4 P0-A 判断

当前默认 residual 通过了数值稳定性检查，但没有通过“非平凡修正幅度”验收：

- warmup 正确；
- active forward/loss/backward finite；
- encoder 可获得梯度；
- gate 近零初始化导致梯度和 correction 近乎消失；
- 2 cm cap 覆盖不了 observation error 的主要分位数；
- 但这些 observation error 混入了 out-of-crop 失败，不能直接据此把 residual bound 调到几十厘米或数米。

## 4. 研究路线决定

当前不应做的事情：

1. 不根据这 64 个 batch 直接放大 residual scale/bound；
2. 不启动当前默认 residual 的正式 seed42 训练；
3. 不把 expanded-crop oracle 收益写成真实时间收益；
4. 不把 GT-history CV recenter 写成 GT-free 在线结果。

P0-B2 已完成，下一步按以下顺序执行：

1. 扩展递归诊断，记录只依赖测试时信息的 confidence、foreground、empty fallback、CV shift 和 proposal agreement；
2. 以离线 GT 作为标签，评估这些信号预测漂移和 next-crop failure 的 AUROC、AUPRC 与 calibration；
3. 只有可靠性代理有效时，固定同一 A1 checkpoint 做无训练 active dual-anchor，验证是否延迟首次失控并缩短连续失败；
4. active 机制通过后才进入 P0-C 稳定 manifest 与 `true/fixed/shuffled` 时间因果矩阵；
5. 在预裁剪可达性改善后，再统计 reachable subset observation error，并决定 bounded residual 是否保留。

## 5. 数据来源

- `output/diagnostics/crop_reachability/standard_train/crop_reachability_summary.json`
- `output/diagnostics/crop_reachability/standard_train/crop_reachability_endpoints.csv`
- `output/diagnostics/crop_reachability/gap1124_train/crop_reachability_summary.json`
- `output/diagnostics/crop_reachability/gap1124_train/crop_reachability_endpoints.csv`
- `output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_summary.json`
- `output/diagnostics/crop_reachability/burst_drop_train/crop_reachability_endpoints.csv`
- `output/diagnostics/p0a_standard_warmup_summary.json`
- `output/diagnostics/p0a_standard_active_summary.json`
- `output/diagnostics/p0a_standard_active.jsonl`
- `logs/diagnostics/`
