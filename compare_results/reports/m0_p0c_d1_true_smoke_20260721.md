# M0 P0-C-D1 `true` 2-tracklet smoke 数据质量与结果审计

日期：2026-07-21
状态：**工程链路通过；M0 对照结论未形成**

## 1. 技术摘要

本次审计读取 `output/diagnostics/m0_endpoints/p0c_d1_true_smoke/` 中的 `m0_endpoints.csv`、`m0_summary.json` 和 `resolved_config.json`。数据来自 P0-C A2 frozen checkpoint、nuScenes mini_val、gap1124 manifest、`true` dynamics time mode，且仅限制为前 2 个 tracklet。

可以确认 exporter 已在服务器真实数据上完成逐 endpoint 输出，关键 identity/hash、数值有限性和 true-time 字段一致性均通过。不能确认 true time 优于 fixed 或 shuffled：本地没有另外两路，也没有 91-tracklet full 输出。当前结果只用于 smoke 与调试，不用于方法 promotion、paired effect 或论文主表。

审计还发现一项正式运行前必须修复的指标一致性问题：exporter aggregate Success 为 `83.0208`，CSV 重算为 `83.1250`，相差 `0.1042 pp`。根因是首帧 aggregate 使用 box backend 返回的实际 self-overlap，而旧 CSV 硬编码 `IoU=1.0`。本地 exporter 已改为复用实际首帧 IoU/距离；三路 smoke 和 full 需要同步后重跑。

## 2. 关键发现与证据

| 项目 | 结果 | 判断 |
| --- | ---: | --- |
| time mode | `true` | 仅有一个模式，不能做因果对照 |
| endpoint / tracklet | `24 / 2` | 足够做链路 smoke，不足以估计总体表现 |
| CSV 字段数 | `102` | endpoint、时间、crop、dynamics、path/provenance 字段已写出 |
| duplicate endpoint key | `0` | 通过；键为 `(tracklet_key, source_frame_index, frame_token)` |
| non-finite IoU / center error | `0 / 0` | 通过 |
| effective-real time 最大绝对差 | `0` | `true` 模式时间路由通过 |
| exporter Success / Precision | `83.0208 / 88.9583` | Success 与 CSV 重算不一致 |
| CSV-recomputed Success / Precision | `83.1250 / 88.9583` | 修复前以 CSV 可复算性为审计口径 |
| mean IoU / center error | `0.8275 / 0.2164 m` | 仅描述当前 smoke |
| empty fallback | `5 / 24` (`20.83%`) | 全部集中在一个 tracklet，需 full 验证 |
| `>2 m` failure | `0` | 两条轨迹内无长尾失败，不能外推 |

没有绘制性能图：单模式、2-tracklet 样本不适合用趋势图表达，表格更能避免把 smoke 误读为稳定性能比较。

### 2.1 Tracklet 级差异

| tracklet key | endpoints | empty fallback | mean IoU | mean error | max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8ce4fe54af77467d90c840465f69677f` | 15 | 5 | 0.8113 | 0.2339 m | 0.5011 m |
| `f4af7fd215ee47aa8b64bac0443d7be8` | 9 | 0 | 0.8545 | 0.1874 m | 0.3297 m |

5 次 fallback 均来自第一条轨迹，其中 3 次 search points 为 0、后两次为 1。fallback 与较低 tracklet 表现同向，但只有一条受影响轨迹，不能声称存在统计关系。它应作为 full run 的预注册诊断项继续检查。

### 2.2 Gap 与点数分桶（描述性）

22 个非首帧 endpoint 的真实 `delta_t`：mean `0.8887 s`、median `0.5261 s`、P95 `2.0006 s`、max `2.0515 s`。离线分桶结果如下：

| `delta_t` bin | n | Success | Precision | mean error | fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `<0.75 s` | 13 | 82.500 | 88.269 | 0.2193 m | 0.231 |
| `0.75–1 s` | 3 | 84.167 | 90.833 | 0.2061 m | 0.333 |
| `1–2 s` | 4 | 77.500 | 85.000 | 0.3034 m | 0.000 |
| `>=2 s` | 2 | 80.000 | 87.500 | 0.2559 m | 0.500 |

这些桶最少只有 2 个 endpoint，不能用于判断 long-gap 单调性或显著性。full 三路比较必须以 endpoint exact match 后的 paired delta 和 tracklet bootstrap 为主，而不是比较这些 smoke 均值。

## 3. 范围、数据与指标定义

- 输入：`p0c_d1_true_smoke/m0_endpoints.csv`、`m0_summary.json`、`resolved_config.json`。
- checkpoint SHA256：`b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad`。
- config SHA256：`69b801...d658`；resolved config SHA256：`bfae5a...5307`。
- gap1124 selection SHA256：`85e5603c...f9649f6f`；manifest content SHA256：`48f80593...b25285`。
- Success：与项目 tracking evaluator 相同的 IoU threshold curve 均值；Precision：center-error threshold curve 均值。
- empty fallback：search input 为空而未执行正常 forward 的 endpoint。
- failure：本次诊断阈值为 center error `>2 m`，连续失败阈值为 2 帧。

## 4. 方法与复核过程

1. 检查 CSV shape、字段、复合主键和单一 checkpoint/config hash。
2. 对 IoU、center error、时间字段进行有限数值与一致性检查。
3. 从 CSV 独立重算 Success/Precision、tracklet 统计、gap/point bins、fallback 与 failure streak。
4. 对比 `m0_summary.json` 的 exporter aggregate 与 CSV 重算。
5. 比较服务器记录的 exporter SHA 与当前本地脚本：服务器 SHA 对应当前脚本内容的 CRLF 版本，说明代码内容一致、仅换行符不同。

## 5. 数据质量结论

| 检查项 | 状态 | 影响 |
| --- | --- | --- |
| schema / endpoint 唯一性 / finite | PASS | 当前 true smoke 可用于工程诊断 |
| checkpoint/config/manifest identity | PASS | 输入身份可追踪 |
| true effective time == real time | PASS | true 路由符合预期 |
| CSV 与 aggregate Success 一致性 | FAIL（已修复代码） | 修复前输出不得与新输出混合做正式比较 |
| true/fixed/shuffled 完整性 | FAIL | 无法计算 paired time-control effect |
| full coverage | FAIL | 只有 2/91 tracklets，无法外推 |
| clean provenance | FAIL | `git.dirty=true` 且 exporter/summarizer 为 untracked；正式输出必须 clean |

总体判定：**适合保留为 smoke 证据，不适合进入方法结论。**

## 6. 局限性、不确定性与稳健性

- 2 个 tracklet 不能支持置信区间、显著性、gap 单调性或 fallback 风险归因。
- 只有 `true`，不能区分“模型对时间值有响应”和“正确物理时间对应关系带来收益”。
- 数据来自 dirty worktree；虽然记录了 commit、dirty status、checkpoint/config/manifest hash，仍不满足正式复现门槛。
- 当前 active crop target recall 仅对 10/22 个非首帧 endpoint 有定义，且均为 1.0；其余 endpoint 目标点本来就不可见，不能把缺失值当成 0。
- DynamicsEncoder 为 feature-concat 路径，部分 observation/residual 字段为空是模型契约所致，不是导出缺失。

## 7. 下一步

1. 只同步修复后的 `tools/export_m0_endpoints.py`（以及更新后的运行指南）；在服务器提交 clean code/config commit。
2. 重新运行 exporter/summarizer self-test，然后从 `true/fixed/shuffled` 三路 2-tracklet smoke 重新开始。
3. 三路 smoke 均通过后，先做 endpoint exact-match、checkpoint/hash、CSV-vs-summary 一致性检查；任一不一致都不进入 full。
4. 再运行 91-tracklet full 三路和 paired summarizer，得到 per-tracklet bootstrap、gap bins、首次失控、连续失败和 fallback 定位。
5. P0-C-D1 full 形成后再继续 A/B/C 四协议 frozen output；M1 目前仅做接口与 zero-init 等价性准备，不启动正式训练，M2 继续锁定。

## 8. 待回答问题

- fixed/shuffled 是否与 true 的 endpoint key/order 完全一致？
- CSV 与 exporter aggregate 在修复后三路是否精确一致（容忍浮点误差 `<=1e-10`）？
- 5 次 fallback 集中于单 tracklet 是偶然 smoke 现象，还是 full 中稳定集中在稀疏/长 gap endpoint？
- true 相对 fixed/shuffled 的 paired delta 是否只来自少数 tracklet，且 bootstrap 区间是否跨 0？
