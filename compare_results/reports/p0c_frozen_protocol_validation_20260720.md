# P0-C Frozen Protocol、Time-Control 与冻结性能判定

更新时间：2026-07-20

## 1. 决策结论

服务器在 clean tracked worktree、commit `343145d` 和 `seqtrack3d` 环境中完成了 role-specific gap1124 manifest、offline shuffled-dt manifest 和真实 nuScenes batch 三路不变量检查。工程判定为：

```text
P0C_FROZEN_PROTOCOL_ENGINEERING_PASS
```

随后同一个 standard-trained A2 seed42 60ep final checkpoint 完成了三路 inference-only 对照。性能判定为：

```text
NO_GO_P0C_A2_TRUE_DT_PROMOTION
```

`true-dt` 相对 fixed 仅为 `+0.438 Success / +0.523 Precision`，相对 shuffled 为 `-0.123 / +0.056`；没有同时优于两个负对照，也没有达到预注册的 `+0.5 / +1.0` 门槛。验证状态：**Ready to share（仅限当前 checkpoint、gap1124、seed42 的预注册决策）**。

## 2. 来源与运行边界

服务器环境：

```text
repository: /home/lishengjie/study/lcyu/CT-SeqTrack
commit: 343145d (protocol: implement frozen P0-C cadence controls)
environment: /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python
tracked worktree before build: clean
dataset/version/split: nuScenes mini / v1.0-mini / mini_val
protocol: gap_pattern [1, 1, 2, 4], seed42, max_gap5
```

服务器保存的证据入口：

```text
output/p0c-protocol-build/build_val_manifest.log
output/p0c-protocol-build/build_test_manifest.log
output/p0c-protocol-build/check_test_manifest.log
output/p0c-protocol-build/build_shuffled_dt.log
output/p0c-protocol-build/check_time_controls.log
protocols/manifests/nuscenes_mini_val_gap1124_seed42.json
protocols/manifests/nuscenes_mini_test_gap1124_seed42.json
protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json
```

本地结构化摘录：

```text
compare_results/data/p0c_manifest_validation_20260720.csv
compare_results/data/p0c_time_control_sample_20260720.csv
compare_results/data/p0c_gap1124_time_control_metrics_20260720.csv
```

完整服务器包已归档到 `server_results/p0c_gap1124_triplet_20260720_172527/`；传输包 SHA256 为 `ab64e9507d41ad699932bd0e240edeb5e751c1ea3c7e75fc996f6519d15ac333`。三个原始 manifest、三份 console/provenance 和 TensorBoard events 均已收到。

## 3. Frozen cadence manifest

val/test 使用同一 split 与同一 gap1124 参数，数据规模完全一致：

| role | tracklets | frames | dropped frames | drop ratio | min length | mean length | content SHA256 | file SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| val | 91 / 106 | 1257 / 2285 | 1028 | 0.449891 | 6 | 13.8132 | `445402e0...64bc0` | `d2a41a11...060eb` |
| test | 91 / 106 | 1257 / 2285 | 1028 | 0.449891 | 6 | 13.8132 | `48f80593...25285` | `c870f078...c5f7` |

两个 role 的 content/file hash 不应相同，因为 role metadata 与 manifest 文件本身不同；相同的是协议和筛选后的规模。test selection SHA256 为：

```text
85e5603c941030b050adab7876a275e654d9da328c859c1101e32940f9649f6f
```

三条示例 tracklet 的物理 gap CV 为 `0.646 / 0.602 / 0.323`，实际 gap 覆盖约 `0.45–2.05 s`。因此该 manifest 的确制造了不规则 cadence，不是只改名字而仍保持近似固定 2Hz。

## 4. Offline shuffled-dt manifest

| field | value |
| --- | ---: |
| endpoint count（含每条 tracklet 首帧） | 1257 |
| transition count | 1166 |
| tracklet count from identity | 1257 - 1166 = 91 |
| content SHA256 | `ee18909b...ff1e2` |
| file SHA256 | `45ed12df...f11e9` |
| permutation SHA256 | `4642aeeb...19467` |

`1257 - 1166 = 91` 与筛选后的 tracklet 数完全一致，说明 endpoint/transition 计数闭合；split-wide mapping 也有独立 permutation hash，可以在三路评测中防止错用 mapping。

## 5. True / fixed / shuffled 真实 batch 验收

同一真实 batch 的时间字段为：

| mode | `delta_t_real` | `delta_t_effective` | current real | current effective |
| --- | --- | --- | ---: | ---: |
| true | `[0.949949, 0.499886, 0.499874]` | `[0.949949, 0.499886, 0.499874]` | 0.949949 | 0.949949 |
| fixed | `[0.949949, 0.499886, 0.499874]` | `[0.500000, 0.500000, 0.500000]` | 0.949949 | 0.500000 |
| shuffled | `[0.949949, 0.499886, 0.499874]` | `[0.499325, 1.998912, 0.498372]` | 0.949949 | 0.499325 |

正式输出：

```text
P0-C true/fixed/shuffled batch invariance: PASS
```

因此当前证据支持以下工程事实：

- 三路共享相同 frames、points、crop、candidate、labels、physical-time supervision 和 main order-time；
- `true` 的 effective time 与 real time 完全相同；
- `fixed` 把相邻 observation step 固定为 `0.5 s`；
- `shuffled` 保留 split 内 gap 分布，但破坏 gap 与具体轨迹 transition 的对应关系；
- 允许变化的只有 dynamics effective-time 字段和 mode ID。

## 6. 冻结 checkpoint 三路性能

三路共同使用：

```text
run: 20260531-2322-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_60ep_bs16_gpu2
checkpoint: lightning_logs/version_0/checkpoints/last.ckpt
checkpoint SHA256: b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad
training: standard cadence, seed42, 60 epochs, final/last checkpoint
```

本地 `last.ckpt` 与 `epoch=59-step=75720.ckpt` 的 SHA256 完全相同。三份 provenance 的 commit、checkpoint、source config、seed、test manifest/selection、`91 tracklets / 1257 frames` 完全一致；resolved config 只在 `dynamics_time_mode`、shuffled manifest 路径和 `log_dir` 上不同。

| mode | Success | Precision | true − mode Success | true − mode Precision | inference progress | empty fallback prints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 55.2247 | 66.8854 | 0.0000 | 0.0000 | 127 s | 94 |
| fixed | 54.7872 | 66.3624 | +0.4375 | +0.5231 | 103 s | 106 |
| shuffled | 55.3480 | 66.8298 | -0.1233 | +0.0557 | 123 s | 92 |

console 与独立 TensorBoard scalar 解析逐值一致。三路总 wall time 为 `17:14:43–17:22:41`，约 7 分 58 秒；运行时间和 empty fallback 数只作为路径诊断，不作为精度判定。

预注册 promotion 要求 true 同时超过 fixed 与 shuffled，且最小差值达到 `Success >= +0.5 / Precision >= +1.0`。实际最小差值为：

```text
min(true - controls) = -0.1233 Success / +0.0557 Precision
promotion            = False
```

## 7. 能说和不能说

当前能说：

- frozen standard-trained A2 的输出会随 effective time 改变，但正确物理时间对应关系没有产生稳定、达到门槛的收益；
- fixed 造成轻微下降，而 shuffled 与 true 几乎持平且 Success 略高，因此不能把 A2 的 gap1124 表现归因于 transition-aligned physical `delta_t`；
- fixed 的 empty-point fallback prints 为 `106`，高于 true 的 `94`，说明时间干预会通过递归预测改变后续 crop 路径；但 shuffled 只有 `92` 且 Success 略高于 true，路径变化没有形成“越接近真实时间越好”的单调关系。该差异是干预后的 downstream 结果，不违反初始 frame/input invariance；
- 当前 feature-concat DynamicsEncoder 的 P0-C promotion 正式 No-Go，不扩展 burst-drop、未见 fixed-gap或多 seed；
- 旧 HTV 的按协议分别训练问题已被修复为公平的冻结 checkpoint 对照。

当前不能说：

- 不能据此断言所有真实时间建模都无效；这里只覆盖一个 A2 checkpoint、一个 seed 和一个 held-out cadence；
- 不能把该结论外推到 `residual_limited`、TWC 或其它时间注入方式；本轮 checkpoint 是 feature-concat A2；
- 不能从总体指标判断哪些 tracklet、long-gap bin 或首次失控点发生变化；当前主评测没有保存 endpoint/per-tracklet 输出；
- 不能把三路运行时间差写成速度收益，运行顺序、缓存和 downstream empty fallback 均不同。

## 8. 决策与下一步

停止 P0-C 的 schedule/multiseed 扩展，也不基于该结果直接调大 residual。若论文诊断确实需要机制证据，只允许补一次输出型复跑：保存 per-tracklet/endpoint Success、Precision、gap bin、首次失控、连续失败和 empty fallback，再做三路 paired delta；不改 checkpoint、模型或阈值。完成该诊断后，再进入 P0-A crop-reachable residual target 的一次性机制收尾。

## 9. Validation QA

- **传输完整性**：tarball SHA256 与服务器 sidecar 一致。
- **manifest 完整性**：三个 file/content SHA256 已从原文件重算；val/test selection SHA256 均重算为 `85e5603c...f9649f6f`，shuffled permutation 重算为 `4642aeeb...19467`。
- **规模闭合**：`1257 frames = 1166 transitions + 91 tracklets`；val/test 均为 `91/106` tracklets。
- **provenance**：三路同 commit `343145d`、checkpoint `b508f958...24ac87ad`、source config `69b801f7...7d658`、seed42 和 endpoints；tracked worktree clean。`dirty_any=true` 仅来自三个未跟踪路径，作为低风险 caveat 保留。
- **指标复核**：console 与 TensorBoard events exact match；差值由原始浮点指标独立重算。
- **主要限制**：没有 endpoint/per-tracklet 预测文件，无法做 paired bootstrap、长 gap 分桶或递归失败定位。
