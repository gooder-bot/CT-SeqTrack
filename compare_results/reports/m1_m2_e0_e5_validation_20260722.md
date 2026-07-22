# M1/M2 E0–E5 服务器工程验收报告

更新时间：2026-07-22

## 1. 总结判定

**总体评估：可用于确认 E0–E5 Engineering GO；不可用于启动正式训练或声称 tracking 性能收益。**

正式状态写为：

```text
PASS_M1_M2_E0_E5_ENGINEERING_GATES
HOLD_FORMAL_TRAINING_PENDING_E6
```

本次验收回答的是 shared world-SE(2)、canonical dynamics label、zero-init dual-clock adapter 和 bounded proposal innovation 是否满足真实 nuScenes loader 下的几何、回退、数值安全与可训练性要求。它没有训练收敛后的 tracking Success/Precision，也没有执行 true/fixed/shuffled-dt 因果负对照。

## 2. 数据源与 provenance

本地证据目录：

```text
output/m1_m2_gates_9a0b26d_20260722_101709/
```

服务器运行信息：

| 字段 | 值 |
| --- | --- |
| Git commit | `9a0b26d175a843ee49c07ca08ce63f5baa3c0168` |
| run stamp | `20260722_101709` |
| 数据 | nuScenes mini，`/home/lishengjie/data/nuscenes-mini` |
| A1 checkpoint SHA256 | `a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82` |
| M2 engineering config SHA256 | `be436cfe82715f0369311779a238188672777df30bb80d479ebbfcb06c1b7182` |
| gap1124 config SHA256 | `03d19ab061b50aa2ad4b91c60abdfd0485fd1c3edeb921aba61a225cb4d53102` |
| burst-drop config SHA256 | `59afbedd4af400b20cba33faafc90667313dfbc8b60524caf60a40537e351000` |
| GPU | standard/fallback=`2`，gap/burst=`3` |

以上三个 config hash 已在本地 commit `9a0b26d` 上重新计算并与服务器 `provenance.txt` 完全匹配；复核开始时本地 tracked worktree 无差异。13 个回传文件的本地 SHA256 索引见 `m1_m2_e0_e5_artifact_sha256_20260722.txt`。

回传包没有包含外层 `/tmp/m1_m2_gates_*.log` 或服务器 tarball SHA256，因此不能复核外层最终五行汇总；但两个 GPU `run.log`、五个 summary、五个逐步 JSONL 和 provenance 均齐全，且已从原始 JSONL 独立重建全部硬门统计，不影响 E0–E5 判定。以后正式运行必须把 console、artifact manifest 和归档 SHA256 一并保留。

## 3. 独立数据质量与计算复核

逐行读取五个 JSONL 后得到：

- 共 `61` 个 batch record、`122` 个样本；各 run 的 step 从 1 连续递增，record 数与 summary 的 `completed_steps` 完全一致。
- 每条记录 `ok=true`；所有 loss finite flag 为 true；总梯度范数与最大绝对梯度均为有限数。
- sample、invalid、empty、resampled、applied、clamp、optimizer-step 计数均可从 JSONL 精确重建并与五个 summary 一致。
- invalid/empty 的 applied maximum、warmup 输出、梯度最大值和 correction bound 均与 summary 一致。
- 五个 summary 都是 schema `ct_seqtrack.m1_m2_innovation_gate_summary` version 1，`requirement_failures=[]` 且 `requirements_passed=true`。

| run | batch / optimizer | 样本 | invalid / empty / resampled | applied / clamp | applied max | bound violation max | encoder / adapter grad max | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 2` | `0.562864 m` | `0` | `0.044475 / 1.185760` | PASS |
| standard warmup | `2 / 2` | 4 | `0 / 1 / 0` | `0 / 2` | `0` | `0` | `0.039217 / 0` | PASS |
| standard fallback | `53 / 0` | 106 | `8 / 16 / 2` | `86 / 10` | `0.562251 m` | `5.96e-8` | `0.214454 / 0.302734` | PASS |
| gap1124 active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 3` | `0.750290 m` | `5.96e-8` | `0.126719 / 3.886739` | PASS |
| burst-drop active | `2 / 2` | 4 | `0 / 1 / 0` | `3 / 3` | `0.937743 m` | `5.96e-8` | `0.051545 / 2.659258` | PASS |

fallback 扫描固定顺序且不更新 optimizer，在第 `53` 个 batch 首次观察到 `2` 个 sampler-resampled 样本后自动停止。这里的 resampled 表示 sampler 在原请求样本断言失败后返回另一条有效训练样本；其验收要求是被显式记录且 forward/loss/backward finite，不是要求模型输出 strict-zero。strict-zero 只适用于 invalid history、empty search、disabled/zero scale 和 warmup。

## 4. E0–E6 逐项判定

### E0 默认回归：PASS

- shared-SE(2) dataset-free 与真实 loader/TWC 检查通过。
- A1 strict-zero model equivalence 通过，checkpoint `320` 个张量全部匹配；新模型共 `334` 个张量，缺失的 `14` 个均属于按设计新建的 DynamicsEncoder/physical-time adapter，unexpected key 为 0。
- 新功能默认关闭和 zero-scale 路径没有改变 A1 motion/output/loss。

### E1 几何不变量：PASS

dataset-free world-SE(2)、identity、radians/degrees、anchor-normalized trajectory，以及真实 loader candidate1/2/3 与 TWC 共享检查均通过。当前证据支持“一个 sample-level 共同世界刚体变换”，不再是逐框局部 offset。

### E2 label 不变量：PASS

canonical displacement/velocity label 与 candidate 坐标协变检查通过；candidate perturbation 不再改变被监督的物理速度/位移定义。

### E3 公式不变量：PASS

- warmup 内真实执行 `2` 次 optimizer step 后，adapter output、adapter effective scale、innovation applied 和 innovation effective scale仍全部精确为 `0`。
- warmup 中 DynamicsEncoder 两步梯度均非零，说明辅助 velocity/displacement 监督仍能学习；adapter 梯度为 0 是结构关闭的预期结果。
- fallback 扫描中 invalid `8`、empty `16`，二者 applied max 都精确为 `0`。
- 正式路径使用 `d_obs + alpha * clip(d_dyn - stopgrad(d_obs), R(dt))`，没有恢复旧 full-displacement addition。

### E4 数值安全：PASS

standard、gap1124、burst-drop 和 fallback 的 `61` 个 batch 全部 finite；最大 bound violation 只有 float32 舍入量级 `5.96e-8`，低于硬门 `1e-6`。invalid、empty 和 sampler-resampled 均已覆盖。

### E5 可训练性：PASS

standard/gap/burst active 路径共完成 `6` 次真实 optimizer update；三协议都观测到 DynamicsEncoder 和 physical-time adapter 的非零有限梯度，并产生非平凡但受 `alpha*R(dt)` 约束的 correction。另有 `2` 次 warmup optimizer update 验证 schedule 语义。

### E6 可复现性：未完成

本次 gate run 已具备 clean commit、A1/config hash、固定 seed、workers0 和不混写输出目录，足以归档工程验收；但正式 seed42 尚未冻结以下项目：

1. 只依据 mini_train oracle 一次性确定的 `alpha` 与 `R(delta_t)`，以及不经指标调优的 adapter/innovation 共享 warmup（预注册计划值 5 epoch）；不得根据本次随机初始化 smoke 的 clamp ratio 调参。
2. 唯一 true-dt 训练配置及只改变 effective-time mapping 的 fixed/shuffled controls。
3. train/eval manifest、candidate/point-sampling、optimizer steps、final checkpoint 和评测 endpoint 口径。
4. 正式运行的 console、artifact manifest、archive SHA256 与 clean provenance。

因此 E6 仍是正式训练的唯一 E-gate 阻塞项；M0-2 四协议/path-variance 也仍需并行收口，M0 整体不能标记完成。

## 5. 结论边界与下一步

本次可以支持：M1/M2 第一工程切片已完成 E0–E5，允许进入唯一正式配置和 E6 冻结阶段。

本次不能支持：M2 已提高 tracking 指标、真实 `delta_t` 有因果收益、`alpha=0.75` 或当前半径已经最优、可以跳过 fixed/shuffled controls、可以启动 M3/M4。

下一步固定为：

1. 只使用已有 mini_train M0-3 oracle 原始向量，对 alpha 和 `R(delta_t)` 做一次预注册确认；共享 warmup 直接冻结为计划值 5 epoch，不经指标调优；不使用 mini_val/test 或本次 smoke 的 clamp ratio。
2. 新建唯一 seed42 formal true-dt config，并派生仅 effective-time mapping 不同的 fixed/shuffled controls；完成 E6 manifest/provenance 规范。
3. E6 文档和 clean commit 通过后，才执行一次 seed42 true/fixed/shuffled；不扫 scale/gate/bound 网格。
4. 并行完成 M0-2 冻结 A/B/C 四协议与 evaluation-only path variance，不重训旧模型。
