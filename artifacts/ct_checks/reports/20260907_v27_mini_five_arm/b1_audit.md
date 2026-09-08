# v27 mini 五臂：B1 数据通路与获取瓶颈审计

审计日期：2026-09-07。仅只读本次 output、当前代码和配置；未修改模型或实验结果，未读取 TrajTrack。可复现提取脚本为 `collect_b1_audit.py`，输出 `b1_audit.json`。本报告的逐帧数据来自每个运行 `lightning_logs/version_0/candidate_diagnostics/epoch_60.csv` 和 `dev_diagnostics/epoch_60_endpoints.csv`。

## 1. 结论

**B1 学习输出已进入真实裁剪；现在的主要获取瓶颈是递归定位失效后搜索域仍围绕错误历史移动，而不是旧版 prepass 遗漏 margin 导致全量 CV fallback。** 第 60 轮四个 B1 臂的 309 个非首帧 endpoint 中，297 帧实际 source=1（learned）、12 帧 source=0（每条轨迹第一个 query 无有效历史位移）。本次没有 source=2 的 CV fallback 帧。

**目前无法判定 CfC 与 GRU 谁更好。** 两份 resolved_config 除后端和运行名/路径外完全一致，满足配置匹配；但各臂 B0 预测及递归历史不相同。当前端点误差、novel recall、背景点数同时受到 B0 轨迹偏差影响。先统一递归输入的离线比较，再谈后端结论。

**现有 B1 dev 的 NLL、coverage 和“motion error”还混合了两种误差定义。** 它们不能直接证明物理运动预测失准或 sigma 严重失校准，见第 3 节。这个报告问题应在下一轮正式对照前修正。

## 2. 真实获取链路与数值

数据范围均为同一个 dev 场景、12 条轨迹、321 帧，其中 309 帧为预测 endpoint；不是官方 mini_val。全局可见目标点共 13,098 个（跨帧计数），230/309 帧有当前目标点。

| epoch 60 指标 | B1-CfC | B1-GRU | B1+B2 | Full |
|---|---:|---:|---:|---:|
| 实际 learned 获取帧 | 297 | 297 | 297 | 297 |
| B0 raw 内目标点 | 1,361 | 3,003 | 1,834 | 1,192 |
| B0 raw 外全局 novel 目标点 | 11,737 | 10,095 | 11,264 | 11,906 |
| 实际 support 内目标点 | 1,565 | 3,176 | 2,378 | 2,208 |
| 实际 novel pool 目标点 | 330 | 624 | 759 | 1,210 |
| novel pool / 全局 novel 目标点 | 2.81% | 6.18% | 6.74% | 10.16% |
| novel pool 背景点 | 22,130 | 17,961 | 36,241 | 33,416 |
| novel pool 有目标的帧数 | 24 | 59 | 60 | 55 |
| 768 预选池目标点 | 330 | 624 | 759 | 1,210 |
| XY 支持内目标点因 Z 被排除 | 56 | 188 | 257 | 158 |

四臂从 novel pool 到 768 预选池的**目标点总量均 100% 保留**。当前这组 Car dev 结果不支持优先增加 768 预算；多数目标早在进入可获取 support 前就已丢失。这个结论只描述这些运行，不能外推 Bus/Trailer 或完整 nuScenes。

按当前 observation XY 误差分层可以定位失效位置：

| observation 误差 >10m 的 endpoint | B1-CfC | B1-GRU | B1+B2 | Full |
|---|---:|---:|---:|---:|
| 帧数 | 137 | 70 | 111 | 122 |
| 全局 novel 目标点 | 10,082 | 8,915 | 9,107 | 9,365 |
| 实际获取 novel 目标点 | 2 | 108 | 33 | 16 |

在这些失效帧，base raw 目标点全部为 0。B1 mean 相对 CV 的平均改变量仅 0.31/0.33/0.76/0.50m（四臂顺序同上），而错误 anchor 已偏离十米以上。B1 的输入又仅包含过去三个预测框和质量，无法凭空观察真实绝对位置。因此，改用另一种循环单元、调大 sigma 或扩大 768 预算都不能单独解决这种失效。

这不是否定现有模块：在 observation 误差 2–10m 的部分帧，确实可以取得 novel 目标点；例如 Full 获得 301+892=1,193 个，而 >10m 帧仅获得 16 个。应首先阻止早期错误更新和建立可解释的恢复路径。

## 3. 已确认的指标语义问题

B1 训练物理标签在 `utils/candidate_utils.py:82` 的 `physical_motion_targets` 中定义为：

`R(predicted_anchor)^T * (GT_current.center - GT_previous.center)`。

`build_b1_physical_contract`（同文件第 252 行）明确保持这个 origin，`datasets/sampler.py:1736` 将结果写入 `motion_main_target_xy`。`models/seqtrack3d.py:5931` 的 mean/NLL loss 使用这一标签；`motion.py:777` 的 sigma 分支和 `motion.py:786` 的 margin 分支分离。

但是 `models/base_model.py:1139` 先以 **预测 reference box** 变换当前 GT，得到 `R^T*(GT_current.center - predicted_previous.center)`；然后在第 1358 行拿它减去 B1 的物理位移输出，计算当前 CSV 的 `b1_nll` 和 coverage。两者相差 `R^T*(GT_previous.center - predicted_previous.center)`，即历史定位误差。

因此当前日志有以下含义：

- `learned_motion_error` / `kinematic_error` 实際是“以预测历史为 origin 的 endpoint 定位误差”。适合检验搜索中心是否接近目标，不能当作物理位移误差。
- 当前 `b1_nll` / `b1_coverage_*` 用物理位移 sigma 去评价混入历史漂移的 endpoint 误差，不能作为统计 sigma 是否校准的直接证据。
- `models/base_model.py:2039` 又把这些 endpoint 字段记录为 `b1_learned_motion_mse/dev` 等，名称容易引起后端结论误读。

| epoch 60，297 个有效行 | B1-CfC | B1-GRU | B1+B2 | Full |
|---|---:|---:|---:|---:|
| 当前 CSV learned endpoint RMSE（m） | 16.814 | 15.072 | 16.819 | 13.809 |
| 当前 CSV CV endpoint RMSE（m） | 16.871 | 15.091 | 17.049 | 13.871 |
| 当前 CSV endpoint NLL | 48.435 | 28.490 | 34.663 | 24.839 |
| 当前 CSV endpoint 95% coverage | 27.95% | 46.13% | 46.13% | 42.09% |

这些值可以诚实报告“support 在漂移状态下很难覆盖目标”，不能直接写“GRU 物理预测优于 CfC”或“CfC 的统计 uncertainty 更差”。

建议下一版同时输出两套字段：`physical_delta_error/NLL/coverage` 使用 sampler 的 `motion_main_target_xy`；`endpoint_error/endpoint_coverage` 使用预测 origin 的位置目标。GT 仅用于诊断标签，不能写回模型或替换历史。后端比较另以同一份 observation 历史 replay 运行两个已训练 B1，并保存输入 digest，确保同 endpoint 且同历史、同点预算。

## 4. margin 学到了什么、尚缺什么证据

训练 TensorBoard 中，最后三轮记录的 batch margin 平均（包含每轮少量全 mask 行）为：

| 指标 | B1-CfC | B1-GRU | B1+B2 | Full |
|---|---:|---:|---:|---:|
| parallel mean（m） | 3.595 | 3.266 | 3.859 | 3.720 |
| perpendicular mean（m） | 0.996 | 0.996 | 0.996 | 0.996 |
| margin pinball loss | 0.137 | 0.131 | 0.141 | 0.133 |

perpendicular 小于 1 的统计均值是无有效行时 masked mean 返回 0 所致，不是预测越过 `[1,3]` 下界；最后三轮 753 条记录中有 3 条全 mask 的零值，非零 batch 的均值为 1.000345 / 1.000344 / 1.000298 / 1.000336m，接近下界。最后一个 batch 四臂 perpendicular 为 1.00022–1.00030m。parallel 明显离开初始化的约 2.04m，说明 margin 分支确实在更新。

源码 `utils/b1_acquisition.py:118` 的 81 候选监督几何只在最大支持表上向量化计算，不参与 forward；`datasets/sampler.py:2182` 将 GT-derived margin 写入损失字段。`models/ct_v2/motion.py:786` 中 temporal context detach 后进入 145→64→2 head，保证不会用 margin loss 改写物理 mean。prepass 导出 margin（`models/seqtrack3d.py:2587`），unbatch finite 检查也包含 margin（第 2696 行），实际 source 和 geometry 在 sampler 第 2220 行按 learned/resolved 分离。

尚不能判断 perpendicular 接近下界是合理标签分布还是稀有正例被大量“无 novel”标签压制，因为训练输出没有保存 margin target 的类别分层分布；已有 `acquisition_supply` 在 B1-only 臂全 0 是 B2 acquisition loss 统计未启用，不能解释为 B1 没裁点。下一版应记录：`no_novel / reachable_novel / outside_maximum_support` 三组数量、target/predicted margin 分布、90% 可达 recall 达标率、不可达比例、GT-visible 且 base-missing 比例。按这些分层后再调整 margin loss 权重或采样平衡，不能只看全局 pinball loss。

当前 XY 支持域内因 Z 排除的目标点只占全局目标的 0.43% / 1.44% / 1.96% / 1.21%，显著小于 support 本身的缺失，因而 Z padding 可做后续局部改进，不应先把它认定为主因。

## 5. 首 query 与稀疏早期错误

无有效 transition 时 source=0 的直接路径是：`utils/ct_search.py:171` 返回 `no_valid_transition`；`resolve_b1_search_support` 第 700 行左右标为 `base_only`；第 737 行不生成 endpoint/tube；corridor 第 1493 行也要求有效最近 transition。这是当前明确设计的冷启动限制，不是 margin 字段丢失。

可以设计一个 **stationary、首帧尺寸与 yaw、最小 margin 的合法冷启动 support**，并以独立 source/reason 标注，不能伪称估计出 CV。所有训练/推理/标签几何应使用同一条路径。但这不能解释或单独修复本组的主要首帧失败：

| tracklet，第一个 query | 全局目标点 | base raw 目标点 | base raw 总点 | 证据解释 |
|---|---:|---:|---:|---|
| 3 | 0 | 0 | 1 | 当前只有背景点，扩当前云无法找到目标 |
| 7 | 8 | 8 | 53 | 目标证据全部已在 base，extension-only 没有目标可补 |
| 8 | 0 | 0 | 0 | 当前不可观测，缺运动历史，不能用 GT 速度补输入 |
| 10 | 0 | 0 | 1 | 当前只有背景点，B0 可能输出大偏移 |

frame2 也有“全局目标存在而已全部在 base”的失效，例如 tracklet1 的 2 点、tracklet5 的 1 点，tracklet7 的 25 点。此时增加 extension-only 选点能力不能直接修复 B0 本已有证据却回归错误的问题。Full 的 tracklet4/frame2 是反例：4 个目标点在 extension 中，仍需结合 B2/动作半径判断能否纠正。

应优先检查并改进 B0 在 N=1/2、全部背景、历史失真时的回归和质量判断；可保留真实稀疏点语义，同时通过因果质量与预测运动一致性约束避免任意大跳。这个建议不等同于把 N<=2 的输入强制清零，也不能使用当前 GT 是否可见作推理门。

## 6. 优先顺序

1. 先定位 B0 训练/推理差异和早期错误更新，统一后端比较的真实历史；参考主审计结论。
2. 修复上述 B1 指标语义，增加 margin 三类监督分层，复用现有 checkpoint 做同历史 B1 replay；不需要先重训来补标签诊断。
3. 加入可审计的冷启动几何，但将“当前完全不可见”和“已在 base 但 B0 错估”单独处理；不能把全部问题交给 B1。
4. 面向中等漂移建立历史质量感知的恢复支持；先比较同一轨迹上的实际 novel recall/背景与延续收益，再决定调整获取 margin、短期可靠历史或恢复动作预算。
5. CfC/GRU 均保留，默认 GRU 可作为工程设置，不把这组未经同历史匹配的结果写成后端性能证明。

完整数值、逐 epoch 摘要及冷启动行见 `b1_audit.json`。训练 batch 的 RMSE 均值不是全帧全局 RMSE；脚本明确保存原始统计口径，未将训练误差当作泛化证据。
