# 全帧 bounded H1 几何独立复核

结论：本次两个 epoch 60 candidate CSV 的 309 个非首帧都可可靠恢复真实有界动作的 3D 中心距离。更直接地，CSV 末尾已经导出了**不受 presence 门控制**的 `success_gain / precision_gain / utility_gain`；应使用这些原生列，而不是将 `dev_diagnostics` 中被候选有效性门替换后的 `candidate_*` 当成完整有界动作。

## 几何合同

`models/base_model.py` 的 `_build_ct_joint_diagnostic_row`：

- `target_xy` 来自 `transform_box(GT, reference_box)` 的 local XY。
- `observation_error` 是 observation local XY 到同一 target local XY 的距离。
- `observation_distance` 是 `estimateAccuracy(..., dim=3)`，即世界坐标下的 3D 中心欧氏距离。
- `bounded_local4` 拷贝 observation 的全部四维，再只把 XY 替换为 `observation_xy + bounded_residual_xy`；local z、yaw 不变。
- `bounded_action_error` 是该 bounded local XY 的误差。
- 本次两臂 resolved config 均为 `IoU_space=3, limit_box=false, use_z=true, degrees=false`。

`datasets/points_utils.py:getOffsetBB` 在 `limit_box=false` 时，仅通过 reference 的旋转和平移将 local offset 转到世界；`transform_box` 使用对应逆刚体变换。欧氏距离对旋转和平移不变。因此令 local z 误差为 e_z：

`observation_distance² = observation_error² + e_z²`

`bounded_distance² = bounded_action_error² + e_z²`

于是：

`bounded_distance = sqrt(bounded_action_error² + observation_distance² - observation_error²)`。

这个证明不要求 reference 是严格 yaw-only：即使参考框含微小 pitch/roll，只要相同刚体变换且 local z 不变，3D 范数仍然保持。不能把这个公式泛化到 limit_box 开启随机 offset 修正、动作同时改变 local z、或二维旧评价器的运行。

## 实际逐帧核对

机器结果：[bounded_geometry_crosscheck.json](bounded_geometry_crosscheck.json)。

| 核对 | B1+B2 | Full |
|---|---:|---:|
| candidate 非首帧 | 309 | 309 |
| 推回的 e_z² 最小值 | 6.10e-6 | 5.33e-10 |
| 用相同公式恢复已知 raw_distance，最大误差 | 4.70e-13 m | 4.14e-13 m |
| 恢复已知 selective_distance，最大误差 | 1.49e-13 m | 0 m |
| candidate CSV 与 endpoint observation_distance 最大差 | 0 m | 0 m |
| endpoint 真正保留 bounded candidate 的帧 | 2 | 12 |
| 这些帧恢复 bounded_distance 与 endpoint 的最大误差 | 1.49e-13 m | 9.33e-14 m |
| 重算 S/P/U gain 与 candidate CSV 原生 gain 最大差 | 2.98e-8 | 0 |

通过候选门的集合可由 candidate CSV 的 `search_valid` 或 `presence_probability>0.5` 识别；本次 `structural_available` 分别为177和140帧，不能拿它当已经通过 presence 的候选集合。

关键反例是 B1+B2 tracklet 0/frame 8：observation 3D distance=0.9349402m，真实 bounded distance=1.6825304m，原生 U gain=-0.2500000；而 endpoint 的 candidate_distance=0.9349402m，已被替换回 observation。这说明若只看 endpoint 的候选收益，会漏掉实际可计算的有界动作风险。

## 可以挽回的证据及边界

无需服务器重跑，就能使用 candidate CSV 的原生收益列得到全部309个非首帧 bounded H1。加上12个首帧的零动作收益，用321帧作为分母：

- B1+B2：help/harm/neutral=26/12/271，U gain 总和=-0.2500000168，全帧平均 **-0.077881625 个百分点**。
- Full：help/harm/neutral=8/6/295，U gain 总和=-0.0250000013，全帧平均 **-0.007788163 个百分点**。

这仅是**各自现有递归轨迹上的逐帧一次动作反事实收益**，不是“删除 presence 门后的完整闭环分数”。动作改变后续 reference、裁剪和 memory 后，后续结果需真实闭环重新评测。也不能用这两项跨 checkpoint 小差异推出 B3 或后端因果结论。
