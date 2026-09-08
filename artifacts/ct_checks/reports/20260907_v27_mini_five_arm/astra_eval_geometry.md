# a43a8ee「astra重构」：B0 推理输入、裁剪与几何专项复核

日期：2026-09-08。只读审计；未改动模型、配置或 output。范围为 `a43a8ee^ → a43a8ee`，结合当前 v27 代码核对后续是否改变相关行为。

## 结论

这次重构确实替换了 B0 的正式评测输入构造路径，并改变其稀疏输入行为，因此不能称作「只修改 B1/B2/B3」。最明显且已复现的新增输入回归，是把预测历史的 `0.2/0.8` prior 改成了 canonical candidate0 的 `0/1` prior。当前 base 为空时强制返回 reference，则是新增的恢复限制。

但本专项没有找到「普通有效点云的 anchor/坐标变换整体错位、历史反序、candidate_bc 换成 GT 特征、当前 GT 泄露进输入」这种可以进一步解释大面积首帧失败的新证据。用父提交真实 eval builder 与当前 v27 真实 sampler 做四个时刻的合成对照，在 dense 输入下，除历史 prior 外，B0 的 XYZ、时间、历史框、尺寸和 box-aware 候选输入一致。不能因为重构大，就把所有旧缺陷归到这个提交。

## 本提交实际修改了哪些 B0 相关内容

| 内容 | 父提交 | astra/v27 | 判断 |
|---|---|---|---|
| 推理装配入口 | `MotionBaseModelMF.build_input_dict` 自行拼接输入 | `build_v27_eval_input → motion_processing_mf` 共用训练 sampler | 统一构造器有价值，但必须保留输入来源语义 |
| 历史 prior | 首个查询 hard，后续预测历史 `0.2/0.8` | eval 固定 candidate0，sampler 因此始终 `0/1` | 明确新增回归；canonical 身份不能代替历史可信度 |
| 当前 N=0 | 老 evaluator 按 XYZ-sum 条件跳过；其他路径可仍预测 | 显式有效性 + B0 eval 中网络结果强制归零位移 | 显式空输入判断正确；无条件 hold 会阻断历史运动预测 |
| N=1/2 点 | `regularize_pc` 清成零点 | 保留真实点并重复采样 | 数据修复应保留；必须补齐前向 padding/unique 语义 |
| 裁剪后回到 anchor | support-local→world→anchor-local 两次往返 | 同一被选 ID 从原坐标直接变换至 anchor-local | 同一几何，减少浮点误差；未发现改错裁剪集合 |
| point ID | 裁剪不维护原始身份 | 裁剪、采样携带原 ID/valid/unique | 必要修复，但不能自动保证 feature 与 ID 对齐 |
| observation 样本准入 | 过多历史 GT box 缺少点时 assert/resample | v27 取消此拒绝条件 | 改变 B0 训练分布；需结合训练专项分析稀疏样本权重 |
| GT/诊断字段 | eval 输入主要只有因果字段 | sampler 一并产生 label 和 GT sidecar | 当前模型因果输入未因 current GT 改变；标签存在不等于泄露 |

相关源码：

- `models/base_model.py:2266`：v27 eval dispatch；旧分支仍在同文件保留。
- `utils/v27_input.py:28–67`：固定 candidate0，构造递归 contract，调用共享 sampler。
- `datasets/sampler.py:265`：`canonicalize=v27`；`sampler.py:310` 后从预测 state 重建历史框；`sampler.py:1613` 后按 candidate 身份生成 prior。
- `datasets/points_utils.py:28`：N>0 保留点；`points_utils.py:317` 后按同一 ID 直接变换到 anchor。
- `models/seqtrack3d.py:3832`：eval 当前 raw N=0 时覆盖 observation。

## 真实构造器对照

可复核脚本：[reproduce_astra_eval_geometry.py](reproduce_astra_eval_geometry.py)。输出：[astra_eval_geometry_reproduction.json](astra_eval_geometry_reproduction.json)。

执行：

```text
artifacts/ct_checks/v27_runtime_env/Scripts/python.exe -m pytest -q artifacts/ct_checks/reports/20260907_v27_mini_five_arm/reproduce_astra_eval_geometry.py --disable-warnings
1 passed
```

方法：直接 `git show a43a8ee^:models/base_model.py`，通过 AST 提取当时 `MotionBaseModelMF.build_input_dict`，保留真实函数体；与当前 `build_v27_eval_input`/真实 sampler 对照。底层当前通用 helper 在本项 dense 场景中的语义与父提交一致；这不是执行完整旧仓库或重跑历史模型。使用既有 CPU fixture，只替代不可用的数据包导入，不替代 crop、sampler 或 builder。

对照包含第1、2、3、8个查询，非零绝对坐标、不同历史 yaw、跨 ±π 的角度，以及启动期不足3帧的重复历史。各帧数千点，避开已知 N≤2 新旧行为差异。

| 比较字段 | 四个查询的最大绝对差 |
|---|---:|
| 点 XYZ | 0，转 float32 后 |
| 点 time | 0 |
| `ref_boxs` | 0 |
| `candidate_bc` | 0 |
| `bbox_size` | 0 |
| `valid_mask` | 0 |
| `timestamps`、`delta_T` | 0 |
| `delta_t_real/effective` | 0 |
| 历史 prior：第1查询 | 0 |
| 历史 prior：第2/3/8查询 | 0.2，hard/soft 语义变化 |

另对4个不同朝向的历史支持域直接比较 `canonicalize=False/True`：保留 ID 集合完全相同，XYZ 最大差为 `2.22e-16 m`。这项修改是数值路径简化，不支持「新裁剪把点放错坐标系」的归因。

## 已核对而不应重复归因的部分

1. 历史始终按 newest→oldest 排列；eval contract 的 offsets 明确为 `[1,2,3]`。`create_history_frame_dict` 和 sampler 的排序一致。
2. 历史预测框的 center、尺寸、yaw 在进入 sampler 时均由递归 state 覆盖。GT source box 只作为构造对象载体，不保留其几何值。
3. candidate0 的 SE(2) 变换为 identity；输入裁剪锚点依然是最近预测框，返回给 host 的参考框也是同一个预测。
4. `candidate_bc` 依然是历史点到预测历史框的距离，当前 candidate_bc 依然全零。`prev_bc/this_bc` 是监督字段，不能把它们与 causal candidate_bc 混淆。
5. 首帧尺寸在父版本的 `observation_safe_bbox_size` 下已启用，不是本提交第一次变化；动态对照也一致。
6. 主干的 `solo_x = x.reshape(...)`、Transformer 不使用 `valid_mask`，不是本提交新增。它们需要修复，但不能作为本次历史分界的独有证据。
7. nuScenes 点云在数据层变换到 world；这里不是把 world box 与 lidar 点云直接混用。本专项只验证 SE(2) yaw 几何。若将来要支持非竖直姿态，`box_world_row` 只保留 yaw 与重新构造纯 z 旋转是额外边界，当前没有实际数据证据将其认定为此次主因。

## 对根因判断的含义

- **能确定**：astra 对 B0 有实质修改，其中历史 prior 来源错配应优先修正；对 empty 的新策略会影响长时间漂移后的恢复。
- **不能确定**：仅靠这两个变化即可解释所有低分；尤其第1次查询的 hard prior 在新旧实现相同，而当前已经存在第1帧直接失败的轨迹。
- **应保留**：统一构造器、明确 N=0、保留真实1/2点、point ID、直接 anchor 坐标变换。
- **应修正**：按历史 provenance 而非 candidate_id 生成 prior；把空观测定位与仅历史运动延续分开；让有效/唯一语义进入特征计算而非只进入 loss/诊断。
- **验证路径**：旧权重+新旧 builder 在共同评测集的同 checkpoint 推理对照，可以量化输入语义变化；新的 loss、feature layout、训练样本规则则需要 scratch 对照，不能依赖只换 builder 的结果。

本报告只缩小 eval 输入几何原因范围，不能替代 root 对训练损失、样本准入、训练预算和历史运行分数的联合审计。
