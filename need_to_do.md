# CT-SeqTrack v2 下一步

更新时间：2026-07-25

只执行下面五个部分。旧阶段的细碎任务已移到 `docs/legacy/need_to_do_20260724.md`，不再作为当前待办。

## 0. v2 逻辑修复门禁（代码已完成）

- [x] B3 observation statistics 改用 effective `dt`，真实 `dt` 只保留为监督和诊断。
- [x] B1–B3 加入 candidate0 clean / 非零 candidate correlated 的训练历史；motion 标签仍为 canonical。
- [x] correlated search history 从实际 candidate anchor 出发，不再使用 GT anchor 构造 tube。
- [x] 0、1、2 个可用搜索点时关闭 proposal innovation，并记录 nominal/applied alpha。
- [x] 轻量单元测试覆盖时间 fallback、相关历史、invalid transition 和空搜索边界。

完成条件：本地单测、配置解析和编译检查通过；服务器真实 batch 两步
optimizer smoke 与 2-tracklet 三时间模式 smoke 通过后，才能开始 B0–B3。
修复前的 B3 `true/fixed/shuffled` 存在真实时间旁路，不能作为 v2 因果证据。

## 1. 正常数据四组消融

- 训练 B0 `baseline`、B1 `motion`、B2 `motion_search`、B3 `full`。
- 固定 seed42、60 epoch、candidate4、正常 nuScenes-mini 和 final checkpoint。
- 汇总 Success/Precision、逐 tracklet paired delta、门控 alpha、search 使用率与 expansion point ratio。

完成条件：确认 CT Motion、Search、Adaptive Gate 各自是正贡献还是应被删除。

## 2. 真实时间控制

- 只对 B3 final checkpoint 运行 `true / fixed / shuffled`。
- 保持 endpoint、点采样 seed/预算、checkpoint 和协议一致；时间控制允许按设计改变 motion prior 与 search tube，因此实际扩展点可以变化。
- 正常集涨点且 true 不低于两个控制，才允许进入正式多 seed。

完成条件：给出“涨点通过/失败”和“真实时间因果通过/未通过”两个独立结论。

## 3. 正式实验

- mini 通过后，只补 B0/B3 的 seed43、44。
- 三 seed 稳定后转 full nuScenes，正常数据作为论文主表。
- Random-20% 仅在 full/normal 结论稳定后补作鲁棒性表，不参与选模。

完成条件：同代码 baseline、完整模型、三 seed 和 full dataset 主结果齐全。

## 4. 第二阶段候选

只有前三部分通过后，才参考 [ChronoTrack](https://arxiv.org/abs/2604.13789) 依次评估下面两个候选。每次只加入一个变量，不同时引入一致性和记忆模块。

### 4.1 非对称时间一致性

- 以正常、连续历史路径作为稳定 teacher，以不规则采样或长间隔路径作为 student，只约束共同 endpoint 的 motion proposal/最终框。
- 第一版只增加训练期 consistency loss，不增加推理网络，不恢复旧 symmetric TWC 的双路监督方式。
- 单独比较 `B3` 与 `B3 + asymmetric consistency`；正常集不涨点或 true-time 控制退化则删除。

### 4.2 紧凑前景记忆

- 一致性模块通过后，再尝试少量 recurrent foreground memory tokens，保存目标前景和运动状态，不缓存完整历史点云。
- 历史前景特征先对齐到最近 anchor，再更新固定数量的 memory tokens；总 token/点数预算保持不变。
- 单独比较上一阶段最佳模型与 `+ compact memory`，重点检查遮挡、稀疏点云和长时间间隔，同时保留正常集 guardrail。

完成条件：每个候选都必须相对进入该阶段的最佳模型独立涨点；失败的模块立即删除，不用另一个模块掩盖其退化。
