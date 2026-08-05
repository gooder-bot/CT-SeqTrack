# CT-SeqTrack 论文计划

更新时间：2026-08-04

> B1–B4 的完成度、代码问题、目标数据流和执行门槛以
> [B1–B4 连接重构与消融计划](docs/B1_B4_REDESIGN_AND_ABLATION_PLAN_20260804.md)
> 为唯一权威来源。正式耦合技术合同见
> [不确定性感知非对称双查询耦合](docs/ASYMMETRIC_DUAL_QUERY_COUPLING_20260804.md)。
> 本文只维护论文叙事、主表和证据标准。

## 1. 当前论文命题

CT-SeqTrack 不把“历史框运动”“proposal refinement”“search expansion”或
“temporal consistency”单独当作创新。候选论文命题是：

> 在 SeqTrack3D 的 order-clock observation 主干之外，使用连续物理时间运动
> 先验产生均值与校准风险；以 motion pre-pass 构造保留 base 的扩展支持域，
> 通过独立 `q_obs/q_search` 分离当前观测与运动引导证据；最后以 observation
> 为安全锚点执行可拒绝、可限步的闭环修正。

简化方法链：

```text
B0 SeqTrack3D observation
        +
B1 continuous-time mean / calibrated risk
        -> B2 base-preserving support + asymmetric dual-query evidence
        -> B3 observation-anchored conservative routing
```

B4 暂不属于主线。只有修正几何、防坍缩目标与速度后独立涨点，才作为可选模块。

## 2. 当前科学状态

| 模块 | 当前结论 | 论文状态 |
| --- | --- | --- |
| B1 | encoder 与递归接口基本完成；标准增益不稳定，sigma 未受 NLL 监督 | 待补 calibrated uncertainty 与 `dt` 因果证据 |
| B2 | raw search 优于 motion，但 B1-centered refined 显著差于 raw；验证 foreground-valid 仅 3.29%，presence AUC 0.497 | 第一优先级，不能宣称已完成 |
| B3 | 工程链基本完成，但 B2 未过 gate，router 未进入正式验证 | 暂停训练与论文结论 |
| B4 | final `51.189/60.886`，表示收缩明显，约 `8.24×` B0 成本 | 当前实现 No-Go，移出主线 |

B2-v3 的关键机制证据是：motion/raw/refined endpoint error 为
`2.9045/2.6496/2.7344`，`refined - raw = +0.0848`，tracklet 95% CI
`[+0.0379,+0.1323]`。因此下一步应先解除 B2 对 B1 的错误裁剪连接，而不是调 B3。

## 3. 论文贡献成立所需证据

### C1：连续物理时间先验

只有 shared checkpoint 下 `true dt` 持续超过 dataset-mean fixed 和
within-dataset shuffled，并在同数据集 stride/gap 与 held-out cadence 上成立，
才能声称 frame-rate-invariant physical-time benefit。

最低报告：mean error、NLL、50/80/95% coverage、按 gap/类别分层结果。若真实时间
因果证据未通过，论文降级为 trajectory prior，不使用“跨帧率不变”强表述。

### C2：风险控制的双尺度搜索证据

B2 必须输出独立的 `raw_search_box`，而不是只输出围绕 B1 裁剪后的候选。需要
证明：

- base + endpoint/tube support 提高 GT reachability；
- final decoder `q_obs` 优于 pre-Transformer coarse query；
- `q_obs/q_search` 双 query 优于把 observation/motion 混入单 query；
- forced-invalid/shuffled B1 时 observation 路径保持稳定；
- recursive replay 缩小训练/部署 structural-valid 差距；
- presence/utility 能区分 helpful 与 harmful search；
- raw search 相对 observation 存在 tracklet-level oracle headroom。

### C3：动作一致的保守闭环

B3 只有在 B2 通过 candidate gate 后才有研究意义。需要证明训练标签、选择动作、
执行候选和 step ratio 是同一个动作，并且 recursive rollout 与部署一致。主要对照
是 no-router、oracle、H1/H3、with/without cooldown，而不是只展示 gate 分布。

### C4：可选表征一致性

B4 必须同时通过正确 canonical geometry、防坍缩、tracking 正收益与 `<2× B0`
速度门槛。未通过则不进入摘要、方法图和主表。

## 4. 论文主表顺序

| 行 | 模型 | 隔离的贡献 |
| --- | --- | --- |
| A0 | matched SeqTrack3D B0 | same-code baseline |
| A1 | B0 + B2（CV support + dual query） | 当前帧扩展搜索证据与 task-specific query |
| A2 | B0 + B1 + B2 | learned physical prior 替代 CV，并控制 support/bias/risk |
| A3 | A2 + B3 | 闭环选择将候选 headroom 转为最终跟踪收益 |
| A4 | A3 + redesigned B4（可选） | 独立验证的表征一致性 |

A1 必须先于 A2：若没有 A1，就无法区分涨点来自 Search evidence，还是来自 learned
motion prior。A4 不是论文完整性的必要条件。

主表统一报告 final Success/Precision、late-3、per-category、gap/稀疏度分层、
FPS 和 paired tracklet bootstrap CI。必须补 same-commit matched B0；历史 B0 只作
guardrail，best checkpoint 只作诊断。

## 5. 内部消融

### B1

- CV / learned GRU or MLP mean；
- true / fixed / shuffled `dt`；
- mean-only / heteroscedastic NLL；
- sigma 不控制任何模块 / 控制 B2 support / 同时进入 B3；
- stride/gap 1、2、4 和 held-out cadence。

### B2

- hand endpoint / learned B1 pre-pass；
- base-only / prior-only / base∪prior；
- pre-Transformer / final `q_obs` / final `q_obs+q_search`；
- fixed width / calibrated sigma / shuffled sigma；
- raw candidate / B1-centered clipped candidate；
- synthetic candidate / frozen B0/B1 recursive replay；
- random negatives / same-category tracklet hard negatives；
- availability-only / separated presence + utility。

### B3

- no router / oracle / learned router；
- H1 / H3；
- q50 / q10；
- cooldown off/on；
- scalar risk features / full embeddings。

### B4（独立附表）

- wrong/correct yaw 只作 bug 复现，不作方法消融；
- raw SmoothL1 / stop-grad + variance-covariance / cycle；
- low-level point feature / projector / memory or decoder token；
- accuracy、feature std/effective rank、step time 同时报告。

## 6. 实验晋级顺序

1. 同 checkpoint 导出 observation、motion、raw search、current clipped refined、
   oracle(obs/raw)，完成 sigma coverage 和 crop reachability。
2. 只移除 B2 B1-centered clip，把 raw search 设为官方候选；不训练 router。
3. 保持当前 support 不变，依次测试 final `q_obs` 和零初始化 `q_obs/q_search`。
4. dual query 有 headroom 后补 B1 NLL calibration。
5. 校准后依次接 B1 pre-pass、fixed/calibrated support、geometry bias、recursive
   replay 和 hard negatives；一次只改一个。
6. B1–B2 候选稳定后才训练 B3。
7. B1–B3 主表稳定后，决定是否给 B4 一次 5-epoch 三臂 kill-test。

详细阈值和停止规则见
[权威重构计划第 5 节](docs/B1_B4_REDESIGN_AND_ABLATION_PLAN_20260804.md#5-执行顺序与晋级门槛)。

## 7. 声明边界

当前不能写入论文结论的表述：

- “B1 已带来稳定涨点”——缺 same-commit matched B0 与多 seed；
- “B2 search 已有效”——raw 有机制信号，但跟踪收益、覆盖和判别仍未通过；
- “B3 已完成”——router 尚未获得合格候选；
- “B4 思路无效”——只能说当前实现 No-Go；
- “物理时间带来跨帧率泛化”——尚缺 true/fixed/shuffled 和 held-out cadence；
- “状态已完全对齐”——目前只有 history tensor 对齐，search support 尚未统一。

## 8. 目标投稿形态

最稳妥的最小论文是 B0 + B1 + B2 + B3，三项贡献分别对应 prior、evidence、
decision，并由 A1/A2/A3 逐级隔离。若 B2 raw/oracle 没有足够 headroom，则及时停止
B3，缩小论文问题；不要为了凑四个模块保留无法被数据支持的 B4。
