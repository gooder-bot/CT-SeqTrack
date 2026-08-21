# CTSEQTRACK B0--B3 方法与证据合同

## 1. 论文主线

暂定题目：**Observation-Anchored Evidence Recovery for Irregular-Time 3D
Single-Object Tracking**。

CTSEQTRACK 将“产生候选”和“决定是否动作”分离。B0 是唯一 nominal output；B1
只提供由递归历史框与物理时间得到的先验；B2 必须在 observation crop 之外找到
新增点证据；B3 对“执行这次修正是否有帮助/伤害”建模。该结构避免把运动预测、
memory 或候选置信度直接误写成可靠性。

## 2. 代码映射

| 论文对象 | 实现 | 输出所有权 |
|---|---|---|
| B0 Observation | `B0Observation` + `SEQTRACK3D` backbone | box、stats、点对齐 feature |
| B1 Physical-Time Prior | `B1PhysicalTimePrior` | mean、direction、log-sigma、valid、source |
| B2 Evidence Acquirer | `B2EvidenceAcquirer` | raw box、structural availability、presence、targetness、点诊断 |
| B3 Selective Updater | `B3SelectiveUpdater` | help/harm、expected gains、bounded residual、final box |

`pipeline_contracts.py` 定义内部 typed contracts；外部 flat dictionary 只是 evaluator
兼容层。B0 contract 在进入 B1--B3 前 detach，B3 再次 detach 全部上游输入。

## 3. B1 合同

B1 输入只来自递归历史框、物理时间与有效性。模型同时导出 learned mean 和 CV
fallback；输出永不直接写 final box。首版 B2 固定 support margins，避免训练早期
未稳定 sigma 改变采样分布。二维 coverage 使用 Mahalanobis 阈值报告；固定正态
`q90` 不被称为二维校准保证。

物理时间贡献的必要证据是 matched-scratch 的 true/fixed/shuffled 对照，并在
held-out cadence 上复现。否则论文只描述实现，不作因果主张。

## 4. B2 合同

B2 raw center 由 extension 点的 targetness-weighted votes 构造。base 和 memory
仅作特征上下文，不能独立回归位置。训练 mask 为：

```text
targetness_mask = every valid extension point
raw/vote_mask   = canonical candidate0 with target in extension
presence_mask   = candidate0 only
B3_mask         = candidate0 only
```

因此 extension 中无目标的 canonical 样本只提供 presence-negative；空间 recovery
candidate 已从正式路径删除。`candidate_valid` 是 structural
availability 与 predicted presence 的合取，而不是“采到了任意扩展点”。

no-extension 条件下 raw candidate 精确退回 observation；若未来修改重新允许 base
或 memory 单独产生位置增益，测试应立即失败。

## 5. Memory 合同

每个 memory token 携带相对历史框坐标、physical age、相对 yaw 的 sin/cos、
inside/context role、history-frame identity 与 valid mask。正式控制：

- `none/empty`：不提供有效 memory；
- `real`：token 与时空元数据正确对应；
- `time_misaligned`：token、channel、token count 与 mask 不变，只错配三个历史块
  与元数据。

Memory 是 B2 条件晋级项，不是独立论文创新。未同时超过两个控制时，最终配置为
`memory=none`。

## 6. B3 与经验风险

B3 特征包括 B0 coarse/refined agreement、B0--B1/B0--B2 disagreement、B1
uncertainty/dt/gap/age、presence、targetness entropy/ESS、点/voxel 数和 observation
statistics。H=1 与 H=3 shadow rollout 生成：

```text
center_gain = mean(error_observation - error_action)
iou_gain    = mean(iou_action - iou_observation)
```

B3 分别学习 helpful、harmful、expected center gain、expected IoU gain。校准 score
固定为 `sigmoid(help_logit) * (1 - sigmoid(harm_logit))`。部署动作还必须通过结构、
presence 和 `radius(dt)` 有界残差条件。

阈值在独立 calibration tracklets 上经验选择，统计单位为 tracklet bootstrap。
论文只报告有限样本经验风险和置信区间，不声称严格 conformal 或分布无关保证。

## 7. 匹配 scratch 不变量

四臂各自随机初始化，不共享 checkpoint。可归因性来自训练事务相同：

- B0 只收 observation loss；
- B1 只收 candidate0 mean/NLL；
- B2 只收 candidate0 on-policy evidence；
- B3 只收 candidate0 H1/H3 action-risk labels；
- B2/B3 只做 shadow 学习，canonical recursive state 永远写 observation；
- 每模块独立 optimizer/scaler/scheduler/clip/step/hash。

共享 prefix 的初始化、step1、step100、epoch-end hash 任一不一致，即判消融不匹配，
不再解释最终分数。

## 8. Claim--evidence map

| 候选主张 | 最低证据 | 失败后的处理 |
|---|---|---|
| observation-preserving recovery | exact fallback 测试 + action rows | 必须修复，不能降级措辞 |
| extension-only evidence 有效 | acquisition、target-bearing、retention、no-extension、raw gain/oracle headroom | final/late-3 后如实报告负结果 |
| selective update 安全且有益 | calibration gate、risk--coverage、center/IoU 单侧 CI | Full 退回 observation |
| physical time 有贡献 | true > fixed/shuffled，跨 cadence | 删除因果措辞 |
| memory metadata 有贡献 | real > empty/time-misaligned，paired CI | `memory=none` |
| 最终涨点稳定 | full nuScenes seeds 42/43/44 + paired CI | 不声称稳定/SOTA |

## 9. 相关工作定位

Memory、box prior、base-expansion attention 分别已有 MBPTrack、HVTrack 与
ChronoTrack 等先例，不能单独作为主创新。SelectiveNet 支持 risk--coverage
视角，Conformal Decision Theory 提供决策风险的相关思想；当前实现只主张经验
校准。最终论文引用必须再次核对原始论文、版本、作者与 BibTeX。
