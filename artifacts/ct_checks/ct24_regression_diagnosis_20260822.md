# CT-SeqTrack v24 回退与 B1/B2 链路诊断

日期：2026-08-22  
范围：最新 v24 B0/B1-only、CT24 B0 2×2、历史 CT22/SeqTrack、当前 HEAD `a00935a`。  
口径：正式比较优先使用同一 37-tracklet atomic dev 的 final epoch 与 late-3；历史 mini_val 数值只作背景。

## 技术摘要

1. 最新 B0 并非“只轻微下降”：相对可比的 CT22 B0 dev，final 下降 `8.721 Success / 7.856 Precision`，late-3 下降 `5.472 / 5.365`。
2. 没有证据表明源码瘦身制造了新的性能损失。最新 B0 `41.079/49.916` 与瘦身前 2×2 的 `reseed1_rngshift1=40.847/48.383` 接近，且两者训练协议相同。瘦身主要删除历史资产并收敛配置入口。
3. 最新正式配置与 2×2 报告的冻结结论冲突：报告选择 `reseed=1, rng_shift=0`，当前 `24_b0.yaml` 和 `24_formal_base.yaml` 却都是 `ct_b0_rng_shift_control: true`。最新实验因此实际复现的是 2×2 的 rng-shift-on cell。
4. 相对 CT22 的主要结构变化来自 candidate/B0 训练重构，而不是 B2 candidate 数量本身：B0 从 canonical 在线事务加三个隔离辅助 forward、加权梯度合成；B2 只读 canonical view0。这个协议改变了 B0 的有效训练目标、随机流、BN 行为和 batch 执行顺序，足以解释 B0 级回退，但仍需 candidate1 对照才能做因果确认。
5. B0 与 B1-only 在相同 seed/commit 下，B0 candidate0 loss 从 step1 就不同，辅助 loss 从 step5 不同。若模块构造、数据顺序、RNG、BN 和 B0 optimizer 状态完全隔离，这种即时分叉不应发生；当前四臂尚未满足 matched-prefix 可归因性。
6. B1 均值没有明显学过 CV：epoch50 learned RMSE `21.315m`、CV RMSE `21.305m`；差 `+0.011m`。均值误差有极重长尾，median `0.889m` 但 mean `9.647m`。
7. B1 sigma 明显欠覆盖：epoch50 经验 95% coverage 只有 `52.0%`，NLL 从 epoch5 的 `689.7` 增至 `2444.3`。这不是简单的 sigma=0，而是预测尺度与长尾误差不匹配。
8. `24_b1.yaml` 明确设置 `search_v3_use_dynamic_sigma: false`。因此 sigma 当前不决定 support 几何；几何使用 learned mean 加固定 `2m/1m` margins。用户观察“sigma 没真正接入几何”属实。
9. epoch50 B1 诊断中，只有 `15.37%` 行出现任何 extension points，平均 extension count `2.53`；真正 extension foreground 非零仅 `1.56%`。因此低激活首先是 support/crop overlap 与数据可观测性问题，不是 B2 网络 gate 单独造成。

## 数据与耦合链路

```text
tracklet + recursive state
  -> candidate view 构造
  -> view0 canonical B0 forward
  -> observation box 写回唯一递归状态
  -> B1 读取递归历史框、真实 delta-t，输出 learned mean / CV / sigma
  -> B1 mean 定义 endpoint/tube；当前 B1 arm 使用固定 margins
  -> expanded support 减去 baseline crop，得到 extension-only points
  -> B2 只读取 view0 extension；base/memory 仅作上下文
  -> B3 在 detached 上游特征上判断 help/harm/gain
  -> 校准通过才施加有界残差，否则精确返回 observation

views1–3 不进入 B1/B2/B3，只通过独立 auxiliary forwards 向 B0 累积加权梯度。
```

关键耦合约束是“参数不冻结，但梯度/状态所有权隔离”：B0、B1、B2、B3 分 optimizer；B1/B2/B3 读取 detached B0 contract；唯一递归提交仍是 observation。当前风险不在概念合同，而在不同 arm 的 B0 随机流、BN buffer、optimizer moment、数据次序是否真的一致。

## 指标模式

| 对照 | Success final | Precision final | Success late-3 | Precision late-3 |
|---|---:|---:|---:|---:|
| CT22 B0 dev | 49.800 | 57.773 | 47.523 | 56.955 |
| v24 B0 dev | 41.079 | 49.916 | 42.051 | 51.590 |
| v24 - CT22 | -8.721 | -7.856 | -5.472 | -5.365 |

2×2 final：

| Arm | Success | Precision | 解释 |
|---|---:|---:|---|
| reseed0/rng0 | 26.065 | 34.684 | 无 reseed，递归训练崩坏 |
| reseed0/rng1 | 32.518 | 35.440 | 无 reseed，仍明显失败 |
| reseed1/rng0 | 34.095 | 46.708 | 报告原本选择的可辩护协议 |
| reseed1/rng1 | 40.847 | 48.383 | 数值最高但属于随机流控制 |
| 最新 v24 B0 | 41.079 | 49.916 | 实际配置为 reseed1/rng1 |

这说明两件事：reseed 是必要训练 curriculum；同时结果对一次无语义 RNG 消耗高度敏感，单 seed 下模型方差很大。最新 B0 与最高 2×2 cell 一致，不能把差距归因于瘦身。

## 原因排序

### P0：candidate/B0 训练事务改变，且跨臂隔离未成立（高概率，部分证实）

- v24 candidate 协议不只是把 `num_candidates` 从 1 改为 4；它改变了在线 batch 拆分、三个额外 B0 forwards、BN running-stat 隔离、辅助梯度累积和随机数执行顺序。
- B0/B1-only 的 candidate0 loss 从 step1 即分叉，说明 B1 的存在改变了本应共享的 B0 prefix。
- `ct_b0_rng_shift_control` 对 final Success 的主效应约 `+6.60`，证明该训练系统对随机流异常敏感。
- 但缺少当前 commit 上的 `B0 candidate1 control` 与 `B0 candidate4` matched run，尚不能把全部 `-8.7/-7.9` 因果归给 candidate4。

### P1：最新实验使用了错误的 2×2 cell（已证实）

- 2×2 报告要求正式关闭 RNG-shift；当前正式 YAML 与最新 provenance 都为 `true`。
- 这不会解释相对 CT22 的全部下降，因为 rng-shift-on 反而是 seed42 高分 cell；但它破坏了注册协议和后续消融可归因性。

### P2：B1 均值任务被长尾递归样本支配（已证实）

- learned mean 与 CV 几乎相同；aux gap RMSE 约 `3.60m`，说明多步预测同样困难。
- median 很小、mean/RMSE 极大，说明少数彻底漂移的递归历史占据损失。
- 直接换 CfC 不会自动修复错误 anchor、不可观测历史或长尾监督。

### P3：sigma 训练与几何消费脱节（已证实）

- 当前 B1 arm 强制 dynamic sigma off；support 使用固定 margins。
- sigma 的经验 coverage 严重不足，不能直接打开 dynamic sigma，否则容易缩小已偏窄的 support，并进一步降低 extension supply。
- 正确顺序应是先校准 sigma，再做 fixed-vs-dynamic 几何对照。

### P4：extension supply 是上游几何/可观测性瓶颈（已证实）

- epoch50 仅 15.37% 行有 extension；只有 1.56% 行有 extension foreground。
- B1-only 中 B2 未启用，所以 acquisition JSON 的 eligible/retained 全零属于预期，不能当作 B2 失败证据。
- `target_in_support` 等历史字段曾存在未写入默认零问题；应以重新计算的真实点/GT诊断和 Full-B3 acquisition 为准。

### P5：源码瘦身（低概率）

- 瘦身前后同一 2×2 cell 数值接近；108 项 contract tests 通过。
- slimming gate 因 HEAD 不等于固定基线 `001951a` 失败是设计行为，不代表回归。
- 尚未完成真实 batch/resume parity，因此不能说风险为零，但它不是当前最符合证据的解释。

## 后续任务顺序与门禁

### 阶段 0：冻结新增创新，修复实验有效性

1. 将正式协议与实际 YAML 的 RNG-shift 冲突处理清楚；如果遵守原 2×2 决策，应设为 false，并使四臂完全一致。
2. 在当前 commit 跑 candidate1 B0 control 与 candidate4 B0，各自 scratch、相同 seed/selection/steps；只比较 final 与 late-3。
3. 增加 init、step1、step100、epoch-end 的 B0 parameter、BN buffer、optimizer state、sampler/RNG hash。
4. 固定真实 batch 比较 B0 与 B1 arm 的 B0 输出、loss、gradient；要求 B0 shared prefix 在允许边界内一致。
5. 在完成上述检查前，不运行 Full-B3/Full 60 epoch，也不接 CfC/BNN/RT-DETR。

Go 条件：candidate4 不显著伤害 B0，或明确选择 candidate1；B0/B1 shared prefix hash 一致；B0 恢复到可信区间。否则回退 candidate 训练实现，不动论文链路。

### 阶段 1：修复 B1 基础任务

1. 按递归 age、gap、历史有效率、B0 当前误差分层报告 learned-vs-CV。
2. 对极端漂移样本采用显式 valid/observability mask 或稳健 loss；不要靠 sigma 吞掉错误标签。
3. 保留 CV residual parameterization 和 zero-init，要求 learned mean 在 final/late-3 都优于 CV。
4. 使用 held-out tracklets 校准二维 covariance；达到合理 coverage 后才开启 dynamic sigma。
5. 运行 fixed margins vs calibrated dynamic sigma；同时报告 support recall、support volume 和 extension foreground recall。

Go 条件：learned mean 稳定优于 CV；NLL 不随训练恶化；95% coverage 接近目标；dynamic sigma 不降低 target-bearing support recall。

### 阶段 2：修复 B2 supply，再接 RT-DETR-style

1. 先运行 Full-B3 的原始 extension acquisition，不训练 B3。
2. 核对 base/expanded/pool/sample 数量、target-bearing rows、retention@K、no-extension fallback、raw/oracle gain。
3. 如果 support 中几乎没有目标，先改 support 几何/冷启动搜索，不改 B2 网络。
4. 有足够 target-bearing supply 后，再加入 RT-DETR-style quality-aware top-K：`targetness - localization uncertainty + observation similarity + motion consistency`。
5. 保持 query 坐标来自真实 extension points，base/memory 不得独立回归新中心。

Go 条件：target-bearing rows 足够、retention@K 高、oracle headroom 为正；否则 RT-DETR-style 只能重新排序背景点，应 No-Go。

### 阶段 3：完成确定性 B3，再接轻量贝叶斯

1. 先训练现有 deterministic B3，并在独立 calibration tracklets 上选阈值。
2. 报告 coverage、harm rate、center/IoU gain 和 tracklet-bootstrap CI。
3. 只有 deterministic B3 存在“coverage 下可控风险”的基本信号，再增加 3-head deep ensemble 或 last-layer Laplace。
4. 贝叶斯版本使用 gain LCB / harm UCB，但仍必须经过 held-out calibration 和 exact fallback。

Go 条件：相同 B2 candidates、相同 coverage 下，Bayesian B3 明显降低 harm 或提高净 gain。

### 阶段 4：最后接 CfC

1. 在 B1 数据、anchor、loss、sigma 校准已经正常后，以参数量匹配的 CfC 替换 GRU。
2. 做 `GRU vs CfC` 与 `true/fixed/shuffled delta-t` 两因素实验。
3. 增加 dropped-frame/held-out cadence；报告 mean RMSE、NLL、coverage、support recall/volume 与最终 tracking。

只有 CfC 在 true time 下超过 fixed/shuffled 且跨 cadence 复现，才写物理连续时间贡献。否则仅作为 backbone 消融。

### 阶段 5：正式论文实验

1. mini seed42：B0、B1、Full-B3、Full，全部 scratch，final/late-3。
2. mini 机制消融：time、sigma geometry、RT-DETR query、deterministic/Bayesian B3。
3. 完整 nuScenes 四臂 seed42。
4. 完整 nuScenes B0/Full seeds43/44，paired tracklet CI。
5. 最后再考虑 memory、其他数据集或多模态扩展。

## 最终判断

- **2×2/candidate 重构：最可能是相对 CT22 的主要回退来源之一，但需 candidate1 matched control 才能完成因果确认。**
- **瘦身：没有显示为主要原因；最新结果反而复现了瘦身前同一 2×2 cell。**
- **candidate 问题：不是 B2 读取 candidate0 这一原则错，而是 B0 的四视图训练事务、随机流、BN/gradient 合成改变了基线。**
- **B1：均值未超过 CV，sigma 欠覆盖，而且当前明确未接入几何。**
- **扩展路径：低激活主要来自 support 与真实目标重叠不足；先修 supply，再做 RT-DETR-style。**
- **创新顺序：RT-DETR-style B2 第一，轻量 Bayesian B3 第二，CfC 最后；但三者都必须排在 B0/candidate 与 B1 基础合同修复之后。**

## 证据来源

- `output/20260822-0330-24_b0-.../run_provenance.json` 与 TensorBoard scalars。
- `output/20260822-0330-24_b1-.../candidate_diagnostics/epoch_*.csv`。
- Git `001951a:compare_results/data/ct22_b0_diagnosis_20260811.json`。
- Git `f320d7e:compare_results/reports/ct24_b0_2x2_seed42_20260818.md`。
- `cfgs/ct_seqtrack/24_b0.yaml`、`24_b1.yaml`、`24_formal_base.yaml`。
- `models/seqtrack3d.py`、`models/base_model.py`、`utils/ct_search.py`、`models/ct_v2/evidence_memory.py`。

## 限制

- 最新 B1 本地副本只包含至 epoch50 的正式验证与 epoch52 训练诊断，不是完整 60-epoch final。
- 当前 dev 是 mini_train 的 37-tracklet atomic dev，不可直接等同标准 mini_val。
- 尚缺当前 commit 的 candidate1 matched run、逐 tracklet paired CI、真实 batch parity 与 resume parity。

## 补充：为什么启动 B0 2×2，以及是否应回到 2×2 前

### 历史动机

2×2 出现在 v24 大规模重构 `96f0baf` 之后。该重构把 CT21/CT22 的联合 online candidate 路径改造成 observation-anchored B0--B3 事务，并把 B0 的递归状态、有限视野 reseed 和无语义 RNG 扰动暴露为独立因素。随后 `9403066` 修复配置继承，`63fcbcd` 修复在线首步状态合同，才运行四格实验。

因此 2×2 的问题不是“candidate4 是否优于旧 baseline”，而是：在已经采用新 candidate/online 事务的前提下，低分究竟来自没有 reseed，还是只来自随机流变化。实验回答了 reseed 必要、RNG 极敏感，但没有回答新 candidate 事务是否保持旧 B0 能力。

### 2×2 前的高分确实存在，但有两种口径

- `25586b2` 下普通 SeqTrack B0：mini_val 106 tracklets，final `51.001/60.893`，75,720 optimizer steps。
- 同 commit 的 B1-only：mini_val 106 tracklets，final `55.397/66.126`，75,720 steps。
- CT22 online B0：atomic dev 37 tracklets，final `49.800/57.773`，51,300 steps。
- 当前 v24 B0：atomic dev 37 tracklets，final `41.079/49.916`，12,780 steps。

旧 SeqTrack/B1 与当前 v24 不同验证人口，不能把 `51/55` 直接减去 `41`；CT22 与当前 atomic dev 更接近，`-8.7/-7.9` 才是较可信的回退量。

### 新 candidate 事务可能造成大幅下降的机制

当前 v24 将 4 views 聚合到一次 optimizer transaction：每 epoch 约 213 次更新。CT22 每 epoch 约 855 次，恰好约 4 倍；旧 SeqTrack/B1 每 epoch 约 1262 次。即使每 epoch 处理的 candidate forward 总量相近，下列量已经改变：

- Adam moment 每个 epoch 的更新次数；
- StepLR 在 epoch20/40 衰减前经历的 optimizer steps；
- canonical 与辅助梯度是逐样本/逐 batch 更新，还是先加权再一次更新；
- BN buffer 是否看到辅助 views；
- 数据顺序与随机增强的组合方式；
- canonical view 的有效权重从独立训练样本变成一次合成梯度中的 0.5。

所以“candidate 只是去掉 B2 扰动，不应影响 B0”只适用于概念层；当前实现同时改了 B0 优化动力学，完全可能产生 5--10 分差距。

### 是否回退

可以回到旧行为做实验，但不应执行整仓 `git reset` 或直接以 `25586b2` 作为新主线。该提交缺少后来修复的首步状态、坐标所有权、extension-only 和 calibration 合同，并包含已经否决的耦合路径。

推荐做“行为回退、接口不回退”三臂验证：

1. `legacy B0 transaction control`：在当前代码接口下恢复 CT22/旧 SeqTrack 的 B0 optimizer-step 粒度、candidate batch 处理与 scheduler 更新预算；B1/B2/B3 全关。
2. `current candidate1 control`：当前 v24 代码、view0 单视图、保持相同 optimizer-step 预算。
3. `current candidate4`：当前正式 B0=4/B2=1。

三臂必须使用同一 37-tracklet dev 或同一 mini_val、同一总 canonical examples、同一 optimizer steps、同一 LR schedule、同一 seed 和 final/late-3。先做 5 epoch trajectory parity，再跑 60 epoch。

若 legacy/current-candidate1 恢复约 49--50 Success，而 candidate4 仍约 41，则可以确认 candidate 聚合事务是主因，并将正式 B0 回退为旧优化行为；B2 仍然只读取 canonical view0，不影响论文的 extension-only 原则。若三臂都低，则应继续定位 v24 B0 数据/递归状态重构，而不能归因 candidate 数量。
