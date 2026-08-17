# CT-SeqTrack 论文可行性、主张边界与后续执行路线

> **TrajTrack 参考状态（2026-08-16）**：自本标注起，TrajTrack 不再作为
> CT-SeqTrack 后续方法设计、Gate/proposal 机制选择、超参数设定或性能有效性的
> 参考依据；仅保留为必须引用的相关工作、历史审计对象和 GT-free 评测警示。
> 下文既有 TrajTrack 内容均为历史记录，不再驱动当前或未来方案。

更新时间：2026-07-20

> **2026-07-21 结果回填**：同提交 TWC A/B/C seed42 已完成。`C-B` final 为 `+8.31 Success / +11.74 Precision`，但 paired-view 的 `B-A` 为 `-15.30/-24.18`，最终 `C-A=-7.00/-12.44`。因此 consistency 相对 paired control 的净效应成立，但 standard guardrail 失败，正式判定 `NO_GO_TWC_MAIN_METHOD_PROMOTION`，不补 seed43/44。本文以下内容保留为当时的预注册决策依据；实际结果、图表和新执行顺序见 [twc_abc_seed42_comparison_20260721.md](twc_abc_seed42_comparison_20260721.md)。

本文回答一个具体问题：在 P0-B4 reliability 与 P0-C feature-concat true-dt promotion 均 No-Go 后，CT-SeqTrack 是否仍有论文机会，以及接下来怎样用最少实验区分“可投稿的方法工作”和“只能保留为诊断的研究快照”。

## 1. 决策结论

**仍有论文机会，但当前结果不能直接组成一篇“完整 CT-SeqTrack 方法稳定优于 SeqTrack3D”的论文。**

现在最有价值的部分不是已经失败的多模块 full model，而是三个更窄、可以被严格验证的对象：

1. 同一 tracklet 内的 irregular cadence / burst drop / long gap 评测协议；
2. 保持 checkpoint、几何输入和 endpoint 不变的 `true/fixed/shuffled-dt` 时间干预；
3. 坐标严格对齐后，不同历史采样路径到同一 endpoint 的 consistency。

论文路线必须按证据分叉：

```text
corrected-TWC 的净贡献 C-B 可复现
    -> 方法路线：variable-rate / resampling-consistent 3D SOT

TWC 不成立，但多模型、多数据集均暴露稳定 rate-robustness 缺口
    -> benchmark/diagnosis 路线

residual correction 的 true-dt 明确超过 fixed/shuffled
    -> 才能恢复 timestamp-native / continuous-time 方法主张

三者都不成立
    -> 当前项目不应继续通过增加 gate、memory、Mamba 或 ODE 追求论文
```

当前推荐的第一选择是 **corrected-TWC 的同提交 A/B/C 因果验证**；第二选择是 benchmark/diagnosis。bounded residual 只保留为由 oracle feasibility 触发的备选，不再默认排在 TWC 前面。

## 2. 为什么仍然有机会

### 2.1 问题仍真实存在

P0-B oracle 和 recursive 诊断已经证明：在 long gap 和 burst drop 下，目标可能在模型 forward 前离开 search crop；扩大 crop 会显著增加背景点，却仍不能覆盖全部困难样本。这个问题不是单纯换 backbone 可以解释的。

P0-C 又提供了一个重要的负结果：当前 A2 会对时间输入变化产生响应，但正确 physical-time alignment 没有超过 shuffled 对照。这不是“真实时间无效”，而是说明只做 feature concat 不足以把物理时间转化为可靠收益。

如果把这些证据扩展到完整数据和多个代表性 tracker，可以形成一个清晰问题：

> 现有 3D SOT 方法在固定帧率上取得的性能，是否能够在同一序列内部的不规则观测间隔下保持？模型是真的使用了物理时间，还是只对时间数值产生响应？

### 2.2 协议基础设施已经比模型结果成熟

当前代码已经实现：

- train/eval cadence 分离；
- stable-token manifest、selection hash 和 role/split/protocol fail-fast；
- real time 与 effective time 双字段；
- offline shuffled gap permutation；
- checkpoint/config/manifest/commit provenance；
- 同一 batch 中除 effective time 外其它输入不变的 invariance 检查；
- TWC 的共享 candidate offset、共享 point-sampling seed、`coordinate_anchor` 与 current XYZ fail-fast。

这些能力能够支持一套可审计的 variable-rate benchmark。它们本身还不足以成为方法贡献，但比当前 feature-concat、gate 和 reliability 分支更接近可发表资产。

### 2.3 当前最有希望的方法信号是 corrected-TWC

坐标修复后的 A1+TWC seed42 相对旧配置级 baseline 的 final Success/Precision 为 `+1.49/+5.03`，late mean 为 `+0.99/+2.67`。它还没有同提交 baseline、paired-view weight0 和多 seed，因此只能作为启动因果实验的信号，不能作为论文数字。

## 3. 代码到论文主张的审计

| 对象 | 代码实际做了什么 | 当前证据 | 可以写什么 | 不能写什么 |
| --- | --- | --- | --- | --- |
| A1-order | `main_time_source=order`，主干使用固定顺序 token | 稳定基线 | 保留 SeqTrack3D order semantics | 不能称为真实时间模型 |
| A2 feature-concat | `DynamicsEncoder` 读取 `delta_t_effective`，`z_dyn` 与 observation feature 拼接 | P0-C true 未超过 shuffled | 失败消融、time-control case study | 不能作为 physical-time 正贡献 |
| bounded residual | `d_dyn=v_pred*dt` 后直接加到已预测完整 displacement 的 `d_obs` 上 | correction 约 `1e-7 m` | 当前定义/初始化的 kill-test | 不能称为已验证的保守时间修正 |
| corrected-TWC | 同 endpoint、共享 t-1 anchor/crop/current XYZ，对不同历史路径的最终框做一致性约束 | A1 seed42 单点正信号 | resampling-path consistency 候选 | 不能把 A1-TWC 收益归因于真实 `delta_t` |
| P0-B3/B4 reliability | previous observation 特征预测可见目标 crop miss | 开发集为正，独立 split No-Go | failure diagnosis | 不能作为 uncertainty/reliability 方法贡献 |
| variable-rate protocol | 在 nuScenes 内冻结 cadence、endpoint identity、time intervention 与 provenance | P0-C 工程 PASS | benchmark 基础 | 不能在只有 mini、单模型时称完整 benchmark |

### 3.1 A1 corrected-TWC 不等于 timestamp-aware

`cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml` 中：

```yaml
main_time_source: order
use_dynamics_encoder: false
use_twc: true
```

所以 A1-TWC 即使最终稳定提升，也只能证明：

> 对同一 endpoint 的不同历史采样路径施加一致性约束，可以提高 rate/resampling robustness。

它不能单独证明模型利用了真实 timestamp。若最终只有 TWC 成立，论文标题和 claim 应从 `continuous-time` 收窄为 `variable-rate` 或 `resampling-consistent`。

### 3.2 当前 residual 的 correction 定义存在结构性歧义

`motion_mlp` 的 `motion_pred[:, :3]` 已经由 `motion_label[:, 0, :3]` 监督为从上一框到当前框的完整 displacement。`DynamicsEncoder` 的 `dynamics_displacement_pred` 也由同一个 displacement label 监督。当前 residual 路径计算：

```text
d_final = d_obs + alpha * scale * clamp(d_dyn)
```

这可能重复叠加完整运动，而不是预测 observation error。更合理、但仍需 oracle 先验收的定义是 proposal interpolation：

```text
r       = clamp(d_dyn - stopgrad(d_obs), R)
d_final = d_obs + alpha * r
```

它等价于在 observation proposal 与 dynamics proposal 之间做有界移动，而不是把两个完整 displacement 相加。

在改代码或正式训练前，应离线计算每个 crop-reachable endpoint 上线段 `d_obs -> d_dyn` 的 oracle 最优点。如果 oracle blend 在 long-gap/sparse 子集也没有稳定净收益，直接终止 residual；如果 oracle 有明显空间，再固定一个定义和初始化做一次 true/fixed/shuffled seed42。

### 3.3 当前 benchmark 仍缺两个关键能力

1. virtual-rate manifest 和 effective-time intervention 当前只接入 nuScenes；`datasets/__init__.py` 的 Waymo 分支尚未传入相同协议参数。
2. 当前 base test loop 主要汇总全局 Success/Precision，尚未统一保存 endpoint/per-tracklet 输出、首次失控、连续失败与跨协议 paired delta。

因此现有代码可以支撑 protocol paper 的核心，但还不能直接支撑完整 benchmark 主表。

## 4. 与相邻工作的差异边界

不能再使用以下宽泛新颖性：

- 首次利用多帧历史：SeqTrack3D 已经使用历史点云和历史框序列；
- 首次研究 HTV：HVTrack 已构造不同 frame interval 的 KITTI-HV；
- 首次用高效长时序模型：MambaTrack3D 已针对 HTV 使用状态空间模型；
- 首次使用 bbox trajectory：TrajTrack 已提出 trajectory-based paradigm；
- 首次 temporal consistency：ChronoTrack 已使用 temporal consistency 和 memory cycle consistency；
- 首次通用 3D SOT benchmark：GSOT3D 已提供大规模通用 benchmark。

仍可防御、但必须通过实验建立的窄边界是：

1. **within-track irregular cadence**，而不是只在整条序列上固定 frame interval；
2. **matched physical-time interventions**：同 checkpoint、同 endpoint、同几何输入下比较 true/fixed/shuffled；
3. **endpoint-conditioned history resampling consistency**：同一当前时刻、同一局部坐标和 current points，仅改变合法历史采样路径；
4. **rate generalization**：一个模型跨 seen/unseen drop schedule，不按协议分别重训。

相关工作入口：

- [SeqTrack3D](https://arxiv.org/abs/2402.16249)
- [HVTrack / 3D SOT under High Temporal Variation](https://arxiv.org/abs/2408.02049)
- [MambaTrack3D](https://arxiv.org/abs/2511.15077)
- [TrajTrack](https://arxiv.org/abs/2509.11453)
- [ChronoTrack](https://openaccess.thecvf.com/content/CVPR2026F/html/Yoo_Temporally_Consistent_Long-Term_Memory_for_3D_Single_Object_Tracking_CVPRF_2026_paper.html)
- [GSOT3D](https://openaccess.thecvf.com/content/ICCV2025/html/Jiao_GSOT3D_Towards_Generic_3D_Single_Object_Tracking_in_the_Wild_ICCV_2025_paper.html)

## 5. 推荐路线 A：方法论文

### 5.1 建议定位

如果 TWC 成立，优先使用：

```text
Variable-rate 3D SOT with Endpoint-consistent History Resampling
```

而不是提前使用：

```text
Continuous-Time 3D Tracking
```

候选贡献可以收敛为：

1. 一个 within-track irregular-cadence、held-out-schedule 的 3D SOT 评测协议；
2. 一个只改变历史路径、严格共享当前 crop/coordinate/points 的 endpoint consistency objective；
3. 一套区分 paired-view augmentation、consistency loss 与 physical-time alignment 的 matched controls。

只有 explicit-dt residual 后续通过 true/fixed/shuffled，才增加第四条 timestamp-conditioned correction 贡献。

### 5.2 第一优先级：TWC A/B/C seed42 因果实验

必须同 commit、同数据 manifest、同 seed、candidate4、optimizer steps 和 checkpoint 规则：

```text
A. single-view A1
B. paired-view A1 + twc_weight=0
C. paired-view A1 + corrected-TWC
```

解释：

- `B-A`：paired history-view augmentation 的收益和代价；
- `C-B`：consistency loss 的净贡献；
- evaluation-only multi-path variance：TWC 是否真的降低同 endpoint 的历史路径敏感性。

建议预注册 Go 条件：

- `C-B` 在 gap1124 和 burst-drop 的 final Success/Precision 都不为负，且至少一个主指标达到有意义的最小增益；
- standard 不出现超过预注册容忍度的退化；
- 同 endpoint center/angle variance 明确下降；
- 收益不能由 best epoch、额外 supervised loss、不同 optimizer steps 或坐标/crop 差异解释。

seed42 不通过，停止 TWC；seed42 通过后才补 seed43/44，并使用 tracklet-level paired bootstrap，而不是 frame-level bootstrap。

### 5.3 第二优先级：显式时间机制只走 oracle-triggered residual

不要恢复 feature concat，也不要重新开发 reliability gate。顺序应是：

1. P0-C-D1 保存 endpoint/per-tracklet 三路输出，定位 true/fixed/shuffled 差异来自哪里；
2. crop-reachable subset 计算 observation/dynamics 两 proposal 的 oracle convex-blend 上限；
3. oracle 不通过则停止 residual；
4. oracle 通过才改成 `d_obs + alpha*clamp(d_dyn-stopgrad(d_obs))`，固定非零但保守初始化；
5. 只跑一个 seed42 的 true/fixed/shuffled；true 必须同时超过两个负对照；
6. 通过后再补 seed 和完整数据。

## 6. 推荐路线 B：benchmark/diagnosis 论文

如果 TWC 与 residual 都没有因果正信号，仍可以保留论文机会，但必须从“我的模型涨点”转成“现有 tracker 是否真正 rate-robust”。

最低完整度：

- 至少两个数据集；当前首选完整 nuScenes，加 Waymo 或 KITTI；
- 至少五类代表方法：two-frame motion、sequence、memory/HTV、trajectory、当前强方法；
- standard、fixed interval、irregular gap、burst drop、random drop、held-out schedule；
- 同 checkpoint 跨 schedule，不能每个协议分别训练后宣称泛化；
- 报告 Success/Precision、相对性能下降、worst-schedule、gap/displacement/sparsity bins、首次失控、连续失败、empty crop 与 FPS；
- 区分 sequence-level realistic evaluation 与 matched-endpoint mechanism diagnosis；
- 发布 manifests、hash、protocol builder、endpoint logs 和复算脚本。

只有 nuScenes-mini、A1/A2 两个模型和若干负结果，不足以构成 benchmark 论文。扩到完整数据、多模型和独立 schedule 后，这条路线才成立。

## 7. 最小执行矩阵与停止条件

### Phase 0：冻结问题与输出能力

- [ ] 完成 P0-C-D1 endpoint/per-tracklet logger；不改变预测路径。
- [ ] 固定论文主问题、主指标、checkpoint 规则和 tracklet-level 统计方法。
- [ ] 明确 A1-TWC 是 rate consistency，不是 explicit timestamp modeling。

### Phase 1：最强候选的单 seed 因果筛选

- [ ] 同提交完成 TWC A/B/C seed42。
- [ ] 在 standard、gap1124、burst-drop 和一个 unseen schedule 上评估同一 final checkpoint。
- [ ] 保存 multi-path prediction variance、center/angle gap、per-tracklet delta 与计算开销。

### Phase 2A：TWC 通过

- [ ] 补 seed43/44，报告 mean±std 和 paired bootstrap CI。
- [ ] 迁移完整 nuScenes，而不是继续在 mini 上调参。
- [ ] 接入第二数据集的 manifest/provenance/endpoint logger。
- [ ] 增加 SeqTrack3D、HVTrack/MambaTrack3D、TrajTrack GT-free 等公平基线。

### Phase 2B：TWC 失败

- [ ] 只做 residual oracle blend，不直接训练。
- [ ] oracle 通过才做一个修正公式的 seed42 true/fixed/shuffled。
- [ ] oracle 或 time control 再失败，正式停止方法扩展，进入 benchmark 路线。

### 总停止条件

以下任一成立，就不再增加复杂模块：

- `C-B` 不优于 0，或 TWC 只复现 paired-view augmentation；
- residual oracle blend 在 long-gap/sparse reachable subset 没有稳定空间；
- residual true-dt 不超过 fixed/shuffled；
- positive signal 只存在于 mini、单 seed、best checkpoint 或单一 tracklet；
- 第二数据集/完整数据方向反转。

## 8. 论文成稿前的证据底线

### 可以投稿完整方法论文

需要同时满足：

- 一个简单机制在同提交负对照中有净贡献；
- 三 seed 和完整数据稳定；
- 至少一个第二数据集或强公开 HTV 协议；
- 一个模型跨未见 cadence，而不是按协议重训；
- mechanism metric 与 tracking metric 同方向；
- 相邻工作对比与代码复现公平。

### 可以投稿 benchmark/analysis 论文

需要同时满足：

- 多数据集、多模型和统一 protocol adapter；
- stable manifests、endpoint logs、hash/provenance 可公开；
- 有可复现的新 failure taxonomy 或 ranking reversal；
- matched-endpoint 与 realistic recursive evaluation 都完整；
- 至少提供一个简单、可信的 baseline 或 calibration recommendation。

### 当前快照还不够

当前只有 mini 数据、部分单 seed 正信号和多个 No-Go。它非常适合作为研究决策依据，但还不能支撑“新模型有效”的投稿结论。

## 9. 最终建议

继续做，但改变下注方式：

1. 不再试图同时挽救 dynamics、reliability、gate、dual-anchor 和 TWC；
2. 先用 A/B/C 判断 corrected-TWC 是否是真正的净方法贡献；
3. 把 protocol/provenance/negative controls 发展成无论方法成败都能保留的论文资产；
4. explicit timestamp 方法只允许通过 residual oracle feasibility 后复活；
5. 根据最终证据选择论文名字，而不是先固定“continuous-time”标题再解释负结果。

最诚实也最有机会的策略是：**保住 variable-rate 问题，严格筛掉无效机制，让最终论文只有一条成立的方法主张。**
