# CT-SeqTrack v24 实验清单

更新时间：2026-08-18

本文件只描述当前 B0--B3 主线。代码完成不等于模块晋级；任何未通过 gate 的
分支都不能进入论文主结果。

## 0. 当前状态

- [x] 建立 `CTSEQTRACK` 与 `b0 / b1 / full_minus_b3 / full` 单变体接口。
- [x] 明确 B0--B3 数据所有权，并保留 evaluator 兼容别名。
- [x] B2 改为 extension-only voting、target-bearing-only regression。
- [x] auxiliary rows 只更新 B2；B3 输入全部 detach。
- [x] memory 使用显式时空元数据并提供 `empty/real/time_misaligned` 对照。
- [x] B3 改为 helpful/harmful/expected-gain 动作风险模型与有界修正。
- [x] calibration artifact、tracklet bootstrap、risk--coverage、fail-closed 已实现。
- [x] optimizer map、梯度范数、step count、parameter hash 与 resume RNG 合同已实现。
- [x] M3、M4、TWC 的正式运行入口已移除；B4 paired views 独立保留。
- [x] 重构前基线为 `243 passed / 1 skipped`；删除旧 M3/M4/TWC、CRPA、旧
  selective-router 测试后，当前 paper-facing suite 为 `109 passed / 1 skipped`。
- [x] B0 2×2 seed42 协议选择实验已完整运行并归档。
- [ ] Acquisition preflight 及其后的模块实验尚未运行；当前不得声称涨点或论文
  结论成立。

## 1. 必须先完成的代码验收

- [x] 全量 `pytest`（`109 passed / 1 skipped`）、`py_compile` 与
  `git diff --check` 通过。
- [ ] 对固定 batch 保存 B0/B1/B2/B3 新入口输出 fixture，并与迁移前 v23 fixture
  核对允许差异。
- [ ] fixture parity 在完整运行环境通过后，物理删除 `SEQTRACK3D` 中仍保留的
  dormant legacy fusion/dynamics 研究分支；当前 `CTSEQTRACK` 已在 dataloader
  构造前拒绝并关闭它们，但尚未冒险删除 B0 共用文件中的全部历史实现。
- [ ] 在真实 Lightning 100-step + resume 测试中核对 optimizer/scaler/scheduler、
  RNG、recursive state 和 module hash。
- [ ] 在 B0、B1-only、Full-B3、Full 的初始化、step1、step100、epoch-end 比较共享
  prefix hash；任一不一致，整组消融作废。
- [ ] selective evaluator 缺失或错配 calibration artifact 时实际启动失败；
  observation fallback 在目标硬件上 bitwise 或规定数值容差内一致。

## 2. B0 2×2：固定代码与协议

- [x] seed42 从头运行 reseed on/off × RNG-shift on/off；四组均完成 60 epoch、
  12,780 optimizer steps 和 12 次 atomic-dev 验证。
- [x] 四组除两个注册因子和运行路径元数据外，candidate0 数据、loss、optimizer、
  scheduler、训练预算与最终 epoch 完全匹配。
- [x] 2×2 历史诊断曾选择 `ct_recursive_reseed_enabled=true`；因果时间候选正式协议已改为
  `ct_recursive_reseed_enabled=false`，避免 GT 重写 B0 history/crop anchor；
  `ct_b0_rng_shift_control=false`；选择的是训练协议，不是可写成贡献的模块。
- [x] 不把 2×2 的任何 checkpoint 作为其他 arm 的初始化；正式四臂仍分别随机
  初始化 B0，并统一使用 final epoch。
- [x] 从现在起不再修改 B0 模型、在线 sampler、candidate0、loss、optimizer、
  scheduler 或该协议；若修改，必须重跑 2×2。
- [x] resolved config、代码 commit、数据 selection hash、RNG state 与四个 final
  checkpoint SHA256 已核验并记录在
  `compare_results/reports/ct24_b0_2x2_seed42_20260818.md`。
- [x] 结果只来自 mini_train 的 37-tracklet atomic dev，不是 mini_val；完整数值、
  因子效应、限制与后续顺序见上述报告。

## 3. Acquisition preflight

- [ ] 使用完整 train/dev tracklets，无 checkpoint、无 `--max-batches` 截断。
- [ ] candidate0 至少 100 个 target-bearing extension rows。
- [ ] candidate0 target-bearing row retention >= 50%。
- [ ] dev c1 boundary-role、c2 outside-role satisfied rows 各至少 100；
  同时报告 gap 选择率、`r_g` 分布、截断率、coverage 与耗时。
- [ ] train extension 点同时含正负类，artifact 生成有限的 targetness 正负权重。
- [ ] 若不通过，只调整预注册 support 几何/采样；不得先训练 Full 再解释。

## 4. 四臂 kill test（全部重新 scratch）

- [ ] B0：3--5 epoch。
- [ ] B1-only：3--5 epoch，最终输出精确等于 observation。
- [ ] Full-B3：3--5 epoch，B2 仅 shadow 学习。
- [ ] Full：3--5 epoch，B3 仅 shadow 学习，不写 recursive state。
- [ ] kill-test checkpoint 不续训为正式实验。
- [ ] 检查 absence rows 的 raw loss/gradient 为零、presence gradient 非零。
- [ ] 检查 auxiliary loss 对 B0/B1/B3 梯度严格为零。
- [ ] 检查 no-extension counterfactual 不产生候选增益。

## 5. seed42 60-epoch 正式筛选

- [ ] 四臂从头重跑，统一使用 final epoch，不跨 arm 选择不同 best epoch。
- [ ] B1 报告 mean-vs-CV RMSE、NLL、二维 coverage、support recall/volume，并按
  time gap、稀疏度、recursive age 分层。
- [ ] B2 报告 supply、target-bearing retention、presence、raw candidate gain、oracle
  headroom、harm 与 no-extension counterfactual。
- [ ] Full-B3 只有同时具备正 oracle headroom 和正 expected gain，才启动 B3 校准。
- [ ] Full 冻结阈值后只评估一次 mini_val；Success 和 Precision 未同时改善则停止，
  不补多 seed。

## 6. B3 calibration gate

- [ ] calibration tracklets 与 train/dev/test manifest 分离。
- [ ] artifact 绑定 final checkpoint SHA、正式 config identity、tracklet manifest SHA、
  score definition 与 artifact SHA。
- [ ] action tracklets >= 30；actions >= 100；coverage >= 1%。
- [ ] tracklet-bootstrap 单侧 95% harmful-rate 上界 <= 5%。
- [ ] center gain 与 IoU gain 单侧 95% 下界均 >= 0。
- [ ] 不通过时 artifact 标记失败，最终方法只输出 observation。
- [ ] 报告完整 risk--coverage 曲线和置信区间，不宣称分布无关保证。

## 7. 机制对照与条件晋级

- [ ] `Full-CV` 从头训练，与 learned physical prior 匹配比较。
- [ ] true/fixed/shuffled physical time 均从头训练；只有 true 稳定胜出且在
  held-out stride/gap 成立，才保留 physical-time 因果表述。
- [ ] memory `real/empty/time_misaligned` 预算、token 数、channel、mask 完全匹配。
- [ ] real 必须同时优于 empty 和 time_misaligned，且 paired CI 方向稳定；否则最终
  `memory=none`。
- [ ] B4 默认关闭；只有独立 point/decoder consistency 机制 gate 通过才讨论附录。

## 8. 完整数据与论文结果

- [ ] 架构和阈值协议冻结后，在完整 nuScenes 运行 seeds 42/43/44。
- [ ] 报告 final-epoch 均值、标准差、tracklet paired CI、risk--coverage 与 failure cases。
- [ ] 当前多次使用的 mini_val 不作为最终未触碰测试集。
- [ ] 每个论文主张都绑定表格/图/manifest/checkpoint hash；负结果同样保留。

## 9. Claim guardrails

在实验通过前禁止写：SOTA、统计稳定涨点、physical-time 因果贡献、memory 贡献、
严格 conformal/分布无关保证。任何情况下都不把 agreement 本身称为 reliability，
不把固定正态分位数称为二维已校准 coverage，不把 predicted-box 内点称为真实
foreground，也不用 CT21/CT22、Search=0 或不同训练轨迹证明当前模块贡献。
