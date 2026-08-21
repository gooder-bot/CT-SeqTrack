# CT-SeqTrack 历史证据索引

本文件只保留会约束当前论文主张的历史结论。原始 CSV、HTML、日志、权重和旧实现
已从活动工作树移出，均可从既有提交
`001951a3aee15fdad6e5d5e32ca02d87bf083f3a` 按下表路径恢复。历史数值不是当前
v24 B0--B3 的正式结果。

## 核心证据

| 对象 | 历史观测 | 允许结论 | 恢复路径 |
|---|---|---|---|
| 历史 SeqTrack/B0 | `53.360 / 64.382`；与后续臂训练轨迹不匹配 | 只作背景数值，不能计算当前模块净增益 | `sum_results.md`；`compare_results/` |
| CT22 matched B0 | `52.196 / 64.707`，相对 SeqTrack 的 paired CI 跨 0 | 建立过可信 observation baseline，但未证明涨点 | `compare_results/data/ct22_ablation_summary_20260811.json` |
| CT22 B1-only | `50.141 / 68.038`；B1 motion error `1.257 m`，CV fallback `1.333 m` | B1 学习先验略优于 fallback；tracking 增益未成立 | 同上 |
| CT22 B2 | 1,986 行中只有 17 行出现新增 extension，target-bearing extension 为 0 | 旧 B2 证据不可辨识，不能支持 recovery claim | `compare_results/data/ct22_minival_test_diagnosis_20260809.json` |
| CT22 B3 | 有效校准证据 `3/704 = 0.426%`，没有合规 Full 结果 | B3 不可评价，不能补写成 0 或正结果 | `compare_results/data/ct22_ablation_summary_20260811.json` |
| 历史 B4 PFTC | final `51.189 / 60.886`；feature std `0.0947 -> 0.0156`；成本约 B0 的 `8.24x` | 当前点级实现 No-Go；B4 只隔离保留，不进入主线 | `compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md` |

## 当前 v24 主张边界

- 新 B0--B3 实验尚未完成，不能声称涨点、稳定、多 seed、SOTA、物理时间因果贡献
  或 memory 贡献。
- B1 必须通过 matched-scratch true/fixed/shuffled 和 held-out cadence 才能支持
  physical-time 因果措辞。
- B2 必须证明 extension-only supply、target-bearing retention 和 raw/oracle gain。
- B3 必须通过独立 tracklet 校准、harm 上界和 center/IoU gain 下界。
- Memory 只有 real 同时超过 empty 与 time-misaligned 才能进入最终配置。

## 恢复方式

读取旧内容只使用现有 Git 对象，不创建新提交或改写历史：

```bash
git show 001951a:path/to/file
```

完整路径、大小、SHA256 和 `commit:path` 已记录在
`docs/slimming_baseline/tracked_files.jsonl`。
