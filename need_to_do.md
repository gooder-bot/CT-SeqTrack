# CT-SeqTrack v25 当前进展与后续实验

## 本地整理已完成

- [x] 基线提交固定为 `9ed2afc`，创建恢复标签和仓库外完整 Git bundle。
- [x] 保存 16 个 v25 清理前 resolved-config 快照。
- [x] 配置整理为 common、mini/full profile 和 16 个兼容入口。
- [x] `ct_variant` 成为正式模块组合入口；旧版本开关退出公开 YAML。
- [x] 删除 `--init_checkpoint`；所有正式 arm 只允许 scratch 或严格同运行恢复。
- [x] B1/B2/B3 contract、forward、loss、sample builder、recursive state、
  optimizer/AMP、online/H3、calibration、provenance 已迁入 `ctseqtrack/`。
- [x] 删除 v24/ct_v2 配置、历史结果目录、transfer 包和已跟踪预训练权重。
- [x] 中期本地回归为 `154 passed, 1 skipped`。
- [x] 删除正式数据路径中永远不会执行的旧 contract 分支；保留 v25 真分支。
- [x] 16 个配置通过适配后逐字段、逐类型等价检查。
- [x] 最终当前测试集为 `98 passed, 1 skipped`；`py_compile`、格式、脚本语法、
  JSON 和 `git diff --check` 均通过。
- [x] `HEAD/main/origin/main` 仍为 `9ed2afc`，暂存区为空；未创建提交、未改写
  历史、未推送。

## 服务器验收

- [ ] 完整 acquisition preflight，不截断 batch。
- [ ] 四臂各 2-batch smoke；Full 至少 20 batches 并覆盖 H3。
- [ ] 四臂 5-epoch epoch-boundary kill/resume 对照。
- [ ] B0、B1、Full-B3、Full 全部从随机初始化正式训练。
- [ ] B3 在 `calibration_select` / `calibration_audit` scene 上两阶段校准。
- [ ] 同一 Full checkpoint 的 observation/raw/selective 评估。
- [ ] scene-paired bootstrap、risk--coverage、module hash audit。

服务器命令和应保存证据见
`docs/CTSEQTRACK_V25_SERVER_ACCEPTANCE.md`。真实数据结果通过前，不声称稳定增益、
SOTA、物理时间因果收益或 calibration 的分布无关保证。
