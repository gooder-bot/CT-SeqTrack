# CT-SeqTrack v25 保守式整理报告

## 结论

当前工作树已收敛为一条 v25 正式实现，外部入口、16 个配置文件名、B0→B1→B2→B3
顺序、数据字段、候选角色、detach/optimizer 所有权、observation recursive state 和
evaluator 输出契约保持不变。本次整理没有创建提交、没有改写提交历史、没有推送；
`HEAD`、`main`、`origin/main` 仍指向
`9ed2afc776a852d0f9551db4e61c56cf4c54312c`。

## 恢复基线

- 恢复标签：`ctseqtrack-v25-pre-cleanup-9ed2afc`，指回源提交 `9ed2afc`。
- 仓库外 bundle：`../CT-SeqTrack-v25-pre-cleanup.bundle`，大小
  `3,091,856,873` bytes，已验证。
- 清理前测试基线：`154 passed, 1 skipped`。
- 16 个 v25 resolved-config 快照：
  `tests/fixtures/ct_v25_resolved_configs.json`。
- 基线与服务器 fixture 要求：
  `tests/fixtures/ct_v25_cleanup_baseline.json`。

历史文件只从当前工作树移除，仍可由标签或 bundle 恢复；没有执行 rebase、reset、
amend、commit 或历史重写。

## 结构收敛

- `models/ctseqtrack.py` 与 `models/seqtrack3d.py` 保持正式兼容入口；
  `models/seqtrack3d.py` 收敛为 324 行的组合层。
- B1/B2/B3、forward/loss、训练与推理样本、recursive runtime、优化事务、校准、
  provenance 和 evaluator diagnostics 已按职责迁入 `ctseqtrack/`。
- `models/base_model.py` 只保留当前 joint-Full evaluator schema，不再承载多代
  proposal diagnostics。
- 训练与推理数据路径只接受 joint contract v3 和 point-evidence contract v2；旧
  contract 的 false 分支已删除，但当前真分支的语句顺序和返回字段不变。
- 正式生产 Python 文件均不超过 1,200 行；当前较大的文件为
  `inference.py` 1,127 行、`search.py` 1,073 行、`outputs.py` 860 行和
  `sample_builder.py` 758 行。
- 当前工作树移除了 v24/ct_v2 配置、旧 dynamics/observability/router/B4/replay
  分支、历史结果、transfer 包、临时图像、旧报告和已跟踪预训练权重。

## 配置与训练约束

- 16 个 `cfgs/ct_seqtrack/25*.yaml` 入口保留；继承最多为
  `entry → mini/full profile → common`。
- `ct_variant` 是唯一正式模块组合入口，公开 YAML 不再暴露历史版本开关或 contract
  版本选择。
- `--init_checkpoint` 及训练初始化加载路径已删除。
- 所有启用模块从随机初始化、epoch 0 开始，且启用参数必须
  `requires_grad=True`；detach 仍按既有梯度所有权合同执行。
- 训练 `--checkpoint` 只允许同实验、同完整语义配置哈希、epoch 边界和 artifact
  身份匹配的恢复；评估仍可加载明确指定的待测 checkpoint。
- epoch-boundary checkpoint 记录 recursive-state boundary；恢复时验证后按与未中断
  运行相同的规则清空 state，同时恢复 optimizer、scheduler、scaler 和 RNG。

## 本地验证

- 16 个配置通过适配后逐字段、逐类型等价检查。
- 聚焦数据/contract 回归：`42 passed`。
- 最终当前 v25 测试集：`98 passed, 1 skipped`。
- Black 格式检查、`compileall`、`git diff --check`、服务器脚本 shell 语法和三份
  JSON artifact 解析均通过。
- 最终测试数量少于清理前基线，是因为只服务旧版本的测试随旧实现一起删除，不表示
  当前 v25 测试失败；中期在删除旧测试前仍得到 `154 passed, 1 skipped`。
- 结构测试持续检查：旧 CLI/导入/contract 分支不回流、公开 YAML 无历史开关、正式
  文件行数上限、sample output 上下文完整、handoff 路径有效以及跨配置 resume 拒绝。

本机只记录过一次环境缺失：直接项目运行环境缺少 `easydict`/完整训练依赖。因此没有
反复安装或探测 CUDA、nuScenes、Lightning，也没有伪造四臂真实张量 bitwise fixture。
四臂中间输出、loss、state-dict/optimizer/梯度归属、一步参数哈希、recursive state
和 RNG 的真实数据等价验收明确留给服务器脚本。

## 服务器剩余验收

服务器仍须完成四臂 2-batch smoke、Full 20-batch/H3、5-epoch kill/resume、完整
acquisition preflight、四臂 scratch training、B3 selection/audit calibration、
observation/raw/selective evaluation、scene bootstrap、risk--coverage 与 module hash
audit。命令与证据清单见 `docs/CTSEQTRACK_V25_SERVER_ACCEPTANCE.md`。

这些实验完成前，不声称性能增益、稳定性、SOTA、物理时间或 memory 的因果收益。
