# CT-SeqTrack 工作约定（2026-09-22）

## 当前范围

- 用户已确认工作树只维护 v31；旧版源码、配置、测试、工具和文档从活动树移除，通过 Git `64ad056` 及 [历史索引](docs/HISTORY_EVIDENCE_INDEX.md) 复现，不新增旧源码归档目录。
- 本轮为行为等价清理，不实施 [最新问题](最新问题.md) 中的算法修订。保留 v31 当前支持的 nuScenes mini/full、KITTI、CfC/GRU、模块臂与时间控制能力，不缩减为仅三份 mini 配置。
- 唯一活动项目为本仓库。所有命令从本仓库执行；不修改兄弟冻结参考项目，不读取 TrajTrack。
- 服务器 `lishengjie@10.109.253.86`、项目 `/home/lishengjie/study/lcyu/CT-SeqTrack` 当前仅允许只读；不得自行上传、安装、启动训练或停止进程。

## 当前模型与实验约定

- 唯一入口为 `main.py`，v31 entry 为 `models/ct_v31/entry.py`，host 为 `models/ctseqtrackv31.py`，联合模型在 `models/ct_v31/`。
- Full 同帧联合训练，四假设共用定位/质量头；不同臂 B0 不要求逐位一致。sigma、离散几何、GT 和跨帧 detach 保留；无旧策略校准依赖。
- 每个臂从 epoch0 随机初始化，不冻结启用参数。`--init_checkpoint` 禁止；`--checkpoint` 仅用于同身份 epoch 边界恢复或评测，工程权重不能进入正式初始化。
- 保留四份 `cfgs/ct_seqtrack/31_*.yaml` 的已有内容。正式预算为 60 轮/batch16/workers4/seed42/FP32、Adam(.5,.999)/eps1e-6/lr1e-4、StepLR20轮×.1、每5轮验证。最终自动评测58/59/60，不用最佳轮次替代 final60。
- 不使用旧 preloading；每 worker 原始点云缓存256MiB。当前网络为 PyTorch 路径，不要求旧 PointNet++ CUDA 扩展。
- 完整定义见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)，操作入口见 [工具面](docs/FORMAL_TOOLING.md)。不恢复旧24–30版本门禁或标定流程。

## 最新事实与研究边界

2026-09-21 已核实 `64ad056` 三臂 mini Car 真实60轮与58–60评测完成；final S/P 为 B0 23.959519/27.834792、CfC 22.491247/29.352297、GRU 22.839169/31.844639。旧“未上传/未训练”已被覆盖，两个 Full 的 final Success 仍低于 B0，性能条件未达成。

当前方向、原始分母与诊断限制见 [待办](need_to_do.md)、[最新问题](最新问题.md) 和 [9/21报告](artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)。不声称涨分、SOTA、时间/记忆因果收益；不把 CPU 探针、候选 oracle 或 BN 交换当新正式成绩。

## 保护范围

- `output/` 和既有 `artifacts/` 的数据、日志、权重、报告、脚本与图均不删除、不覆盖，不运行 `git clean` 清除 ignored 证据。
- 本次变更前完整源码与两份未提交状态文档保存在 `artifacts/ct_checks/20260922_v31_slimming/before/`，仅只读。它补充 Git 中未记录的9/21新增内容，不能仅凭旧 HEAD 恢复当前研究状态。
- 新工程检查产物放入 `artifacts/ct_checks/` 的独立目录。旧配置/文档恢复使用 `git show COMMIT:path`；不改写历史、不覆盖正式结果。

## 验证

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

清理需要核对配置与 checkpoint 身份、固定输入输出/loss/梯度/BN/Adam、递归提交及 epoch 边界恢复。真实数据或 CUDA 未执行的检查须明确说明，不能由本地通过推断。旧 `verify_ct_slimming.py` 固定 HEAD=`001951a` 的门禁仅属历史，不作为 v31 验收。

项目文档和注释以中文为主，编辑时匹配周围语言。本次审计统一记录在 `artifacts/ct_checks/20260922_v31_slimming/`，不要散布多个相互覆盖的“最新完成”结论。
