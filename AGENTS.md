# CT-SeqTrack 工作约定（2026-09-22，v32 B0 修复）

## 当前范围

- 用户已批准 v32 B0 修复方案。当前活动实现原位演进，保留 `models/ct_v31/` 与 host 文件路径；新增独立 SeqTrack 对照，不复制一套旧 CT 算法树。v31 通过 Git `b1d886e` 复现。
- 本轮修 B0、递推训练、必要的共享 decoder 几何适配及独立对照。B1/B2/B3 机制与其监督不修订。保留 nuScenes mini/full、KITTI、CfC/GRU、模块臂与时间控制能力。
- 唯一活动项目为本仓库。所有命令从本仓库执行；不修改兄弟冻结参考项目，不读取 TrajTrack。
- 服务器 `lishengjie@10.109.253.86`、项目 `/home/lishengjie/study/lcyu/CT-SeqTrack` 当前仅允许只读；不得自行上传、安装、启动训练或停止进程。

## 当前模型与实验约定

- 唯一入口为 `main.py`，entry 保留 `models/ct_v31/entry.py` 路径，host 为 `models/ctseqtrackv31.py`；当前模型/实验身份为v32。
- Full 同帧联合训练，四假设共用定位/质量头；不同臂 B0 不要求逐位一致。sigma、离散几何、GT 和跨帧 detach 保留；无旧策略校准依赖。
- 每个臂从 epoch0 随机初始化，不冻结启用参数。`--init_checkpoint` 禁止；`--checkpoint` 仅用于同身份 epoch 边界恢复或评测，工程权重不能进入正式初始化。
- 保留四份 `cfgs/ct_seqtrack/31_*.yaml` 的已有内容；当前入口拒绝旧身份，使用 `32_*` 配置及 v32 checkpoint schema。按用户最新安排，本轮为 SeqTrack reference、B0、Full-GRU、Full-CfC 各 seed42，物理 GPU 依次 0/0/1/1。60轮/batch16/workers4/FP32、Adam(.5,.999)/eps1e-6/lr1e-4、StepLR20轮×.1、每5轮验证。最终自动评测58/59/60，不用最佳轮次替代 final60。原 B0/reference seed52 复验另列后续，不将本轮四组当作双 seed 验收。
- 不使用旧 preloading；每 worker 原始点云缓存256MiB。当前网络为 PyTorch 路径，不要求旧 PointNet++ CUDA 扩展。
- 完整定义见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)，后台命令与 tail 见 [v32 mini 运行说明](docs/CTSEQTRACK_V32_MINI_LAUNCH.md)，操作入口见 [工具面](docs/FORMAL_TOOLING.md)。不恢复旧24–30版本门禁或标定流程。

## 最新事实与研究边界

2026-09-21 已核实 `64ad056` 三臂 mini Car 真实60轮与58–60评测完成；final S/P 为 B0 23.959519/27.834792、CfC 22.491247/29.352297、GRU 22.839169/31.844639。旧“未上传/未训练”已被覆盖，两个 Full 的 final Success 仍低于 B0，性能条件未达成。

当前方向、原始分母与诊断限制见 [待办](need_to_do.md)、[最新问题](最新问题.md) 和 [9/21报告](artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)。不声称涨分、SOTA、时间/记忆因果收益；不把 CPU 探针、候选 oracle 或 BN 交换当新正式成绩。

## 保护范围

- `output/` 和既有 `artifacts/` 的数据、日志、权重、报告、脚本与图均不删除、不覆盖，不运行 `git clean` 清除 ignored 证据。
- v31精简前完整源码与两份当时未提交状态文档保存在 `artifacts/ct_checks/20260922_v31_slimming/before/`，仅只读；v32修改前活动树为Git `b1d886e`。
- 新工程检查产物放入 `artifacts/ct_checks/` 的独立目录。旧配置/文档恢复使用 `git show COMMIT:path`；不改写历史、不覆盖正式结果。

## 验证

```bash
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

修复需要核对局部/世界几何、loss/梯度、稀疏观测、BN、Adam、递归提交及epoch恢复，独立SeqTrack另核原数学与采样审计。真实数据或CUDA未执行须明确说明。旧固定HEAD精简门禁只属历史，不作为v32验收。

项目文档和注释以中文为主。v31精简审计保留只读；v32当前实现和验证结论统一记录在 `docs/B0_V32_REPAIR.md`，新合成入口产物位于独立 `artifacts/ct_checks/v32_integration/`。
