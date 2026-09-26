# v34 B0：显式观测支持与共同 query 上下文

**当前配方更新：**用户已将初始S/W两组扩为八组mini：SeqTrack正常/减半，S/W各1e-4、5e-5、2.5e-5；见 [八组协议](B0_V34_LR_GRID.md)。下文结构与梯度设计保持，原两份5e-5配置仍是结构主对照，新增LR不改模型。最新上传及命令统一以 [启动说明](CTSEQTRACK_V34_MINI_LAUNCH.md) 为准。

2026-09-26。按用户批准方案新增 v34；v33 与独立 SeqTrack 的计算、配置、权重身份和已有证据继续保留。本轮只在本地实现，不上传、不安装服务器环境、不启动或停止训练。实现完成不代表涨分，正式结果尚待 S/W 两组 scratch60。

## 结构与梯度

在原 decoder 的当前四个候选（q0 + 三个 mode）角点 query 中加入共同的 64 维上下文。三个 history query 不读新增上下文；不加 query self-attention、新 KV token、候选、定位头或 hold 门。

- 每个历史槽：anchor-local 中心 XYZ/首帧 LWH、相对 yaw 的 sin/cos、`log1p(age_seconds/time_scale)`、唯一点量、框内软 FG 支持量、平均熵，共 9 维。不存在历史清零后附 exists 位，按 oldest-first 拼成 30 维。合法历史框即使没有点也保留几何与 exists。
- 历史编码：`Linear(30,64) → LayerNorm → GELU`。
- 当前编码：已有 MiniPointNet pooled256 与当前 3 维支持拼接，`Linear(259,64) → LayerNorm → GELU`；没有序列点时清零该分支。
- 两路拼接经 `Linear(128,64)`，最终融合层权重/bias 全零初始化。新模块构造在旧初始化循环之后，用独立 CPU RNG 作用域保持既有权重及后续 B1/B2 初始化流。

点量为 `log1p(n)/log1p(N)`；软 FG 支持为 `log1p(sum(p_fg))/log1p(N)`；熵为有效点平均 Bernoulli 熵除以 log2。历史支持只统计输入历史框内的 FG mass，当前支持统计整个当前 crop；熵均在各帧全部有效点计算，无点为零。支持由当前 forward 重编码的原始点生成，不缓存过去网络特征。支持统计、历史框、时间、几何 membership 均 detach，不能用 GT 前景或离线 raw 点数替代。

`ObservationFeatures` 增加可选 `coarse_features[B,256]`、`history_support[B,3,3]`、`current_support[B,3]`；`DecoderOutput` 增加可选 `query_context_norm[B]`。v33 不生成这些新上下文、不构造其参数；v34 逐帧仅新增小型被动支持摘要和上下文幅度。

pooled256 保持 live，新增 main/mode/quality→Mini/BC 的同帧梯度路径。这是有意改变耦合；概率统计仍 detach，history loss 不通过新分支训练 Mini/BC。新上下文不读取 live vote/mode center，mode query detach、输出 center live 与 pre-vote 边界保持。

## 原两份结构主对照（仍保留，八组扩展见上方）

| 组 | 配置 | 成熟窗口 | 唯一机制差异 |
|---|---|---|---|
| S | `34_b0_context_mini.yaml` | 1/3/3/8 | 相对 v33 C 新增上述结构包 |
| W | `34_b0_context_w4_mini.yaml` | 1/4/4/8 | 相对 S 调整完整短窗口课程 |

共同：mini Car、seed42、batch16、workers4、FP32、scratch60、Adam betas=(0.5,0.999)、eps=1e-6、wd=0；无 warmup。第 1–20/21–50/51–60 轮分别使用 5e-5/5e-6/5e-7。保留四分支、十轮课程、112 尾部预留及 BN 策略。每轮 19,108 行、1,195 更新，每组总计 71,700 更新。

成熟完整预测三历史覆盖由 14.381% 升到 25.183%；W 同时更早进入递推。旧短窗口课程为第 1–5/6–9/10+ 轮对应 1/2/3，新为第 1–3/4–6/7–9/10+ 轮对应 1/2/3/4。因此 W−S 检验完整课程，不能只归因最终窗口。首测实际曝光数量没有增加。

新模型/实验族分别为 `ctseqtrackv34` / `ct_seqtrack_v34`，模型身份 `ct_seqtrack.joint_identity.v34`，checkpoint 使用 `ct_v34_runtime`。v33/reference 保留原身份；S/W 和跨版本 checkpoint 不能交叉续训。已有 R/C/旧 v32 B0 直接复用，不重跑参考、不把工程权重当初始化。

## 验收与去留

比较固定 final60 与 late-3（58/59/60）完整精度 S/P，初始化帧仍按原正式口径计入。总体四项均须达到 R，同时固定曾移动的 31 条轨迹全程（657 个预测帧）的四项不低于 C。移动定义始终为相邻 GT XY 位移≥0.15m，GT 仅用于被动分组。

| 门槛 | final60 S/P | late-3 S/P |
|---|---:|---:|
| R 总体 | 51.823851 / 61.341356 | 51.884391 / 61.966448 |
| C 曾移动轨迹 | 39.912481 / 45.711568 | 38.479199 / 44.679097 |

表内只展示六位小数；工具从原始帧复算，不拿四舍五入表值判断通过。raw0/1–2/3–9/≥10、首测缺测当帧与轨迹、移动轨迹低位移片段、阈值跨越、fine/coarse、失跟右删失及错误 supported 均按固定共同帧条件报告。缺失诊断不能填零。阈值跨越不等于真实起步/停车。

S 达标而 W 不达标则保留 S；两组均达标按 final Success、再 Precision 排序；总体达标但移动守底失败仍不晋级。两组均失败则先检查新增历史/语义实际依赖，再登记下一项改变，不扩 LR/模块矩阵。结构适合 Full 不能豁免掉分；单 seed 达标不代表普适稳定收益。

## 范围与验证

保持 unique/mask、严格身份、物理时间、坐标、首帧尺寸、q0 direct、共享头、唯一 accepted commit、无跨帧图、现有 loss/pooling/quality 目标。quality 在 strong 后筛选过弱及候选间身份效用可比性，进入 Full 前另行处理；本轮不在 mini 评测集调阈值。

必要验证包含空/稀疏/重复点、历史有框无点、GT 标签不影响 forward、梯度与初始化/RNG、旧 C 真实 checkpoint/配置 SHA/固定输入输出、跨身份恢复拒绝、同组 epoch 恢复、60轮预算，以及 pytest/compileall/diff。真实 nuScenes CUDA batch 由用户运行一次；不能把旧 v33 服务器检查登记为 v34 已通过。

启动、单批检查与结果比较见 [v34 操作说明](CTSEQTRACK_V34_MINI_LAUNCH.md)。本地全套408 passed、8项Lightning/CUDA环境跳过；真实网络CPU epoch恢复逐位一致，详情见 [实施报告](../artifacts/ct_checks/20260926-154028_v34_implementation/REPORT.md)。真实CUDA检查与S/W正式训练尚未执行。
