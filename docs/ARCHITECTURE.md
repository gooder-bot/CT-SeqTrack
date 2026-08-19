# CT-SeqTrack v25 代码结构

正式入口保持为 `net_model: CTSEQTRACK`、`models/ctseqtrack.py` 和
`models/seqtrack3d.py`。`SEQTRACK3D` 只负责组合正式组件并保留原有公开类名、
模块属性名和 state-dict key；研究逻辑按职责放在 `ctseqtrack/`。

```text
ctseqtrack/
  config.py                  # ct_variant 适配和正式配置约束
  contracts.py               # B0/B1/B2/B3 张量所有权合同
  model/
    builder.py               # 模块构造及稳定属性名
    observation.py           # B0 observation 与 B1 prior 调度
    prior.py                 # B1 physical-time prior
    evidence.py              # B2 evidence 与 B3 updater
    forward.py               # B0→B1→B2→B3 的固定执行顺序
    base_losses.py           # B0/B1 loss transaction
    losses.py                # B2/B3 mask、detach 与 loss transaction
    prepass.py               # 只读历史框和时间的 B1 causal prepass
  data/
    sample_builder.py        # motion_processing_mf 正式训练样本
    auxiliary.py             # B1 auxiliary microbatch 的纯构造逻辑
    outputs.py               # flat sample dictionary
    inference.py             # 正式在线推理输入
    search.py                # B1/B2 support 几何与确定性 extension sampling
    recursive.py             # observation recursive state
  runtime/
    online.py                # causal candidates、H3 shadow 与 state commit
    optimization.py          # 独立 optimizer/AMP/clip/scheduler transaction
    checkpointing.py         # 同运行 epoch-boundary resume
    diagnostics.py           # 模块哈希和训练诊断
    evaluation.py            # 论文使用的 joint-Full evaluator 字段
    acquisition.py           # checkpoint-free acquisition preflight
    calibration.py           # B1/B3 calibration 与 risk–coverage
    contracts.py             # scratch/resume/promotion identity
    provenance.py            # 运行身份记录
    scene_bootstrap.py       # scene-paired bootstrap
```

历史 `models/ct_v2/`、旧 dynamics/observability、replay cache、B4 和多代
proposal evaluator 已从当前工作树删除；需要时只能通过
`ctseqtrack-v25-pre-cleanup-9ed2afc` 标签或外部 Git bundle 恢复。

## 固定数据链路

```text
dataset
→ scene_v2 partition
→ online recursive sampler
→ physical-frame deterministic sampling
→ motion_processing_mf
→ candidate0/c1/c2
→ B0/B1/B2/B3
→ candidate0 observation recursive state
```

- c1/c2 只提供因果辅助训练，不写 recursive state。
- B2 loss 不进入 B0/B1；B3 loss 不进入上游。
- detach 是梯度所有权合同，不是冻结；所有启用参数均可训练。
- no-extension 精确返回 B0 observation；B3 calibration 缺失或身份不匹配时
  fail-closed。

## 配置与恢复

16 个 `25*.yaml` 入口保持不变，继承深度为：

```text
entry → v25_mini/v25_full → v25_common
```

公开 YAML 只用 `ct_variant` 选择 `b0 / b1 / full_minus_b3 / full`。
joint contract v3 和 point-evidence contract v2 由配置适配层固定，正式数据代码
不再保留旧 contract 的运行分支。
不存在模型初始化 checkpoint 入口。训练时 `--checkpoint` 只接受同配置、
同实验、epoch 边界以及 artifact/hash 完全匹配的故障恢复；测试时用于加载
待评估模型。
