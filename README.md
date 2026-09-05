# CT-SeqTrack

## 当前轮次：v27（2026-09-05）

当前实现与实验入口以 [v27 方法](docs/CTSEQTRACK_V27_METHOD.md) 和
[正式协议](docs/EXPERIMENT_PROTOCOL.md) 顶部为准。先运行 mini Car 的
`27_b0`、`27_b1_gru`、`27_b1_cfc`、`27_full_minus_b3`、`27_full` 五臂；
再使用对应 `*_nuscenes_full.yaml` 对 Car、Pedestrian、Truck、Trailer、Bus
分别从 epoch0 训练。每臂60 epoch、seed42，所有启用模块同时训练。

v27 修复原始 point ID、稀疏输入、B1 获取接口与几何语义，新增选点前局部几何和
目标/上下文条件；B2 仍只让新增点投票，长车投票范围由首帧尺寸决定。
Full−B3 评测为 `bounded_always`，Full 使用绑定 checkpoint 的 v27 效用策略。
训练仍统一 observation-recursive；B1 不直接混入输出，B0 host 独占状态写入。

mini 的训练/阈值拟合/诊断为6/1/1 scenes，官方2个mini_val只评测；full使用
train_track全部350 scenes，17/18阈值拟合与诊断scene也参与参数训练，官方150个val
只评测。外部参考是 `cfgs/27_seqtrack_reference{_nuscenes_full}.yaml`，使用普通
SeqTrack loss并关闭CT模块。主指标为benchmark兼容口径，另报精确几何审计。

入口仍为 `main.py`，例如：

```bash
python main.py --cfg cfgs/ct_seqtrack/27_full.yaml --path /home/lishengjie/data/nuscenes-mini --tag ct27_full_mini_car_seed42
```

v27 正式训练、涨分与论文主张均尚待实验验证。v24/v25/v26配置和输出保持历史证据，
不能用于初始化v27。新增核心接口为 `models/ct_v2/evidence_v27.py`、
`models/ct_v2/action_v27.py`、`utils/b1_acquisition.py`、`utils/v27_protocol.py`。

实际模型训练、checkpoint保存与同运行恢复的二次审计和使用说明见
[训练就绪记录](docs/CTSEQTRACK_V27_TRAINING_READINESS.md)。
本次GPU2/GPU3五臂mini后台命令见[mini启动说明](docs/CTSEQTRACK_V27_MINI_LAUNCH.md)。

## 历史 v24—v26 说明

以下正文保留历史记录；其中的“当前”、实验顺序与旧promotion条件只对应其原始版本，
不覆盖上面的v27协议。

CT-SeqTrack studies observation-anchored evidence recovery for irregular-time
3D single-object tracking. The active repository contains one formal chain:

```text
SeqTrack / B0 Observation
  -> B1 Physical-Time Prior
  -> B2 Extension-Only Evidence
  -> B3 Calibrated Selective Update
```

B4 is retained as an isolated experiment and is disabled in every formal
configuration. The v26 method has not completed its registered full-nuScenes
experiments, so this repository does not currently claim a gain,
stability, SOTA, or a causal benefit from physical time or memory.

## Method contract

- B0 is the nominal observation tracker and the only recursive state writer.
- B1 reads prediction-backed history boxes and physical timestamps. It supplies
  a prior and uncertainty but never replaces the observation.
- B1 uses a fixed kinematic anchor plus a bounded normalized residual. Its
  statistical sigma is separate from a q=0.90 bounded acquisition-margin head.
  v26 adds a causal three-frame backup corridor for catastrophic drift. GRU is
  the formal temporal backend for this round.
- B2 must recover identifiable evidence from extension-only points. A
  deterministic 768-point novel pre-pool is reduced to 256 points by relation,
  spatial-coverage and stateless-exploration selection, then aggregated by
  mode-consistent robust voting. Base points and memory are context only.
- B3 consumes detached upstream evidence and may apply only a calibrated,
  bounded residual. Missing or mismatched calibration returns B0 exactly.

The canonical candidate is `b0_view_id=0`. Three auxiliary B0 views stop after
B0, and the objective remains
`0.5*L0 + (L1+L2+L3)/6`. B1/B2/B3 run once per online endpoint on the
canonical view only.

## Active interfaces

- `models/seqtrack3d.py`: SeqTrack/B0 host and isolated B4 hook.
- `models/ctseqtrack.py`: paper-facing composition root.
- `models/ct_v2/pipeline.py`: B0/B1 paper-facing components.
- `models/ct_v2/cfc.py`: dependency-free optional B1 temporal cell.
- `models/ct_v2/evidence_memory.py`: B2 evidence and B3 selective update.
- `models/ct_v2/pipeline_contracts.py`: typed internal ownership contracts.
- `utils/action_calibration.py`: held-out action calibration and fail-closed validation.
- `utils/online_contract.py`: scratch/resume and cross-arm identity contracts.

The evaluator continues to receive the existing flat dictionaries and metric
names. Required compatibility aliases remain part of the public runtime.

## Formal variants

| Variant | Trainable modules from random initialization | Training/deployed output |
|---|---|---|
| `b0` | B0 | observation |
| `b1` | B0+B1 | observation |
| `full_minus_b3` | B0+B1+B2 | raw B2 search output |
| `full` | B0+B1+B2+B3 | observation until calibrated selective evaluation |

The registered v26 mini configs are `26_b0.yaml`, `26_b1_gru.yaml`,
`26_b1_cfc.yaml`, `26_full_minus_b3.yaml`, and `26_full.yaml` under
`cfgs/ct_seqtrack/`. Their matching full-nuScenes configs use the
`*_nuscenes_full.yaml` suffix; the separate external reference is
`cfgs/26_seqtrack_strict_nuscenes_full.yaml`. The two B1-only configs are
backend diagnostics; CfC is not the v26 main method. The v24/v25 configs and
outputs remain frozen evidence and cannot initialize v26. B4 keeps
`cfgs/ct_v2/19_b4_decoder_alignment.yaml` and
`20_b4_decoder_anticollapse.yaml`.

All formal arms use `scratch_only`. `--init_checkpoint` is forbidden.
`--checkpoint` is accepted only for exact same-run epoch-boundary resume or
for evaluation. Enabled B0/B1/B2/B3 parameters are never frozen.

The retained v25 B1 backend ablation is selected without duplicating configs:

```bash
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path DATA_ROOT --tag b1_gru --b1-backend gru
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path DATA_ROOT --tag b1_cfc --b1-backend cfc
```

Both historical commands construct only the selected backend and train every
enabled module from epoch 0. The v26 integrated arms use GRU; the separately
registered v26 B1-CfC arm is a scratch-only backend diagnostic and never
initializes another arm. A calibration checkpoint is evaluation-only and
cannot be used for resume or initialization.

## Experiment order

The authoritative protocol is
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). In summary:

1. use the fixed candidate protocol in every arm: four B0 views and one
   canonical B2 view;
2. run the zero-step full-data launch preflight for each arm (no sample
   forward and no checkpoint is created);
3. train the requested B0, B1-GRU, B1-CfC, Full-B3 and Full arms independently
   from epoch0 on full nuScenes, Car, seed42, for 60 epochs; SeqTrack-strict
   remains the separately registered external reference;
4. export disjoint calibration/dev action rows for every final/late-3 Full
   checkpoint and install thresholds only after dev promotion;
5. report final/late-3 and tracklet-paired intervals without a cross-seed claim.

The only v26 launch gate is `tools/preflight_v26_full.py`; it performs
configuration, dependency, data-layout, model-construction and optimizer-group
checks without training. There is no mini run, kill-test, warm-start checkpoint
or intermediate metric stopping gate.
B1 backend promotion is a post-run decision from held-out mechanism metrics;
analysis artifacts never initialize another run or alter a run in progress.

## Validation

```bash
python tools/preflight_v26_full.py --arm full --path FULL_NUSCENES_ROOT
python tools/verify_ct_slimming.py verify
python -m pytest -q
```

Formal training explicitly sets `min_epochs=max_epochs=60`, `max_steps=-1`,
validation every two epochs, one visible trainer device and no early-stopping
callback. `last.ckpt` remains
the same-run resume point; `formal_checkpoints/epoch=058.ckpt` through
`epoch=060.ckpt` are retained for the registered late-3/final analysis.

Real-batch forward/backward, 100-step/resume parity and point/box visualization
are required server-side acceptance checks before a formal long run. They are not represented as
locally completed when the full Lightning/nuScenes environment is unavailable.
Engineering checkpoints are discarded and may not initialize formal runs.

Detailed method and evidence boundaries:

- [Safe-SeqTrack v25 runtime protocol](docs/SAFE_SEQTRACK_V25_PROTOCOL.md)
- [v26 bounded evidence recovery](docs/CTSEQTRACK_V26_METHOD.md)
- [B0--B3 method](docs/CTSEQTRACK_B0_B3_METHOD.md)
- [formal experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [formal tooling](docs/FORMAL_TOOLING.md)
- [source slimming gate](docs/SOURCE_SLIMMING_GATE.md)
- [active status](need_to_do.md)
- [historical evidence index](docs/HISTORY_EVIDENCE_INDEX.md)
- [slimming baseline](docs/slimming_baseline/README.md)
