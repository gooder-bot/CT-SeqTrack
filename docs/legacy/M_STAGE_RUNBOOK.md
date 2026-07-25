# CT-SeqTrack 历史 M 阶段运行手册

当前工程状态：M3 matched A/B/C 训练、standard/gap1124 配对评测、M4 四臂在线递归评测均已接通。代码已经通过 dataset-free invariants、配置合同、自测、Python 编译和 Bash 语法检查；本机没有 nuScenes 数据与 CUDA，因此最终训练分数需要在服务器产生。

## 1. 方法与比较对象

M3 从选定的 M2 checkpoint 继续训练，三个臂只改变被研究因素：

| 臂 | 数据路径 | 蒸馏权重 | 用途 |
|---|---|---:|---|
| A | 单视图 `[1,2,3]` | 0 | 普通 continuation baseline |
| B | 配对 `[1,2,3]` / `[1,3,5]` | 0 | 隔离 paired data/compute effect |
| C | 与 B 相同 | 0.05 | endpoint distillation method |

主方法效应是 `C-B`，实际部署收益是 `C-A`，相对冻结 M2 初始化的训练收益是 `C-INIT`。M3 使用 EMA canonical teacher、无 GT 的 foreground/proposal-agreement hybrid confidence，并同时蒸馏 refined endpoint 与低权重 coarse endpoint。

M4 不训练参数，在同一 checkpoint 上比较：

| 臂 | 状态滤波 | trajectory tube |
|---|---:|---:|
| off | 否 | 否 |
| filter | 是 | 否 |
| tube | 否 | 是 |
| filter_tube | 是 | 是 |

默认同时运行 `real` 与 `fixed` 两种状态时钟。网络点数预算始终为 1024，GT 只用于离线指标和 oracle 诊断。

## 2. 服务器一条命令完成

先确认代码已提交或服务器工作树干净，然后执行：

```bash
cd /path/to/CT-SeqTrack
INIT_CKPT=/absolute/path/to/selected_m2/last.ckpt \
DATA_ROOT=/home/lishengjie/data/nuscenes-mini \
GPU=0 \
bash tools/run_m_stage_pipeline.sh
```

默认参数为 `SEED=42`、`BATCH_SIZE=16`、`WORKERS=12`、`EPOCHS=60`、`M3_WEIGHT=0.05`、`RUN_M4=1`、`M4_CLOCKS="real fixed"`。显存不足时只降低 `BATCH_SIZE`，三臂必须使用同一个值。

只跑 M3：

```bash
INIT_CKPT=/absolute/path/to/m2.ckpt RUN_M4=0 \
DATA_ROOT=/home/lishengjie/data/nuscenes-mini GPU=0 \
bash tools/run_m_stage_pipeline.sh
```

已有 M3 checkpoint、只跑 M4：

```bash
MODEL_CKPT=/absolute/path/to/last.ckpt \
DATA_ROOT=/home/lishengjie/data/nuscenes-mini \
GPU=0 M4_CLOCKS="real fixed" \
bash tools/run_m4_matched_evaluation.sh
```

`REQUIRE_CLEAN=1` 是正式实验默认值。只做临时 smoke 时可显式设置 `REQUIRE_CLEAN=0`，但该结果不要放入论文表格。

## 3. 输出与读数

总控输出位于：

```text
output/m_stage_<commit>_<timestamp>/
├── m3_train/
├── m3_eval/analysis/
├── m4_eval/analysis/
├── pipeline_contract.txt
└── artifact_manifest.sha256
```

优先阅读：

- `m3_standard_matched_abc_report.md`
- `m3_gap1124_matched_abc_report.md`
- `m4_real_standard_matched_report.md`
- `m4_real_gap1124_matched_report.md`
- 对应的 `*_summary.json` 和 `*_tracklet_deltas.csv`

汇总器会先检查 endpoint identity，再用 tracklet 作为 bootstrap 单位运行 20,000 次重采样。Success/Precision 的正向 delta 才是涨分；oracle center recall 仅用于解释 tube 是否扩大了有效搜索覆盖。

## 4. 正式结果判读

M3 至少报告：

1. `C-B`：蒸馏项在完全相同 paired 路径上的净效应；
2. `C-A`：方法能否胜过普通 continuation；
3. `A-INIT` 与 `C-INIT`：继续训练本身和完整方法相对 M2 的变化；
4. standard 与 gap1124 两个协议的 tracklet-bootstrap CI。

M4 至少报告每个 variant 相对 `off` 的差值，并检查：

1. standard 是否保持或提高；
2. gap1124 是否提高；
3. tube 的新增覆盖是否转化为 Success/Precision；
4. `filter_tube` 的收益是否超过单独 `filter` 或 `tube`。

若第一轮需要调参，保持 A/B 不变，只对 C 的 `M3_WEIGHT` 做小范围搜索，例如 `0.02/0.05/0.10`；M4 先调固定 Q/R 与 tube 尺寸，不改网络或 checkpoint。最终候选必须用独立 seed 重复，不能用同一 mini-val 反复选择后直接作为最终论文结论。
