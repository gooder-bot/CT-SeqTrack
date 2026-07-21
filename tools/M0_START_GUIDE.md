# M0 冻结输出阶段：同步与服务器运行指南

更新时间：2026-07-21

本轮只实现 M0 的统一 endpoint/per-tracklet 输出与离线配对汇总，不修改模型、训练损失或正式递归预测路径。M1 的 augmentation/adapter 训练、M2 proposal innovation、M3 path distillation 仍保持锁定。

## 0. 2026-07-21 当前状态

P0-C-D1 已完成，不要再重复跑本节第 4 节的 A2 full：true/fixed/shuffled 各 `91` 个 tracklet、`1257` 个 endpoint，三路 endpoint/order/hash 和时间干预检查通过；true−fixed 为 `+0.4376/+0.5231`，true−shuffled 为 `-0.1233/+0.0557`，正式结论为 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`。服务器仓库当时为 dirty，但 exact exporter/config/checkpoint/manifest/CSV hash 已归档，且配对效应复现此前 clean aggregate；该结果足以完成冻结诊断，论文正式归档保留 provenance caveat。

当前从第 5 节继续：先在专用 clean worktree 构建 M0 strong/unseen cadence manifests，再执行第 6–8 节的 A/B/C frozen matrix 与 path variance。与此并行，只做 M0-3 proposal oracle 和 M0-4 candidate 审计；不要开始 M1/M2 正式训练。

## 1. 本轮新增文件

```text
tools/export_m0_endpoints.py
tools/summarize_m0_endpoints.py
tools/M0_START_GUIDE.md
```

运行时复用仓库中已有文件：

```text
tools/diagnose_crop_reachability.py
tools/diagnose_recursive_crop_reachability.py
datasets/protocol_utils.py
utils/metrics.py
```

如果服务器代码已经包含 commit `65c2420` 或其父提交 `343145d` 的完整 P0-C/diagnostic 代码，只同步三个新增文件即可。

## 2. 同步与 Formal run 前的版本要求

### 推荐：使用冻结 source bundle

本地已把 M0 endpoint 工具与 P0-C-D1 归档冻结到分支
`m0-frozen-20260721`。优先上传 source bundle，而不是再次手工覆盖服务器文件：

```text
output/diagnostics/m0_transfers/m0_frozen_source_20260721.bundle
output/diagnostics/m0_transfers/m0_frozen_source_20260721.bundle.sha256
```

从本地 PowerShell 上传：

```powershell
scp .\output\diagnostics\m0_transfers\m0_frozen_source_20260721.bundle `
    .\output\diagnostics\m0_transfers\m0_frozen_source_20260721.bundle.sha256 `
    lishengjie@<SERVER_HOST>:/home/lishengjie/study/lcyu/
```

服务器端先验证 checksum 和 bundle prerequisite，再从 bundle 建立 detached clean
worktree；这不会切换或覆盖原 `CT-SeqTrack` 工作目录：

```bash
SOURCE_REPO=/home/lishengjie/study/lcyu/CT-SeqTrack
M0_WORKTREE=/home/lishengjie/study/lcyu/CT-SeqTrack-m0-20260721
M0_BUNDLE=/home/lishengjie/study/lcyu/m0_frozen_source_20260721.bundle

cd /home/lishengjie/study/lcyu
sha256sum -c m0_frozen_source_20260721.bundle.sha256
git bundle verify "$M0_BUNDLE"
git -C "$SOURCE_REPO" cat-file -e 65c2420^{commit}

git -C "$SOURCE_REPO" fetch "$M0_BUNDLE" \
  refs/heads/m0-frozen-20260721:refs/m0/m0-frozen-20260721
git -C "$SOURCE_REPO" worktree add --detach \
  "$M0_WORKTREE" refs/m0/m0-frozen-20260721

cd "$M0_WORKTREE"
git status --short
git rev-parse HEAD
```

`sha256sum -c` 必须输出 `OK`，`git status --short` 必须为空。随后从本指南
第 5 节开始构建绑定该 commit 的 strong/unseen cadence manifests。

### 备选：手工同步工具并提交

如果同一 Linux 账户或同一仓库目录还有别人正在使用，推荐先在服务器建立专用 worktree。它不会切换原 `CT-SeqTrack` 目录的分支，也不会改动其他终端中的环境变量：

```bash
SOURCE_REPO=/home/lishengjie/study/lcyu/CT-SeqTrack
M0_WORKTREE=/home/lishengjie/study/lcyu/CT-SeqTrack-m0-20260721

git -C "$SOURCE_REPO" worktree add \
  -b m0-frozen-20260721 "$M0_WORKTREE" HEAD
cd "$M0_WORKTREE"
```

随后把下方 `scp` 的服务器目标改为 `$M0_WORKTREE/tools/`，并在该 worktree 内提交、运行。checkpoint 仍使用原仓库中的绝对路径。worktree 会在共享仓库中增加一个 branch/worktree 记录，但不会修改原工作目录中的文件；仍建议先告知同账户协作者，不要让两个人同时操作同一个 M0 worktree。

从本地 PowerShell 同步三个文件（把 `<SERVER_HOST>` 换成服务器地址）：

```powershell
cd D:\desktop\research\CT-SeqTrack

scp .\tools\export_m0_endpoints.py `
    .\tools\summarize_m0_endpoints.py `
    .\tools\M0_START_GUIDE.md `
    lishengjie@<SERVER_HOST>:/home/lishengjie/study/lcyu/CT-SeqTrack/tools/
```

也可以用 WinSCP 手动上传，但必须保持上面三个相对路径和文件名不变。上传后先在服务器把这三个文件提交为一个独立 commit；这样 manifest、endpoint 输出和源码版本才能绑定。开发 smoke 可以 dirty，正式 manifest 和 full output 必须 clean。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

git status --short
git add \
  tools/export_m0_endpoints.py \
  tools/summarize_m0_endpoints.py \
  tools/M0_START_GUIDE.md
git diff --cached --check
git commit -m "tools: add frozen M0 endpoint diagnostics"
```

如果服务器不允许提交，仍可完成 smoke，但不要生成正式 manifest 或把 full 输出作为论文证据。

服务器预检：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack

git status --short
git rev-parse HEAD

python -m py_compile \
  tools/export_m0_endpoints.py \
  tools/summarize_m0_endpoints.py

python tools/export_m0_endpoints.py --self-test
python tools/summarize_m0_endpoints.py --self-test
```

预期：

```text
M0 endpoint exporter self-test: PASS
M0 endpoint summarizer self-test: PASS
```

建议环境：

```bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

## 3. 冻结 checkpoint

```bash
A1_OLD_CKPT=output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0/checkpoints/last.ckpt
A2_CKPT=output/20260531-2322-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_60ep_bs16_gpu2/lightning_logs/version_0/checkpoints/last.ckpt

TWC_ROOT=output/paper_twc_abc_20260720_183711
A_CKPT=$TWC_ROOT/A_single_seed42/lightning_logs/version_0/checkpoints/last.ckpt
B_CKPT=$TWC_ROOT/B_paired_weight0_seed42/lightning_logs/version_0/checkpoints/last.ckpt
C_CKPT=$TWC_ROOT/C_corrected_twc_seed42/lightning_logs/version_0/checkpoints/last.ckpt
```

必须核对：

```bash
sha256sum "$A2_CKPT" "$A_CKPT" "$B_CKPT" "$C_CKPT"
```

预期：

```text
A2 P0-C: b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad
A:        08b27a6548bcfd9e11c4cec618e5edf7ce5055d5dbc17b6e81d6200b4e9d7de1
B:        24f2c20d4f2bb658220e884d4471e6a9c340797081136432f986f6b8b0fa04c9
C:        a26c59de0fa779a6e6069385af6fa21bfdf7e4b570baf4a7dd1c18c6f297b2ca
```

先做模型加载 smoke：

```bash
python tools/export_m0_endpoints.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A_CKPT" --device cuda:0 --model-load-smoke

python tools/export_m0_endpoints.py \
  --cfg cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml \
  --weights "$A2_CKPT" --device cuda:0 --model-load-smoke
```

## 4. M0-1：P0-C-D1 true/fixed/shuffled

P0-C-D1 必须复用原始 gap1124 selection 与 shuffled mapping：

```bash
P0C_CFG=cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml
P0C_SHUFFLED=protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json
DATA_ROOT=/home/lishengjie/data/nuscenes-mini
```

原 manifest 记录的 commit 为 `343145d`，而 endpoint logger 位于更新的 clean commit。已核对 `343145d..65c2420`，本轮也只新增 `tools/` 文件；两段变化都没有修改 model/dataset inference 代码。因此 D1 允许显式使用 `--allow-manifest-commit-mismatch`，但 exporter 会同时保存当前 git commit、manifest hash、selection hash 和 checkpoint hash。不得换 manifest、checkpoint、seed 或阈值。

### 4.1 2-tracklet smoke

2026-07-21 首个 `true` smoke 已证明模型加载和 endpoint 写出链路可用，但本地复核发现旧 exporter 的首帧 CSV 曾固定写为 `IoU=1.0`，而 aggregate 使用 box backend 实际 self-overlap，导致 Success 相差 `0.1042 pp`。该问题已在 `tools/export_m0_endpoints.py` 修复。同步后必须重新运行 **true/fixed/shuffled 三路 smoke**；不要把旧 true-only smoke 与新 fixed/shuffled 混成 paired comparison。

运行前确认当前 shell 已启用 pipeline 失败传播，避免 Python 异常被 `tee` 的零退出码掩盖：

```bash
set -o pipefail
```

```bash
for mode in true fixed shuffled; do
  extra=()
  if [ "$mode" = shuffled ]; then
    extra+=(--dynamics-time-manifest "$P0C_SHUFFLED")
  fi

  python tools/export_m0_endpoints.py \
    --cfg "$P0C_CFG" --weights "$A2_CKPT" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --max-tracklets 2 \
    --run-label "p0c_a2_${mode}" --protocol-name gap1124 \
    --dynamics-time-mode "$mode" --dynamics-fixed-delta-t 0.5 \
    --allow-manifest-commit-mismatch \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "p0c_d1_${mode}_smoke" \
    "${extra[@]}"
done
```

检查：

```bash
for mode in true fixed shuffled; do
  test -s "output/diagnostics/m0_endpoints/p0c_d1_${mode}_smoke/m0_endpoints.csv"
  test -s "output/diagnostics/m0_endpoints/p0c_d1_${mode}_smoke/m0_summary.json"
  test -s "output/diagnostics/m0_endpoints/p0c_d1_${mode}_smoke/resolved_config.json"
done
```

### 4.2 full 输出

```bash
mkdir -p output/diagnostics/logs

for mode in true fixed shuffled; do
  extra=()
  if [ "$mode" = shuffled ]; then
    extra+=(--dynamics-time-manifest "$P0C_SHUFFLED")
  fi

  python tools/export_m0_endpoints.py \
    --cfg "$P0C_CFG" --weights "$A2_CKPT" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 \
    --run-label "p0c_a2_${mode}" --protocol-name gap1124 \
    --dynamics-time-mode "$mode" --dynamics-fixed-delta-t 0.5 \
    --allow-manifest-commit-mismatch \
    --require-clean-git \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "p0c_d1_${mode}_full" \
    "${extra[@]}" \
    2>&1 | tee "output/diagnostics/logs/m0_p0c_d1_${mode}.log"
done
```

三路 CSV 都应包含 `1257` 行（含每条 tracklet 的初始化帧）和 `91` 个 tracklet。三路 checkpoint hash 必须相同，endpoint key/order 必须 exact match。

### 4.3 D1 配对汇总

```bash
python tools/summarize_m0_endpoints.py \
  --input true=output/diagnostics/m0_endpoints/p0c_d1_true_full/m0_endpoints.csv \
  --input fixed=output/diagnostics/m0_endpoints/p0c_d1_fixed_full/m0_endpoints.csv \
  --input shuffled=output/diagnostics/m0_endpoints/p0c_d1_shuffled_full/m0_endpoints.csv \
  --comparison true:fixed \
  --comparison true:shuffled \
  --require-same-checkpoint \
  --failure-threshold 2.0 --failure-consecutive 2 \
  --bootstrap-iterations 10000 --seed 42 \
  --output-dir output/diagnostics/m0_analysis \
  --tag p0c_d1_gap1124_20260721 \
  2>&1 | tee output/diagnostics/logs/m0_p0c_d1_summary.log
```

## 5. M0-2：构建 strong/unseen cadence manifest

在新增工具已经提交、服务器 tracked worktree clean 后运行。以下三个 manifest 都绑定新的 evaluation commit。

```bash
mkdir -p protocols/manifests

python tools/build_virtual_rate_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
  --role test --split mini_val --mode gap_pattern \
  --gap-pattern 1 1 2 4 --seed 42 --max-gap 5 \
  --output protocols/manifests/nuscenes_mini_test_gap1124_m0_seed42.json

python tools/build_virtual_rate_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml \
  --role test --split mini_val --mode burst_drop \
  --seed 42 --max-gap 5 \
  --output protocols/manifests/nuscenes_mini_test_burst_drop_m0_seed42.json

python tools/build_virtual_rate_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
  --role test --split mini_val --mode gap_pattern \
  --gap-pattern 2 --seed 42 --max-gap 2 \
  --output protocols/manifests/nuscenes_mini_test_fixedgap2_m0_seed42.json
```

注意：fixedgap2 是 cadence protocol，不是 `dynamics_time_mode=fixed`。前者改变保留的帧，后者只干预模型读取的 effective time，两者不能混写。

## 6. M0-2：A/B/C standard smoke 与 path variance

先对 standard 做 2-tracklet smoke：

```bash
declare -A CKPT
CKPT[A]="$A_CKPT"
CKPT[B]="$B_CKPT"
CKPT[C]="$C_CKPT"

for model in A B C; do
  python tools/export_m0_endpoints.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
    --weights "${CKPT[$model]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --max-tracklets 2 \
    --run-label "$model" --protocol-name standard \
    --passive-path-variance --path-a-offsets 1,2,3 --path-b-offsets 1,3,5 \
    --fail-on-path-mismatch \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "twc_${model}_standard_smoke"
done
```

每个 summary 必须满足：

```text
passive_path_variance.anchor_gap_max = 0
passive_path_variance.current_point_gap_max = 0
```

注意：新 exporter 显式固定了全局 seed，得到的是新的标准化 frozen inference。历史训练期 validation 的随机状态受到训练过程影响，因此不要求新输出逐位等于旧事件文件；必须保证的是同一 exporter 下 A/B/C 的 endpoint、协议、seed 与 checkpoint 口径一致。

## 7. M0-2：A/B/C 四协议 full matrix

### 7.1 standard

```bash
for model in A B C; do
  python tools/export_m0_endpoints.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
    --weights "${CKPT[$model]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 \
    --run-label "$model" --protocol-name standard \
    --passive-path-variance --fail-on-path-mismatch \
    --require-clean-git \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "twc_${model}_standard_full" \
    2>&1 | tee "output/diagnostics/logs/m0_twc_${model}_standard.log"
done
```

### 7.2 gap1124

```bash
GAP_MANIFEST=protocols/manifests/nuscenes_mini_test_gap1124_m0_seed42.json

for model in A B C; do
  python tools/export_m0_endpoints.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
    --protocol-cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
    --virtual-rate-manifest "$GAP_MANIFEST" --require-manifest-commit-match \
    --weights "${CKPT[$model]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 \
    --run-label "$model" --protocol-name gap1124 \
    --passive-path-variance --fail-on-path-mismatch \
    --require-clean-git \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "twc_${model}_gap1124_full" \
    2>&1 | tee "output/diagnostics/logs/m0_twc_${model}_gap1124.log"
done
```

### 7.3 burst-drop

```bash
BURST_MANIFEST=protocols/manifests/nuscenes_mini_test_burst_drop_m0_seed42.json

for model in A B C; do
  python tools/export_m0_endpoints.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
    --protocol-cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml \
    --virtual-rate-manifest "$BURST_MANIFEST" --require-manifest-commit-match \
    --weights "${CKPT[$model]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 \
    --run-label "$model" --protocol-name burst_drop \
    --passive-path-variance --fail-on-path-mismatch \
    --require-clean-git \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "twc_${model}_burst_drop_full" \
    2>&1 | tee "output/diagnostics/logs/m0_twc_${model}_burst_drop.log"
done
```

### 7.4 unseen fixedgap2

```bash
FIXEDGAP_MANIFEST=protocols/manifests/nuscenes_mini_test_fixedgap2_m0_seed42.json

for model in A B C; do
  python tools/export_m0_endpoints.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
    --virtual-rate-mode gap_pattern --virtual-rate-gap-pattern 2 \
    --virtual-rate-max-gap 2 \
    --virtual-rate-manifest "$FIXEDGAP_MANIFEST" \
    --require-manifest-commit-match \
    --weights "${CKPT[$model]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 \
    --run-label "$model" --protocol-name fixedgap2 \
    --passive-path-variance --fail-on-path-mismatch \
    --require-clean-git \
    --output-dir output/diagnostics/m0_endpoints \
    --tag "twc_${model}_fixedgap2_full" \
    2>&1 | tee "output/diagnostics/logs/m0_twc_${model}_fixedgap2.log"
done
```

## 8. 每个协议的 A/B/C 汇总

```bash
for protocol in standard gap1124 burst_drop fixedgap2; do
  python tools/summarize_m0_endpoints.py \
    --input A="output/diagnostics/m0_endpoints/twc_A_${protocol}_full/m0_endpoints.csv" \
    --input B="output/diagnostics/m0_endpoints/twc_B_${protocol}_full/m0_endpoints.csv" \
    --input C="output/diagnostics/m0_endpoints/twc_C_${protocol}_full/m0_endpoints.csv" \
    --comparison B:A --comparison C:B --comparison C:A \
    --failure-threshold 2.0 --failure-consecutive 2 \
    --bootstrap-iterations 10000 --seed 42 \
    --output-dir output/diagnostics/m0_analysis \
    --tag "twc_abc_${protocol}_20260721" \
    2>&1 | tee "output/diagnostics/logs/m0_twc_abc_${protocol}_summary.log"
done
```

汇总器会在配对前强制检查 endpoint key 和顺序 exact match。A/B/C checkpoint 本来就不同，因此这里不能加 `--require-same-checkpoint`。

## 9. 结果完整性检查

```bash
for tag in \
  p0c_d1_true_full p0c_d1_fixed_full p0c_d1_shuffled_full \
  twc_A_standard_full twc_B_standard_full twc_C_standard_full \
  twc_A_gap1124_full twc_B_gap1124_full twc_C_gap1124_full \
  twc_A_burst_drop_full twc_B_burst_drop_full twc_C_burst_drop_full \
  twc_A_fixedgap2_full twc_B_fixedgap2_full twc_C_fixedgap2_full; do
  test -s "output/diagnostics/m0_endpoints/$tag/m0_endpoints.csv"
  test -s "output/diagnostics/m0_endpoints/$tag/m0_summary.json"
  test -s "output/diagnostics/m0_endpoints/$tag/resolved_config.json"
done
```

还要检查：

```text
- P0-C 三路 endpoint exact match、checkpoint hash 相同；
- A/B/C 同协议 endpoint exact match；
- path_anchor_gap_max == 0；
- path_current_point_gap_max == 0；
- CSV 不存在重复 (tracklet_key, source_frame_index, frame_token)；
- JSON 无 NaN/Infinity；
- full run 的 git.dirty == false；
- resolved config hash、manifest selection hash 与 checkpoint hash 已保存。
```

## 10. 打包回传

```bash
mkdir -p transfer

tar -czf transfer/m0_frozen_outputs_20260721.tar.gz \
  output/diagnostics/m0_endpoints \
  output/diagnostics/m0_analysis \
  output/diagnostics/logs/m0_*.log \
  protocols/manifests/nuscenes_mini_test_gap1124_m0_seed42.json \
  protocols/manifests/nuscenes_mini_test_burst_drop_m0_seed42.json \
  protocols/manifests/nuscenes_mini_test_fixedgap2_m0_seed42.json

sha256sum transfer/m0_frozen_outputs_20260721.tar.gz \
  > transfer/m0_frozen_outputs_20260721.tar.gz.sha256
```

把 `.tar.gz` 和 `.sha256` 一起下载。本轮不需要回传 checkpoint。

## 11. 本轮之后的解锁边界

完成并复核本指南的输出后，下一批代码才是：

```text
tools/diagnose_proposal_oracle.py
tools/audit_candidate_dynamics.py
```

proposal oracle 与 candidate 审计完成前，不修改：

```text
models/dynamics.py
models/seqtrack3d.py 的 residual 公式
models/observability.py
TWC loss
正式训练配置
```
