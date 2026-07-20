#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lishengjie/study/lcyu/CT-SeqTrack}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
A1_CKPT="${A1_CKPT:-$REPO_ROOT/output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0/checkpoints/last.ckpt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MAX_TRACKLETS="${MAX_TRACKLETS:-}"

cd "$REPO_ROOT"
mkdir -p logs/diagnostics

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -s "$A1_CKPT" ]]; then
  echo "Missing A1 checkpoint: $A1_CKPT" >&2
  exit 1
fi

FIT_STANDARD_CSV="output/diagnostics/reliability_signals/standard_p0b3/reliability_endpoints.csv"
if [[ ! -s "$FIT_STANDARD_CSV" ]]; then
  echo "Missing P0-B3 standard fitting CSV: $FIT_STANDARD_CSV" >&2
  exit 1
fi

"$PYTHON_BIN" tools/diagnose_crop_reachability.py --self-test
"$PYTHON_BIN" tools/diagnose_reliability_signals.py --self-test
"$PYTHON_BIN" tools/summarize_reliability_signals.py --self-test
"$PYTHON_BIN" tools/validate_observation_reliability.py --self-test

"$PYTHON_BIN" tools/diagnose_reliability_signals.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order.yaml \
  --weights "$A1_CKPT" --device "$DEVICE" --model-load-smoke

declare -A CFG
CFG[standard]="cfgs/seqtrack3d_nuscenes_a1_order.yaml"
CFG[gap1124]="cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml"
CFG[burst_drop]="cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml"

LIMIT_ARGS=()
TAG_SUFFIX=""
if [[ -n "$MAX_TRACKLETS" ]]; then
  LIMIT_ARGS=(--max-tracklets "$MAX_TRACKLETS")
  TAG_SUFFIX="_smoke${MAX_TRACKLETS}"
fi

for protocol in standard gap1124 burst_drop; do
  reference_tag="${protocol}_val_reference${TAG_SUFFIX}"
  reliability_tag="${protocol}_p0b4_val${TAG_SUFFIX}"
  reference_csv="output/diagnostics/crop_reachability/${reference_tag}/crop_reachability_endpoints.csv"
  reference_summary="output/diagnostics/crop_reachability/${reference_tag}/crop_reachability_summary.json"
  reliability_csv="output/diagnostics/reliability_signals/${reliability_tag}/reliability_endpoints.csv"
  reliability_summary="output/diagnostics/reliability_signals/${reliability_tag}/reliability_summary.json"

  "$PYTHON_BIN" tools/diagnose_crop_reachability.py \
    --cfg "${CFG[$protocol]}" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --require-full-history "${LIMIT_ARGS[@]}" \
    --tag "$reference_tag" \
    2>&1 | tee "logs/diagnostics/p0b4_${protocol}_val_reference${TAG_SUFFIX}.log"

  test -s "$reference_csv"
  test -s "$reference_summary"

  "$PYTHON_BIN" tools/diagnose_reliability_signals.py \
    --cfg "${CFG[$protocol]}" \
    --weights "$A1_CKPT" \
    --reference-endpoints-csv "$reference_csv" \
    --path "$DATA_ROOT" --version v1.0-mini --split mini_val \
    --require-full-history --device "$DEVICE" "${LIMIT_ARGS[@]}" \
    --tag "$reliability_tag" \
    2>&1 | tee "logs/diagnostics/p0b4_${protocol}_mini_val${TAG_SUFFIX}.log"

  test -s "$reliability_csv"
  test -s "$reliability_summary"
done

TAG_SUFFIX_FOR_CHECK="$TAG_SUFFIX" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

suffix = os.environ["TAG_SUFFIX_FOR_CHECK"]
hashes = set()
for protocol in ("standard", "gap1124", "burst_drop"):
    path = Path(
        "output/diagnostics/reliability_signals"
    ) / f"{protocol}_p0b4_val{suffix}" / "reliability_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["reference_match"]["exact_match"] is not True:
        raise SystemExit(f"reference exact-match failed: {path}")
    hashes.add(summary["weights_sha256"])
if len(hashes) != 1:
    raise SystemExit(f"checkpoint hash mismatch across protocols: {sorted(hashes)}")
print(f"reference/checkpoint integrity: PASS ({next(iter(hashes))})")
PY

if [[ -n "$MAX_TRACKLETS" ]]; then
  echo "P0-B4 smoke completed for MAX_TRACKLETS=$MAX_TRACKLETS."
  echo "Run again without MAX_TRACKLETS for the confirmatory full mini_val evaluation."
  exit 0
fi

"$PYTHON_BIN" tools/validate_observation_reliability.py \
  --fit-standard "$FIT_STANDARD_CSV" \
  --eval standard=output/diagnostics/reliability_signals/standard_p0b4_val/reliability_endpoints.csv \
  --eval gap1124=output/diagnostics/reliability_signals/gap1124_p0b4_val/reliability_endpoints.csv \
  --eval burst_drop=output/diagnostics/reliability_signals/burst_drop_p0b4_val/reliability_endpoints.csv \
  --feature-set observation_v1 \
  --strong-protocols gap1124,burst_drop \
  --target-recall 0.80 --l2 0.001 \
  --go-auroc 0.75 --go-auprc-margin 0.15 \
  --max-ece 0.10 --max-fpr 0.30 --min-operating-recall 0.70 \
  --output-dir output/diagnostics/reliability_signals/validation \
  --tag observation_v1_minitrain_to_minival \
  2>&1 | tee logs/diagnostics/p0b4_observation_v1_validation.log

test -s output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_summary.json
test -s output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_calibrator.json
test -s output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_report.md

sha256sum \
  tools/diagnose_reliability_signals.py \
  tools/summarize_reliability_signals.py \
  tools/validate_observation_reliability.py \
  output/diagnostics/reliability_signals/standard_p0b4_val/reliability_endpoints.csv \
  output/diagnostics/reliability_signals/gap1124_p0b4_val/reliability_endpoints.csv \
  output/diagnostics/reliability_signals/burst_drop_p0b4_val/reliability_endpoints.csv \
  output/diagnostics/reliability_signals/validation/observation_v1_minitrain_to_minival_summary.json

echo "P0-B4 independent observation-reliability validation completed."
