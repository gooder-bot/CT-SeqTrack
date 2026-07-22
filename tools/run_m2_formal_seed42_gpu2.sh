#!/usr/bin/env bash
# Launch the one allowed M1/M2 mini seed42 true-dt formal training run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
A1_CKPT="${A1_CKPT:?Set A1_CKPT to the frozen A1-order last.ckpt}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the reviewed E6 commit}"
GPU="${GPU:-2}"
FORMAL_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_formal_true.yaml"
ORACLE_ROOT="output/diagnostics/m0_proposal_oracle/m0_oracle_gap1124_seed42"

MANIFEST_DIR="protocols/manifests"
MANIFESTS=(
  "${MANIFEST_DIR}/m2_nuscenes_mini_test_gap1124_seed42.json"
  "${MANIFEST_DIR}/m2_nuscenes_mini_test_burst_drop_seed42.json"
  "${MANIFEST_DIR}/m2_nuscenes_mini_test_standard_shuffled_dt_seed42.json"
  "${MANIFEST_DIR}/m2_nuscenes_mini_test_gap1124_shuffled_dt_seed42.json"
  "${MANIFEST_DIR}/m2_nuscenes_mini_test_burst_drop_shuffled_dt_seed42.json"
)

ACTUAL_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_GIT_COMMIT}" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "Git commit mismatch: expected ${EXPECTED_GIT_COMMIT}, got ${ACTUAL_GIT_COMMIT}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal training requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi
for manifest in "${MANIFESTS[@]}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing formal manifest: ${manifest}" >&2
    echo "Run tools/prepare_m2_formal_manifests.sh first." >&2
    exit 1
  fi
done

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m2_formal_true_seed42_${ACTUAL_GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists; formal outputs must never be mixed: ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

"${PYTHON_BIN}" tools/check_m2_formal_freeze.py \
  --a1-checkpoint "${A1_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --expected-commit "${EXPECTED_GIT_COMMIT}" \
  --require-clean-git --require-server-inputs \
  --output "${OUT_ROOT}/e6_preflight.json"

"${PYTHON_BIN}" tools/freeze_m2_formal_parameters.py \
  --endpoints "${ORACLE_ROOT}/proposal_oracle_endpoints.csv" \
  --summary "${ORACLE_ROOT}/proposal_oracle_summary.json" \
  --output-json "${OUT_ROOT}/m2_e6_parameter_freeze.json" \
  --output-md "${OUT_ROOT}/m2_e6_parameter_freeze.md"

# The tool already fail-closes on the two raw input hashes and all acceptance
# checks.  Do not require byte equality across NumPy/Python versions: bootstrap
# quantile serialization may differ by a final floating-point ulp.

{
  echo "schema=ct_seqtrack.m2_formal_training_contract.v1"
  echo "git_commit=${ACTUAL_GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "seed=42"
  echo "batch_size=16"
  echo "workers=12"
  echo "epochs=60"
  echo "steps_per_epoch=1262"
  echo "expected_optimizer_steps=75720"
  echo "checkpoint_selection=last.ckpt_only"
  echo "training_time_mode=true"
  echo "control_policy=fixed_and_shuffled_are_same_checkpoint_evaluation_only"
  sha256sum \
    "${A1_CKPT}" \
    "${FORMAL_CFG}" \
    compare_results/reports/m2_e6_parameter_freeze_20260722.json \
    "${MANIFESTS[@]}"
} >"${OUT_ROOT}/formal_contract.txt"

echo "Starting the unique M2 formal run on physical GPU ${GPU}"
echo "output: ${OUT_ROOT}"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
"${PYTHON_BIN}" -u main.py \
  --cfg "${FORMAL_CFG}" \
  --init_checkpoint "${A1_CKPT}" \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --save_top_k 0 \
  --log_dir "${OUT_ROOT}" \
  2>&1 | tee "${OUT_ROOT}/console.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

echo "${TRAIN_STATUS}" >"${OUT_ROOT}/training_exit_code.txt"
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Formal training failed with status ${TRAIN_STATUS}" >&2
  exit "${TRAIN_STATUS}"
fi

mapfile -t LAST_CHECKPOINTS < <(find "${OUT_ROOT}" -type f -name last.ckpt | sort)
if [[ "${#LAST_CHECKPOINTS[@]}" -ne 1 ]]; then
  echo "Expected exactly one last.ckpt, found ${#LAST_CHECKPOINTS[@]}" >&2
  printf '%s\n' "${LAST_CHECKPOINTS[@]}" >&2
  exit 1
fi
FINAL_CKPT="${LAST_CHECKPOINTS[0]}"

"${PYTHON_BIN}" - "${FINAL_CKPT}" "${OUT_ROOT}/final_checkpoint.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
try:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(checkpoint_path, map_location="cpu")
global_step = int(payload.get("global_step", -1))
epoch = int(payload.get("epoch", -1))
if global_step != 75720 or epoch != 59:
    raise SystemExit(
        f"Unexpected final checkpoint state: epoch={epoch}, global_step={global_step}")
digest_builder = hashlib.sha256()
with checkpoint_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest_builder.update(chunk)
digest = digest_builder.hexdigest()
result = {
    "schema": "ct_seqtrack.m2_formal_final_checkpoint",
    "schema_version": 1,
    "path": str(checkpoint_path),
    "sha256": digest,
    "epoch": epoch,
    "global_step": global_step,
    "selection_rule": "last.ckpt at epoch60/step75720; no best-checkpoint selection",
}
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"${OUT_ROOT}/artifact_manifest.sha256"

ARCHIVE_ROOT="${PROJECT_ROOT}/output/formal_archives"
mkdir -p "${ARCHIVE_ROOT}"
ARCHIVE_PATH="${ARCHIVE_ROOT}/$(basename "${OUT_ROOT}").tar.gz"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
sha256sum "${ARCHIVE_PATH}" >"${ARCHIVE_PATH}.sha256"

echo "M2 formal seed42 true-dt training: COMPLETE"
echo "final checkpoint: ${FINAL_CKPT}"
echo "archive: ${ARCHIVE_PATH}"
echo "next: run tools/run_m2_formal_time_controls_gpu3.sh with FINAL_CKPT=${FINAL_CKPT}"
