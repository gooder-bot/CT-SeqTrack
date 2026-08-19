#!/usr/bin/env bash
set -euo pipefail

# Server-only staged acceptance runner. It intentionally performs no package,
# CUDA, or dataset discovery; paths and prerequisite artifacts are explicit.
stage="${1:-}"
: "${CT_DATA_ROOT:?set CT_DATA_ROOT to the nuScenes root}"
CT_ARTIFACT_ROOT="${CT_ARTIFACT_ROOT:-artifacts/v25_acceptance}"
CT_RUN_ROOT="${CT_RUN_ROOT:-output/v25_acceptance}"
CT_SEED="${CT_SEED:-42}"
mkdir -p "${CT_ARTIFACT_ROOT}" "${CT_RUN_ROOT}"

preflight="${CT_PREFLIGHT:-${CT_ARTIFACT_ROOT}/acquisition_preflight.json}"
promotion="${CT_B2_PROMOTION:-${CT_ARTIFACT_ROOT}/b2_method_promotion.json}"

train_arm() {
  local config="$1"
  local name="$2"
  shift 2
  python main.py --cfg "${config}" --path "${CT_DATA_ROOT}" \
    --seed "${CT_SEED}" --log_dir "${CT_RUN_ROOT}/${name}" "$@"
}

case "${stage}" in
  preflight)
    python tools/export_ct_acquisition_preflight_rows.py \
      --config cfgs/ct_seqtrack/25_full_minus_b3.yaml \
      --path "${CT_DATA_ROOT}" \
      --output "${CT_ARTIFACT_ROOT}/acquisition_rows.jsonl" \
      --data-manifest-output "${CT_ARTIFACT_ROOT}/acquisition_data_manifest.json"
    python tools/preflight_ct_acquisition.py \
      --rows "${CT_ARTIFACT_ROOT}/acquisition_rows.jsonl" \
      --data-manifest "${CT_ARTIFACT_ROOT}/acquisition_data_manifest.json" \
      --config cfgs/ct_seqtrack/25_full_minus_b3.yaml \
      --path "${CT_DATA_ROOT}" --output "${preflight}"
    ;;
  smoke)
    : "${CT_B2_PROMOTION:?set CT_B2_PROMOTION to a passed method-only manifest}"
    train_arm cfgs/ct_seqtrack/25_b0.yaml b0_smoke \
      --epoch 1 --limit_train_batches 2 --limit_val_batches 1
    train_arm cfgs/ct_seqtrack/25_b1.yaml b1_smoke \
      --epoch 1 --limit_train_batches 2 --limit_val_batches 1
    train_arm cfgs/ct_seqtrack/25_full_minus_b3.yaml full_minus_b3_smoke \
      --epoch 1 --limit_train_batches 2 --limit_val_batches 1 \
      --acquisition_preflight "${preflight}"
    train_arm cfgs/ct_seqtrack/25_full.yaml full_smoke \
      --epoch 1 --limit_train_batches 2 --limit_val_batches 1 \
      --acquisition_preflight "${preflight}" \
      --b2_method_promotion "${promotion}"
    ;;
  full20)
    : "${CT_B2_PROMOTION:?set CT_B2_PROMOTION to a passed method-only manifest}"
    train_arm cfgs/ct_seqtrack/25_full.yaml full_20batch \
      --epoch 1 --limit_train_batches 20 --limit_val_batches 1 \
      --acquisition_preflight "${preflight}" \
      --b2_method_promotion "${promotion}"
    ;;
  resume)
    : "${CT_RESUME_CONFIG:?set CT_RESUME_CONFIG to the original v25 YAML}"
    : "${CT_RESUME_CHECKPOINT:?set CT_RESUME_CHECKPOINT to its epoch-boundary checkpoint}"
    resume_args=()
    if [[ -n "${CT_PREFLIGHT:-}" ]]; then
      resume_args+=(--acquisition_preflight "${CT_PREFLIGHT}")
    fi
    if [[ -n "${CT_B2_PROMOTION:-}" ]]; then
      resume_args+=(--b2_method_promotion "${CT_B2_PROMOTION}")
    fi
    python main.py --cfg "${CT_RESUME_CONFIG}" --path "${CT_DATA_ROOT}" \
      --seed "${CT_SEED}" --checkpoint "${CT_RESUME_CHECKPOINT}" \
      --log_dir "${CT_RUN_ROOT}/resume" \
      "${resume_args[@]}"
    ;;
  train)
    : "${CT_B2_PROMOTION:?set CT_B2_PROMOTION to a passed method-only manifest}"
    train_arm cfgs/ct_seqtrack/25_b0.yaml b0
    train_arm cfgs/ct_seqtrack/25_b1.yaml b1
    train_arm cfgs/ct_seqtrack/25_full_minus_b3.yaml full_minus_b3 \
      --acquisition_preflight "${preflight}"
    train_arm cfgs/ct_seqtrack/25_full.yaml full \
      --acquisition_preflight "${preflight}" \
      --b2_method_promotion "${promotion}"
    ;;
  *)
    echo "usage: bash tools/run_ct_v25_server_acceptance.sh {preflight|smoke|full20|resume|train}" >&2
    exit 2
    ;;
esac
