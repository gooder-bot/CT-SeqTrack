#!/usr/bin/env bash
# 可选串行辅助工具；用户当前的同卡并行命令见 docs/CTSEQTRACK_V39_LAUNCH.md。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${CTSEQ_PYTHON:-/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python}"
DATA_ROOT="${CTSEQ_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${CTSEQ_V39_RUN_ROOT:-$ROOT/output/v39_sequential_${STAMP}_$$}"

usage() {
  printf '%s\n' 'Usage: bash tools/launch_v39_sequential.sh [--python PYTHON] [--path DATA_ROOT] [--run-root NEW_DIRECTORY]'
  printf '%s\n' 'Always physical GPU0. Runs CUDA contract tests, a lifecycle probe and paired profile, then scratch Full A followed by Full B.'
  printf '%s\n' 'Refuses occupied GPU0 or an existing run root. Never kills processes, resumes old weights, installs packages, or changes other GPUs.'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    --path) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; DATA_ROOT="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; RUN_ROOT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

cd -- "$ROOT"
[[ -x "$PYTHON_BIN" ]] || { printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2; exit 2; }
[[ -d "$DATA_ROOT/v1.0-mini" && -d "$DATA_ROOT/samples" ]] || {
  printf 'Expected nuScenes mini metadata and samples under: %s\n' "$DATA_ROOT" >&2; exit 2;
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'Run root already exists; use a new directory: %s\n' "$RUN_ROOT" >&2; exit 2; }
command -v nvidia-smi >/dev/null

assert_gpu0_idle() {
  local gpu_uuid all_processes occupied
  gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)"
  [[ -n "$gpu_uuid" ]] || { printf '%s\n' 'Cannot identify physical GPU0.' >&2; return 1; }
  all_processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)"
  occupied="$(printf '%s\n' "$all_processes" | awk -F ', *' -v wanted="$gpu_uuid" '$1 == wanted { print }')"
  if [[ -n "$occupied" ]]; then
    printf 'Physical GPU0 has existing compute processes; refusing to start:\n%s\n' "$occupied" >&2
    return 1
  fi
}

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native,max_split_size_mb:64}"

"$PYTHON_BIN" tools/register_v39.py --verify
assert_gpu0_idle
CHECK_ROOT="$ROOT/artifacts/ct_checks/v39_launch_${STAMP}_$$"
mkdir -p -- "$ROOT/artifacts/ct_checks"
mkdir -- "$CHECK_ROOT"
"$PYTHON_BIN" -c 'import torch, flash_attn; assert torch.cuda.is_available(), "CUDA required"; assert torch.cuda.is_bf16_supported(), "native BF16 required"'
"$PYTHON_BIN" -m pytest -q \
  tests/test_ct_v39_efficiency.py tests/test_ct_v39_g.py \
  tests/test_ct_v39_data_contracts.py tests/test_ct_v39_integration.py \
  --junitxml "$CHECK_ROOT/cuda_tests.xml" 2>&1 | tee "$CHECK_ROOT/cuda_tests.log"
"$PYTHON_BIN" - "$CHECK_ROOT/cuda_tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
required = {
    'test_cuda_fa2_varlen_forward_backward_matches_math',
    'test_cuda_packed_bf16_training_keeps_fp32_outputs_stats_and_finite_gradients',
    'test_cuda_coarse_recovery_strict_deterministic_repeats',
}
cases = {case.get('name'): case for case in ET.parse(sys.argv[1]).iter('testcase')}
for name in sorted(required):
    if name not in cases or any(cases[name].find(tag) is not None for tag in ('skipped', 'failure', 'error')):
        raise SystemExit('required CUDA test did not pass without skip: ' + name)
print('Required CUDA/FA2, packed BF16 and recovery tests passed without skips.')
PY
assert_gpu0_idle
"$PYTHON_BIN" -u tools/check_v39_full_batch.py \
  --cfg cfgs/ct_seqtrack/39_full_a_mini.yaml --path "$DATA_ROOT" \
  --device cuda:0 --threads 2 --output "$CHECK_ROOT/full_batch.json"
assert_gpu0_idle
"$PYTHON_BIN" -u tools/benchmark_v39.py \
  --cfg cfgs/ct_seqtrack/39_full_a_mini.yaml --path "$DATA_ROOT" \
  --training --steps 12 --warmup 2 --output "$CHECK_ROOT/paired_profile"
"$PYTHON_BIN" tools/register_v39.py --verify
assert_gpu0_idle
mkdir -p -- "$(dirname -- "$RUN_ROOT")"
mkdir -- "$RUN_ROOT"

for recipe in full_a full_b; do
  assert_gpu0_idle
  "$PYTHON_BIN" tools/register_v39.py --verify
  run_dir="$RUN_ROOT/${recipe}"
  mkdir -- "$run_dir"
  printf 'Starting scratch %s on physical GPU0; output: %s\n' "$recipe" "$run_dir"
  # pipefail ensures a failed A (including automatic evaluation) prevents B.
  "$PYTHON_BIN" -u main.py --cfg "cfgs/ct_seqtrack/39_${recipe}_mini.yaml" \
    --path "$DATA_ROOT" --devices 0 --log_dir "$run_dir" 2>&1 | tee "$run_dir/train.log"
done
printf 'Both v39 scratch runs completed sequentially: %s\n' "$RUN_ROOT"
