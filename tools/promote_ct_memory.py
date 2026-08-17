"""Build the optional-memory promotion artifact from paired tracklet rows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.action_calibration import sha256_file
from utils.memory_promotion import evaluate_memory_promotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--real-checkpoint", required=True)
    parser.add_argument("--empty-checkpoint", required=True)
    parser.add_argument("--time-misaligned-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    with Path(args.rows).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    artifact = evaluate_memory_promotion(
        rows,
        checkpoint_sha256={
            "real": sha256_file(args.real_checkpoint),
            "empty": sha256_file(args.empty_checkpoint),
            "time_misaligned": sha256_file(
                args.time_misaligned_checkpoint),
        },
        seed=args.seed, resamples=args.bootstrap_resamples)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if not artifact["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
