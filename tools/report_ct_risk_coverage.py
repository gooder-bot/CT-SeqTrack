"""Export the B3 empirical risk--coverage curve from diagnostic rows."""

import argparse
import csv
import json
from pathlib import Path

from ctseqtrack.runtime.calibration import risk_coverage_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--presence-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.rows)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    curve = risk_coverage_curve(
        rows, presence_threshold=args.presence_threshold,
        seed=args.seed, resamples=args.bootstrap_resamples)
    payload = {
        "schema": "ct_seqtrack.risk_coverage.v1",
        "presence_threshold": args.presence_threshold,
        "curve": curve,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
