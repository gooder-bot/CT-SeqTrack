"""Read-only reproductions for the 2026-09-05 project review.

Run from CT-SeqTrack with the local CPU Python. No training or output writes.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant
from utils.action_calibration import tracklet_bootstrap_bounds
from utils.config import load_yaml_config


def main():
    cfg = load_yaml_config(ROOT / "cfgs/ct_seqtrack/26_full.yaml")
    cases = []
    for requested in ("true", "fixed", "shuffled"):
        candidate = deepcopy(cfg)
        # main.parse_config() applies this CLI override before normalization.
        candidate["dynamics_time_mode"] = requested
        configure_ct_variant(candidate)
        cases.append({
            "requested_dynamics_time_mode": requested,
            "effective_dynamics_time_mode": candidate["dynamics_time_mode"],
            "effective_ct_time_mode": candidate["ct_time_mode"],
        })
    rows = [
        {"tracklet_id": str(tracklet), "center_gain": 0.1, "iou_gain": 0.01}
        for tracklet in range(30) for _ in range(4)
    ]
    print(json.dumps({
        "time_cli_override": cases,
        "zero_harm_percentile_bootstrap": tracklet_bootstrap_bounds(
            rows, resamples=200),
        "interpretation": (
            "A zero empirical bootstrap upper endpoint is not evidence of "
            "zero population risk. The following binomial number is only "
            "an illustration for 30 independent binary units, not a bound "
            "for the project's action-weighted tracking risk."
        ),
        "illustrative_30_zero_failures_binomial_upper_95": (
            1.0 - 0.05 ** (1.0 / 30.0)),
    }, indent=2))


if __name__ == "__main__":
    main()
