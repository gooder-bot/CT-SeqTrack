"""Summarize B1 prior/support diagnostics with registered strata."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"B1 diagnostic lacks numeric {key}") from exc
    return value


def _optional_values(rows, key):
    if not rows or any(key not in row for row in rows):
        return None
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    return values if np.isfinite(values).all() else None


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows):
    valid = [row for row in rows if (
        _number(row, "b1_valid") > 0
        and all(math.isfinite(_number(row, key)) for key in (
            "learned_motion_error", "kinematic_error", "b1_nll")))]
    if not valid:
        return {"count": 0, "valid_rate": 0.0}

    def values(key):
        return np.asarray([_number(row, key) for row in valid], dtype=float)

    learned = values("learned_motion_error")
    cv = values("kinematic_error")
    coverage = {
        level: float(values(f"b1_coverage_{level}").mean())
        for level in ("50", "80", "95")
    }
    summary = {
        "count": len(valid),
        "valid_rate": float(len(valid) / len(rows)),
        "learned_rmse": float(np.sqrt(np.mean(learned ** 2))),
        "cv_rmse": float(np.sqrt(np.mean(cv ** 2))),
        "learned_minus_cv_rmse": float(
            np.sqrt(np.mean(learned ** 2)) - np.sqrt(np.mean(cv ** 2))),
        "nll": float(values("b1_nll").mean()),
        "target_in_support_recall": float(values(
            "target_in_support").mean()),
        "support_volume": float(values("support_volume").mean()),
        "coverage": coverage,
        "coverage_ece": float(np.mean([
            abs(coverage["50"] - 0.50),
            abs(coverage["80"] - 0.80),
            abs(coverage["95"] - 0.95),
        ])),
    }
    optional = {
        "pool_target_count": "pool_target_count",
        "sampled_target_count": "sampled_target_count",
        "extension_unique_count": "evidence_extension_unique_count",
    }
    for output_key, row_key in optional.items():
        observed = _optional_values(valid, row_key)
        if observed is not None:
            summary[output_key] = float(observed.mean())
    if summary["support_volume"] > 0:
        summary["support_recall_per_volume"] = float(
            summary["target_in_support_recall"] / summary["support_volume"]
        )
        if "sampled_target_count" in summary:
            summary["sampled_targets_per_volume"] = float(
                summary["sampled_target_count"] / summary["support_volume"]
            )
    return summary


def stratify(rows, key, bins=3):
    numeric = np.asarray([_number(row, key) for row in rows], dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"B1 stratum {key} must be finite")
    edges = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return [{
            "low": float(edges[0]), "high": float(edges[0]),
            **summarize(rows)}]
    output = []
    for index in range(len(edges) - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        selected = [row for row in rows if (
            _number(row, key) >= low and (
                _number(row, key) <= high if index == len(edges) - 2
                else _number(row, key) < high))]
        output.append({"low": low, "high": high, **summarize(selected)})
    return output


def build_report(rows):
    if not rows:
        raise ValueError("B1 diagnostic rows are empty")
    return {
        "schema": "ct_seqtrack.b1_report.v1",
        "overall": summarize(rows),
        "strata": {
            "time_gap": stratify(rows, "query_delta_t"),
            "sparsity": stratify(rows, "current_target_points"),
            "recursive_age": stratify(rows, "recursive_age"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = build_report(load_rows(args.rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
