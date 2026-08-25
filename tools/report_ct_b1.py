"""Summarize repaired B1-v25 prior/support diagnostics and promotion strata."""

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


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tracklet_value(row):
    if "tracklet_key" in row:
        return str(row["tracklet_key"])
    if "tracklet_id" in row:
        return str(row["tracklet_id"])
    return None


def paired_tracklet_bootstrap_rmse(
        rows, candidate_key, reference_key,
        samples=2000, seed=20260825):
    """Cluster-bootstrap a paired RMSE difference over tracklets."""
    valid = [row for row in rows if (
        _tracklet_value(row) is not None
        and math.isfinite(_number(row, candidate_key))
        and math.isfinite(_number(row, reference_key)))]
    if not valid:
        return None
    tracklets = sorted({_tracklet_value(row) for row in valid})
    index = {key: position for position, key in enumerate(tracklets)}
    candidate_sum = np.zeros(len(tracklets), dtype=np.float64)
    reference_sum = np.zeros(len(tracklets), dtype=np.float64)
    counts = np.zeros(len(tracklets), dtype=np.float64)
    for row in valid:
        position = index[_tracklet_value(row)]
        candidate_sum[position] += _number(row, candidate_key) ** 2
        reference_sum[position] += _number(row, reference_key) ** 2
        counts[position] += 1.0
    point = float(
        np.sqrt(candidate_sum.sum() / counts.sum())
        - np.sqrt(reference_sum.sum() / counts.sum()))
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(samples), dtype=np.float64)
    for bootstrap_index in range(int(samples)):
        selected = rng.integers(
            0, len(tracklets), size=len(tracklets))
        denominator = counts[selected].sum()
        deltas[bootstrap_index] = (
            np.sqrt(candidate_sum[selected].sum() / denominator)
            - np.sqrt(reference_sum[selected].sum() / denominator))
    return {
        "unit": "tracklet",
        "samples": int(samples),
        "tracklets": len(tracklets),
        "point": point,
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975))],
        "upper_lt_zero": bool(np.quantile(deltas, 0.975) < 0.0),
    }


def summarize(rows):
    valid = [row for row in rows if (
        _number(row, "b1_valid") > 0
        and all(math.isfinite(_number(row, key)) for key in (
            "learned_motion_error", "kinematic_error", "b1_nll")))]
    if not valid:
        return {"count": 0}

    def values(key):
        return np.asarray([_number(row, key) for row in valid], dtype=float)

    learned = values("learned_motion_error")
    cv = values("kinematic_error")
    nll = values("b1_nll")
    positive_nll = np.maximum(nll, 0.0)
    top_count = max(1, int(math.ceil(0.01 * len(positive_nll))))
    top_share = float(np.sort(positive_nll)[-top_count:].sum() / max(
        float(positive_nll.sum()), 1e-12))
    result = {
        "count": len(valid),
        "learned_rmse": float(np.sqrt(np.mean(learned ** 2))),
        "cv_rmse": float(np.sqrt(np.mean(cv ** 2))),
        "learned_minus_cv_rmse": float(
            np.sqrt(np.mean(learned ** 2)) - np.sqrt(np.mean(cv ** 2))),
        "paired_help_rate": float(np.mean(learned < cv)),
        "learned_vs_cv_paired_bootstrap": (
            paired_tracklet_bootstrap_rmse(
                valid, "learned_motion_error", "kinematic_error")),
        "nll": float(nll.mean()),
        "top1pct_nll_share": top_share,
        "target_in_support_recall": float(values(
            "target_in_support").mean()),
        "support_volume": float(values("support_volume").mean()),
        "coverage": {
            level: float(values(f"b1_coverage_{level}").mean())
            for level in ("50", "80", "95")
        },
    }
    optional_quantiles = (
        "residual_unit_parallel", "residual_unit_perpendicular",
        "sigma_parallel", "sigma_perpendicular",
        "target_residual_unit_parallel",
        "target_residual_unit_perpendicular",
    )
    result["quantiles"] = {}
    for key in optional_quantiles:
        if all(key in row for row in valid):
            array = values(key)
            result["quantiles"][key] = {
                label: float(np.quantile(array, quantile))
                for label, quantile in (
                    ("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
            }
    saturation = []
    tail_axes = []
    for row in valid:
        for axis in ("parallel", "perpendicular"):
            recoverable_key = f"residual_recoverable_{axis}"
            residual_key = f"residual_unit_{axis}"
            if recoverable_key in row and residual_key in row:
                recoverable = _number(row, recoverable_key) > 0
                tail_axes.append(not recoverable)
                if recoverable:
                    saturation.append(abs(_number(row, residual_key)) >= 0.95)
    result["recoverable_saturation_rate"] = (
        float(np.mean(saturation)) if saturation else None)
    result["tail_axis_fraction"] = (
        float(np.mean(tail_axes)) if tail_axes else None)
    if all("b1_mahalanobis_sq" in row for row in valid):
        mahalanobis_sq = values("b1_mahalanobis_sq")
        levels = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
                  0.70, 0.80, 0.90, 0.95, 0.99)
        result["coverage_curve"] = [{
            "nominal": level,
            "observed": float(np.mean(
                mahalanobis_sq <= -2.0 * math.log(1.0 - level))),
        } for level in levels]
    return result


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


def _paired_backend_comparison(candidate_rows, reference_rows):
    def row_key(row):
        tracklet = _tracklet_value(row)
        if tracklet is None:
            raise ValueError(
                "backend comparison requires tracklet_key or tracklet_id")
        return (
            tracklet, int(_number(row, "frame_id")),
            int(_number(row, "candidate_id"))
            if "candidate_id" in row else 0)

    reference = {row_key(row): row for row in reference_rows}
    if len(reference) != len(reference_rows):
        raise ValueError("reference B1 rows contain duplicate endpoints")
    candidate_keys = [row_key(row) for row in candidate_rows]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("candidate B1 rows contain duplicate endpoints")
    if set(candidate_keys) != set(reference):
        raise ValueError(
            "backend comparison requires identical endpoint identities")
    paired = []
    for row in candidate_rows:
        key = row_key(row)
        other = reference[key]
        candidate_valid = _number(row, "b1_valid") > 0
        reference_valid = _number(other, "b1_valid") > 0
        if candidate_valid != reference_valid:
            raise ValueError(
                "backend comparison B1-valid masks are not identical")
        if not candidate_valid:
            continue
        paired.append({
            "tracklet_key": key[0],
            "candidate_error": _number(row, "learned_motion_error"),
            "reference_error": _number(other, "learned_motion_error"),
            "candidate_nll": _number(row, "b1_nll"),
            "reference_nll": _number(other, "b1_nll"),
            **{
                f"candidate_coverage_{level}": _number(
                    row, f"b1_coverage_{level}")
                for level in ("50", "80", "95")
            },
            **{
                f"reference_coverage_{level}": _number(
                    other, f"b1_coverage_{level}")
                for level in ("50", "80", "95")
            },
        })
    if not paired:
        raise ValueError("backend comparison has no matched valid endpoints")
    bootstrap = paired_tracklet_bootstrap_rmse(
        paired, "candidate_error", "reference_error")
    nominal = {"50": 0.50, "80": 0.80, "95": 0.95}
    candidate_coverage = {
        level: float(np.mean([
            row[f"candidate_coverage_{level}"] for row in paired]))
        for level in nominal}
    reference_coverage = {
        level: float(np.mean([
            row[f"reference_coverage_{level}"] for row in paired]))
        for level in nominal}
    candidate_ece = float(np.mean([
        abs(candidate_coverage[key] - nominal[key]) for key in nominal]))
    reference_ece = float(np.mean([
        abs(reference_coverage[key] - nominal[key]) for key in nominal]))
    candidate_nll = float(np.mean([
        row["candidate_nll"] for row in paired]))
    reference_nll = float(np.mean([
        row["reference_nll"] for row in paired]))
    return {
        "matched_endpoints": len(paired),
        "candidate_minus_reference_rmse": bootstrap,
        "candidate_nll": candidate_nll,
        "reference_nll": reference_nll,
        "candidate_coverage": candidate_coverage,
        "reference_coverage": reference_coverage,
        "candidate_coverage_ece": candidate_ece,
        "reference_coverage_ece": reference_ece,
        "promotion": {
            "passed": bool(
                bootstrap["upper_lt_zero"]
                and candidate_nll <= reference_nll
                and candidate_ece <= reference_ece),
            "rmse_ci_upper_lt_zero": bootstrap["upper_lt_zero"],
            "nll_not_worse": bool(candidate_nll <= reference_nll),
            "coverage_ece_not_worse": bool(candidate_ece <= reference_ece),
        },
    }


def build_report(rows, reference_rows=None):
    if not rows:
        raise ValueError("B1 diagnostic rows are empty")
    age_rows = [row for row in rows if (
        "recursive_age_valid" not in row
        or _number(row, "recursive_age_valid") > 0)]
    strata = {
        "time_gap": stratify(rows, "query_delta_t"),
        "recursive_age": stratify(age_rows, "recursive_age"),
        "kinematic_error_tail": stratify(rows, "kinematic_error", bins=4),
        "sparsity": stratify(rows, "current_target_points"),
    }
    if all("gap_ratio" in row for row in rows):
        strata["gap_ratio"] = stratify(rows, "gap_ratio", bins=4)
    report = {
        "schema": "ct_seqtrack.b1_report.v2",
        "overall": summarize(rows),
        "strata": strata,
    }
    if reference_rows is not None:
        report["backend_comparison"] = _paired_backend_comparison(
            rows, reference_rows)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument(
        "--reference-rows",
        help="matched GRU/reference rows for paired backend promotion")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = build_report(
        load_rows(args.rows),
        load_rows(args.reference_rows) if args.reference_rows else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
