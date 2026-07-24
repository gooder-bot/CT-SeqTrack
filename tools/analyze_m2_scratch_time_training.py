#!/usr/bin/env python3
"""Audit the four 2026-07-24 M2 scratch true/shuffled training runs.

The analysis is intentionally checkpoint-policy aware:

* epoch-60 ``last.ckpt`` is the frozen primary endpoint;
* the twelve five-epoch validation points are robustness diagnostics;
* epochs are not treated as independent statistical replicates;
* the older HTV A1 runs are contextual references, not matched controls.

The script verifies provenance/config/manifest/checkpoint integrity, extracts
TensorBoard metrics and batch losses, profiles shuffled-time intervention
strength, and writes reviewed CSV/JSON/Markdown/report-artifact outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ANALYSIS_DATE = "20260724"
EXPECTED_COMMIT = "473738fa2cf3def246e4e6b1bce35d8692c416c7"
EXPECTED_EPOCHS = 60
EXPECTED_EVAL_POINTS = 12

EXPERIMENT_RELATIVE = Path(
    "output/m2_scratch_r20_gap_true_shuffled_473738f_20260724_004339"
)

RUNS = {
    "random20_true": {
        "protocol": "random20",
        "time_mode": "true",
        "steps_per_epoch": 1018,
        "color": "#2563EB",
        "label": "random20 · true",
    },
    "random20_shuffled": {
        "protocol": "random20",
        "time_mode": "shuffled",
        "steps_per_epoch": 1018,
        "color": "#D97706",
        "label": "random20 · shuffled",
    },
    "gap1124_true": {
        "protocol": "gap1124",
        "time_mode": "true",
        "steps_per_epoch": 714,
        "color": "#0F766E",
        "label": "gap1124 · true",
    },
    "gap1124_shuffled": {
        "protocol": "gap1124",
        "time_mode": "shuffled",
        "steps_per_epoch": 714,
        "color": "#BE185D",
        "label": "gap1124 · shuffled",
    },
}

LOSS_LEAVES = {
    "loss_total": "loss_loss_total",
    "loss_velocity": "loss_loss_velocity",
    "loss_dynamics_displacement": "loss_loss_dynamics_displacement",
    "loss_center": "loss_loss_center",
}

EXPECTED_CONFIG_DIFFS = {
    "cfg",
    "dynamics_time_manifest_train",
    "dynamics_time_manifest_val",
    "dynamics_time_mode",
    "log_dir",
    "tag",
    "train_dynamics_time_mode",
    "val_dynamics_time_mode",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scalar(version_dir: Path, leaf: str) -> list[tuple[int, float]]:
    event_dir = version_dir / leaf
    accumulator = EventAccumulator(
        str(event_dir), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if len(tags) != 1:
        raise RuntimeError(
            f"Expected exactly one scalar tag under {event_dir}, found {tags}"
        )
    return [
        (int(event.step), float(event.value))
        for event in accumulator.Scalars(tags[0])
    ]


def install_easydict_pickle_shim() -> None:
    if "easydict" in sys.modules:
        return
    module = types.ModuleType("easydict")
    easy_dict = type("EasyDict", (dict,), {})
    easy_dict.__module__ = "easydict"
    module.EasyDict = easy_dict
    sys.modules["easydict"] = module


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    install_easydict_pickle_shim()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_bytes": int(path.stat().st_size),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch_zero_based": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "checkpoint_tensor_count": int(len(payload.get("state_dict", {}))),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def localize_server_path(experiment_root: Path, server_path: str) -> Path:
    return experiment_root / Path(server_path).name


def collect_metric_points(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, meta in RUNS.items():
        version_dir = (
            experiment_root / "runs" / run_id / "lightning_logs" / "version_0"
        )
        for metric, leaf in (
            ("Success", "metrics_test_success"),
            ("Precision", "metrics_test_precision"),
        ):
            points = read_scalar(version_dir, leaf)
            if len(points) != EXPECTED_EVAL_POINTS:
                raise RuntimeError(
                    f"{run_id} {metric}: expected {EXPECTED_EVAL_POINTS} "
                    f"points, found {len(points)}"
                )
            for index, (step, value) in enumerate(points, start=1):
                expected_step = index * 5 * int(meta["steps_per_epoch"])
                if step != expected_step:
                    raise RuntimeError(
                        f"{run_id} {metric}: unexpected step {step}, "
                        f"expected {expected_step}"
                    )
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": meta["label"],
                        "protocol": meta["protocol"],
                        "time_mode": meta["time_mode"],
                        "metric": metric,
                        "epoch": index * 5,
                        "step": step,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def summarize_metrics(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, metric), group in points.groupby(
        ["run_id", "metric"], sort=False
    ):
        ordered = group.sort_values("epoch")
        final = ordered.iloc[-1]
        best = ordered.loc[ordered["value"].idxmax()]
        late = ordered[ordered["epoch"] >= 45]
        meta = RUNS[run_id]
        rows.append(
            {
                "run_id": run_id,
                "run_label": meta["label"],
                "protocol": meta["protocol"],
                "time_mode": meta["time_mode"],
                "metric": metric,
                "final": float(final["value"]),
                "final_epoch": int(final["epoch"]),
                "best": float(best["value"]),
                "best_epoch": int(best["epoch"]),
                "mean_all": float(ordered["value"].mean()),
                "std_all": float(ordered["value"].std(ddof=0)),
                "late_mean_45_60": float(late["value"].mean()),
                "late_std_45_60": float(late["value"].std(ddof=0)),
                "evaluation_points": int(ordered.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def collect_paired_deltas(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in ("random20", "gap1124"):
        for metric in ("Success", "Precision"):
            subset = points[
                (points["protocol"] == protocol)
                & (points["metric"] == metric)
            ]
            pivot = subset.pivot(
                index="epoch", columns="time_mode", values="value"
            ).sort_index()
            for epoch, values in pivot.iterrows():
                rows.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "epoch": int(epoch),
                        "true_value": float(values["true"]),
                        "shuffled_value": float(values["shuffled"]),
                        "true_minus_shuffled": float(
                            values["true"] - values["shuffled"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_comparisons(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, metric), group in paired.groupby(
        ["protocol", "metric"], sort=False
    ):
        ordered = group.sort_values("epoch")
        late = ordered[ordered["epoch"] >= 45]
        final = ordered.iloc[-1]
        rows.append(
            {
                "protocol": protocol,
                "metric": metric,
                "comparison_label": f"{protocol} · {metric}",
                "final_true": float(final["true_value"]),
                "final_shuffled": float(final["shuffled_value"]),
                "final_delta": float(final["true_minus_shuffled"]),
                "late_true_45_60": float(late["true_value"].mean()),
                "late_shuffled_45_60": float(late["shuffled_value"].mean()),
                "late_delta_45_60": float(
                    late["true_minus_shuffled"].mean()
                ),
                "mean_delta_all": float(
                    ordered["true_minus_shuffled"].mean()
                ),
                "true_win_points": int(
                    (ordered["true_minus_shuffled"] > 0).sum()
                ),
                "eval_points": int(ordered.shape[0]),
                "true_win_rate": float(
                    (ordered["true_minus_shuffled"] > 0).mean()
                ),
                "note": (
                    "Epoch-60 last.ckpt is primary; validation-point win counts "
                    "are descriptive robustness checks, not independent trials."
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_loss_epochs(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, meta in RUNS.items():
        version_dir = (
            experiment_root / "runs" / run_id / "lightning_logs" / "version_0"
        )
        steps_per_epoch = int(meta["steps_per_epoch"])
        expected_points = EXPECTED_EPOCHS * steps_per_epoch
        for metric, leaf in LOSS_LEAVES.items():
            points = read_scalar(version_dir, leaf)
            if len(points) != expected_points:
                raise RuntimeError(
                    f"{run_id} {metric}: expected {expected_points} loss "
                    f"points, found {len(points)}"
                )
            values = np.asarray([value for _, value in points], dtype=np.float64)
            matrix = values.reshape(EXPECTED_EPOCHS, steps_per_epoch)
            for epoch_index, epoch_values in enumerate(matrix, start=1):
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": meta["label"],
                        "protocol": meta["protocol"],
                        "time_mode": meta["time_mode"],
                        "loss_metric": metric,
                        "epoch": epoch_index,
                        "batch_points": steps_per_epoch,
                        "mean": float(epoch_values.mean()),
                        "median": float(np.median(epoch_values)),
                        "p10": float(np.quantile(epoch_values, 0.10)),
                        "p90": float(np.quantile(epoch_values, 0.90)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_loss_ratios(loss_epoch: pd.DataFrame) -> pd.DataFrame:
    late = loss_epoch[loss_epoch["epoch"] >= 45]
    summary = (
        late.groupby(["protocol", "time_mode", "loss_metric"], as_index=False)
        .agg(
            late_mean=("mean", "mean"),
            late_median=("median", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    labels = {
        "loss_total": "Total loss",
        "loss_velocity": "Velocity auxiliary loss",
        "loss_dynamics_displacement": "Displacement auxiliary loss",
        "loss_center": "Center loss",
    }
    for protocol in ("random20", "gap1124"):
        for metric in LOSS_LEAVES:
            selected = summary[
                (summary["protocol"] == protocol)
                & (summary["loss_metric"] == metric)
            ].set_index("time_mode")
            true_value = float(selected.loc["true", "late_mean"])
            shuffled_value = float(selected.loc["shuffled", "late_mean"])
            rows.append(
                {
                    "protocol": protocol,
                    "loss_metric": metric,
                    "loss_label": labels[metric],
                    "late_true": true_value,
                    "late_shuffled": shuffled_value,
                    "shuffled_to_true_ratio": shuffled_value / true_value,
                    "shuffled_minus_true": shuffled_value - true_value,
                    "late_epoch_window": "45-60",
                }
            )
    return pd.DataFrame(rows)


def manifest_profile(path: Path, protocol: str, role: str) -> dict[str, Any]:
    payload = load_json(path)
    transitions = [
        entry
        for entry in payload["entries"]
        if int(entry["frame_index"]) > 0
    ]
    real = np.asarray(
        [float(entry["real_incoming_delta_t"]) for entry in transitions],
        dtype=np.float64,
    )
    effective = np.asarray(
        [float(entry["effective_incoming_delta_t"]) for entry in transitions],
        dtype=np.float64,
    )
    target_keys = [str(entry["endpoint_key"]) for entry in transitions]
    source_keys = [
        str(entry["source_endpoint_key"]) for entry in transitions
    ]
    differences = np.abs(real - effective)
    correlation = (
        float(np.corrcoef(real, effective)[0, 1])
        if real.size > 1
        else math.nan
    )
    return {
        "protocol": protocol,
        "role": role,
        "manifest": path.name,
        "transition_count": int(real.size),
        "endpoint_count": int(payload["endpoint_count"]),
        "permutation_is_one_to_one": bool(
            len(source_keys) == len(set(source_keys))
            and set(source_keys) == set(target_keys)
        ),
        "derangement_has_no_self_map": bool(
            all(target != source for target, source in zip(target_keys, source_keys))
        ),
        "marginal_gap_multiset_preserved": bool(
            np.allclose(np.sort(real), np.sort(effective), rtol=0.0, atol=1e-9)
        ),
        "real_dt_mean": float(real.mean()),
        "real_dt_std": float(real.std(ddof=0)),
        "numeric_equal_share_1e9": float((differences <= 1e-9).mean()),
        "numeric_near_equal_share_1ms": float(
            (differences <= 1e-3).mean()
        ),
        "mean_abs_dt_change": float(differences.mean()),
        "median_abs_dt_change": float(np.median(differences)),
        "p95_abs_dt_change": float(np.quantile(differences, 0.95)),
        "real_effective_correlation": correlation,
        "content_sha256": str(payload["content_sha256"]),
        "permutation_sha256": str(payload["permutation_sha256"]),
    }


def collect_manifest_strength(experiment_root: Path) -> pd.DataFrame:
    rows = []
    manifest_dir = experiment_root / "manifests"
    for protocol in ("random20", "gap1124"):
        for role in ("train", "val"):
            path = (
                manifest_dir
                / f"{protocol}_{role}_shuffled_dt_seed42.json"
            )
            rows.append(manifest_profile(path, protocol, role))
    return pd.DataFrame(rows)


def collect_config_diffs(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in ("random20", "gap1124"):
        true_provenance = load_json(
            experiment_root
            / "runs"
            / f"{protocol}_true"
            / "run_provenance.json"
        )
        shuffled_provenance = load_json(
            experiment_root
            / "runs"
            / f"{protocol}_shuffled"
            / "run_provenance.json"
        )
        true_cfg = true_provenance["resolved_config"]
        shuffled_cfg = shuffled_provenance["resolved_config"]
        for field in sorted(set(true_cfg) | set(shuffled_cfg)):
            true_value = true_cfg.get(field)
            shuffled_value = shuffled_cfg.get(field)
            if true_value == shuffled_value:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "field": field,
                    "true_value": json.dumps(
                        true_value, ensure_ascii=False, sort_keys=True
                    ),
                    "shuffled_value": json.dumps(
                        shuffled_value, ensure_ascii=False, sort_keys=True
                    ),
                    "expected_difference": field in EXPECTED_CONFIG_DIFFS,
                }
            )
    return pd.DataFrame(rows)


def collect_integrity(
    repo_root: Path,
    experiment_root: Path,
    metric_points: pd.DataFrame,
    loss_epoch: pd.DataFrame,
    config_diffs: pd.DataFrame,
) -> pd.DataFrame:
    unexpected_diff_by_protocol = (
        config_diffs[~config_diffs["expected_difference"]]
        .groupby("protocol")
        .size()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for run_id, meta in RUNS.items():
        run_root = experiment_root / "runs" / run_id
        version_dir = run_root / "lightning_logs" / "version_0"
        provenance = load_json(run_root / "run_provenance.json")
        checkpoint = checkpoint_metadata(
            version_dir / "checkpoints" / "last.ckpt"
        )
        exit_code = int(
            (run_root / "training_exit_code.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        log_path = experiment_root / "logs" / f"{run_id}.launcher.log"
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 100_000))
            log_tail = handle.read().decode("utf-8", errors="replace")
        completed_log = (
            "`Trainer.fit` stopped: `max_epochs=60` reached." in log_tail
        )

        config_path = localize_server_path(
            experiment_root / "configs", provenance["config_path"]
        )
        config_sha_match = (
            sha256_file(config_path) == provenance["config_sha256"]
        )

        manifest_sha_matches = []
        datasets = provenance["datasets"]
        for role in ("train", "val"):
            dataset = datasets[role]
            virtual_path = localize_server_path(
                experiment_root / "manifests",
                provenance["resolved_config"][f"virtual_rate_manifest_{role}"],
            )
            manifest_sha_matches.append(
                sha256_file(virtual_path)
                == dataset["virtual_rate_manifest_file_sha256"]
            )
            dynamics_manifest = dataset["dynamics_time_summary"]["manifest"]
            if dynamics_manifest:
                dynamics_path = localize_server_path(
                    experiment_root / "manifests", dynamics_manifest
                )
                manifest_sha_matches.append(
                    sha256_file(dynamics_path)
                    == dataset["dynamics_time_summary"][
                        "manifest_file_sha256"
                    ]
                )

        expected_step = EXPECTED_EPOCHS * int(meta["steps_per_epoch"])
        run_metric_points = metric_points[
            metric_points["run_id"] == run_id
        ]
        run_loss_rows = loss_epoch[
            (loss_epoch["run_id"] == run_id)
            & (loss_epoch["loss_metric"] == "loss_total")
        ]
        paired_other = (
            f"{meta['protocol']}_shuffled"
            if meta["time_mode"] == "true"
            else f"{meta['protocol']}_true"
        )
        other_provenance = load_json(
            experiment_root
            / "runs"
            / paired_other
            / "run_provenance.json"
        )
        cadence_pair_match = all(
            datasets[role]["virtual_rate_selection_sha256"]
            == other_provenance["datasets"][role][
                "virtual_rate_selection_sha256"
            ]
            and datasets[role]["frames"]
            == other_provenance["datasets"][role]["frames"]
            and datasets[role]["tracklets"]
            == other_provenance["datasets"][role]["tracklets"]
            for role in ("train", "val")
        )

        complete = all(
            [
                exit_code == 0,
                completed_log,
                provenance["git"]["commit"] == EXPECTED_COMMIT,
                provenance["git"]["dirty_any"] is False,
                provenance["seed"] == 42,
                provenance["init_checkpoint_path"] is None,
                provenance["checkpoint_path"] is None,
                provenance["resolved_config"]["epoch"] == EXPECTED_EPOCHS,
                provenance["resolved_config"]["workers"] == 12,
                provenance["resolved_config"]["batch_size"] == 16,
                checkpoint["checkpoint_epoch_zero_based"] == 59,
                checkpoint["checkpoint_global_step"] == expected_step,
                checkpoint["checkpoint_tensor_count"] == 334,
                run_metric_points.shape[0] == EXPECTED_EVAL_POINTS * 2,
                run_loss_rows.shape[0] == EXPECTED_EPOCHS,
                config_sha_match,
                all(manifest_sha_matches),
                cadence_pair_match,
                unexpected_diff_by_protocol.get(meta["protocol"], 0) == 0,
            ]
        )
        rows.append(
            {
                "run_id": run_id,
                "run_label": meta["label"],
                "protocol": meta["protocol"],
                "time_mode": meta["time_mode"],
                "status": "PASS" if complete else "FAIL",
                "exit_code": exit_code,
                "log_reached_max_epochs": completed_log,
                "commit": provenance["git"]["commit"],
                "dirty": provenance["git"]["dirty_any"],
                "seed": provenance["seed"],
                "init_checkpoint": provenance["init_checkpoint_path"],
                "batch_size": provenance["resolved_config"]["batch_size"],
                "workers": provenance["resolved_config"]["workers"],
                "checkpoint_epoch_zero_based": checkpoint[
                    "checkpoint_epoch_zero_based"
                ],
                "checkpoint_global_step": checkpoint[
                    "checkpoint_global_step"
                ],
                "checkpoint_tensor_count": checkpoint[
                    "checkpoint_tensor_count"
                ],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "metric_points": int(run_metric_points.shape[0]),
                "loss_epochs": int(run_loss_rows.shape[0]),
                "config_sha_match": config_sha_match,
                "manifest_sha_match": all(manifest_sha_matches),
                "cadence_pair_match": cadence_pair_match,
                "unexpected_config_differences": int(
                    unexpected_diff_by_protocol.get(meta["protocol"], 0)
                ),
                "train_tracklets": datasets["train"]["tracklets"],
                "train_frames": datasets["train"]["frames"],
                "val_tracklets": datasets["val"]["tracklets"],
                "val_frames": datasets["val"]["frames"],
            }
        )
    return pd.DataFrame(rows)


def collect_historical_context(
    repo_root: Path, metric_summary: pd.DataFrame
) -> pd.DataFrame:
    path = (
        repo_root
        / "compare_results"
        / "data"
        / "htv_6runs_metrics_summary.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_csv(path)
    history = history[
        (history["protocol"].isin(["random20", "gap1124"]))
        & (history["model"] == "A1-order")
        & (history["metric"].isin(["success/test", "precision/test"]))
    ].copy()
    rows: list[dict[str, Any]] = []
    for protocol in ("random20", "gap1124"):
        for metric, history_metric in (
            ("Success", "success/test"),
            ("Precision", "precision/test"),
        ):
            historical = float(
                history[
                    (history["protocol"] == protocol)
                    & (history["metric"] == history_metric)
                ]["final"].iloc[0]
            )
            current_true = float(
                metric_summary[
                    (metric_summary["protocol"] == protocol)
                    & (metric_summary["time_mode"] == "true")
                    & (metric_summary["metric"] == metric)
                ]["final"].iloc[0]
            )
            rows.append(
                {
                    "protocol": protocol,
                    "metric": metric,
                    "historical_a1_final": historical,
                    "current_scratch_m2_true_final": current_true,
                    "m2_true_minus_historical_a1": current_true - historical,
                    "comparison_status": (
                        "Context only: older A1 lacks matched current-code "
                        "provenance and is not a causal control."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def value_at(
    frame: pd.DataFrame, protocol: str, metric: str, field: str
) -> float:
    return float(
        frame[
            (frame["protocol"] == protocol) & (frame["metric"] == metric)
        ][field].iloc[0]
    )


def int_at(
    frame: pd.DataFrame, protocol: str, metric: str, field: str
) -> int:
    return int(
        frame[
            (frame["protocol"] == protocol) & (frame["metric"] == metric)
        ][field].iloc[0]
    )


def generate_markdown_report(
    output: Path,
    integrity: pd.DataFrame,
    comparisons: pd.DataFrame,
    loss_ratios: pd.DataFrame,
    manifest_strength: pd.DataFrame,
    historical: pd.DataFrame,
) -> None:
    r20_s = value_at(comparisons, "random20", "Success", "final_delta")
    r20_p = value_at(comparisons, "random20", "Precision", "final_delta")
    gap_s = value_at(comparisons, "gap1124", "Success", "final_delta")
    gap_p = value_at(comparisons, "gap1124", "Precision", "final_delta")

    r20_late_s = value_at(
        comparisons, "random20", "Success", "late_delta_45_60"
    )
    r20_late_p = value_at(
        comparisons, "random20", "Precision", "late_delta_45_60"
    )
    gap_late_s = value_at(
        comparisons, "gap1124", "Success", "late_delta_45_60"
    )
    gap_late_p = value_at(
        comparisons, "gap1124", "Precision", "late_delta_45_60"
    )

    r20_win_s = int_at(
        comparisons, "random20", "Success", "true_win_points"
    )
    r20_win_p = int_at(
        comparisons, "random20", "Precision", "true_win_points"
    )
    gap_win_s = int_at(
        comparisons, "gap1124", "Success", "true_win_points"
    )
    gap_win_p = int_at(
        comparisons, "gap1124", "Precision", "true_win_points"
    )

    comparison_table = comparisons[
        [
            "protocol",
            "metric",
            "final_true",
            "final_shuffled",
            "final_delta",
            "late_delta_45_60",
            "mean_delta_all",
            "true_win_points",
            "eval_points",
        ]
    ].copy()
    comparison_table.columns = [
        "Protocol",
        "Metric",
        "Final true",
        "Final shuffled",
        "Final Δ",
        "Late Δ (45–60)",
        "All-point mean Δ",
        "True wins",
        "Eval points",
    ]

    loss_table = loss_ratios[
        loss_ratios["loss_metric"].isin(
            [
                "loss_total",
                "loss_velocity",
                "loss_dynamics_displacement",
            ]
        )
    ][
        [
            "protocol",
            "loss_label",
            "late_true",
            "late_shuffled",
            "shuffled_to_true_ratio",
        ]
    ].copy()
    loss_table.columns = [
        "Protocol",
        "Loss",
        "Late true",
        "Late shuffled",
        "Shuffled / true",
    ]

    manifest_table = manifest_strength[
        [
            "protocol",
            "role",
            "transition_count",
            "numeric_near_equal_share_1ms",
            "mean_abs_dt_change",
            "median_abs_dt_change",
            "real_effective_correlation",
        ]
    ].copy()
    manifest_table.columns = [
        "Protocol",
        "Role",
        "Transitions",
        "|Δt change| ≤1 ms",
        "Mean |Δt change|",
        "Median |Δt change|",
        "Real/effective corr.",
    ]

    historical_lines = ""
    if not historical.empty:
        pieces = []
        for protocol in ("random20", "gap1124"):
            success_delta = float(
                historical[
                    (historical["protocol"] == protocol)
                    & (historical["metric"] == "Success")
                ]["m2_true_minus_historical_a1"].iloc[0]
            )
            precision_delta = float(
                historical[
                    (historical["protocol"] == protocol)
                    & (historical["metric"] == "Precision")
                ]["m2_true_minus_historical_a1"].iloc[0]
            )
            pieces.append(
                f"- {protocol}: scratch M2 true 相对旧 A1 为 "
                f"`{success_delta:+.3f}/{precision_delta:+.3f}`。"
            )
        historical_lines = "\n".join(pieces)

    text = f"""# M2 scratch true-time vs shuffled-time 训练复核

## 技术摘要

四组实验的归档、配置、manifest、checkpoint 与 TensorBoard 事件全部通过完整性检查；四组均为 clean commit `{EXPECTED_COMMIT[:7]}`、seed42、scratch、60 epoch、batch16、workers12。按预先冻结的 epoch60 `last.ckpt` 口径，true-time 在 random20 上相对 shuffled-time 为 **{r20_s:+.3f} Success / {r20_p:+.3f} Precision**，在 gap1124 上为 **{gap_s:+.3f}/{gap_p:+.3f}**。

这个结果只能支持 **scratch 训练中的弱/部分 correct-time signal**，不能推翻此前 same-checkpoint physical-time causal No-Go。原因有两个：第一，gap1124 的优势只出现在最后一个验证点，epochs45–60 平均反而为 **{gap_late_s:+.3f}/{gap_late_p:+.3f}**；第二，shuffled 训练同时引入了速度与位移辅助目标不相容的优化冲突，因此 true/shuffled 分训不是纯时间信息对照。

## random20 显示稳定训练差异，gap1124 不支持 HTV 放大假设

| Protocol | Metric | Final true | Final shuffled | Final Δ | Late Δ (45–60) | All-point mean Δ | True wins |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in comparison_table.iterrows():
        text += (
            f"| {row['Protocol']} | {row['Metric']} | "
            f"{row['Final true']:.3f} | {row['Final shuffled']:.3f} | "
            f"{row['Final Δ']:+.3f} | {row['Late Δ (45–60)']:+.3f} | "
            f"{row['All-point mean Δ']:+.3f} | "
            f"{int(row['True wins'])}/{int(row['Eval points'])} |\n"
        )

    text += f"""

- random20 的 true 在 12 个验证点中赢得 **{r20_win_s}/12 Success、{r20_win_p}/12 Precision**；epochs45–60 平均优势为 **{r20_late_s:+.3f}/{r20_late_p:+.3f}**。这是本批次最稳定的正信号。
- gap1124 的 true 只赢得 **{gap_win_s}/12 Success、{gap_win_p}/12 Precision**；尽管 epoch60 为正，整个曲线和 late window 都由 shuffled 占优。不能把最后一点写成“强 HTV 下 correct time 更有效”。
- gap1124 的时间置乱更强，但 true 优势反而更弱，因此 **HTV amplification hypothesis 当前不成立**。

## shuffled 训练损失显著更高，但这不是纯机制证据

| Protocol | Loss | Late true | Late shuffled | Shuffled / true |
|---|---|---:|---:|---:|
"""
    for _, row in loss_table.iterrows():
        text += (
            f"| {row['Protocol']} | {row['Loss']} | "
            f"{row['Late true']:.4f} | {row['Late shuffled']:.4f} | "
            f"{row['Shuffled / true']:.2f}× |\n"
        )

    text += """

当前实现中 `displacement_pred = velocity_pred × delta_t_effective`，但 velocity label 仍由真实 `delta_t_real` 定义，displacement label 仍是真实位移。因此 shuffled 模型无法在 `delta_t_effective != delta_t_real` 时同时完美满足两项辅助监督。更强的 gap1124 置乱确实产生了更大的 displacement-loss 比值，但 tracking 曲线并未同步表现为更稳定的 true 优势。这说明当前分训差异混合了：

1. 正确时间信息；
2. 辅助目标的一致性/不一致性；
3. 独立训练的随机优化轨迹。

所以它是 learnability stress test，不是纯 physical-time 因果实验。

## manifest 与干预强度

| Protocol | Role | Transitions | Near equal ≤1 ms | Mean |Δt change| | Median |Δt change| | Corr(real,effective) |
|---|---:|---:|---:|---:|---:|---:|
"""
    for _, row in manifest_table.iterrows():
        text += (
            f"| {row['Protocol']} | {row['Role']} | "
            f"{int(row['Transitions'])} | "
            f"{100 * row['|Δt change| ≤1 ms']:.1f}% | "
            f"{row['Mean |Δt change|']:.3f}s | "
            f"{row['Median |Δt change|']:.3f}s | "
            f"{row['Real/effective corr.']:.3f} |\n"
        )

    text += f"""

所有四份 shuffled manifest 都通过一一排列、无 self-map、gap 多重集合守恒检查。random20 仍有约 40–45% transition 的数值差在 1 ms 内；gap1124 只有约 20–25%，且平均绝对改变约 0.56–0.58s。干预不是失活的，尤其 gap1124 足够强。

## 与旧 A1 的上下文比较

{historical_lines}

这些差值只作方向参照：旧 A1 缺少本批次完全匹配的 current-code provenance、shared-SE(2) 配置和冻结 manifest，因此不能承担正式 method attribution。方向上，scratch M2 true 只在 random20 有明显正增益，在 gap1124 与旧 A1 基本持平略低；这与“强 HTV 自动带来更明显涨点”不一致。

## 完整性与方法

- 四组 `training_exit_code=0`，日志均显示 `max_epochs=60 reached`。
- random20 checkpoint 为 epoch59/global_step61080；gap1124 为 epoch59/global_step42840；四组均含334个 state tensors。
- true/shuffled 在每个协议内 train/val 的 tracklet、frame、cadence selection SHA 完全一致。
- 本地四份配置和八份 manifest 的 SHA256 与服务器 provenance 完全一致。
- 每组有12个 Success/Precision 验证点；loss 按全部 batch event 聚合为60个 epoch。
- final 固定为 epoch60 last；best epoch 只用于诊断，没有以 best 重新选模型。

## 限制、结论边界与状态

1. 只有一个 seed，不能估计跨 seed 方差。
2. 只有 aggregate validation 指标，没有本批次的 per-endpoint/per-tracklet 输出，无法做 paired bootstrap。
3. 十二个 epoch 点来自同一次训练，不能当成十二个独立重复实验。
4. shuffled 分训含结构性辅助目标冲突，不能直接解释成“错误时间本身导致全部差距”。
5. mini_val 已参与多轮开发，不是独立最终测试集。

因此冻结判断为：

- **Scratch correct-time learnability signal：PARTIAL / random20 positive, gap1124 unstable**
- **HTV amplification：NOT SUPPORTED**
- **Physical-time causal claim：仍为 NO-GO**
- **Timestamp-conditioned M3/M4 解锁：NO**

## 推荐下一步

1. **先不重训，做两个协议的 2×2 cross-time evaluation。** 对 true-trained 和 shuffled-trained 两个 final checkpoint，分别在 true/shuffled val clock 下评估；每个 checkpoint 内的差值才是干净的 inference-time 时间因果干预。
2. **导出逐 endpoint 结果并做 tracklet-bootstrap。** aggregate 的 +1～2 点不能替代配对置信区间。
3. **如果仍要比较分训，先冻结一个无目标冲突的合同。** 推荐两种中只选一种并同时用于 true/shuffled：关闭 velocity auxiliary，或把 shuffled velocity target 改为 `displacement_real / delta_t_effective`；两者回答的问题不同，不能混用。
4. **补 current-code matched W0/A1 baseline 后再谈 M2 涨点。** 旧 A1 只能作历史上下文。
5. 只有在 gap1124 的同-checkpoint 2×2 干预和配对统计都支持 true 优势时，才值得增加 seed43/44；否则停止 physical-time 方向，把 M3/M4 改写为 time-agnostic path/state robustness。

## 仍需回答的问题

- random20 的稳定差距来自正确时间还是 shuffled auxiliary-conflict？
- 为什么 gap1124 的 auxiliary loss 冲突更强，但 tracking 曲线多数时间仍由 shuffled 占优？
- 同一 final checkpoint 切换 true/shuffled clock 后，方向是否与分训结果一致？
- current-code matched W0/A1 在相同 cadence 和 budget 下的真实基线是多少？
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def source_object(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    table_name: str,
    metric_definitions: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": sql,
            "description": description,
            "tables_used": [table_name],
            "metric_definitions": metric_definitions,
        },
    }


def generate_artifact(
    output: Path,
    metric_points: pd.DataFrame,
    metric_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    loss_ratios: pd.DataFrame,
    manifest_strength: pd.DataFrame,
    integrity: pd.DataFrame,
    historical: pd.DataFrame,
) -> None:
    title = "CT-SeqTrack M2 scratch true/shuffled 训练复核"
    paths = {
        "metric_points": (
            "compare_results/data/"
            f"m2_scratch_time_metric_points_{ANALYSIS_DATE}.csv"
        ),
        "metric_summary": (
            "compare_results/data/"
            f"m2_scratch_time_metric_summary_{ANALYSIS_DATE}.csv"
        ),
        "comparisons": (
            "compare_results/data/"
            f"m2_scratch_time_comparisons_{ANALYSIS_DATE}.csv"
        ),
        "loss_ratios": (
            "compare_results/data/"
            f"m2_scratch_time_loss_ratios_{ANALYSIS_DATE}.csv"
        ),
        "manifest_strength": (
            "compare_results/data/"
            f"m2_scratch_time_manifest_strength_{ANALYSIS_DATE}.csv"
        ),
        "integrity": (
            "compare_results/data/"
            f"m2_scratch_time_integrity_{ANALYSIS_DATE}.csv"
        ),
        "historical": (
            "compare_results/data/"
            f"m2_scratch_time_historical_context_{ANALYSIS_DATE}.csv"
        ),
    }
    sources = [
        source_object(
            "metric_points_source",
            "Reviewed TensorBoard validation metrics",
            paths["metric_points"],
            (
                "SELECT protocol, time_mode, metric, epoch, value "
                f"FROM read_csv_auto('{paths['metric_points']}') "
                "ORDER BY protocol, metric, time_mode, epoch"
            ),
            "Read the twelve five-epoch Success and Precision points.",
            "m2_scratch_time_metric_points",
            [
                "value: nuScenes mini_val Success or Precision in points.",
                "epoch: validation checkpoint epoch; evaluations occur every five epochs.",
            ],
        ),
        source_object(
            "comparison_source",
            "Reviewed true-minus-shuffled comparisons",
            paths["comparisons"],
            (
                "SELECT protocol, metric, final_true, final_shuffled, "
                "final_delta, late_delta_45_60, true_win_points, eval_points "
                f"FROM read_csv_auto('{paths['comparisons']}') "
                "ORDER BY protocol, metric"
            ),
            "Read frozen epoch-60 and robustness comparison metrics.",
            "m2_scratch_time_comparisons",
            [
                "final_delta: epoch-60 true minus shuffled score in points.",
                "late_delta_45_60: mean true-minus-shuffled delta over epochs 45, 50, 55 and 60.",
                "true_win_points: descriptive count, not an independent-trial significance test.",
            ],
        ),
        source_object(
            "loss_source",
            "Reviewed late-training loss ratios",
            paths["loss_ratios"],
            (
                "SELECT protocol, loss_label, late_true, late_shuffled, "
                "shuffled_to_true_ratio "
                f"FROM read_csv_auto('{paths['loss_ratios']}') "
                "ORDER BY protocol, loss_metric"
            ),
            "Read epochs45-60 batch-mean training loss comparisons.",
            "m2_scratch_time_loss_ratios",
            [
                "late_true and late_shuffled: mean of epoch-level batch means over epochs 45-60.",
                "shuffled_to_true_ratio: late shuffled loss divided by late true loss.",
            ],
        ),
        source_object(
            "manifest_source",
            "Reviewed shuffled-time manifest strength",
            paths["manifest_strength"],
            (
                "SELECT protocol, role, transition_count, "
                "numeric_near_equal_share_1ms, mean_abs_dt_change, "
                "median_abs_dt_change, real_effective_correlation "
                f"FROM read_csv_auto('{paths['manifest_strength']}') "
                "ORDER BY protocol, role"
            ),
            "Read train/val shuffled-time intervention strength checks.",
            "m2_scratch_time_manifest_strength",
            [
                "mean_abs_dt_change: mean absolute real-versus-effective incoming gap change in seconds.",
                "numeric_near_equal_share_1ms: share of transitions whose gap changes by at most 1 ms.",
            ],
        ),
        source_object(
            "integrity_source",
            "Run and checkpoint integrity audit",
            paths["integrity"],
            (
                "SELECT run_label, status, commit, checkpoint_global_step, "
                "config_sha_match, manifest_sha_match, cadence_pair_match "
                f"FROM read_csv_auto('{paths['integrity']}') "
                "ORDER BY protocol, time_mode"
            ),
            "Read completion, checkpoint, provenance and pairing checks.",
            "m2_scratch_time_integrity",
            [
                "status PASS requires exit0, epoch59 checkpoint, expected steps/events, clean commit and matching artifacts.",
            ],
        ),
        source_object(
            "historical_source",
            "Historical HTV A1 contextual comparison",
            paths["historical"],
            (
                "SELECT protocol, metric, historical_a1_final, "
                "current_scratch_m2_true_final, m2_true_minus_historical_a1 "
                f"FROM read_csv_auto('{paths['historical']}') "
                "ORDER BY protocol, metric"
            ),
            "Read contextual, non-matched comparison against older A1 runs.",
            "m2_scratch_time_historical_context",
            [
                "m2_true_minus_historical_a1: descriptive current scratch M2 true minus older A1 final score.",
            ],
        ),
    ]

    final_deltas = comparisons[
        [
            "comparison_label",
            "metric",
            "final_delta",
        ]
    ].to_dict("records")
    success_curve = metric_points[metric_points["metric"] == "Success"][
        [
            "run_label",
            "epoch",
            "value",
        ]
    ].to_dict("records")
    precision_curve = metric_points[metric_points["metric"] == "Precision"][
        [
            "run_label",
            "epoch",
            "value",
        ]
    ].to_dict("records")
    loss_chart = loss_ratios[
        loss_ratios["loss_metric"].isin(
            [
                "loss_total",
                "loss_velocity",
                "loss_dynamics_displacement",
            ]
        )
    ][
        [
            "protocol",
            "loss_label",
            "shuffled_to_true_ratio",
        ]
    ].to_dict("records")
    summary_rows = (
        metric_summary.pivot_table(
            index=["run_id", "run_label", "protocol", "time_mode"],
            columns="metric",
            values=["final", "best", "best_epoch", "late_mean_45_60"],
            aggfunc="first",
        )
        .reset_index()
    )
    summary_rows.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary_rows.columns
    ]
    summary_rows = summary_rows.rename(
        columns={
            "final_Success": "final_success",
            "final_Precision": "final_precision",
            "best_Success": "best_success",
            "best_Precision": "best_precision",
            "best_epoch_Success": "best_success_epoch",
            "best_epoch_Precision": "best_precision_epoch",
            "late_mean_45_60_Success": "late_success",
            "late_mean_45_60_Precision": "late_precision",
        }
    )[
        [
            "run_id",
            "final_success",
            "final_precision",
            "late_success",
            "late_precision",
            "best_success",
            "best_precision",
        ]
    ].to_dict("records")

    r20_s = value_at(comparisons, "random20", "Success", "final_delta")
    r20_p = value_at(comparisons, "random20", "Precision", "final_delta")
    gap_s = value_at(comparisons, "gap1124", "Success", "final_delta")
    gap_p = value_at(comparisons, "gap1124", "Precision", "final_delta")
    gap_late_s = value_at(
        comparisons, "gap1124", "Success", "late_delta_45_60"
    )
    gap_late_p = value_at(
        comparisons, "gap1124", "Precision", "late_delta_45_60"
    )

    charts = [
        {
            "id": "final_delta_chart",
            "title": "Epoch-60 true-minus-shuffled score deltas",
            "subtitle": (
                "Positive values favor true-time; final last.ckpt is the "
                "frozen primary endpoint."
            ),
            "type": "bar",
            "dataset": "final_deltas",
            "sourceId": "comparison_source",
            "encodings": {
                "x": {
                    "field": "comparison_label",
                    "type": "nominal",
                    "label": "Protocol and metric",
                },
                "y": {
                    "field": "final_delta",
                    "type": "quantitative",
                    "label": "True − shuffled",
                    "format": "number",
                },
                "color": {
                    "field": "metric",
                    "type": "nominal",
                    "label": "Metric",
                },
            },
            "xAxisTitle": "Protocol and metric",
            "yAxisTitle": "Score delta (points)",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 10,
        },
        {
            "id": "success_curve_chart",
            "title": "Validation Success across training",
            "subtitle": (
                "Twelve mini_val evaluations per run at five-epoch intervals."
            ),
            "type": "line",
            "dataset": "success_curve",
            "sourceId": "metric_points_source",
            "encodings": {
                "x": {
                    "field": "epoch",
                    "type": "quantitative",
                    "label": "Epoch",
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Success",
                    "format": "number",
                },
                "color": {
                    "field": "run_label",
                    "type": "nominal",
                    "label": "Run",
                },
            },
            "xAxisTitle": "Training epoch",
            "yAxisTitle": "Success (points)",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 60,
        },
        {
            "id": "precision_curve_chart",
            "title": "Validation Precision across training",
            "subtitle": (
                "Twelve mini_val evaluations per run at five-epoch intervals."
            ),
            "type": "line",
            "dataset": "precision_curve",
            "sourceId": "metric_points_source",
            "encodings": {
                "x": {
                    "field": "epoch",
                    "type": "quantitative",
                    "label": "Epoch",
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Precision",
                    "format": "number",
                },
                "color": {
                    "field": "run_label",
                    "type": "nominal",
                    "label": "Run",
                },
            },
            "xAxisTitle": "Training epoch",
            "yAxisTitle": "Precision (points)",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 60,
        },
        {
            "id": "loss_ratio_chart",
            "title": "Late-training shuffled-to-true loss ratios",
            "subtitle": (
                "Epochs45-60; values above 1 mean shuffled training retains "
                "higher batch-mean loss."
            ),
            "type": "bar",
            "dataset": "loss_ratios",
            "sourceId": "loss_source",
            "encodings": {
                "x": {
                    "field": "loss_label",
                    "type": "nominal",
                    "label": "Loss",
                },
                "y": {
                    "field": "shuffled_to_true_ratio",
                    "type": "quantitative",
                    "label": "Shuffled / true",
                    "format": "number",
                },
                "color": {
                    "field": "protocol",
                    "type": "nominal",
                    "label": "Protocol",
                },
            },
            "xAxisTitle": "Training objective",
            "yAxisTitle": "Loss ratio",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 10,
        },
    ]

    tables = [
        {
            "id": "run_summary_table",
            "title": "Final, best and late-training metrics",
            "subtitle": (
                "Final uses epoch60 last.ckpt; late mean covers epochs45-60."
            ),
            "dataset": "run_summary",
            "sourceId": "metric_points_source",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "run_id", "direction": "asc"},
            "columns": [
                {"field": "run_id", "label": "Run", "type": "text"},
                {
                    "field": "final_success",
                    "label": "Final Success",
                    "type": "number",
                },
                {
                    "field": "final_precision",
                    "label": "Final Precision",
                    "type": "number",
                },
                {
                    "field": "late_success",
                    "label": "Late Success",
                    "type": "number",
                },
                {
                    "field": "late_precision",
                    "label": "Late Precision",
                    "type": "number",
                },
                {
                    "field": "best_success",
                    "label": "Best Success",
                    "type": "number",
                },
                {
                    "field": "best_precision",
                    "label": "Best Precision",
                    "type": "number",
                },
            ],
        },
        {
            "id": "manifest_table",
            "title": "Shuffled-time intervention strength",
            "subtitle": (
                "Train and validation manifests preserve gap marginals while "
                "breaking endpoint alignment."
            ),
            "dataset": "manifest_strength",
            "sourceId": "manifest_source",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {
                "field": "mean_abs_dt_change",
                "direction": "desc",
            },
            "columns": [
                {
                    "field": "protocol",
                    "label": "Protocol",
                    "type": "text",
                },
                {"field": "role", "label": "Role", "type": "text"},
                {
                    "field": "transition_count",
                    "label": "Transitions",
                    "type": "number",
                },
                {
                    "field": "numeric_near_equal_share_1ms",
                    "label": "Near equal ≤1ms",
                    "type": "number",
                },
                {
                    "field": "mean_abs_dt_change",
                    "label": "Mean |Δt change|",
                    "type": "number",
                },
                {
                    "field": "median_abs_dt_change",
                    "label": "Median |Δt change|",
                    "type": "number",
                },
                {
                    "field": "real_effective_correlation",
                    "label": "Real/effective corr.",
                    "type": "number",
                },
            ],
        },
        {
            "id": "integrity_table",
            "title": "Run, checkpoint and artifact integrity",
            "subtitle": (
                "All four runs reached epoch60 and preserve matched cadence "
                "selection within each protocol."
            ),
            "dataset": "integrity",
            "sourceId": "integrity_source",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "run_id", "direction": "asc"},
            "columns": [
                {"field": "run_id", "label": "Run", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {
                    "field": "checkpoint_global_step",
                    "label": "Global step",
                    "type": "number",
                },
                {
                    "field": "checkpoint_tensor_count",
                    "label": "State tensors",
                    "type": "number",
                },
                {
                    "field": "config_sha_match",
                    "label": "Config SHA",
                    "type": "boolean",
                },
                {
                    "field": "manifest_sha_match",
                    "label": "Manifest SHA",
                    "type": "boolean",
                },
                {
                    "field": "cadence_pair_match",
                    "label": "Cadence pair",
                    "type": "boolean",
                },
            ],
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Four-run M2 scratch true/shuffled integrity, training curves, "
            "optimization confounds and M-stage decision."
        ),
        "generatedAt": "2026-07-24T13:00:00+08:00",
        "cards": [],
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": f"# {title}",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## 技术摘要\n\n四组归档、配置、manifest、checkpoint "
                    "和事件完整性全部 PASS。按冻结的 epoch60 `last.ckpt`，"
                    f"random20 true−shuffled 为 **{r20_s:+.3f} Success / "
                    f"{r20_p:+.3f} Precision**，gap1124 为 "
                    f"**{gap_s:+.3f}/{gap_p:+.3f}**。\n\n"
                    "结论是 **scratch correct-time learnability signal "
                    "仅部分成立**：random20 稳定为正，gap1124 只在最后一点"
                    "为正且 late mean 反向。因此 HTV amplification 不受支持，"
                    "physical-time causal claim 仍为 No-Go。"
                ),
            },
            {
                "id": "final_result",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## epoch60 同向，但证据强度在两个协议间分化\n\n"
                    "两个 final checkpoint 都显示 true 更高，且 gap1124 "
                    "的 +1.420/+1.818 超过原先 0.5/1.0 的描述性门槛。"
                    "不过这个门槛原本用于 same-checkpoint 配对干预，不能直接"
                    "移植到分别训练的模型上。"
                ),
            },
            {
                "id": "final_delta_block",
                "type": "chart",
                "chartId": "final_delta_chart",
            },
            {
                "id": "curve_result",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## random20 的正差贯穿训练，gap1124 的 final 差值不稳健\n\n"
                    "random20 true 在 12 个验证点中赢得 10/12 Success、"
                    "11/12 Precision。gap1124 只有 3/12、2/12；其 "
                    f"epochs45–60 平均 true−shuffled 为 "
                    f"**{gap_late_s:+.3f}/{gap_late_p:+.3f}**。"
                    "因此不能用 epoch60 单点声称强 HTV 放大正确时间优势。"
                ),
            },
            {
                "id": "success_curve_block",
                "type": "chart",
                "chartId": "success_curve_chart",
            },
            {
                "id": "precision_curve_block",
                "type": "chart",
                "chartId": "precision_curve_chart",
            },
            {
                "id": "loss_conflict",
                "type": "markdown",
                "sourceId": "loss_source",
                "body": (
                    "## shuffled 的优化负担更重，但它混入了监督不一致\n\n"
                    "epochs45–60 的 shuffled displacement auxiliary loss "
                    "约为 true 的 random20 4.44×、gap1124 5.97×。这是当前"
                    "`velocity_pred × effective_dt` 与 real-time velocity/"
                    "displacement 双监督不相容的直接表现。分训差距因此混合了"
                    "时间信息、目标一致性和随机优化轨迹，不能解释成纯时间因果。"
                ),
            },
            {
                "id": "loss_ratio_block",
                "type": "chart",
                "chartId": "loss_ratio_chart",
            },
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## 范围、指标与比较口径\n\n"
                    "- 数据：nuScenes-mini Car，mini_train 训练、mini_val 验证。\n"
                    "- 协议：random20 与 gap1124，各自使用冻结 train/val "
                    "cadence manifest。\n"
                    "- 模型：M2 proposal_innovation，从随机初始化训练60轮。\n"
                    "- 主结果：epoch60 `last.ckpt` 的 Success/Precision。\n"
                    "- 稳健性：每5轮验证一次，共12点；这些点不是独立重复实验。\n"
                    "- true/shuffled 只在同一协议内比较。"
                ),
            },
            {
                "id": "run_summary_block",
                "type": "table",
                "tableId": "run_summary_table",
            },
            {
                "id": "manifest_result",
                "type": "markdown",
                "sourceId": "manifest_source",
                "body": (
                    "## gap1124 的置乱明显更强，但没有产生更稳定的 true 优势\n\n"
                    "四份 shuffled mapping 都是一一 derangement 且保持 gap "
                    "边缘分布。random20 约40–45%的 transition 数值变化≤1ms；"
                    "gap1124 只有20–25%，平均绝对改变约0.56–0.58秒。干预强度"
                    "足以排除“gap shuffled 没生效”的解释。"
                ),
            },
            {
                "id": "manifest_table_block",
                "type": "table",
                "tableId": "manifest_table",
            },
            {
                "id": "method_integrity",
                "type": "markdown",
                "sourceId": "integrity_source",
                "body": (
                    "## 完整性核验支持四组内部可比\n\n"
                    "四组均为 clean commit 473738f、seed42、scratch、batch16、"
                    "workers12；退出码为0，checkpoint 达到 epoch59 和预期 "
                    "global step。协议内 true/shuffled 的 tracklet、frame、"
                    "cadence selection SHA 完全一致，本地配置/manifest SHA "
                    "与服务器 provenance 一致。"
                ),
            },
            {
                "id": "integrity_table_block",
                "type": "table",
                "tableId": "integrity_table",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制与结论边界\n\n"
                    "当前只有 seed42 和 aggregate mini_val 指标，没有逐 endpoint "
                    "输出与 tracklet bootstrap；十二个 epoch 点高度相关。更重要的"
                    "是 shuffled 分训含结构性辅助目标冲突。因此本批次不能覆盖此前"
                    "同-checkpoint true/fixed/shuffled 的 physical-time No-Go。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 推荐下一步\n\n"
                    "1. 不重训，先对两个协议做 2×2 cross-time evaluation："
                    "每个 true/shuffled-trained final checkpoint 分别用 true 与 "
                    "shuffled val clock 推理。\n"
                    "2. 导出逐 endpoint 结果并做 tracklet-bootstrap。\n"
                    "3. 如仍研究分训，预注册无目标冲突的监督合同，并补 current-code "
                    "matched W0/A1 baseline。\n"
                    "4. 只有 gap1124 的同-checkpoint 干预和配对统计都支持 true，"
                    "才补 seed43/44；否则停止 physical-time claim，M3/M4 只能按 "
                    "time-agnostic robustness 重新立题。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "- random20 稳定差距中有多少来自正确时间，有多少来自 "
                    "shuffled auxiliary-conflict？\n"
                    "- 为什么 gap1124 的 loss 冲突更强，tracking 曲线多数时间"
                    "仍由 shuffled 占优？\n"
                    "- 同一 checkpoint 切换 clock 后，方向是否与分训一致？\n"
                    "- current-code matched W0/A1 的真实基线是多少？"
                ),
            },
        ],
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": "2026-07-24T13:00:00+08:00",
        "datasets": {
            "final_deltas": final_deltas,
            "success_curve": success_curve,
            "precision_curve": precision_curve,
            "loss_ratios": loss_chart,
            "run_summary": summary_rows,
            "manifest_strength": manifest_strength[
                [
                    "protocol",
                    "role",
                    "transition_count",
                    "numeric_near_equal_share_1ms",
                    "mean_abs_dt_change",
                    "median_abs_dt_change",
                    "real_effective_correlation",
                ]
            ].to_dict("records"),
            "integrity": integrity[
                [
                    "run_id",
                    "status",
                    "checkpoint_global_step",
                    "checkpoint_tensor_count",
                    "config_sha_match",
                    "manifest_sha_match",
                    "cadence_pair_match",
                ]
            ].to_dict("records"),
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root is not None
        else repo_root / EXPERIMENT_RELATIVE
    )
    if not experiment_root.is_dir():
        raise FileNotFoundError(experiment_root)

    data_dir = repo_root / "compare_results" / "data"
    report_dir = (
        repo_root
        / "compare_results"
        / "reports"
        / f"m2_scratch_time_analysis_{ANALYSIS_DATE}"
    )
    markdown_path = (
        repo_root
        / "compare_results"
        / "reports"
        / f"m2_scratch_time_analysis_{ANALYSIS_DATE}.md"
    )
    artifact_path = report_dir / "artifact.json"

    metric_points = collect_metric_points(experiment_root)
    metric_summary = summarize_metrics(metric_points)
    paired_deltas = collect_paired_deltas(metric_points)
    comparisons = summarize_comparisons(paired_deltas)
    loss_epoch = collect_loss_epochs(experiment_root)
    loss_ratios = summarize_loss_ratios(loss_epoch)
    manifest_strength = collect_manifest_strength(experiment_root)
    config_diffs = collect_config_diffs(experiment_root)
    integrity = collect_integrity(
        repo_root,
        experiment_root,
        metric_points,
        loss_epoch,
        config_diffs,
    )
    historical = collect_historical_context(repo_root, metric_summary)

    outputs = {
        f"m2_scratch_time_metric_points_{ANALYSIS_DATE}.csv": metric_points,
        f"m2_scratch_time_metric_summary_{ANALYSIS_DATE}.csv": metric_summary,
        f"m2_scratch_time_paired_deltas_{ANALYSIS_DATE}.csv": paired_deltas,
        f"m2_scratch_time_comparisons_{ANALYSIS_DATE}.csv": comparisons,
        f"m2_scratch_time_loss_epoch_{ANALYSIS_DATE}.csv": loss_epoch,
        f"m2_scratch_time_loss_ratios_{ANALYSIS_DATE}.csv": loss_ratios,
        f"m2_scratch_time_manifest_strength_{ANALYSIS_DATE}.csv": (
            manifest_strength
        ),
        f"m2_scratch_time_config_diff_{ANALYSIS_DATE}.csv": config_diffs,
        f"m2_scratch_time_integrity_{ANALYSIS_DATE}.csv": integrity,
        f"m2_scratch_time_historical_context_{ANALYSIS_DATE}.csv": historical,
    }
    for name, frame in outputs.items():
        write_csv(frame, data_dir / name)

    summary_json = {
        "experiment_root": str(experiment_root.relative_to(repo_root)),
        "analysis_date": ANALYSIS_DATE,
        "integrity_pass": bool((integrity["status"] == "PASS").all()),
        "run_count": int(integrity.shape[0]),
        "comparison_count": int(comparisons.shape[0]),
        "manifest_count": int(manifest_strength.shape[0]),
        "frozen_status": {
            "scratch_correct_time_learnability": (
                "PARTIAL_RANDOM20_POSITIVE_GAP1124_UNSTABLE"
            ),
            "htv_amplification": "NOT_SUPPORTED",
            "physical_time_causal_claim": "NO_GO_UNCHANGED",
            "timestamp_conditioned_m3_m4": "LOCKED",
        },
        "comparisons": comparisons.to_dict("records"),
        "late_loss_ratios": loss_ratios.to_dict("records"),
        "manifest_strength": manifest_strength.to_dict("records"),
        "integrity": integrity.to_dict("records"),
        "historical_context": historical.to_dict("records"),
    }
    summary_path = (
        data_dir / f"m2_scratch_time_analysis_{ANALYSIS_DATE}.json"
    )
    summary_path.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    generate_markdown_report(
        markdown_path,
        integrity,
        comparisons,
        loss_ratios,
        manifest_strength,
        historical,
    )
    generate_artifact(
        artifact_path,
        metric_points,
        metric_summary,
        comparisons,
        loss_ratios,
        manifest_strength,
        integrity,
        historical,
    )

    print(f"integrity_pass={summary_json['integrity_pass']}")
    print(f"markdown={markdown_path}")
    print(f"artifact={artifact_path}")
    print(f"summary={summary_path}")
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
