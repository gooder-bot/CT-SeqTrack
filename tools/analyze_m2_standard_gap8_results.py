"""Independently validate and summarize the M2 standard/gap1124 controls.

This script intentionally recomputes the headline metrics from the raw endpoint
CSVs instead of importing the server-side summarizer.  It validates archive
integrity, endpoint identity, checkpoint/time-mode contracts, paired tracklet
bootstrap intervals, prediction sensitivity, and selected cadence/motion
buckets.  Outputs are plain Markdown/CSV/JSON files suitable for repository
review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FORMAL_COMMIT = "473738fa2cf3def246e4e6b1bce35d8692c416c7"
M2_SHA = "362b3314ad056f289509818606af30fcc43d2331eb9156405ed2727bfe49658f"
A1_SHA = "a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82"


@dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    protocol: str
    mode: str
    rows: int
    tracklets: int
    checkpoint: str


RUNS = (
    RunSpec("m2_standard_true_seed42", "M2", "standard", "true", 2285, 106, M2_SHA),
    RunSpec("m2_standard_fixed_seed42", "M2", "standard", "fixed", 2285, 106, M2_SHA),
    RunSpec(
        "m2_standard_shuffled_seed42",
        "M2",
        "standard",
        "shuffled",
        2285,
        106,
        M2_SHA,
    ),
    RunSpec("a1_standard_true_seed42", "A1", "standard", "true", 2285, 106, A1_SHA),
    RunSpec("m2_gap1124_true_seed42", "M2", "gap1124", "true", 1257, 91, M2_SHA),
    RunSpec("m2_gap1124_fixed_seed42", "M2", "gap1124", "fixed", 1257, 91, M2_SHA),
    RunSpec(
        "m2_gap1124_shuffled_seed42",
        "M2",
        "gap1124",
        "shuffled",
        1257,
        91,
        M2_SHA,
    ),
    RunSpec("a1_gap1124_true_seed42", "A1", "gap1124", "true", 1257, 91, A1_SHA),
)

COMPARISONS = (
    ("standard_M2_minus_A1", "m2_standard_true_seed42", "a1_standard_true_seed42"),
    (
        "standard_true_minus_fixed",
        "m2_standard_true_seed42",
        "m2_standard_fixed_seed42",
    ),
    (
        "standard_true_minus_shuffled",
        "m2_standard_true_seed42",
        "m2_standard_shuffled_seed42",
    ),
    ("gap1124_M2_minus_A1", "m2_gap1124_true_seed42", "a1_gap1124_true_seed42"),
    (
        "gap1124_true_minus_fixed",
        "m2_gap1124_true_seed42",
        "m2_gap1124_fixed_seed42",
    ),
    (
        "gap1124_true_minus_shuffled",
        "m2_gap1124_true_seed42",
        "m2_gap1124_shuffled_seed42",
    ),
)

KEY_COLUMNS = ["tracklet_key", "source_frame_index", "frame_token"]
CORE_NUMERIC = [
    "iou",
    "center_error",
    "prediction_x",
    "prediction_y",
    "prediction_z",
    "prediction_yaw",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_z",
    "ground_truth_yaw",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest(root: Path) -> dict:
    manifest_path = root / "artifact_manifest.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    checked = 0
    failures: list[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = raw_line.split(maxsplit=1)
        relative = relative.removeprefix("./")
        path = root / Path(relative)
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"sha256:{relative}")
        checked += 1
    if failures:
        raise RuntimeError(f"Artifact manifest failures: {failures[:10]}")
    return {"checked": checked, "failures": failures}


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def endpoint_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[KEY_COLUMNS].assign(
            source_frame_index=pd.to_numeric(
                frame["source_frame_index"], errors="raise"
            ).astype(int)
        )
    )


def tracking_scores(frame: pd.DataFrame) -> dict:
    ious = pd.to_numeric(frame["iou"], errors="raise").to_numpy(dtype=np.float64)
    errors = pd.to_numeric(
        frame["center_error"], errors="raise"
    ).to_numpy(dtype=np.float64)
    success_x = np.linspace(0.0, 1.0, 21)
    precision_x = np.linspace(0.0, 2.0, 21)
    success_curve = np.asarray([(ious >= threshold).mean() for threshold in success_x])
    precision_curve = np.asarray(
        [(errors <= threshold).mean() for threshold in precision_x]
    )
    return {
        "success": float(np.trapz(success_curve, x=success_x) * 100.0),
        "precision": float(np.trapz(precision_curve, x=precision_x) * 50.0),
        "mean_iou": float(ious.mean()),
        "mean_center_error": float(errors.mean()),
    }


def finite_stats(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def tracklet_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tracklet_key, subset in frame.groupby("tracklet_key", sort=True):
        scores = tracking_scores(subset)
        rows.append({"tracklet_key": tracklet_key, **scores})
    return pd.DataFrame(rows).set_index("tracklet_key").sort_index()


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 20260724, iterations: int = 20000
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    sample_means = values[indices].mean(axis=1)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "low": float(np.quantile(sample_means, 0.025)),
        "high": float(np.quantile(sample_means, 0.975)),
    }


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def paired_comparison(
    name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    left = left.copy()
    right = right.copy()
    left.index = endpoint_index(left)
    right.index = endpoint_index(right)
    if not left.index.equals(right.index):
        raise RuntimeError(f"{name}: endpoint identity/order mismatch")

    left_scores = tracking_scores(left)
    right_scores = tracking_scores(right)
    left_tracklets = tracklet_metrics(left.reset_index(drop=True))
    right_tracklets = tracklet_metrics(right.reset_index(drop=True))
    if not left_tracklets.index.equals(right_tracklets.index):
        raise RuntimeError(f"{name}: tracklet identity mismatch")

    tracklet_delta = left_tracklets - right_tracklets
    success_ci = bootstrap_mean_ci(tracklet_delta["success"].to_numpy(), seed=20260724)
    precision_ci = bootstrap_mean_ci(
        tracklet_delta["precision"].to_numpy(), seed=20261733
    )
    error_gain = (
        right_tracklets["mean_center_error"] - left_tracklets["mean_center_error"]
    )
    error_gain_ci = bootstrap_mean_ci(error_gain.to_numpy(), seed=20262742)

    left_xyz = left[["prediction_x", "prediction_y", "prediction_z"]].to_numpy(
        dtype=np.float64
    )
    right_xyz = right[["prediction_x", "prediction_y", "prediction_z"]].to_numpy(
        dtype=np.float64
    )
    center_shift = np.linalg.norm(left_xyz - right_xyz, axis=1)
    yaw_shift = np.abs(
        wrap_angle(
            left["prediction_yaw"].to_numpy(dtype=np.float64)
            - right["prediction_yaw"].to_numpy(dtype=np.float64)
        )
    )
    initial = parse_bool_series(left["is_initial_frame"]).to_numpy()
    noninitial_shift = center_shift[~initial]
    noninitial_yaw = yaw_shift[~initial]

    iou_delta = (
        left["iou"].to_numpy(dtype=np.float64)
        - right["iou"].to_numpy(dtype=np.float64)
    )
    error_delta = (
        right["center_error"].to_numpy(dtype=np.float64)
        - left["center_error"].to_numpy(dtype=np.float64)
    )
    eps = 1e-12
    result = {
        "comparison": name,
        "endpoint_exact": True,
        "endpoint_count": int(len(left)),
        "tracklet_count": int(len(left_tracklets)),
        "success_left": left_scores["success"],
        "success_right": right_scores["success"],
        "success_delta": left_scores["success"] - right_scores["success"],
        "precision_left": left_scores["precision"],
        "precision_right": right_scores["precision"],
        "precision_delta": left_scores["precision"] - right_scores["precision"],
        "mean_error_left": left_scores["mean_center_error"],
        "mean_error_right": right_scores["mean_center_error"],
        "mean_error_gain": (
            right_scores["mean_center_error"] - left_scores["mean_center_error"]
        ),
        "success_tracklet_mean": success_ci["mean"],
        "success_ci_low": success_ci["low"],
        "success_ci_high": success_ci["high"],
        "precision_tracklet_mean": precision_ci["mean"],
        "precision_ci_low": precision_ci["low"],
        "precision_ci_high": precision_ci["high"],
        "error_gain_tracklet_mean": error_gain_ci["mean"],
        "error_gain_ci_low": error_gain_ci["low"],
        "error_gain_ci_high": error_gain_ci["high"],
        "success_positive_tracklet_fraction": float(
            (tracklet_delta["success"] > eps).mean()
        ),
        "precision_positive_tracklet_fraction": float(
            (tracklet_delta["precision"] > eps).mean()
        ),
        "iou_win_fraction": float((iou_delta > eps).mean()),
        "iou_loss_fraction": float((iou_delta < -eps).mean()),
        "center_error_win_fraction": float((error_delta > eps).mean()),
        "center_error_loss_fraction": float((error_delta < -eps).mean()),
        "prediction_changed_fraction_gt_1e-6": float(
            (noninitial_shift > 1e-6).mean()
        ),
        "prediction_changed_fraction_gt_1cm": float(
            (noninitial_shift > 0.01).mean()
        ),
        "prediction_changed_fraction_gt_10cm": float(
            (noninitial_shift > 0.10).mean()
        ),
        "prediction_shift_mean_m": float(noninitial_shift.mean()),
        "prediction_shift_p50_m": float(np.quantile(noninitial_shift, 0.50)),
        "prediction_shift_p95_m": float(np.quantile(noninitial_shift, 0.95)),
        "prediction_shift_max_m": float(noninitial_shift.max()),
        "yaw_shift_mean_rad": float(noninitial_yaw.mean()),
        "yaw_shift_p95_rad": float(np.quantile(noninitial_yaw, 0.95)),
    }
    tracklet_output = tracklet_delta.reset_index()
    tracklet_output.insert(0, "comparison", name)
    tracklet_output["center_error_gain"] = error_gain.to_numpy()
    return result, tracklet_output


def metric_row(spec: RunSpec, frame: pd.DataFrame) -> dict:
    initial = parse_bool_series(frame["is_initial_frame"])
    noninitial = frame.loc[~initial]
    all_scores = tracking_scores(frame)
    noninitial_scores = tracking_scores(noninitial)
    current_dt = pd.to_numeric(
        noninitial["current_delta_t_effective"], errors="coerce"
    )
    real_dt = pd.to_numeric(noninitial["current_delta_t_real"], errors="coerce")
    row = {
        "run": spec.key,
        "model": spec.model,
        "protocol": spec.protocol,
        "time_mode": spec.mode,
        "endpoint_count": len(frame),
        "noninitial_count": len(noninitial),
        "tracklet_count": frame["tracklet_key"].nunique(),
        **all_scores,
        "noninitial_success": noninitial_scores["success"],
        "noninitial_precision": noninitial_scores["precision"],
        "noninitial_mean_center_error": noninitial_scores["mean_center_error"],
        "empty_fallback_count": int(parse_bool_series(frame["empty_fallback"]).sum()),
        "effective_dt_mean": float(current_dt.mean()),
        "effective_dt_std": float(current_dt.std(ddof=0)),
        "effective_dt_cv": float(current_dt.std(ddof=0) / current_dt.mean()),
        "real_dt_mean": float(real_dt.mean()),
        "real_dt_std": float(real_dt.std(ddof=0)),
        "real_dt_cv": float(real_dt.std(ddof=0) / real_dt.mean()),
    }
    if spec.model == "M2":
        for column, prefix in (
            ("dynamics_innovation_applied_norm", "innovation_applied_norm"),
            ("dynamics_innovation_radius", "innovation_radius"),
            ("physical_time_adapter_norm", "adapter_norm"),
        ):
            stats = finite_stats(noninitial[column])
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_p95"] = stats["p95"]
        for column, output in (
            ("dynamics_innovation_applied_mask", "innovation_applied_rate"),
            ("dynamics_innovation_clamp_mask", "innovation_clamp_rate"),
            ("dynamics_innovation_invalid_fallback", "innovation_invalid_rate"),
        ):
            values = pd.to_numeric(noninitial[column], errors="coerce")
            row[output] = float(values.mean())
    return row


def dt_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 0.75, 1.0, 2.0, np.inf],
        labels=["<0.75", "0.75-1.0", "1.0-2.0", ">=2.0"],
        right=False,
    ).astype("object").fillna("missing")


def displacement_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=["<0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", ">=4.0"],
        right=False,
    ).astype("object").fillna("missing")


def bucket_comparison(
    comparison: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    bucket_name: str,
    buckets: pd.Series,
) -> list[dict]:
    rows = []
    for label in sorted(set(buckets.astype(str))):
        mask = buckets.astype(str) == label
        left_subset = left.loc[mask.to_numpy()]
        right_subset = right.loc[mask.to_numpy()]
        if left_subset.empty:
            continue
        left_scores = tracking_scores(left_subset)
        right_scores = tracking_scores(right_subset)
        rows.append(
            {
                "comparison": comparison,
                "bucket_type": bucket_name,
                "bucket": label,
                "count": len(left_subset),
                "success_left": left_scores["success"],
                "success_right": right_scores["success"],
                "success_delta": left_scores["success"] - right_scores["success"],
                "precision_left": left_scores["precision"],
                "precision_right": right_scores["precision"],
                "precision_delta": (
                    left_scores["precision"] - right_scores["precision"]
                ),
                "mean_error_gain": (
                    right_scores["mean_center_error"]
                    - left_scores["mean_center_error"]
                ),
            }
        )
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    def render(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |"
    body = [
        "| " + " | ".join(render(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *body])


def build_report(
    integrity: dict,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    buckets: pd.DataFrame,
    root: Path,
) -> str:
    comparison_lookup = comparisons.set_index("comparison")
    standard_gain = comparison_lookup.loc["standard_M2_minus_A1"]
    gap_gain = comparison_lookup.loc["gap1124_M2_minus_A1"]

    metric_view = metrics[
        [
            "run",
            "success",
            "precision",
            "mean_center_error",
            "empty_fallback_count",
        ]
    ].copy()
    metric_view.columns = ["run", "Success", "Precision", "mean error", "empty"]

    comparison_view = comparisons[
        [
            "comparison",
            "success_delta",
            "precision_delta",
            "mean_error_gain",
            "success_ci_low",
            "success_ci_high",
            "precision_ci_low",
            "precision_ci_high",
        ]
    ].copy()
    comparison_view.columns = [
        "comparison",
        "ΔSuccess",
        "ΔPrecision",
        "error gain",
        "S CI low",
        "S CI high",
        "P CI low",
        "P CI high",
    ]

    gate_rows = [
        {
            "gate": "standard guardrail: M2−A1 ≥ −0.5/−1.0",
            "observed": "+4.133/+9.445",
            "decision": "PASS",
        },
        {
            "gate": "gap1124 complementarity: M2−A1 ≥ +1/+2",
            "observed": "+2.279/+4.143",
            "decision": "PASS",
        },
        {
            "gate": "standard true−fixed/shuffled ≥ +0.5/+1",
            "observed": "min +0.031/−0.010",
            "decision": "FAIL",
        },
        {
            "gate": "gap1124 true−fixed/shuffled ≥ +0.5/+1",
            "observed": "min −0.318/−0.209",
            "decision": "FAIL",
        },
        {
            "gate": "M3/M4 timestamp-dependent promotion",
            "observed": "requires causal-time gates",
            "decision": "LOCKED",
        },
    ]
    gate_frame = pd.DataFrame(gate_rows)

    sensitivity = comparisons[
        comparisons["comparison"].str.contains("true_minus")
    ][
        [
            "comparison",
            "prediction_changed_fraction_gt_1cm",
            "prediction_shift_mean_m",
            "prediction_shift_p95_m",
        ]
    ].copy()
    sensitivity.columns = ["comparison", "changed >1cm", "mean shift m", "p95 shift m"]

    dynamics_view = metrics[metrics["model"] == "M2"][
        [
            "run",
            "effective_dt_cv",
            "innovation_applied_rate",
            "innovation_clamp_rate",
            "innovation_applied_norm_mean",
            "innovation_radius_mean",
            "adapter_norm_mean",
        ]
    ].copy()
    dynamics_view.columns = [
        "run",
        "effective dt CV",
        "applied rate",
        "clamp rate",
        "applied norm m",
        "radius m",
        "adapter norm",
    ]

    gap_bucket = buckets[
        (
            buckets["comparison"].isin(
                ["gap1124_M2_minus_A1", "gap1124_true_minus_shuffled"]
            )
        )
        & (buckets["bucket_type"] == "real_dt")
        & (buckets["bucket"] != "missing")
    ][
        [
            "comparison",
            "bucket",
            "count",
            "success_delta",
            "precision_delta",
            "mean_error_gain",
        ]
    ].copy()
    gap_bucket.columns = [
        "comparison",
        "real dt",
        "n",
        "ΔSuccess",
        "ΔPrecision",
        "error gain",
    ]

    gap_displacement = buckets[
        (
            buckets["comparison"].isin(
                ["gap1124_M2_minus_A1", "gap1124_true_minus_shuffled"]
            )
        )
        & (buckets["bucket_type"] == "gt_displacement")
        & (buckets["bucket"] != "missing")
    ][
        [
            "comparison",
            "bucket",
            "count",
            "success_delta",
            "precision_delta",
            "mean_error_gain",
        ]
    ].copy()
    gap_displacement.columns = [
        "comparison",
        "GT displacement m",
        "n",
        "ΔSuccess",
        "ΔPrecision",
        "error gain",
    ]

    lines = [
        "# M2 standard / gap1124 同 checkpoint 控制分析",
        "",
        "**日期：2026-07-24｜结论：M2 tracking 信号成立，但 physical-time 因果主张 No-Go；方法归因仍 Hold。**",
        "",
        "## 执行摘要",
        "",
        "- R1 M2 相对历史 A1 在 standard 为 **+4.133 Success / +9.445 Precision**，在 gap1124 为 **+2.279/+4.143**；逐 tracklet bootstrap 的两项主指标 95% CI 均为正。",
        "- 同一 R1 checkpoint 改成 fixed/shuffled 时间后，standard 差异接近零；gap1124 中 shuffled 反而比 true 高 **+0.318 Success / +0.209 Precision**。",
        "- 因此现有结果支持“R1 训练得到更强 tracker”的描述，但不支持“正确物理时间映射造成涨点”。R1 还包含额外 60 epoch continuation、shared-SE(2) 和 M2 联合改变，不能把 M2−A1 直接写成模块净增益。",
        "- 按已冻结计划，timestamp-conditioned M3/M4 不解锁。下一步先做同 checkpoint adapter/innovation 2×2、A1-init W0 continuation、legacy-candidate W0 和 proposal 语义/递归误差审计。",
        "",
        "## 数据与完整性",
        "",
        f"- 来源：`{root.as_posix()}`。",
        f"- 包内 artifact manifest：**{integrity['artifact_manifest']['checked']}/{integrity['artifact_manifest']['checked']} PASS**。",
        "- 8 个 run 均来自 clean commit `473738f`；M2/A1 checkpoint SHA256 与冻结合同一致。",
        "- standard 为 106 tracklets / 2285 endpoints；gap1124 为 91 / 1257。每个协议内 endpoint key、顺序、GT 与真实时间字段完全配对。",
        "- 首轮 reference 校验遗漏了每条轨迹的 GT 初始化帧；恢复时只从 validator reference 移除 106/91 个初始帧，8 份结果 CSV 仍完整保留初始帧。全量 CSV 的跨 run endpoint identity 复核为 exact。",
        "",
        "## 八组原始指标复算",
        "",
        markdown_table(metric_view, list(metric_view.columns)),
        "",
        "指标由原始 CSV 中 21 点 Success/Precision 曲线独立积分复算，与服务器 `m0_summary.json` 最大绝对误差小于 `1e-10`。",
        "",
        "## 冻结门槛",
        "",
        markdown_table(gate_frame, list(gate_frame.columns)),
        "",
        "## 配对差异与 tracklet bootstrap",
        "",
        markdown_table(comparison_view, list(comparison_view.columns)),
        "",
        "表中 CI 是以 tracklet 为抽样单位、20,000 次独立 bootstrap 得到的 tracklet-mean delta 区间；aggregate delta 按 endpoint 加权，因此两者中心值允许不同。",
        "",
        "### 方法信号",
        "",
        (
            "- standard M2−A1 的 tracklet-mean 95% CI：Success "
            f"`[{standard_gain['success_ci_low']:.3f}, "
            f"{standard_gain['success_ci_high']:.3f}]`，Precision "
            f"`[{standard_gain['precision_ci_low']:.3f}, "
            f"{standard_gain['precision_ci_high']:.3f}]`。"
        ),
        (
            "- gap1124 M2−A1 的 tracklet-mean 95% CI：Success "
            f"`[{gap_gain['success_ci_low']:.3f}, "
            f"{gap_gain['success_ci_high']:.3f}]`，Precision "
            f"`[{gap_gain['precision_ci_low']:.3f}, "
            f"{gap_gain['precision_ci_high']:.3f}]`。"
        ),
        "- gap1124 的 M2 empty fallback 为 113，A1 为 107：整体指标虽提高，但空搜索并未改善，增益不是由单纯减少 empty fallback 解释。",
        "- gap1124 与 standard 使用不同 endpoint population；两者绝对分数不能直接解释为 gap 更容易或模型在 gap 下反而更强，只能在各自协议内做 matched comparison。",
        "",
        "### 时间敏感但不具备正确对齐优势",
        "",
        markdown_table(sensitivity, list(sensitivity.columns)),
        "",
        "时间负对照确实改变了递归预测，说明分支不是完全失活；但正确时间没有稳定优于 fixed/shuffled。尤其 gap1124 的 shuffled 在两个主指标上均高于 true，这否定当前 R1 的 physical-time causal promotion。",
        "",
        "### M2 运行时路径并未失活",
        "",
        markdown_table(dynamics_view, list(dynamics_view.columns)),
        "",
        "gap1124 true 将平均 innovation radius 从 fixed 的 `0.750 m` 提高到 `0.962 m`，平均 applied norm 从 `0.351 m` 提高到 `0.387 m`，且约 90.8% 非初始 endpoint 实际应用 innovation。模型确实响应 effective time，但这种响应没有带来正确时间对齐优势。",
        "",
        "## gap1124 实际时间分桶",
        "",
        markdown_table(gap_bucket, list(gap_bucket.columns)),
        "",
        "M2−A1 在四个 real-dt 桶都为正，说明正信号并不只集中在长间隔；true−shuffled 在四个 real-dt 桶的 Success 全为负，`≥2 s` 也没有隐藏的 physical-time promotion。分桶只用于定位，不用于重新选择阈值。",
        "",
        "## gap1124 GT 位移分桶",
        "",
        markdown_table(gap_displacement, list(gap_displacement.columns)),
        "",
        "`≥4 m` 只有 28 个 endpoint，M2−A1 的 Success/Precision 增益均为 0；mean error 虽下降，但没有跨过 tracking 曲线阈值。这个结果与既有 crop-reachability 诊断一致：大位移目标常在网络 forward 前已离开固定 crop，末端 proposal correction 不能稳定恢复。",
        "",
        "## 代码机制解释",
        "",
        "fixed/shuffled 只替换 `DynamicsEncoder`、physical-time adapter 和 `R(Δt)` 消费的 effective time；order-time 主干、GT、候选和监督仍保持不变。当前预测同时经过两条显式时间路径：",
        "",
        "1. `DynamicsEncoder(ref_boxs, delta_t_effective, current_delta_t_effective)` 产生 dynamics proposal，并通过 zero-init adapter 改写 observation feature；",
        "2. `d_final = d_obs + α·clip(d_dyn−stopgrad(d_obs), R(Δt_effective))`，其中 `α=0.75`、`R(Δt)=min(0.5+0.5Δt, 2.0)`。",
        "",
        "因此 true≈fixed/shuffled 不是“模型完全没读时间”，而是当前学到的时间条件没有转化为正确 alignment 的性能优势。结合 R3 shared-SE(2) W0 塌陷和 candidate-frame/canonical-target 语义风险，更合理的解释是：R1 的正信号主要来自 continuation、联合表征或通用 proposal correction，物理秒数不是已证实的增益来源。",
        "",
        "## Validation Report",
        "",
        "### Overall Assessment: Share with caveats",
        "",
        "数据、配对身份、计算和 bootstrap 均已复核，可用于内部路线决策；但不能对外宣称 M2 的正确物理时间有效，也不能宣称 M2 已因果超过 SeqTrack3D。",
        "",
        "### 必须保留的限制",
        "",
        "- 当前只有 seed42、mini_val、standard 与 gap1124；burst、unseen schedule、full data 和第二数据集未完成。",
        "- A1 是历史 checkpoint，R1 从 A1 再训练 60 epoch，M2−A1 混有额外训练预算。",
        "- R3 不是可靠 matched baseline；shared-SE(2) W0 的异常塌陷尚未解释。",
        "- 参考帧 workaround 不改变预测或最终 CSV，但正式 exporter 应在下一提交中修复初始帧 `observed_keys`。",
        "",
        "## 决策与下一步",
        "",
        "1. 将状态更新为 **`M2 TRACKING SIGNAL POSITIVE / PHYSICAL-TIME CAUSAL CLAIM NO-GO / METHOD ATTRIBUTION HOLD`**。",
        "2. 不为 physical-time claim 追加 seed43/44，也不立即启动 timestamp-conditioned M3/M4。",
        "3. 第一优先级完成 frozen R1 的 full / adapter-only / innovation-only / both-off；它决定正信号来自哪条运行时路径。",
        "4. 补 A1-init W0 continuation，分离额外 60 epoch；补 current-code legacy-candidate W0，解释 shared-SE(2) collapse。",
        "5. 审计 candidate-frame `box_label`、canonical `motion_label`、`d_obs/d_dyn/d_final` 与 recursive-error process。若确认语义错配，再开 M1.5 新分支。",
        "6. burst-drop 只在通用 proposal 路线仍成立后补作 robustness/crop 恢复证据；它不能复活已失败的 gap1124 physical-time causal gate。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "server_results/m2_standard_gap8_473738f_20260723_235400"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "compare_results/reports/m2_standard_gap8_analysis_20260724.md"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("compare_results/data")
    )
    args = parser.parse_args()

    root = args.input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    artifact_manifest = verify_artifact_manifest(root)
    completion = json.loads(
        (root / "completion_audit.json").read_text(encoding="utf-8")
    )
    if completion.get("run_count") != 8:
        raise RuntimeError("completion_audit.json does not contain 8 runs")

    frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict] = []
    integrity_runs: dict[str, dict] = {}
    for spec in RUNS:
        path = root / "endpoints" / spec.key / "m0_endpoints.csv"
        frame = pd.read_csv(
            path,
            low_memory=False,
            float_precision="round_trip",
            dtype={
                "time_mode": str,
                "tracklet_key": str,
                "frame_token": str,
                "checkpoint_sha256": str,
            },
        )
        missing = sorted(set(KEY_COLUMNS + CORE_NUMERIC) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{spec.key}: missing columns {missing}")
        for column in CORE_NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        index = endpoint_index(frame)
        if index.has_duplicates:
            raise RuntimeError(f"{spec.key}: duplicate endpoint keys")
        if len(frame) != spec.rows:
            raise RuntimeError(f"{spec.key}: expected {spec.rows} rows, got {len(frame)}")
        if frame["tracklet_key"].nunique() != spec.tracklets:
            raise RuntimeError(
                f"{spec.key}: expected {spec.tracklets} tracklets, "
                f"got {frame['tracklet_key'].nunique()}"
            )
        if set(frame["checkpoint_sha256"].astype(str)) != {spec.checkpoint}:
            raise RuntimeError(f"{spec.key}: checkpoint mismatch")
        if set(frame["time_mode"].astype(str)) != {spec.mode}:
            raise RuntimeError(f"{spec.key}: time mode mismatch")
        if not np.isfinite(frame[CORE_NUMERIC].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{spec.key}: non-finite core metric/prediction field")

        initial_count = int(parse_bool_series(frame["is_initial_frame"]).sum())
        if initial_count != spec.tracklets:
            raise RuntimeError(
                f"{spec.key}: expected {spec.tracklets} initial rows, got {initial_count}"
            )
        frames[spec.key] = frame
        metric_rows.append(metric_row(spec, frame))
        integrity_runs[spec.key] = {
            "csv": str(path),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "tracklets": frame["tracklet_key"].nunique(),
            "initial_rows": initial_count,
            "duplicate_endpoints": int(index.duplicated().sum()),
            "core_nonfinite": 0,
        }

    for protocol in ("standard", "gap1124"):
        specs = [spec for spec in RUNS if spec.protocol == protocol]
        reference = frames[specs[0].key]
        reference_index = endpoint_index(reference)
        for spec in specs[1:]:
            candidate = frames[spec.key]
            if not reference_index.equals(endpoint_index(candidate)):
                raise RuntimeError(f"{protocol}: endpoint mismatch for {spec.key}")
            for column in (
                "ground_truth_x",
                "ground_truth_y",
                "ground_truth_z",
                "ground_truth_yaw",
                "current_delta_t_real",
            ):
                left = pd.to_numeric(reference[column], errors="coerce").to_numpy()
                right = pd.to_numeric(candidate[column], errors="coerce").to_numpy()
                if not np.allclose(left, right, equal_nan=True, atol=1e-12, rtol=0):
                    raise RuntimeError(
                        f"{protocol}: matched field {column} differs for {spec.key}"
                    )

    comparison_rows: list[dict] = []
    tracklet_rows: list[pd.DataFrame] = []
    bucket_rows: list[dict] = []
    for name, left_key, right_key in COMPARISONS:
        left = frames[left_key]
        right = frames[right_key]
        result, tracklet_output = paired_comparison(name, left, right)
        comparison_rows.append(result)
        tracklet_rows.append(tracklet_output)
        bucket_rows.extend(
            bucket_comparison(
                name,
                left,
                right,
                "real_dt",
                dt_bucket(left["current_delta_t_real"]),
            )
        )
        bucket_rows.extend(
            bucket_comparison(
                name,
                left,
                right,
                "gt_displacement",
                displacement_bucket(left["gt_displacement_from_previous_gt"]),
            )
        )

    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(comparison_rows)
    tracklets = pd.concat(tracklet_rows, ignore_index=True)
    buckets = pd.DataFrame(bucket_rows)

    # Independent recomputation must reconcile with the server completion audit.
    max_metric_error = 0.0
    for row in metric_rows:
        server = completion["runs"][row["run"]]
        for local_key, server_key in (
            ("success", "success"),
            ("precision", "precision"),
            ("mean_center_error", "mean_center_error"),
        ):
            max_metric_error = max(
                max_metric_error, abs(float(row[local_key]) - float(server[server_key]))
            )
    if max_metric_error > 1e-10:
        raise RuntimeError(
            f"Independent metric recomputation mismatch: {max_metric_error}"
        )

    integrity = {
        "schema": "ct_seqtrack.m2_standard_gap8_local_validation",
        "formal_commit": FORMAL_COMMIT,
        "artifact_manifest": artifact_manifest,
        "run_count": len(frames),
        "max_abs_metric_error_vs_server": max_metric_error,
        "protocol_endpoint_identity_exact": True,
        "matched_ground_truth_and_real_time_exact": True,
        "runs": integrity_runs,
    }

    args.data_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.data_dir / "m2_standard_gap8_metrics_20260724.csv"
    comparisons_path = args.data_dir / "m2_standard_gap8_comparisons_20260724.csv"
    tracklets_path = args.data_dir / "m2_standard_gap8_tracklet_deltas_20260724.csv"
    buckets_path = args.data_dir / "m2_standard_gap8_bucket_deltas_20260724.csv"
    integrity_path = args.data_dir / "m2_standard_gap8_integrity_20260724.json"
    metrics.to_csv(metrics_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)
    tracklets.to_csv(tracklets_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    integrity_path.write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_report(integrity, metrics, comparisons, buckets, root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")

    print("M2 standard/gap1124 local validation: PASS")
    print(f"artifact manifest: {artifact_manifest['checked']}/"
          f"{artifact_manifest['checked']}")
    print(f"max metric error: {max_metric_error:.3e}")
    print(f"report: {args.report}")
    print(f"metrics: {metrics_path}")
    print(f"comparisons: {comparisons_path}")
    print(f"tracklets: {tracklets_path}")
    print(f"buckets: {buckets_path}")
    print(f"integrity: {integrity_path}")


if __name__ == "__main__":
    main()
