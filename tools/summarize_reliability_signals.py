import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TASK_FEATURES = {
    "trigger": (
        ("prev_log_search_points", "prev_obs_search_point_count", "log1p"),
        ("prev_empty_fallback", "prev_obs_empty_fallback", "bool"),
        ("prev_forward_ran", "prev_obs_forward_ran", "bool"),
        ("prev_log_soft_fg_count", "prev_obs_soft_fg_count", "log1p"),
        ("prev_log_estimated_fg_points", "prev_obs_estimated_fg_points", "log1p"),
        ("prev_mean_fg_score", "prev_obs_mean_fg_score", "identity"),
        ("prev_fg_entropy", "prev_obs_fg_entropy_mean", "identity"),
        ("prev_fg_margin", "prev_obs_fg_margin_mean", "identity"),
        (
            "prev_motion_dynamic_probability",
            "prev_obs_motion_dynamic_probability",
            "identity",
        ),
        ("log_current_delta_t", "current_delta_t", "log1p"),
        ("log_cv_speed", "cv_speed", "log1p"),
        ("log_cv_shift", "cv_shift", "log1p"),
        ("pred_cv_available", "pred_cv_available", "bool"),
    ),
    "current_evidence": (
        ("log_search_points", "obs_search_point_count", "log1p"),
        ("empty_fallback", "obs_empty_fallback", "bool"),
        ("forward_ran", "obs_forward_ran", "bool"),
        ("log_soft_fg_count", "obs_soft_fg_count", "log1p"),
        ("log_estimated_fg_points", "obs_estimated_fg_points", "log1p"),
        ("mean_fg_score", "obs_mean_fg_score", "identity"),
        ("fg_hard_ratio", "obs_fg_hard_ratio", "identity"),
        ("fg_entropy", "obs_fg_entropy_mean", "identity"),
        ("fg_margin", "obs_fg_margin_mean", "identity"),
        (
            "motion_dynamic_probability",
            "obs_motion_dynamic_probability",
            "identity",
        ),
        ("log_current_delta_t", "current_delta_t", "log1p"),
        ("log_cv_speed", "cv_speed", "log1p"),
        ("log_cv_shift", "cv_shift", "log1p"),
    ),
    "selector": (
        ("obs_log_search_points", "obs_search_point_count", "log1p"),
        ("obs_empty_fallback", "obs_empty_fallback", "bool"),
        ("obs_log_soft_fg_count", "obs_soft_fg_count", "log1p"),
        ("obs_mean_fg_score", "obs_mean_fg_score", "identity"),
        ("obs_fg_hard_ratio", "obs_fg_hard_ratio", "identity"),
        ("obs_fg_entropy", "obs_fg_entropy_mean", "identity"),
        ("obs_fg_margin", "obs_fg_margin_mean", "identity"),
        ("traj_log_search_points", "traj_search_point_count", "log1p"),
        ("traj_empty_fallback", "traj_empty_fallback", "bool"),
        ("traj_log_soft_fg_count", "traj_soft_fg_count", "log1p"),
        ("traj_mean_fg_score", "traj_mean_fg_score", "identity"),
        ("traj_fg_hard_ratio", "traj_fg_hard_ratio", "identity"),
        ("traj_fg_entropy", "traj_fg_entropy_mean", "identity"),
        ("traj_fg_margin", "traj_fg_margin_mean", "identity"),
        ("log_anchor_distance", "anchor_center_distance", "log1p"),
        ("log_candidate_distance", "candidate_center_distance", "log1p"),
        ("candidate_yaw_difference", "candidate_yaw_difference", "identity"),
        ("log_cv_speed", "cv_speed", "log1p"),
        ("log_cv_shift", "cv_shift", "log1p"),
        ("log_current_delta_t", "current_delta_t", "log1p"),
    ),
}


TASK_LABELS = {
    "trigger": "current_obs_crop_miss",
    "current_evidence": "current_obs_crop_miss",
    "selector": "selector_label",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def parse_float(value):
    if value is None or str(value).strip() == "":
        return np.nan
    boolean = parse_bool(value)
    if boolean is not None and str(value).strip().lower() in (
        "true",
        "false",
        "yes",
        "no",
    ):
        return float(boolean)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def transform_value(value, transform):
    if transform == "bool":
        boolean = parse_bool(value)
        return float(boolean) if boolean is not None else np.nan
    value = parse_float(value)
    if not np.isfinite(value):
        return np.nan
    if transform == "identity":
        return float(value)
    if transform == "log1p":
        return float(np.log1p(max(value, 0.0)))
    raise ValueError(f"Unsupported transform: {transform}")


def parse_input_spec(value):
    if "=" not in value:
        raise ValueError(f"Expected protocol=csv_path, got: {value}")
    protocol, path = value.split("=", 1)
    protocol = protocol.strip()
    if not protocol:
        raise ValueError(f"Empty protocol in input spec: {value}")
    return protocol, Path(path).resolve()


def read_protocol_rows(path, protocol):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise RuntimeError(f"No rows found for protocol={protocol}: {path}")
    for row in rows:
        row["protocol"] = protocol
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_rows(rows, task):
    label_name = TASK_LABELS[task]
    selected = []
    for row in rows:
        label = parse_bool(row.get(label_name))
        if label is None:
            continue
        if task in ("trigger", "current_evidence"):
            visible = parse_bool(row.get("current_target_visible"))
            if visible is not True:
                continue
        if task == "selector":
            visible = parse_bool(row.get("current_target_visible"))
            dual_reachable = parse_bool(row.get("dual_has_target_point"))
            if visible is not True or dual_reachable is not True:
                continue
        copied = dict(row)
        copied["_label"] = float(label)
        selected.append(copied)
    return selected


def build_feature_matrix(rows, feature_specs):
    matrix = np.empty((len(rows), len(feature_specs)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, (_, source, transform) in enumerate(feature_specs):
            matrix[row_index, feature_index] = transform_value(row.get(source), transform)
    return matrix


def stable_fold(group, folds, seed):
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_preprocessor(matrix):
    medians = np.zeros(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        finite = matrix[:, column][np.isfinite(matrix[:, column])]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    filled = np.where(np.isfinite(matrix), matrix, medians[None, :])
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0)
    scales[scales < 1e-8] = 1.0
    return medians, means, scales


def apply_preprocessor(matrix, medians, means, scales):
    filled = np.where(np.isfinite(matrix), matrix, medians[None, :])
    return (filled - means[None, :]) / scales[None, :]


def fit_logistic_regression(features, labels, l2=1e-3, max_iter=100):
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Logistic regression requires both label classes.")
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1
    )
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    regularizer[0, 0] = 0.0

    for _ in range(max_iter):
        probabilities = sigmoid(design @ weights)
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient += regularizer @ weights
        curvature = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        hessian = (design.T @ (design * curvature[:, None])) / len(labels)
        hessian += regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return weights


def predict_logistic(features, weights):
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1
    )
    return sigmoid(design @ weights)


def average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def roc_auc(labels, scores):
    labels = labels.astype(np.int64)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def average_precision(labels, scores):
    labels = labels.astype(np.int64)
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    precision = cumulative_positive / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positive_count)


def expected_calibration_error(labels, probabilities, bins=10):
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
        )
    return float(error)


def select_threshold(labels, probabilities, target_recall):
    positive_scores = probabilities[labels == 1]
    if positive_scores.size == 0:
        return 1.0
    candidates = np.unique(probabilities)[::-1]
    selected = float(candidates[-1])
    for threshold in candidates:
        predictions = probabilities >= threshold
        recall = float(np.sum(predictions & (labels == 1)) / positive_scores.size)
        if recall >= target_recall:
            selected = float(threshold)
            break
    return selected


def classification_metrics(labels, probabilities, decisions=None):
    labels = labels.astype(np.int64)
    prevalence = float(np.mean(labels))
    metrics = {
        "count": int(len(labels)),
        "positive_count": int(np.sum(labels)),
        "prevalence": prevalence,
        "auroc": roc_auc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "prevalence_brier": float(prevalence * (1.0 - prevalence)),
        "ece_10": expected_calibration_error(labels, probabilities, bins=10),
    }
    if decisions is not None:
        decisions = decisions.astype(bool)
        positives = labels == 1
        negatives = labels == 0
        true_positive = int(np.sum(decisions & positives))
        false_positive = int(np.sum(decisions & negatives))
        metrics.update(
            {
                "activation_rate": float(np.mean(decisions)),
                "operating_recall": (
                    float(true_positive / np.sum(positives)) if np.any(positives) else None
                ),
                "operating_precision": (
                    float(true_positive / np.sum(decisions)) if np.any(decisions) else None
                ),
                "operating_false_positive_rate": (
                    float(false_positive / np.sum(negatives)) if np.any(negatives) else None
                ),
            }
        )
    return metrics


def fit_task(
    protocol_rows,
    task,
    standard_protocol,
    folds,
    seed,
    l2,
    target_recall,
):
    feature_specs = TASK_FEATURES[task]
    selected = {
        protocol: task_rows(rows, task) for protocol, rows in protocol_rows.items()
    }
    standard_rows = selected.get(standard_protocol, [])
    if not standard_rows:
        raise RuntimeError(
            f"No labeled rows for task={task}, standard_protocol={standard_protocol}"
        )

    oof = {
        protocol: {"labels": [], "probabilities": [], "decisions": [], "groups": []}
        for protocol in selected
    }
    fold_reports = []
    for fold in range(folds):
        train_rows = [
            row
            for row in standard_rows
            if stable_fold(row["tracklet_key"], folds, seed) != fold
        ]
        if len({int(row["_label"]) for row in train_rows}) < 2:
            raise RuntimeError(f"Fold {fold} task={task} training data lacks both classes.")

        train_raw = build_feature_matrix(train_rows, feature_specs)
        train_labels = np.asarray([row["_label"] for row in train_rows], dtype=np.float64)
        medians, means, scales = fit_preprocessor(train_raw)
        train_features = apply_preprocessor(train_raw, medians, means, scales)
        weights = fit_logistic_regression(train_features, train_labels, l2=l2)
        train_probability = predict_logistic(train_features, weights)
        threshold = select_threshold(train_labels, train_probability, target_recall)

        fold_report = {
            "fold": fold,
            "train_count": len(train_rows),
            "train_positive_count": int(np.sum(train_labels)),
            "threshold": threshold,
            "test_counts": {},
        }
        for protocol, rows in selected.items():
            test_rows = [
                row
                for row in rows
                if stable_fold(row["tracklet_key"], folds, seed) == fold
            ]
            fold_report["test_counts"][protocol] = len(test_rows)
            if not test_rows:
                continue
            test_raw = build_feature_matrix(test_rows, feature_specs)
            test_features = apply_preprocessor(test_raw, medians, means, scales)
            probability = predict_logistic(test_features, weights)
            decisions = probability >= threshold
            oof[protocol]["labels"].extend(row["_label"] for row in test_rows)
            oof[protocol]["probabilities"].extend(probability.tolist())
            oof[protocol]["decisions"].extend(decisions.tolist())
            oof[protocol]["groups"].extend(row["tracklet_key"] for row in test_rows)
        fold_reports.append(fold_report)

    metrics = {}
    for protocol, values in oof.items():
        labels = np.asarray(values["labels"], dtype=np.float64)
        probability = np.asarray(values["probabilities"], dtype=np.float64)
        decisions = np.asarray(values["decisions"], dtype=bool)
        if labels.size == 0:
            metrics[protocol] = None
            continue
        metrics[protocol] = classification_metrics(labels, probability, decisions)
        metrics[protocol]["tracklet_count"] = len(set(values["groups"]))

    final_raw = build_feature_matrix(standard_rows, feature_specs)
    final_labels = np.asarray([row["_label"] for row in standard_rows], dtype=np.float64)
    medians, means, scales = fit_preprocessor(final_raw)
    final_features = apply_preprocessor(final_raw, medians, means, scales)
    final_weights = fit_logistic_regression(final_features, final_labels, l2=l2)
    final_probability = predict_logistic(final_features, final_weights)
    final_threshold = select_threshold(final_labels, final_probability, target_recall)

    model = {
        "task": task,
        "label": TASK_LABELS[task],
        "training_protocol": standard_protocol,
        "feature_specs": [
            {"name": name, "source": source, "transform": transform}
            for name, source, transform in feature_specs
        ],
        "imputation_medians": medians.tolist(),
        "standardization_means": means.tolist(),
        "standardization_scales": scales.tolist(),
        "intercept": float(final_weights[0]),
        "coefficients": final_weights[1:].tolist(),
        "operating_threshold": float(final_threshold),
        "target_recall": float(target_recall),
        "l2": float(l2),
    }
    return {
        "task": task,
        "feature_count": len(feature_specs),
        "folds": fold_reports,
        "protocol_metrics": metrics,
        "frozen_standard_calibrator": model,
    }


def passive_complementarity(rows):
    visible = [
        row for row in rows if parse_bool(row.get("current_target_visible")) is True
    ]
    if not visible:
        return None
    obs_any = np.asarray(
        [parse_bool(row.get("obs_has_target_point")) is True for row in visible],
        dtype=bool,
    )
    traj_any = np.asarray(
        [parse_bool(row.get("traj_has_target_point")) is True for row in visible],
        dtype=bool,
    )
    obs_recall = np.asarray(
        [parse_float(row.get("obs_target_point_recall")) for row in visible],
        dtype=np.float64,
    )
    traj_recall = np.asarray(
        [parse_float(row.get("traj_target_point_recall")) for row in visible],
        dtype=np.float64,
    )
    dual_recall = np.fmax(obs_recall, traj_recall)
    return {
        "visible_endpoint_count": len(visible),
        "obs_has_target_point_rate": float(np.mean(obs_any)),
        "traj_has_target_point_rate": float(np.mean(traj_any)),
        "dual_union_has_target_point_rate": float(np.mean(obs_any | traj_any)),
        "traj_only_endpoint_count": int(np.sum(traj_any & ~obs_any)),
        "obs_only_endpoint_count": int(np.sum(obs_any & ~traj_any)),
        "both_miss_endpoint_count": int(np.sum(~obs_any & ~traj_any)),
        "obs_target_point_recall_mean": float(np.nanmean(obs_recall)),
        "traj_target_point_recall_mean": float(np.nanmean(traj_recall)),
        "dual_oracle_target_point_recall_mean": float(np.nanmean(dual_recall)),
        "dual_oracle_recall_gain_over_obs": float(
            np.nanmean(dual_recall) - np.nanmean(obs_recall)
        ),
    }


def evaluate_verdict(
    task_results,
    complementarity,
    strong_protocols,
    go_auroc,
    go_auprc_margin,
    raw_cv_union_gain,
):
    trigger_metrics = task_results["trigger"]["protocol_metrics"]
    protocol_checks = {}
    reliability_go = True
    for protocol in strong_protocols:
        metrics = trigger_metrics.get(protocol)
        if metrics is None:
            protocol_checks[protocol] = {
                "pass": False,
                "reason": "missing trigger metrics",
            }
            reliability_go = False
            continue
        auprc_margin = metrics["auprc"] - metrics["prevalence"]
        passed = metrics["auroc"] >= go_auroc and auprc_margin >= go_auprc_margin
        protocol_checks[protocol] = {
            "pass": bool(passed),
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "prevalence": metrics["prevalence"],
            "auprc_margin": float(auprc_margin),
            "required_auroc": go_auroc,
            "required_auprc_margin": go_auprc_margin,
        }
        reliability_go = reliability_go and passed

    raw_cv_checks = {}
    raw_cv_go = True
    for protocol in strong_protocols:
        metrics = complementarity.get(protocol)
        if metrics is None:
            raw_cv_checks[protocol] = {"pass": False, "reason": "missing complementarity"}
            raw_cv_go = False
            continue
        gain = metrics["dual_oracle_recall_gain_over_obs"]
        passed = gain >= raw_cv_union_gain
        raw_cv_checks[protocol] = {
            "pass": bool(passed),
            "dual_oracle_recall_gain_over_obs": gain,
            "required_gain": raw_cv_union_gain,
        }
        raw_cv_go = raw_cv_go and passed

    if not reliability_go:
        decision = "NO_GO_RELIABILITY_PROXY"
        next_step = (
            "Stop learned gate/dual-anchor work. Pivot to benchmark/diagnosis or a simple "
            "pre-registered fallback without claiming reliability-aware prevention."
        )
    elif not raw_cv_go:
        decision = "RELIABILITY_GO_RAW_CV_ANCHOR_NO_GO"
        next_step = (
            "Do not activate the raw recent-history CV anchor. Test one independent "
            "reliability-updated timestamp-aware Kalman/frozen-state anchor as a bounded kill-test."
        )
    else:
        decision = "GO_ACTIVE_RAW_CV_DUAL_ANCHOR"
        next_step = (
            "Freeze the standard-trained calibrator and implement active dual-anchor inference "
            "with the same A1 checkpoint and cost-matched controls."
        )
    return {
        "decision": decision,
        "reliability_go": bool(reliability_go),
        "raw_cv_anchor_go": bool(raw_cv_go),
        "reliability_protocol_checks": protocol_checks,
        "raw_cv_complementarity_checks": raw_cv_checks,
        "next_step": next_step,
    }


def format_number(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_markdown(summary):
    lines = [
        f"# P0-B3 Reliability Diagnostic - {summary['tag']}",
        "",
        f"Decision: **{summary['verdict']['decision']}**",
        "",
        summary["verdict"]["next_step"],
        "",
        "## Trigger: previous observation quality predicts current crop miss",
        "",
        "| Protocol | N | Prevalence | AUROC | AUPRC | AUPRC - prevalence | Brier | ECE | Activation | Miss recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    trigger = summary["tasks"]["trigger"]["protocol_metrics"]
    for protocol, metrics in trigger.items():
        if metrics is None:
            continue
        lines.append(
            "| {protocol} | {count} | {prevalence} | {auroc} | {auprc} | {margin} | "
            "{brier} | {ece} | {activation} | {recall} |".format(
                protocol=protocol,
                count=metrics["count"],
                prevalence=format_number(metrics["prevalence"]),
                auroc=format_number(metrics["auroc"]),
                auprc=format_number(metrics["auprc"]),
                margin=format_number(metrics["auprc"] - metrics["prevalence"]),
                brier=format_number(metrics["brier"]),
                ece=format_number(metrics["ece_10"]),
                activation=format_number(metrics.get("activation_rate")),
                recall=format_number(metrics.get("operating_recall")),
            )
        )

    lines.extend(
        [
            "",
            "The trigger uses only `prev_obs_*`, current real `delta_t`, and current CV geometry. "
            "It does not use current foreground evidence or any GT-derived field.",
            "",
            "## Passive raw-CV crop complementarity",
            "",
            "| Protocol | Visible N | Obs recall | Traj recall | Dual oracle recall | Gain over obs | Traj-only endpoints | Both miss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for protocol, metrics in summary["passive_complementarity"].items():
        if metrics is None:
            continue
        lines.append(
            "| {protocol} | {count} | {obs} | {traj} | {dual} | {gain} | {traj_only} | {both} |".format(
                protocol=protocol,
                count=metrics["visible_endpoint_count"],
                obs=format_number(metrics["obs_target_point_recall_mean"]),
                traj=format_number(metrics["traj_target_point_recall_mean"]),
                dual=format_number(metrics["dual_oracle_target_point_recall_mean"]),
                gain=format_number(metrics["dual_oracle_recall_gain_over_obs"]),
                traj_only=metrics["traj_only_endpoint_count"],
                both=metrics["both_miss_endpoint_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- All thresholds and calibrators are fitted on grouped standard `mini_train` tracklets only.",
            "- The same stable tracklet hash assigns folds across protocols; frames are never randomly split.",
            "- `current_evidence` and `selector` are post-crop diagnostics. They cannot be used to claim a pre-crop trigger.",
            "- The trajectory branch is passive and never updates recursive history in these CSVs.",
            "- This report is a mechanism decision, not a tracking-performance result.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test():
    labels = np.asarray([0, 0, 1, 1], dtype=np.float64)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    if roc_auc(labels, scores) != 1.0 or average_precision(labels, scores) != 1.0:
        raise RuntimeError("ranking metric self-test failed")

    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    weights = fit_logistic_regression(features, labels, l2=1e-2)
    probability = predict_logistic(features, weights)
    if not (probability[0] < probability[1] < probability[2] < probability[3]):
        raise RuntimeError("logistic regression self-test failed")
    threshold = select_threshold(labels, probability, target_recall=1.0)
    if np.sum((probability >= threshold) & (labels == 1)) != 2:
        raise RuntimeError("threshold self-test failed")

    synthetic = []
    for index in range(60):
        positive = index % 3 == 0
        synthetic.append(
            {
                "tracklet_key": f"tracklet-{index}",
                "current_target_visible": "True",
                "current_obs_crop_miss": str(positive),
                "prev_obs_search_point_count": "1" if positive else "100",
                "prev_obs_empty_fallback": str(positive),
                "prev_obs_forward_ran": str(not positive),
                "prev_obs_soft_fg_count": "0" if positive else "40",
                "prev_obs_estimated_fg_points": "0" if positive else "12",
                "prev_obs_mean_fg_score": "0.01" if positive else "0.8",
                "prev_obs_fg_entropy_mean": "0.1",
                "prev_obs_fg_margin_mean": "0.9",
                "prev_obs_motion_dynamic_probability": "0.5",
                "current_delta_t": "0.5",
                "cv_speed": "2.0",
                "cv_shift": "1.0",
                "pred_cv_available": "True",
            }
        )
    fitted = fit_task(
        {"standard": synthetic, "gap1124": synthetic, "burst_drop": synthetic},
        "trigger",
        "standard",
        folds=5,
        seed=42,
        l2=1e-3,
        target_recall=0.8,
    )
    if fitted["protocol_metrics"]["gap1124"]["auroc"] < 0.99:
        raise RuntimeError("grouped task fitting self-test failed")
    print("reliability summary self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P0-B3 reliability signals with stable tracklet-grouped folds. Models and "
            "thresholds are fitted on the standard protocol only and applied unchanged to all "
            "other protocols. No scikit-learn dependency is required."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeat as protocol=/path/to/reliability_endpoints.csv",
    )
    parser.add_argument("--standard-protocol", default="standard")
    parser.add_argument("--strong-protocols", default="gap1124,burst_drop")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--go-auroc", type=float, default=0.75)
    parser.add_argument("--go-auprc-margin", type=float, default=0.15)
    parser.add_argument("--raw-cv-union-gain", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", default="output/diagnostics/reliability_signals/analysis"
    )
    parser.add_argument("--tag", default="p0b3_reliability")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if len(args.input) < 2:
        parser.error("At least two --input protocol=csv entries are required.")
    if args.folds < 2:
        parser.error("--folds must be at least 2.")
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in (0, 1].")

    input_paths = {}
    protocol_rows = {}
    for spec in args.input:
        protocol, path = parse_input_spec(spec)
        if protocol in input_paths:
            raise ValueError(f"Duplicate protocol input: {protocol}")
        input_paths[protocol] = path
        protocol_rows[protocol] = read_protocol_rows(path, protocol)
    if args.standard_protocol not in protocol_rows:
        raise ValueError(
            f"standard protocol {args.standard_protocol!r} is absent from --input entries"
        )

    task_results = {}
    for task in ("trigger", "current_evidence", "selector"):
        try:
            task_results[task] = fit_task(
                protocol_rows,
                task,
                args.standard_protocol,
                args.folds,
                args.seed,
                args.l2,
                args.target_recall,
            )
        except RuntimeError as error:
            if task == "trigger":
                raise
            task_results[task] = {
                "task": task,
                "error": str(error),
                "note": (
                    "This secondary diagnostic lacked enough labeled class support. "
                    "It does not invalidate or alter the pre-crop trigger verdict."
                ),
            }

    complementarity = {
        protocol: passive_complementarity(rows)
        for protocol, rows in protocol_rows.items()
    }
    strong_protocols = [
        item.strip() for item in args.strong_protocols.split(",") if item.strip()
    ]
    verdict = evaluate_verdict(
        task_results,
        complementarity,
        strong_protocols,
        args.go_auroc,
        args.go_auprc_margin,
        args.raw_cv_union_gain,
    )

    summary = {
        "tag": args.tag,
        "inputs": {
            protocol: {
                "path": str(path),
                "sha256": sha256_file(path),
                "row_count": len(protocol_rows[protocol]),
            }
            for protocol, path in input_paths.items()
        },
        "standard_protocol": args.standard_protocol,
        "strong_protocols": strong_protocols,
        "folds": args.folds,
        "fold_assignment": "sha256(seed|tracklet_key) modulo folds",
        "seed": args.seed,
        "l2": args.l2,
        "target_recall": args.target_recall,
        "go_thresholds": {
            "auroc": args.go_auroc,
            "auprc_margin_over_prevalence": args.go_auprc_margin,
            "raw_cv_dual_oracle_recall_gain": args.raw_cv_union_gain,
        },
        "tasks": task_results,
        "passive_complementarity": complementarity,
        "verdict": verdict,
        "leakage_guard": (
            "Trigger features are hard-coded to prev_obs_* plus current timestamp/CV geometry. "
            "No field containing GT, target, error, miss, drift, better, or selector labels is "
            "accepted as a trigger feature. Standard is the only fitting protocol."
        ),
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in args.tag
    ).strip("_") or "p0b3_reliability"
    json_path = output_dir / f"{safe_tag}_summary.json"
    report_path = output_dir / f"{safe_tag}_report.md"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2, allow_nan=False)
    report_path.write_text(build_markdown(summary), encoding="utf-8")

    print(json.dumps(verdict, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"summary json: {json_path}")
    print(f"report markdown: {report_path}")


if __name__ == "__main__":
    main()
