import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import summarize_reliability_signals as reliability


FEATURE_SETS = {
    "observation_v1": (
        ("prev_log_search_points", "prev_obs_search_point_count", "log1p"),
        ("prev_empty_fallback", "prev_obs_empty_fallback", "bool"),
        ("prev_mean_fg_score", "prev_obs_mean_fg_score", "identity"),
        ("prev_fg_margin", "prev_obs_fg_margin_mean", "identity"),
        (
            "prev_motion_dynamic_probability",
            "prev_obs_motion_dynamic_probability",
            "identity",
        ),
    ),
}

LABEL_NAME = "current_obs_crop_miss"
VISIBLE_NAME = "current_target_visible"
GROUP_NAME = "tracklet_key"
FORBIDDEN_FEATURE_TOKENS = (
    "current_delta_t",
    "cv_speed",
    "cv_shift",
    "pred_cv_available",
    "forward_ran",
    "target",
    "error",
    "miss",
    "drift",
    "selector",
    "candidate",
    "gt_",
)


def validate_feature_specs(feature_specs):
    names = [name for name, _, _ in feature_specs]
    sources = [source for _, source, _ in feature_specs]
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique.")
    if len(sources) != len(set(sources)):
        raise ValueError("Feature sources must be unique.")
    for _, source, _ in feature_specs:
        lowered = source.lower()
        matched = [token for token in FORBIDDEN_FEATURE_TOKENS if token in lowered]
        if matched:
            raise ValueError(
                f"Feature source {source!r} contains forbidden token(s): {matched}"
            )


def select_labeled_rows(rows):
    selected = []
    for row in rows:
        if reliability.parse_bool(row.get(VISIBLE_NAME)) is not True:
            continue
        label = reliability.parse_bool(row.get(LABEL_NAME))
        if label is None:
            continue
        copied = dict(row)
        copied["_label"] = float(label)
        selected.append(copied)
    return selected


def validate_columns(rows, feature_specs, description):
    if not rows:
        raise RuntimeError(f"No rows found for {description}.")
    required = {LABEL_NAME, VISIBLE_NAME, GROUP_NAME}
    required.update(source for _, source, _ in feature_specs)
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise RuntimeError(f"{description} is missing required columns: {missing}")


def labeled_groups(rows):
    return {row[GROUP_NAME] for row in select_labeled_rows(rows)}


def ensure_fit_eval_disjoint(fit_rows, eval_rows, protocol):
    overlap = labeled_groups(fit_rows).intersection(labeled_groups(eval_rows))
    if overlap:
        examples = sorted(overlap)[:5]
        raise RuntimeError(
            f"Fit/eval tracklet leakage for protocol={protocol}: "
            f"{len(overlap)} overlapping groups, examples={examples}"
        )


def fit_calibrator(rows, feature_specs, l2, target_recall):
    selected = select_labeled_rows(rows)
    if not selected:
        raise RuntimeError("No visible labeled rows in the fitting input.")
    labels = np.asarray([row["_label"] for row in selected], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Fitting input must contain both label classes.")

    raw_features = reliability.build_feature_matrix(selected, feature_specs)
    medians, means, scales = reliability.fit_preprocessor(raw_features)
    features = reliability.apply_preprocessor(raw_features, medians, means, scales)
    weights = reliability.fit_logistic_regression(features, labels, l2=l2)
    probabilities = reliability.predict_logistic(features, weights)
    threshold = reliability.select_threshold(labels, probabilities, target_recall)
    decisions = probabilities >= threshold

    model = {
        "feature_set": "observation_v1",
        "label": LABEL_NAME,
        "feature_specs": [
            {"name": name, "source": source, "transform": transform}
            for name, source, transform in feature_specs
        ],
        "imputation_medians": medians.tolist(),
        "standardization_means": means.tolist(),
        "standardization_scales": scales.tolist(),
        "intercept": float(weights[0]),
        "coefficients": weights[1:].tolist(),
        "operating_threshold": float(threshold),
        "target_recall": float(target_recall),
        "l2": float(l2),
    }
    metrics = reliability.classification_metrics(labels, probabilities, decisions)
    metrics["tracklet_count"] = len({row[GROUP_NAME] for row in selected})
    metrics["evaluation_role"] = "fit_resubstitution_only_not_confirmatory"
    return model, metrics


def apply_calibrator(rows, feature_specs, model):
    selected = select_labeled_rows(rows)
    if not selected:
        raise RuntimeError("No visible labeled rows in an evaluation input.")
    labels = np.asarray([row["_label"] for row in selected], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Evaluation input must contain both label classes.")

    raw_features = reliability.build_feature_matrix(selected, feature_specs)
    medians = np.asarray(model["imputation_medians"], dtype=np.float64)
    means = np.asarray(model["standardization_means"], dtype=np.float64)
    scales = np.asarray(model["standardization_scales"], dtype=np.float64)
    weights = np.asarray(
        [model["intercept"], *model["coefficients"]], dtype=np.float64
    )
    features = reliability.apply_preprocessor(raw_features, medians, means, scales)
    probabilities = reliability.predict_logistic(features, weights)
    decisions = probabilities >= float(model["operating_threshold"])
    metrics = reliability.classification_metrics(labels, probabilities, decisions)
    metrics["tracklet_count"] = len({row[GROUP_NAME] for row in selected})
    metrics["evaluation_role"] = "frozen_independent_evaluation"
    return metrics


def protocol_check(metrics, thresholds):
    auroc = metrics.get("auroc")
    auprc = metrics.get("auprc")
    prevalence = metrics.get("prevalence")
    ece = metrics.get("ece_10")
    fpr = metrics.get("operating_false_positive_rate")
    recall = metrics.get("operating_recall")
    checks = {
        "auroc": auroc is not None and auroc >= thresholds["go_auroc"],
        "auprc_margin": (
            auprc is not None
            and prevalence is not None
            and auprc - prevalence >= thresholds["go_auprc_margin"]
        ),
        "ece": ece is not None and ece <= thresholds["max_ece"],
        "fpr": fpr is not None and fpr <= thresholds["max_fpr"],
        "operating_recall": (
            recall is not None and recall >= thresholds["min_operating_recall"]
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "auprc_margin": (
            float(auprc - prevalence)
            if auprc is not None and prevalence is not None
            else None
        ),
    }


def get_git_state(repo_root):
    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_lines": status.splitlines() if status else [],
    }


def format_number(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_markdown(summary):
    lines = [
        f"# Observation Reliability Validation - {summary['tag']}",
        "",
        f"Decision: **{summary['verdict']['decision']}**",
        "",
        "The calibrator, preprocessing statistics, and operating threshold were fitted once on",
        "the standard fitting CSV. Every evaluation protocol used that frozen model unchanged.",
        "",
        "## Frozen feature set",
        "",
    ]
    for feature in summary["calibrator"]["feature_specs"]:
        lines.append(
            f"- `{feature['name']}` <- `{feature['source']}` ({feature['transform']})"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| protocol | N | prevalence | AUROC | AUPRC | AUPRC-prev | Brier | ECE | activation | recall | precision | FPR | pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for protocol, metrics in summary["evaluation_metrics"].items():
        check = summary["verdict"]["protocol_checks"].get(protocol, {})
        margin = (
            metrics["auprc"] - metrics["prevalence"]
            if metrics.get("auprc") is not None
            else None
        )
        lines.append(
            "| {protocol} | {count} | {prevalence} | {auroc} | {auprc} | "
            "{margin} | {brier} | {ece} | {activation} | {recall} | "
            "{precision} | {fpr} | {passed} |".format(
                protocol=protocol,
                count=metrics["count"],
                prevalence=format_number(metrics["prevalence"]),
                auroc=format_number(metrics["auroc"]),
                auprc=format_number(metrics["auprc"]),
                margin=format_number(margin),
                brier=format_number(metrics["brier"]),
                ece=format_number(metrics["ece_10"]),
                activation=format_number(metrics["activation_rate"]),
                recall=format_number(metrics["operating_recall"]),
                precision=format_number(metrics["operating_precision"]),
                fpr=format_number(metrics["operating_false_positive_rate"]),
                passed=check.get("pass", "report-only"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"Strong protocols: `{', '.join(summary['strong_protocols'])}`.",
            "All strong protocols must pass every pre-registered AUROC, AUPRC-margin, ECE,",
            "FPR, and operating-recall check. Evaluation data never refit preprocessing,",
            "weights, or threshold.",
            "",
            "This validates only a visible-target next-crop-risk proxy. It does not validate",
            "timestamp causality, complete-occlusion uncertainty, a trajectory anchor, or active tracking.",
            "",
        ]
    )
    return "\n".join(lines)


def make_synthetic_rows(prefix, count):
    rows = []
    for index in range(count):
        positive = index % 4 == 0 or index % 11 == 0
        empty = positive and index % 3 == 0
        rows.append(
            {
                GROUP_NAME: f"{prefix}-tracklet-{index // 2}",
                VISIBLE_NAME: "True",
                LABEL_NAME: str(positive),
                "prev_obs_search_point_count": "2" if positive else "120",
                "prev_obs_empty_fallback": str(empty),
                "prev_obs_mean_fg_score": "" if empty else ("0.05" if positive else "0.8"),
                "prev_obs_fg_margin_mean": "" if empty else ("0.1" if positive else "0.9"),
                "prev_obs_motion_dynamic_probability": (
                    "" if empty else ("0.8" if positive else "0.2")
                ),
            }
        )
    return rows


def self_test():
    feature_specs = FEATURE_SETS["observation_v1"]
    validate_feature_specs(feature_specs)
    fit_rows = make_synthetic_rows("fit", 120)
    eval_rows = make_synthetic_rows("eval", 80)
    validate_columns(fit_rows, feature_specs, "synthetic fit")
    validate_columns(eval_rows, feature_specs, "synthetic eval")
    ensure_fit_eval_disjoint(fit_rows, eval_rows, "synthetic")
    model, _ = fit_calibrator(fit_rows, feature_specs, l2=1e-3, target_recall=0.8)
    metrics = apply_calibrator(eval_rows, feature_specs, model)
    if metrics["auroc"] is None or metrics["auroc"] < 0.9:
        raise AssertionError(f"Unexpected synthetic AUROC: {metrics['auroc']}")
    first = apply_calibrator(eval_rows, feature_specs, model)
    second = apply_calibrator(eval_rows, feature_specs, model)
    if first != second:
        raise AssertionError("Frozen evaluation is not deterministic.")
    try:
        ensure_fit_eval_disjoint(fit_rows, fit_rows, "leakage-test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Fit/eval overlap was not rejected.")
    print("observation reliability validation self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a non-redundant observation-only reliability calibrator once on a standard "
            "training CSV, then evaluate it unchanged on disjoint protocol CSVs."
        )
    )
    parser.add_argument("--fit-standard", default="")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        help="Repeat as protocol=/path/to/reliability_endpoints.csv",
    )
    parser.add_argument(
        "--feature-set", choices=sorted(FEATURE_SETS), default="observation_v1"
    )
    parser.add_argument("--strong-protocols", default="gap1124,burst_drop")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--go-auroc", type=float, default=0.75)
    parser.add_argument("--go-auprc-margin", type=float, default=0.15)
    parser.add_argument("--max-ece", type=float, default=0.10)
    parser.add_argument("--max-fpr", type=float, default=0.30)
    parser.add_argument("--min-operating-recall", type=float, default=0.70)
    parser.add_argument(
        "--output-dir",
        default="output/diagnostics/reliability_signals/validation",
    )
    parser.add_argument("--tag", default="observation_v1_validation")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.fit_standard:
        parser.error("--fit-standard is required unless --self-test is used.")
    if not args.eval:
        parser.error("At least one --eval protocol=csv entry is required.")
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in (0, 1].")
    if args.l2 < 0.0:
        parser.error("--l2 must be non-negative.")
    bounded_arguments = {
        "--go-auroc": args.go_auroc,
        "--go-auprc-margin": args.go_auprc_margin,
        "--max-ece": args.max_ece,
        "--max-fpr": args.max_fpr,
        "--min-operating-recall": args.min_operating_recall,
    }
    for name, value in bounded_arguments.items():
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be in [0, 1].")

    feature_specs = FEATURE_SETS[args.feature_set]
    validate_feature_specs(feature_specs)
    fit_path = Path(args.fit_standard).resolve()
    fit_rows = reliability.read_protocol_rows(fit_path, "fit_standard")
    validate_columns(fit_rows, feature_specs, "fit standard")

    eval_paths = {}
    eval_rows = {}
    for spec in args.eval:
        protocol, path = reliability.parse_input_spec(spec)
        if protocol in eval_paths:
            raise ValueError(f"Duplicate evaluation protocol: {protocol}")
        rows = reliability.read_protocol_rows(path, protocol)
        validate_columns(rows, feature_specs, f"evaluation protocol={protocol}")
        ensure_fit_eval_disjoint(fit_rows, rows, protocol)
        eval_paths[protocol] = path
        eval_rows[protocol] = rows

    strong_protocols = [
        item.strip() for item in args.strong_protocols.split(",") if item.strip()
    ]
    missing_strong = sorted(set(strong_protocols).difference(eval_rows))
    if missing_strong:
        raise ValueError(f"Missing strong evaluation protocols: {missing_strong}")

    model, fit_metrics = fit_calibrator(
        fit_rows, feature_specs, l2=args.l2, target_recall=args.target_recall
    )
    evaluation_metrics = {
        protocol: apply_calibrator(rows, feature_specs, model)
        for protocol, rows in eval_rows.items()
    }
    thresholds = {
        "go_auroc": args.go_auroc,
        "go_auprc_margin": args.go_auprc_margin,
        "max_ece": args.max_ece,
        "max_fpr": args.max_fpr,
        "min_operating_recall": args.min_operating_recall,
    }
    protocol_checks = {
        protocol: protocol_check(evaluation_metrics[protocol], thresholds)
        for protocol in strong_protocols
    }
    passed = all(check["pass"] for check in protocol_checks.values())
    verdict = {
        "decision": (
            "GO_PASSIVE_INDEPENDENT_STATE_ANCHOR"
            if passed
            else "NO_GO_OBSERVATION_RELIABILITY_VALIDATION"
        ),
        "pass": bool(passed),
        "protocol_checks": protocol_checks,
        "next_step": (
            "Implement one passive reliability-updated independent state anchor."
            if passed
            else "Stop the reliability-controlled anchor path; do not retune on evaluation data."
        ),
    }

    repo_root = Path(__file__).resolve().parents[1]
    summary = {
        "tag": args.tag,
        "feature_set": args.feature_set,
        "fit_input": {
            "path": str(fit_path),
            "sha256": reliability.sha256_file(fit_path),
            "row_count": len(fit_rows),
        },
        "evaluation_inputs": {
            protocol: {
                "path": str(path),
                "sha256": reliability.sha256_file(path),
                "row_count": len(eval_rows[protocol]),
            }
            for protocol, path in eval_paths.items()
        },
        "script_sha256": reliability.sha256_file(Path(__file__).resolve()),
        "dependency_sha256": reliability.sha256_file(
            Path(reliability.__file__).resolve()
        ),
        "git": get_git_state(repo_root),
        "strong_protocols": strong_protocols,
        "thresholds": thresholds,
        "calibrator": model,
        "fit_metrics": fit_metrics,
        "evaluation_metrics": evaluation_metrics,
        "verdict": verdict,
        "leakage_guard": {
            "fit_eval_tracklets_disjoint": True,
            "evaluation_refit": False,
            "evaluation_threshold_tuning": False,
            "feature_sources": [source for _, source, _ in feature_specs],
            "label_scope": "visible-target current observation crop miss",
        },
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in args.tag
    ).strip("_") or "observation_v1_validation"
    summary_path = output_dir / f"{safe_tag}_summary.json"
    model_path = output_dir / f"{safe_tag}_calibrator.json"
    report_path = output_dir / f"{safe_tag}_report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    model_path.write_text(
        json.dumps(model, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"calibrator: {model_path}")
    print(f"report: {report_path}")
    print(f"decision: {verdict['decision']}")


if __name__ == "__main__":
    main()
