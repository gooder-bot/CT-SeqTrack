"""Finite-sample empirical calibration for CT-SeqTrack B3 actions."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "ct_seqtrack.action_calibration.v2"
SCORE_DEFINITION = (
    "sigmoid(help_logit)*(1-sigmoid(harm_logit))")
CONSENSUS_FEATURE_SCHEMA = (
    "vote_consistency,covariance_xx,covariance_xy,covariance_yy,"
    "inlier_ratio,candidate_margin,compatible_hypothesis_count")
CALIBRATION_CODE_FILES = (
    "models/seqtrack3d.py",
    "models/ctseqtrack.py",
    "models/ct_v2/evidence_memory.py",
    "models/ct_v2/motion.py",
    "models/ct_v2/pipeline_contracts.py",
    "utils/action_calibration.py",
    "utils/box_membership.py",
    "utils/ct_search.py",
)


def require_selective_calibration(calibrated, inference_mode):
    mode = str(inference_mode).strip().lower()
    if mode in ("full", "full_selective", "selective") and not bool(
            calibrated):
        raise RuntimeError(
            "selective evaluation requires a matching passed action "
            "calibration artifact")


def _json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256_json(payload):
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_calibration_code_sha256(root=None):
    """Hash score, feature, acquisition and decision source content."""
    root = (Path(__file__).resolve().parents[1]
            if root is None else Path(root).resolve())
    digest = hashlib.sha256()
    for relative in CALIBRATION_CODE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def action_calibration_config_identity(config):
    """Canonical resolved-config identity, excluding deployment plumbing."""
    values = dict(config) if not isinstance(config, dict) else dict(config)
    excluded = {
        "cfg", "path", "tag", "log_dir", "checkpoint", "init_checkpoint",
        "test", "test_split", "preloading", "proposal_inference_mode",
        "ct_action_calibration_path",
        "ct_calibration_tracklet_manifest_sha256",
        "ct_dev_tracklet_manifest_sha256", "ct_source_checkpoint_epoch",
        "acquisition_preflight", "b2_method_promotion",
        "ct_acquisition_preflight_manifest",
        "ct_b2_method_promotion_manifest", "ct_candidate0_b0_source",
        "ct_mechanism_tracklets_observed",
        "ct_mechanism_prediction_frames_observed",
        "ct_mechanism_selection_sha256",
        "ct_observation_steps_per_epoch_observed",
        "ct_mechanism_steps_per_epoch_observed",
    }
    return {
        str(key): value for key, value in sorted(values.items())
        if str(key) not in excluded
    }


def _finite_rows(rows):
    required = (
        "tracklet_id", "structural_available", "presence_score",
        "action_score", "center_gain", "iou_gain")
    normalized = []
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError("calibration row missing: " + ", ".join(missing))
        item = dict(row)
        for key in required[1:]:
            item[key] = float(item[key])
        if not all(np.isfinite(item[key]) for key in required[1:]):
            raise ValueError("calibration rows must be finite")
        item["tracklet_id"] = str(item["tracklet_id"])
        normalized.append(item)
    if not normalized:
        raise ValueError("calibration rows are empty")
    return normalized


def _selected(rows, presence_threshold, action_threshold):
    return [row for row in rows if (
        row["structural_available"] > 0
        and row["presence_score"] >= presence_threshold
        and row["action_score"] >= action_threshold)]


def _metrics(selected, total_rows):
    center = np.asarray([row["center_gain"] for row in selected], dtype=float)
    iou = np.asarray([row["iou_gain"] for row in selected], dtype=float)
    harmful = (center < 0.0) | (iou < 0.0)
    return {
        "actions": int(len(selected)),
        "tracklets": int(len({row["tracklet_id"] for row in selected})),
        "coverage": float(len(selected) / max(int(total_rows), 1)),
        "harmful_rate": float(harmful.mean()) if len(selected) else 1.0,
        "center_gain": float(center.mean()) if len(selected) else 0.0,
        "iou_gain": float(iou.mean()) if len(selected) else 0.0,
    }


def tracklet_bootstrap_bounds(selected, seed=42, resamples=2000):
    """One-sided 95% empirical bounds with tracklets as sampling units."""
    groups = {}
    for row in selected:
        groups.setdefault(row["tracklet_id"], []).append(row)
    keys = sorted(groups)
    if not keys:
        return {
            "harmful_rate_upper_95": 1.0,
            "center_gain_lower_95": -1e30,
            "iou_gain_lower_95": -1e30,
        }
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(resamples), 3), dtype=np.float64)
    for index in range(int(resamples)):
        picked = rng.choice(keys, size=len(keys), replace=True)
        rows = [row for key in picked for row in groups[str(key)]]
        metric = _metrics(rows, len(rows))
        samples[index] = (
            metric["harmful_rate"], metric["center_gain"],
            metric["iou_gain"])
    return {
        "harmful_rate_upper_95": float(np.quantile(samples[:, 0], 0.95)),
        "center_gain_lower_95": float(np.quantile(samples[:, 1], 0.05)),
        "iou_gain_lower_95": float(np.quantile(samples[:, 2], 0.05)),
    }


def calibrate_actions(
        rows, checkpoint_sha256, config_sha256, tracklet_manifest_sha256,
        dev_rows=None, dev_tracklet_manifest_sha256=None,
        code_version="unknown",
        code_content_sha256=None,
        consensus_feature_schema=CONSENSUS_FEATURE_SCHEMA,
        seed=42, resamples=2000, min_tracklets=30, min_actions=100,
        min_coverage=0.01, max_harmful_upper=0.05):
    """Choose a risk-feasible threshold pair on calibration tracklets."""
    rows = _finite_rows(rows)
    compatibility_mode = dev_rows is None
    dev_rows = rows if compatibility_mode else _finite_rows(dev_rows)
    dev_manifest_sha = (
        str(tracklet_manifest_sha256) if compatibility_mode else
        str(dev_tracklet_manifest_sha256))
    if not dev_manifest_sha:
        raise ValueError("dev tracklet manifest SHA is required")
    calibration_ids = {row["tracklet_id"] for row in rows}
    dev_ids = {row["tracklet_id"] for row in dev_rows}
    if not compatibility_mode and calibration_ids & dev_ids:
        raise ValueError("calibration and dev tracklets must be disjoint")
    if (not compatibility_mode
            and str(tracklet_manifest_sha256) == dev_manifest_sha):
        raise ValueError("calibration and dev manifests must be disjoint")
    presence_values = np.quantile(
        [row["presence_score"] for row in rows],
        np.linspace(0.0, 0.95, 20))
    action_values = np.quantile(
        [row["action_score"] for row in rows],
        np.linspace(0.0, 0.99, 50))
    feasible = []
    evaluated = []
    for presence_threshold in np.unique(presence_values):
        for action_threshold in np.unique(action_values):
            selected = _selected(
                rows, float(presence_threshold), float(action_threshold))
            metrics = _metrics(selected, len(rows))
            if (metrics["actions"] < int(min_actions)
                    or metrics["tracklets"] < int(min_tracklets)
                    or metrics["coverage"] < float(min_coverage)):
                continue
            bounds = tracklet_bootstrap_bounds(
                selected, seed=seed, resamples=resamples)
            candidate = {
                "presence_threshold": float(presence_threshold),
                "action_threshold": float(action_threshold),
                **metrics, **bounds,
            }
            evaluated.append(candidate)
            if (bounds["harmful_rate_upper_95"] <= max_harmful_upper
                    and bounds["center_gain_lower_95"] >= 0.0
                    and bounds["iou_gain_lower_95"] >= 0.0):
                feasible.append(candidate)
    if feasible:
        chosen = max(
            feasible,
            key=lambda item: (
                item["center_gain"] + item["iou_gain"], item["coverage"]))
    else:
        chosen = {
            "presence_threshold": 1.0,
            "action_threshold": 1.0,
            **_metrics([], len(rows)),
            **tracklet_bootstrap_bounds([], seed=seed, resamples=resamples),
        }
    calibration_passed = bool(feasible)
    locked_presence = float(chosen["presence_threshold"])
    locked_action = float(chosen["action_threshold"])
    dev_selected = _selected(
        dev_rows, locked_presence, locked_action)
    dev_metrics = _metrics(dev_selected, len(dev_rows))
    dev_bounds = tracklet_bootstrap_bounds(
        dev_selected, seed=seed, resamples=resamples)
    dev_promotion_passed = bool(
        calibration_passed
        and dev_metrics["actions"] >= int(min_actions)
        and dev_metrics["tracklets"] >= int(min_tracklets)
        and dev_metrics["coverage"] >= float(min_coverage)
        and dev_bounds["harmful_rate_upper_95"] <= float(max_harmful_upper)
        and dev_bounds["center_gain_lower_95"] >= 0.0
        and dev_bounds["iou_gain_lower_95"] >= 0.0)
    passed = bool(calibration_passed and dev_promotion_passed)
    deployed_presence = locked_presence if passed else 1.0
    deployed_action = locked_action if passed else 1.0
    artifact = {
        "schema": SCHEMA,
        "passed": passed,
        "calibration_passed": calibration_passed,
        "dev_promotion_passed": dev_promotion_passed,
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "calibration_tracklet_manifest_sha256": str(
            tracklet_manifest_sha256),
        "dev_tracklet_manifest_sha256": dev_manifest_sha,
        "score_definition": SCORE_DEFINITION,
        "consensus_feature_schema": str(consensus_feature_schema),
        "code_version": str(code_version),
        "code_content_sha256": str(
            code_content_sha256 or action_calibration_code_sha256()),
        "selection_partition": "calibration_thresholds_dev_promotion",
        "thresholds": {
            "presence": deployed_presence,
            "action": deployed_action,
        },
        "locked_calibration_thresholds": {
            "presence": locked_presence,
            "action": locked_action,
        },
        "calibration_metrics": {
            key: value for key, value in chosen.items()
            if not key.endswith("threshold")},
        "dev_metrics": {**dev_metrics, **dev_bounds},
        "requirements": {
            "min_tracklets": int(min_tracklets),
            "min_actions": int(min_actions),
            "min_coverage": float(min_coverage),
            "max_harmful_rate_upper_95": float(max_harmful_upper),
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
        },
        "evaluated_threshold_pairs": len(evaluated),
        "compatibility_self_check": bool(compatibility_mode),
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def risk_coverage_curve(
        rows, presence_threshold=0.0, seed=42, resamples=2000,
        points=50):
    """Return tracklet-bootstrap risk/gain statistics over action coverage."""
    rows = _finite_rows(rows)
    eligible = [row for row in rows if (
        row["structural_available"] > 0
        and row["presence_score"] >= float(presence_threshold))]
    if not eligible:
        return []
    thresholds = np.unique(np.quantile(
        [row["action_score"] for row in eligible],
        np.linspace(0.0, 1.0, int(points))))[::-1]
    curve = []
    for threshold in thresholds:
        selected = [row for row in eligible
                    if row["action_score"] >= float(threshold)]
        curve.append({
            "action_threshold": float(threshold),
            **_metrics(selected, len(rows)),
            **tracklet_bootstrap_bounds(
                selected, seed=seed, resamples=resamples),
        })
    return curve


def validate_action_calibration(
        artifact, checkpoint_sha256, config_sha256,
        tracklet_manifest_sha256, dev_tracklet_manifest_sha256=None,
        code_version=None,
        code_content_sha256=None,
        consensus_feature_schema=CONSENSUS_FEATURE_SCHEMA):
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA:
        raise ValueError("missing ct_seqtrack.action_calibration.v2 artifact")
    supplied_hash = artifact.get("artifact_sha256")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if supplied_hash != sha256_json(payload):
        raise ValueError("action calibration artifact hash mismatch")
    expected = {
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "calibration_tracklet_manifest_sha256": str(
            tracklet_manifest_sha256),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"action calibration {key} mismatch")
    if artifact.get("score_definition") != SCORE_DEFINITION:
        raise ValueError("action calibration score definition mismatch")
    if artifact.get("consensus_feature_schema") != str(
            consensus_feature_schema):
        raise ValueError("action calibration consensus feature schema mismatch")
    if (dev_tracklet_manifest_sha256 is not None
            and artifact.get("dev_tracklet_manifest_sha256")
            != str(dev_tracklet_manifest_sha256)):
        raise ValueError(
            "action calibration dev_tracklet_manifest_sha256 mismatch")
    if (code_version is not None
            and artifact.get("code_version") != str(code_version)):
        raise ValueError("action calibration code_version mismatch")
    expected_code_content = str(
        code_content_sha256 or action_calibration_code_sha256())
    if artifact.get("code_content_sha256") != expected_code_content:
        raise ValueError("action calibration code content hash mismatch")
    if (artifact.get("selection_partition")
            != "calibration_thresholds_dev_promotion"):
        raise ValueError(
            "action calibration must lock on calibration and promote on dev")
    if not bool(artifact.get("passed")):
        raise ValueError("action calibration did not pass promotion criteria")
    return artifact


def load_action_calibration(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
