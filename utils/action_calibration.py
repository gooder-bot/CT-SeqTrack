"""Finite-sample empirical calibration for CT-SeqTrack B3 actions."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "ct_seqtrack.action_calibration.v1"
SELECTION_SCHEMA = "ct_seqtrack.action_threshold_selection.v1"
AUDIT_SCHEMA = "ct_seqtrack.action_calibration.v2"
SCORE_DEFINITION = (
    "sigmoid(help_logit)*(1-sigmoid(harm_logit))")


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


def action_calibration_config_identity(config):
    """Hash only fields that define scores or the bounded action."""
    def get(name, default=None):
        return config.get(name, default) if isinstance(config, dict) else (
            getattr(config, name, default))

    fields = (
        "ct_variant", "ct_joint_contract_version", "ct_protocol_version",
        "ct_memory_mode", "ct_partition_scheme", "ct_b2_target_bb_scale",
        "ct_prior_mode", "ct_time_mode", "ct_search_presence_threshold",
        "ct_router_radius_base", "ct_router_radius_per_second",
        "ct_router_radius_max", "ct_router_hidden_dim",
        "ct_router_help_margin", "ct_router_h3_margin",
    )
    return {name: get(name) for name in fields}


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


def _finite_scene_rows(rows, partition, *, require_selective=False,
                       thresholds=None):
    normalized = _finite_rows(rows)
    for item in normalized:
        if "partition_group_key" not in item:
            raise ValueError(
                "v25 calibration rows require partition_group_key (scene)")
        item["partition_group_key"] = str(item["partition_group_key"])
        if str(item.get("partition", "")) != str(partition):
            raise ValueError(
                f"v25 calibration row is not in {partition}")
        if require_selective:
            mode = str(item.get(
                "rollout_mode", item.get("proposal_inference_mode", ""))
            ).strip().lower()
            if mode != "selective":
                raise ValueError(
                    "calibration_audit requires selective continuous rollout")
            if thresholds is not None:
                for field, expected in (
                        ("calibrated_presence_threshold",
                         thresholds["presence"]),
                        ("calibrated_action_threshold",
                         thresholds["action"])):
                    if field not in item or not np.isclose(
                            float(item[field]), float(expected),
                            rtol=0.0, atol=1e-12):
                        raise ValueError(
                            "audit rows were not generated with the fixed "
                            "selection thresholds")
    return normalized


def _scene_metrics(selected, total_rows):
    metrics = _metrics(selected, total_rows)
    metrics["scenes"] = int(len({
        row["partition_group_key"] for row in selected}))
    return metrics


def scene_bootstrap_bounds(selected, seed=42, resamples=2000):
    """One-sided 95% bounds with physical scenes as sampling units."""
    groups = {}
    for row in selected:
        groups.setdefault(row["partition_group_key"], []).append(row)
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
        sampled_rows = [
            row for key in picked for row in groups[str(key)]]
        metric = _scene_metrics(sampled_rows, len(sampled_rows))
        samples[index] = (
            metric["harmful_rate"], metric["center_gain"],
            metric["iou_gain"])
    return {
        "harmful_rate_upper_95": float(np.quantile(samples[:, 0], 0.95)),
        "center_gain_lower_95": float(np.quantile(samples[:, 1], 0.05)),
        "iou_gain_lower_95": float(np.quantile(samples[:, 2], 0.05)),
    }


def select_action_thresholds(
        rows, checkpoint_sha256, config_sha256,
        selection_scene_manifest_sha256, seed=42, resamples=2000,
        min_scenes=10, min_actions=100, min_coverage=0.01,
        max_harmful_upper=0.05):
    """Choose thresholds on calibration_select scenes only."""
    rows = _finite_scene_rows(rows, "calibration_select")
    presence_values = np.unique(np.quantile(
        [row["presence_score"] for row in rows],
        np.linspace(0.0, 0.95, 20)))
    action_values = np.unique(np.quantile(
        [row["action_score"] for row in rows],
        np.linspace(0.0, 0.99, 50)))
    feasible = []
    evaluated = 0
    for presence_threshold in presence_values:
        for action_threshold in action_values:
            selected = _selected(
                rows, float(presence_threshold), float(action_threshold))
            metrics = _scene_metrics(selected, len(rows))
            if (metrics["actions"] < int(min_actions)
                    or metrics["scenes"] < int(min_scenes)
                    or metrics["coverage"] < float(min_coverage)):
                continue
            evaluated += 1
            bounds = scene_bootstrap_bounds(
                selected, seed=seed, resamples=resamples)
            candidate = {
                "presence_threshold": float(presence_threshold),
                "action_threshold": float(action_threshold),
                **metrics, **bounds,
            }
            if (bounds["harmful_rate_upper_95"] <= max_harmful_upper
                    and bounds["center_gain_lower_95"] >= 0.0
                    and bounds["iou_gain_lower_95"] >= 0.0):
                feasible.append(candidate)
    if feasible:
        chosen = max(feasible, key=lambda item: (
            item["center_gain"] + item["iou_gain"], item["coverage"]))
    else:
        chosen = {
            "presence_threshold": 1.0,
            "action_threshold": 1.0,
            **_scene_metrics([], len(rows)),
            **scene_bootstrap_bounds([], seed=seed, resamples=resamples),
        }
    artifact = {
        "schema": SELECTION_SCHEMA,
        "passed": bool(feasible),
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "selection_scene_manifest_sha256": str(
            selection_scene_manifest_sha256),
        "score_definition": SCORE_DEFINITION,
        "selection_partition": "calibration_select",
        "selection_scene_keys": sorted({
            row["partition_group_key"] for row in rows}),
        "thresholds": {
            "presence": chosen["presence_threshold"],
            "action": chosen["action_threshold"],
        },
        "selection_diagnostics": {
            key: value for key, value in chosen.items()
            if not key.endswith("threshold")},
        "requirements": {
            "min_scenes": int(min_scenes),
            "min_actions": int(min_actions),
            "min_coverage": float(min_coverage),
            "max_harmful_rate_upper_95": float(max_harmful_upper),
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
        },
        "evaluated_threshold_pairs": int(evaluated),
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def validate_threshold_selection(
        artifact, checkpoint_sha256, config_sha256,
        selection_scene_manifest_sha256):
    if (not isinstance(artifact, dict)
            or artifact.get("schema") != SELECTION_SCHEMA):
        raise ValueError("missing v25 threshold-selection artifact")
    supplied_hash = artifact.get("artifact_sha256")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if supplied_hash != sha256_json(payload):
        raise ValueError("threshold-selection artifact hash mismatch")
    expected = {
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "selection_scene_manifest_sha256": str(
            selection_scene_manifest_sha256),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"threshold selection {key} mismatch")
    if (artifact.get("selection_partition") != "calibration_select"
            or artifact.get("score_definition") != SCORE_DEFINITION
            or not bool(artifact.get("passed"))):
        raise ValueError("threshold selection is not eligible for audit")
    scene_keys = artifact.get("selection_scene_keys")
    if (not isinstance(scene_keys, list) or not scene_keys
            or len(scene_keys) != len(set(str(key) for key in scene_keys))):
        raise ValueError(
            "threshold selection lacks a unique non-empty scene population")
    return artifact


def audit_action_thresholds(
        rows, selection_artifact, checkpoint_sha256, config_sha256,
        selection_scene_manifest_sha256, audit_scene_manifest_sha256,
        seed=42, resamples=2000, min_scenes=10, min_actions=100,
        min_coverage=0.01, max_harmful_upper=0.05):
    """Evaluate fixed thresholds once on disjoint selective audit scenes."""
    validate_threshold_selection(
        selection_artifact, checkpoint_sha256, config_sha256,
        selection_scene_manifest_sha256)
    if str(selection_scene_manifest_sha256) == str(
            audit_scene_manifest_sha256):
        raise ValueError(
            "calibration_select and calibration_audit manifest identities "
            "must differ")
    thresholds = dict(selection_artifact["thresholds"])
    rows = _finite_scene_rows(
        rows, "calibration_audit", require_selective=True,
        thresholds=thresholds)
    audit_scene_keys = sorted({
        row["partition_group_key"] for row in rows})
    selection_scene_keys = {
        str(key) for key in selection_artifact["selection_scene_keys"]}
    overlap = selection_scene_keys.intersection(audit_scene_keys)
    if overlap:
        raise ValueError(
            "calibration_select and calibration_audit rows share scenes: "
            + ", ".join(sorted(overlap)[:5]))
    selected = _selected(
        rows, float(thresholds["presence"]),
        float(thresholds["action"]))
    metrics = _scene_metrics(selected, len(rows))
    bounds = scene_bootstrap_bounds(
        selected, seed=seed, resamples=resamples)
    criteria = {
        "audit_scenes_at_least_minimum": metrics["scenes"] >= int(min_scenes),
        "audit_actions_at_least_minimum": metrics["actions"] >= int(min_actions),
        "audit_coverage_at_least_minimum": metrics["coverage"] >= float(min_coverage),
        "audit_harmful_rate_upper_95_at_most_maximum": (
            bounds["harmful_rate_upper_95"] <= float(max_harmful_upper)),
        "audit_center_gain_lower_95_nonnegative": (
            bounds["center_gain_lower_95"] >= 0.0),
        "audit_iou_gain_lower_95_nonnegative": (
            bounds["iou_gain_lower_95"] >= 0.0),
    }
    artifact = {
        "schema": AUDIT_SCHEMA,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "selection_scene_manifest_sha256": str(
            selection_scene_manifest_sha256),
        "audit_scene_manifest_sha256": str(audit_scene_manifest_sha256),
        "threshold_selection_artifact_sha256": selection_artifact[
            "artifact_sha256"],
        "score_definition": SCORE_DEFINITION,
        "selection_partition": "calibration_select",
        "audit_partition": "calibration_audit",
        "selection_scene_keys": sorted(selection_scene_keys),
        "audit_scene_keys": audit_scene_keys,
        "thresholds": thresholds,
        "audit_metrics": {**metrics, **bounds},
        "requirements": {
            "min_scenes": int(min_scenes),
            "min_actions": int(min_actions),
            "min_coverage": float(min_coverage),
            "max_harmful_rate_upper_95": float(max_harmful_upper),
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
        },
        "safety_population": "calibration_audit_only",
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def calibrate_actions(
        rows, checkpoint_sha256, config_sha256, tracklet_manifest_sha256,
        seed=42, resamples=2000, min_tracklets=30, min_actions=100,
        min_coverage=0.01, max_harmful_upper=0.05):
    """Choose a risk-feasible threshold pair on calibration tracklets."""
    rows = _finite_rows(rows)
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
    artifact = {
        "schema": SCHEMA,
        "passed": bool(feasible),
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "tracklet_manifest_sha256": str(tracklet_manifest_sha256),
        "score_definition": SCORE_DEFINITION,
        "selection_partition": "calibration",
        "thresholds": {
            "presence": chosen["presence_threshold"],
            "action": chosen["action_threshold"],
        },
        "metrics": {key: value for key, value in chosen.items()
                    if not key.endswith("threshold")},
        "requirements": {
            "min_tracklets": int(min_tracklets),
            "min_actions": int(min_actions),
            "min_coverage": float(min_coverage),
            "max_harmful_rate_upper_95": float(max_harmful_upper),
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
        },
        "evaluated_threshold_pairs": len(evaluated),
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
        tracklet_manifest_sha256, selection_scene_manifest_sha256=None):
    if not isinstance(artifact, dict) or artifact.get("schema") not in (
            SCHEMA, AUDIT_SCHEMA):
        raise ValueError("missing supported CT action calibration artifact")
    supplied_hash = artifact.get("artifact_sha256")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if supplied_hash != sha256_json(payload):
        raise ValueError("action calibration artifact hash mismatch")
    if artifact.get("schema") == AUDIT_SCHEMA:
        expected = {
            "checkpoint_sha256": str(checkpoint_sha256),
            "config_sha256": str(config_sha256),
            "audit_scene_manifest_sha256": str(tracklet_manifest_sha256),
            "selection_scene_manifest_sha256": str(
                selection_scene_manifest_sha256),
        }
    else:
        expected = {
            "checkpoint_sha256": str(checkpoint_sha256),
            "config_sha256": str(config_sha256),
            "tracklet_manifest_sha256": str(tracklet_manifest_sha256),
        }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"action calibration {key} mismatch")
    if artifact.get("score_definition") != SCORE_DEFINITION:
        raise ValueError("action calibration score definition mismatch")
    if artifact.get("schema") == AUDIT_SCHEMA:
        if (artifact.get("selection_scene_manifest_sha256")
                == artifact.get("audit_scene_manifest_sha256")):
            raise ValueError(
                "selection and audit scene manifests must be disjoint")
        if (artifact.get("selection_partition") != "calibration_select"
                or artifact.get("audit_partition") != "calibration_audit"
                or artifact.get("safety_population")
                != "calibration_audit_only"):
            raise ValueError(
                "v25 calibration must separate selection and audit scenes")
        selection_scenes = {
            str(key) for key in artifact.get("selection_scene_keys", [])}
        audit_scenes = {
            str(key) for key in artifact.get("audit_scene_keys", [])}
        if (not selection_scenes or not audit_scenes
                or selection_scenes.intersection(audit_scenes)):
            raise ValueError(
                "v25 calibration scene populations are empty or overlap")
    elif artifact.get("selection_partition") != "calibration":
        raise ValueError("action calibration must use calibration tracklets")
    if not bool(artifact.get("passed")):
        raise ValueError("action calibration did not pass promotion criteria")
    return artifact


def load_action_calibration(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
