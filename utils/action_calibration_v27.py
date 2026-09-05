"""v27 全帧效用校准，真实闭环选择，dev 只诊断，不设风险晋升门。"""
from __future__ import annotations

import math
import json
from pathlib import Path
import numpy as np

from utils.action_calibration import (
    sha256_json, sha256_file,
    action_calibration_config_identity as _legacy_config_identity,
)

SCHEMA = "ct_seqtrack.action_calibration.v27"
ROWS_SCHEMA = "ct_seqtrack.action_rows.v27"
SCORE_DEFINITION = "0.5*(tanh(success_gain_logit)+tanh(precision_gain_logit))"
METRIC_MODE = "benchmark_compat"
COMPARATOR = ">"
CODE_FILES = (
    "models/ct_v2/action_v27.py", "utils/tracking_metrics_v27.py",
    "utils/action_calibration_v27.py", "utils/v27_protocol.py", "utils/v27_training.py",
    "utils/v27_input.py", "utils/v27_diagnostics.py", "utils/v27_evaluation.py",
    "utils/v27_eval_reporting.py", "utils/v27_quality.py", "utils/checkpoint_loading.py", "utils/b1_acquisition.py",
    "utils/point_identity.py", "utils/metrics.py", "utils/action_calibration.py",
    "utils/ct_search.py", "utils/sampling_utils.py", "utils/recursive_state.py",
    "utils/box_membership.py", "utils/training_isolation.py", "utils/online_contract.py",
    "models/seqtrack3d.py", "models/base_model.py", "models/ctseqtrack.py",
    "models/ct_variant.py", "models/ct_v2/evidence_memory.py", "models/ct_v2/evidence_v27.py",
    "models/ct_v2/motion.py", "models/ct_v2/pipeline.py", "models/ct_v2/pipeline_contracts.py",
    "datasets/__init__.py", "datasets/sampler.py", "datasets/nuscenes_lidar_mf.py",
    "datasets/data_classes.py", "datasets/points_utils.py", "datasets/protocol_utils.py",
    "tools/ct_action_v27_runtime.py", "tools/calibrate_ct_actions.py", "tools/export_ct_action_rows.py",
)


def action_calibration_config_identity(config):
    """排除仅决定导出/角色的字段，场景身份由独立 manifest 严格绑定。"""
    values = _legacy_config_identity(config)
    excluded = {"ct_protocol_role", "ct_scene_manifest_path", "ct_scene_manifest_sha256",
                "export_proposal_diagnostics", "export_v3_candidate_diagnostics"}
    return {key: value for key, value in values.items() if key not in excluded}


def code_file_hashes(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in CODE_FILES}


def code_content_sha256(root=None):
    return sha256_json(code_file_hashes(root))


def normalize_policy(policy):
    if not isinstance(policy, dict) or policy.get("kind") not in ("threshold", "always", "never"):
        raise ValueError("action policy must be threshold, always or never")
    kind = policy["kind"]
    if kind != "threshold":
        if set(policy) != {"kind"}:
            raise ValueError("always/never policies cannot carry a threshold")
        return {"kind": kind}
    if set(policy) != {"kind", "threshold"}:
        raise ValueError("threshold policy requires exactly kind and threshold")
    threshold = float(policy["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("policy threshold must be finite; use always/never")
    return {"kind": kind, "threshold": threshold}


def policy_mask(scores, structural_available, policy, is_initial=None):
    policy = normalize_policy(policy)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(structural_available, dtype=bool) & np.isfinite(scores)
    if is_initial is not None:
        valid &= ~np.asarray(is_initial, dtype=bool)
    if policy["kind"] == "never":
        return np.zeros_like(valid)
    if policy["kind"] == "always":
        return valid
    return valid & (scores > policy["threshold"])


def validate_scene_manifest(manifest):
    if manifest.get('version') not in ('v1.0-mini', 'v1.0-trainval'):
        raise ValueError('v27 supports only v1.0-mini or v1.0-trainval')
    if manifest.get("schema") != "ct_seqtrack.scene_protocol.v27":
        raise ValueError("v27 scene manifest schema mismatch")
    body = dict(manifest)
    supplied = body.pop("content_sha256", None)
    if supplied != sha256_json(body):
        raise ValueError("v27 scene manifest hash mismatch")
    scenes = manifest.get("scenes", {})
    roles = {role: set(scenes.get(role, [])) for role in ("train", "calibration", "dev", "test")}
    if any(len(roles[k]) != len(scenes.get(k, [])) for k in roles):
        raise ValueError("duplicate scene IDs")
    mini = manifest.get("version") == "v1.0-mini"
    expected = (6, 1, 1, 2) if mini else (350, 17, 18, 150)
    if tuple(len(roles[k]) for k in roles) != expected:
        raise ValueError("v27 scene role counts do not match mini/full protocol")
    overlap = manifest.get("parameter_training_overlap")
    if type(overlap) is not bool or overlap != (not mini):
        raise ValueError("v27 parameter training overlap declaration mismatch")
    if roles["calibration"] & roles["dev"]:
        raise ValueError("threshold fit and diagnostic scenes must be disjoint")
    if roles["test"] & (roles["train"] | roles["calibration"] | roles["dev"]):
        raise ValueError("official evaluation scenes cannot enter training or tuning")
    if mini and roles["train"] & (roles["calibration"] | roles["dev"]):
        raise ValueError("mini calibration/dev must be held out from all training streams")
    if not mini and not (roles["calibration"] | roles["dev"]) <= roles["train"]:
        raise ValueError("full calibration/dev must be declared training subsets")
    return manifest


def normalize_rows(rows):
    required = ("frame_id", "is_initial", "structural_available", "action_score",
                "observation_success", "observation_precision", "candidate_success",
                "candidate_precision", "final_success", "final_precision", "action_applied")
    normalized, frames = [], {}
    for source in rows:
        row = dict(source)
        for key in required:
            if key not in row:
                raise ValueError(f"v27 endpoint missing {key}")
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"nonfinite v27 endpoint {key}")
        row["frame_id"] = int(row["frame_id"])
        row["tracklet_id"] = str(row["tracklet_id"])
        row["scene_id"] = str(row["scene_id"])
        for key in required:
            if key.endswith("success") or key.endswith("precision"):
                if not 0 <= row[key] <= 1:
                    raise ValueError("per-frame S/P must be in [0,1], not percentages")
        if bool(row["is_initial"]) != (row["frame_id"] == 0):
            raise ValueError("first-frame marker mismatch")
        if row["is_initial"] and (row["structural_available"] or row["action_applied"]):
            raise ValueError("first frame cannot execute an action")
        if row["action_applied"] and row["structural_available"] <= 0:
            raise ValueError("an applied action requires structural evidence")
        frames.setdefault(row["tracklet_id"], []).append(row["frame_id"])
        normalized.append(row)
    if not normalized:
        raise ValueError("v27 endpoints are empty")
    if any(sorted(ids) != list(range(len(ids))) for ids in frames.values()):
        raise ValueError("all contiguous frames including frame0 must be exported once")
    return normalized


def summarize_rows(rows, policy=None):
    """分母含首帧和无候选帧；返回正式百分制 S/P/U。"""
    rows = normalize_rows(rows)
    if policy is None:
        applied = np.asarray([r["action_applied"] > 0 for r in rows])
        success = np.asarray([r["final_success"] for r in rows])
        precision = np.asarray([r["final_precision"] for r in rows])
    else:
        applied = policy_mask([r["action_score"] for r in rows],
                              [r["structural_available"] > 0 for r in rows], policy,
                              [r["is_initial"] > 0 for r in rows])
        success = np.asarray([r["candidate_success"] if a else r["observation_success"]
                              for r, a in zip(rows, applied)])
        precision = np.asarray([r["candidate_precision"] if a else r["observation_precision"]
                                for r, a in zip(rows, applied)])
    gain = np.asarray([.5 * (r["candidate_success"] - r["observation_success"]
                             + r["candidate_precision"] - r["observation_precision"])
                       for r in rows])
    s, p = float(success.mean() * 100), float(precision.mean() * 100)
    result = {"S": s, "P": p, "U": (s + p) / 2, "frames": len(rows),
              "actions": int(applied.sum()), "coverage": float(applied.mean()),
              "tracklets": len({r["tracklet_id"] for r in rows}),
              "scenes": len({r["scene_id"] for r in rows}),
              "action_harm_rate": float((gain[applied] < -1e-6).mean()) if applied.any() else None,
              "action_harm_mean_severity": float(np.maximum(-gain[applied], 0).mean()) if applied.any() else 0.,
              "one_step_net_utility": float((gain * applied).mean())}
    for name in ("success", "precision"):
        difference = np.asarray([r[f"candidate_{name}"] - r[f"observation_{name}"] for r in rows])
        result[f"action_{name}_harm_rate"] = float((difference[applied] < -1e-6).mean()) if applied.any() else None
    if all("exact_final_success" in row and "exact_final_precision" in row for row in rows) and policy is None:
        result["geometry_exact_S"] = 100 * float(np.mean([r["exact_final_success"] for r in rows]))
        result["geometry_exact_P"] = 100 * float(np.mean([r["exact_final_precision"] for r in rows]))
    return result


def paired_scene_bootstrap(rows, baseline_rows, seed=42, resamples=2000):
    """同场景完整闭环差值的经验区间；不设晋升门，不作独立帧保证。"""
    rows, baseline_rows = normalize_rows(rows), normalize_rows(baseline_rows)
    baseline = {(r["tracklet_id"], r["frame_id"]): r for r in baseline_rows}
    if {(r["tracklet_id"], r["frame_id"]) for r in rows} != set(baseline):
        raise ValueError("paired bootstrap requires identical endpoints")
    groups = {}
    for row in rows:
        ref = baseline[(row["tracklet_id"], row["frame_id"])]
        if row["scene_id"] != ref["scene_id"]:
            raise ValueError("paired endpoint scene identity mismatch")
        group = groups.setdefault(row["scene_id"], np.zeros(3))
        group += [row["final_success"] - ref["final_success"],
                  row["final_precision"] - ref["final_precision"], 1.]
    values = np.stack([groups[key] for key in sorted(groups)])
    picked = np.random.default_rng(int(seed)).integers(0, len(values), size=(int(resamples), len(values)))
    samples = values[picked].sum(1)
    differences = samples[:, :2] / samples[:, 2:] * 100
    utility = differences.mean(1)
    return {"method": "paired_scene_percentile_bootstrap", "scenes": len(values),
            "resamples": int(resamples), "seed": int(seed),
            "singleton_scene": len(values) == 1,
            "delta_S_95": np.quantile(differences[:, 0], [.025, .975]).tolist(),
            "delta_P_95": np.quantile(differences[:, 1], [.025, .975]).tolist(),
            "delta_U_95": np.quantile(utility, [.025, .975]).tolist()}


def _rank(summary):
    return summary["U"], summary["S"], summary["P"], -summary["actions"]


def shortlist_policies(rows, quantiles=41, top_k=3):
    rows = normalize_rows(rows)
    valid_scores = [r["action_score"] for r in rows if r["structural_available"] and not r["is_initial"]]
    policies = [{"kind": "never"}, {"kind": "always"}]
    if valid_scores:
        policies += [{"kind": "threshold", "threshold": float(value)}
                     for value in np.unique(np.quantile(valid_scores, np.linspace(0, 1, quantiles)))]
    unique, seen = [], set()
    for policy in policies:
        mask = policy_mask([r["action_score"] for r in rows],
                           [r["structural_available"] > 0 for r in rows], policy,
                           [r["is_initial"] > 0 for r in rows])
        key = mask.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append({"policy": policy, "metrics": summarize_rows(rows, policy)})
    non_never = [item for item in unique if item["policy"]["kind"] != "never"]
    non_never.sort(key=lambda item: _rank(item["metrics"]), reverse=True)
    return non_never[:top_k], unique


def _check_role(rows, role, manifest):
    rows = normalize_rows(rows)
    allowed = set(manifest["scenes"][role])
    if not {r["scene_id"] for r in rows} <= allowed:
        raise ValueError(f"{role} rows contain a scene from another protocol role")
    return rows


def calibrate_actions_v27(calibration_rows, runner, *, checkpoint_sha256,
                          config_sha256, scene_manifest, code_sha256=None):
    """runner(role, policy) 必须重新从首帧执行完整轨迹，返回全帧 endpoint。"""
    validate_scene_manifest(scene_manifest)
    calibration_rows = _check_role(calibration_rows, "calibration", scene_manifest)
    shortlisted, screen = shortlist_policies(calibration_rows)
    policies = [{"kind": "never"}] + [x["policy"] for x in shortlisted]
    evaluated, evaluated_rows = [], {}
    expected_ids = {(r["tracklet_id"], r["frame_id"]) for r in calibration_rows}
    for policy in policies:
        rows = _check_role(runner("calibration", policy), "calibration", scene_manifest)
        if {(r["tracklet_id"], r["frame_id"]) for r in rows} != expected_ids:
            raise ValueError("closed-loop policy runs must evaluate identical complete endpoints")
        evaluated.append({"policy": policy, "metrics": summarize_rows(rows)})
        evaluated_rows[sha256_json(policy)] = rows
    chosen = max(evaluated, key=lambda item: _rank(item["metrics"]))
    # Locked policy only. Never baseline provides interpretable dev net gain; it does not refit.
    dev_rows = _check_role(runner("dev", chosen["policy"]), "dev", scene_manifest)
    dev_baseline = _check_role(runner("dev", {"kind": "never"}), "dev", scene_manifest)
    if {(r["tracklet_id"], r["frame_id"]) for r in dev_rows} != {
            (r["tracklet_id"], r["frame_id"]) for r in dev_baseline}:
        raise ValueError("dev policies must evaluate identical endpoints")
    artifact = {
        "schema": SCHEMA, "action_policy": chosen["policy"],
        "checkpoint_sha256": str(checkpoint_sha256), "config_sha256": str(config_sha256),
        "code_content_sha256": code_sha256 or code_content_sha256(),
        "source_files": list(CODE_FILES),
        "score_definition": SCORE_DEFINITION, "metric_mode": METRIC_MODE,
        "comparator": COMPARATOR, "metric_threshold_count": 21,
        "scene_manifest": scene_manifest,
        "scene_manifest_sha256": scene_manifest["content_sha256"],
        "parameter_training_overlap": scene_manifest["parameter_training_overlap"],
        "selection_role": "training_internal_threshold_fit" if scene_manifest["parameter_training_overlap"] else "held_out_threshold_fit",
        "diagnostic_role": "training_internal_closed_loop_diagnostic" if scene_manifest["parameter_training_overlap"] else "held_out_closed_loop_diagnostic",
        "selection_complete": True, "dev_refit": False,
        "screening": screen, "calibration_closed_loop": evaluated,
        "calibration_selected_metrics": chosen["metrics"],
        "dev_locked_metrics": summarize_rows(dev_rows),
        "dev_never_metrics": summarize_rows(dev_baseline),
        "calibration_gain_interval": paired_scene_bootstrap(
            evaluated_rows[sha256_json(chosen["policy"])],
            evaluated_rows[sha256_json({"kind": "never"})]),
        "dev_gain_interval": paired_scene_bootstrap(dev_rows, dev_baseline),
        "calibration_rows_sha256": sha256_json(calibration_rows),
        "calibration_tracklet_keys_sha256": sha256_json(sorted({r["tracklet_id"] for r in calibration_rows})),
        "dev_tracklet_keys_sha256": sha256_json(sorted({r["tracklet_id"] for r in dev_rows})),
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    return artifact


def validate_action_calibration_v27(artifact, checkpoint_sha256, config_sha256,
                                    scene_manifest_sha256=None, code_sha256=None):
    if artifact.get("schema") != SCHEMA:
        raise ValueError("v27 action calibration schema mismatch")
    body = dict(artifact)
    digest = body.pop("artifact_sha256", None)
    if digest != sha256_json(body):
        raise ValueError("v27 calibration content hash mismatch")
    expected = {"checkpoint_sha256": str(checkpoint_sha256), "config_sha256": str(config_sha256),
                "score_definition": SCORE_DEFINITION, "metric_mode": METRIC_MODE,
                "comparator": COMPARATOR, "metric_threshold_count": 21,
                "code_content_sha256": code_sha256 or code_content_sha256()}
    if scene_manifest_sha256 is not None:
        expected["scene_manifest_sha256"] = str(scene_manifest_sha256)
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"v27 calibration {key} mismatch")
    scene = validate_scene_manifest(artifact["scene_manifest"])
    if (artifact.get("scene_manifest_sha256") != scene["content_sha256"]
            or artifact.get("parameter_training_overlap") != scene["parameter_training_overlap"]):
        raise ValueError("v27 calibration scene provenance mismatch")
    if not artifact.get("selection_complete") or artifact.get("dev_refit") is not False:
        raise ValueError("v27 requires completed closed-loop selection and locked dev evaluation")
    normalize_policy(artifact["action_policy"])
    return artifact


def install_v27_action_calibration(model, config, *, scene_splits=None, code_sha256=None):
    """构造时只安装与 checkpoint/config/source/官方场景清单匹配的策略。

    失败时保留可诊断状态并精确 observation fallback。显式 calibration runner
    之后可 install_policy 执行候选闭环；此处不提供默认 q>0 的隐式部署。
    """
    from utils.v27_protocol import build_scene_manifest
    def get(key, default=None):
        return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
    router = model.ct_joint_router
    router.install_policy({"kind": "never"})
    router.calibrated.fill_(False)
    model._ct_action_calibration = None
    status = {"schema": SCHEMA, "loaded": False, "fallback": "observation", "reason": "missing_calibration_artifact"}
    model._ct_action_calibration_status = status
    path = get("ct_action_calibration_path")
    if not path:
        model._ct_action_calibration_error = status["reason"]
        return status
    try:
        checkpoint = get("checkpoint")
        if not checkpoint:
            raise ValueError("v27 calibration requires the evaluation checkpoint path")
        if scene_splits is None:
            from nuscenes.utils.splits import create_splits_scenes
            scene_splits = create_splits_scenes()
        scene_manifest = build_scene_manifest(scene_splits, get("version"), get("ct_partition_seed", 42))
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_action_calibration_v27(artifact, sha256_file(checkpoint),
            sha256_json(action_calibration_config_identity(config)),
            scene_manifest_sha256=scene_manifest["content_sha256"], code_sha256=code_sha256)
        router.install_policy(artifact["action_policy"])
        model._ct_action_calibration = artifact
        model._ct_action_calibration_error = None
        status.update(loaded=True, fallback=None, reason="ok", action_policy=artifact["action_policy"],
                      artifact_sha256=artifact["artifact_sha256"],
                      parameter_training_overlap=artifact["parameter_training_overlap"],
                      scene_manifest_sha256=scene_manifest["content_sha256"])
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        router.install_policy({"kind": "never"})
        router.calibrated.fill_(False)
        status["reason"] = f"{type(error).__name__}:{error}"
        model._ct_action_calibration_error = status["reason"]
    return status
