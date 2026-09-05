import hashlib
import json
from pathlib import Path


VIRTUAL_RATE_DEFAULTS = {
    "virtual_rate_mode": "none",
    "virtual_rate_gap_pattern": [1, 1, 2, 4],
    "virtual_rate_stride": 2,
    "virtual_rate_drop_every": 5,
    "virtual_rate_drop_prob": 0.0,
    "virtual_rate_seed": 42,
    "virtual_rate_max_gap": 5,
    "virtual_rate_keep_first": True,
    "virtual_rate_keep_last": True,
    "virtual_rate_min_tracklet_len": 0,
    "virtual_rate_burst_keep_lengths": [3, 2, 3],
    "virtual_rate_burst_skip_lengths": [2, 3, 3],
}


def _has(config, key):
    if isinstance(config, dict):
        return key in config
    return hasattr(config, key)


def _get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_protocol_role(role):
    role = str(role or "eval").strip().lower().replace("-", "_")
    if role in ("training", "train_motion_mf"):
        return "train"
    if role in ("validation", "valid"):
        return "val"
    if role in ("evaluation", "evaluate"):
        return "eval"
    if role not in ("train", "val", "test", "eval", "calibration", "dev"):
        raise ValueError(f"Unsupported protocol role: {role}")
    return role


def _role_value(config, base_key, role, default=None):
    """Resolve a split-aware setting while preserving legacy configurations.

    Precedence is role-specific, then train/eval-wide, then the legacy unprefixed
    key. Manifest keys additionally accept the requested
    ``virtual_rate_manifest_{train,val,test}`` spelling.
    """
    role = normalize_protocol_role(role)
    candidates = [f"{role}_{base_key}"]
    if base_key == "virtual_rate_manifest":
        candidates.insert(0, f"virtual_rate_manifest_{role}")
    if base_key == "dynamics_time_manifest":
        candidates.insert(0, f"dynamics_time_manifest_{role}")
    if role in ("val", "test", "eval", "calibration", "dev"):
        candidates.append(f"eval_{base_key}")
    elif role == "train":
        candidates.append(f"train_{base_key}")
    candidates.append(base_key)

    for key in candidates:
        if _has(config, key):
            value = _get(config, key)
            if value is not None:
                return value
    return default


def resolve_virtual_rate_kwargs(config, role):
    role = normalize_protocol_role(role)
    values = {
        key: _role_value(config, key, role, default)
        for key, default in VIRTUAL_RATE_DEFAULTS.items()
    }
    values["virtual_rate_manifest"] = _role_value(
        config, "virtual_rate_manifest", role, "")
    values["virtual_rate_manifest_strict"] = bool(_role_value(
        config, "virtual_rate_manifest_strict", role, True))
    values["virtual_rate_manifest_allow_create"] = bool(_role_value(
        config, "virtual_rate_manifest_allow_create", role, False))
    values["virtual_rate_manifest_require_commit_match"] = bool(_role_value(
        config, "virtual_rate_manifest_require_commit_match", role, False))
    values["protocol_role"] = role
    return values


def resolve_dynamics_time_kwargs(config, role):
    role = normalize_protocol_role(role)
    return {
        "dynamics_time_mode": _role_value(config, "dynamics_time_mode", role, "true"),
        "dynamics_fixed_delta_t": float(_role_value(
            config, "dynamics_fixed_delta_t", role,
            _get(config, "default_time_step", _get(config, "time_step", 0.5)))),
        "dynamics_time_manifest": _role_value(
            config, "dynamics_time_manifest", role, ""),
        "dynamics_time_manifest_strict": bool(_role_value(
            config, "dynamics_time_manifest_strict", role, True)),
        "dynamics_time_manifest_require_commit_match": bool(_role_value(
            config, "dynamics_time_manifest_require_commit_match", role, False)),
    }


def canonical_json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def payload_with_content_sha256(payload):
    payload = dict(payload)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def verify_content_sha256(payload, label="manifest"):
    expected = str(payload.get("content_sha256", ""))
    if not expected:
        raise ValueError(f"{label} does not contain content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    actual = canonical_sha256(unhashed)
    if actual != expected:
        raise ValueError(
            f"{label} content SHA256 mismatch: recorded={expected}, actual={actual}")
    return actual


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
