"""Small YAML composition helper for experiment configurations."""

from pathlib import Path

import yaml


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(file_name, _stack=None):
    """Load a YAML mapping with optional relative ``_base_`` inheritance.

    Existing flat YAML files keep their exact behavior.  New paper-facing
    configs can inherit one or more base files and override only the module
    switches that define an ablation.
    """
    path = Path(file_name).expanduser().resolve()
    stack = list(_stack or [])
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + [path])
        raise ValueError(f"cyclic YAML inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")

    base_entries = payload.pop("_base_", [])
    if isinstance(base_entries, (str, Path)):
        base_entries = [base_entries]
    if not isinstance(base_entries, list):
        raise TypeError(f"_base_ must be a path or list of paths: {path}")

    resolved = {}
    for entry in base_entries:
        base_path = (path.parent / str(entry)).resolve()
        resolved = _deep_merge(
            resolved,
            load_yaml_config(base_path, _stack=stack + [path]),
        )
    return _deep_merge(resolved, payload)
