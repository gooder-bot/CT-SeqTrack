"""Capture the resolved CT-SeqTrack v25 configuration baseline.

This intentionally records configuration semantics only.  It is dependency
light, deterministic, and safe to run on machines without CUDA or nuScenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml_config


CONFIG_ROOT = PROJECT_ROOT / "cfgs" / "ct_seqtrack"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "ct_v25_resolved_configs.json"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def build_snapshot():
    configs = {}
    for path in sorted(CONFIG_ROOT.glob("25*.yaml")):
        resolved = load_yaml_config(path)
        configs[path.name] = {
            "key_count": len(resolved),
            "sha256": sha256_json(resolved),
            "resolved": resolved,
        }
    if len(configs) != 16:
        raise RuntimeError(f"expected 16 v25 entry configs, found {len(configs)}")
    return {
        "schema": "ct_seqtrack.v25_resolved_config_baseline.v1",
        "source_commit": git_head(),
        "config_count": len(configs),
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_snapshot(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
