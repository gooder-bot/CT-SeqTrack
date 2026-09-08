"""Read-only git/provenance comparison for the v25/v26/v27 B0 audit."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REVS = ("049de82", "b8222bb", "b445ecd", "8b8b8d9")
RUNS = (
    "20260824-0219-25_b0*", "20260825-1931-25_b0*",
    "20260903-2301-26_b0*", "20260905-173106-27_b0*",
)


def source(rev, path):
    return subprocess.check_output(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT,
        text=True, encoding="utf-8")


def functions(text):
    lines = text.splitlines()
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    found[node.name + "." + child.name] = (
                        ast.dump(child, include_attributes=False),
                        lines[child.lineno - 1:child.end_lineno], child.lineno)
        elif isinstance(node, ast.FunctionDef):
            found[node.name] = (ast.dump(node, include_attributes=False),
                                lines[node.lineno - 1:node.end_lineno], node.lineno)
    return found


def main():
    paths = (
        "models/seqtrack3d.py", "models/base_model.py",
        "models/backbone/pointnet.py", "models/attn/Models.py",
        "datasets/sampler.py", "datasets/points_utils.py",
    )
    result = {"revisions": REVS, "function_changes": {}, "config_changes": []}
    for path in paths:
        texts = [source(rev, path) for rev in REVS]
        parsed = [functions(text) for text in texts]
        entries = []
        for i in range(3):
            old, new = parsed[i:i + 2]
            changed = {name: {"old_line": old[name][2], "new_line": new[name][2]}
                       for name in sorted(old.keys() & new.keys())
                       if old[name][0] != new[name][0]}
            entries.append({"from": REVS[i], "to": REVS[i + 1],
                            "changed": changed,
                            "added": sorted(new.keys() - old.keys()),
                            "removed": sorted(old.keys() - new.keys())})
        result["function_changes"][path] = entries
        if path == "models/seqtrack3d.py":
            methods = ("forward", "compute_loss", "training_step",
                       "configure_optimizers", "_finalize_observation_output",
                       "_ct_candidate_weighted_observation_loss")
            diffs = {}
            for method in methods:
                name = "SEQTRACK3D." + method
                diffs[method] = "\n".join(difflib.unified_diff(
                    parsed[1][name][1], parsed[2][name][1],
                    fromfile="b8222bb", tofile="b445ecd", n=3))
            result["v25_low_to_v26_method_diffs"] = diffs
    configs = []
    for pattern in RUNS:
        path = next((ROOT / "output").glob(pattern + "/run_provenance.json"))
        provenance = json.loads(path.read_text(encoding="utf-8"))
        configs.append(provenance["resolved_config"])
    for i in range(3):
        old, new = configs[i:i + 2]
        result["config_changes"].append({
            "from": RUNS[i], "to": RUNS[i + 1],
            "differences": {k: {"old": old.get(k), "new": new.get(k)}
                            for k in sorted(old.keys() | new.keys())
                            if old.get(k) != new.get(k)}})
    result["scope"] = (
        "Static code and actual run configuration comparison; this does not "
        "replay old CUDA training or quantify score causality.")
    destination = OUT / "b0_versions_math_evidence.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
