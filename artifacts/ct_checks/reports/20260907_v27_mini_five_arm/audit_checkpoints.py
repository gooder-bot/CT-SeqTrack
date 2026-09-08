"""Read-only v27 training-completion and B0-alignment audit (2026-09-07)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event

# Only reconstruct saved hyperparameter dictionaries; do not import or run models.
try:
    import easydict  # noqa: F401
except ImportError:
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error
    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
B0_PREFIXES = ("seg_pointnet.", "mini_pointnet.", "motion_state_mlp.",
               "motion_mlp.", "feature_pointnet.", "Transformer.")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str(x.dtype).encode())
            h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def finite_audit(value, prefix=""):
    bad = []
    count = 0
    if torch.is_tensor(value):
        count += value.numel()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            n, invalid = finite_audit(item, f"{prefix}/{key}")
            count += n
            bad.extend(invalid)
    elif isinstance(value, (list, tuple)):
        for key, item in enumerate(value):
            n, invalid = finite_audit(item, f"{prefix}/{key}")
            count += n
            bad.extend(invalid)
    return count, bad


def scalars(run, name, limit=1500):
    data = []
    for path in sorted((run / "lightning_logs/version_0" / name).glob("events*")):
        for raw in RawEventFileLoader(str(path)).Load():
            event = Event.FromString(raw)
            for value in event.summary.value:
                data.append({"step": event.step, "value": value.simple_value})
            if len(data) >= limit:
                return data[:limit]
    return data


def compare_state(a, b):
    rows = []
    grouped = {}
    for name, x in a.items():
        y = b[name]
        equal = torch.equal(x, y)
        role = "bn_count" if name.endswith("num_batches_tracked") else (
            "bn_stat" if name.endswith(("running_mean", "running_var")) else "parameter")
        g = grouped.setdefault(role, {"tensors": 0, "equal_tensors": 0,
                                     "elements": 0, "different_elements": 0,
                                     "squared_error": 0.0, "squared_reference": 0.0,
                                     "max_abs": 0.0})
        delta = (x.double() - y.double()).abs()
        g["tensors"] += 1
        g["equal_tensors"] += int(equal)
        g["elements"] += x.numel()
        g["different_elements"] += int((x != y).sum())
        g["squared_error"] += float(delta.square().sum())
        g["squared_reference"] += float(x.double().square().sum())
        g["max_abs"] = max(g["max_abs"], float(delta.max()))
        rows.append({"name": name, "role": role, "equal": equal,
                     "max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
    for g in grouped.values():
        g["relative_l2"] = math.sqrt(g.pop("squared_error") / max(g.pop("squared_reference"), 1e-30))
    return {"groups": grouped, "largest_differences": sorted(rows, key=lambda r: r["max_abs"], reverse=True)[:15]}


def main():
    torch.set_num_threads(1)
    result = {"schema": "ct_seqtrack.v27.checkpoint_fairness_audit.v1", "runs": {}, "comparisons_to_b0": {}}
    baseline = None
    for run in sorted((ROOT / "output").glob("20260905-173*-27_*-mini_car_seed42_60ep_bs16")):
        arm = run.name.split("-27_")[1].split("-mini_")[0]
        ckpt = load(run / "formal_checkpoints/epoch=060.ckpt")
        provenance = json.loads((run / "run_provenance.json").read_text(encoding="utf-8"))
        text = (run / "train.log").read_text(encoding="utf-8", errors="replace")
        log_lines = text.replace("\r", "\n").splitlines()
        model_n, model_bad = finite_audit(ckpt["state_dict"])
        optim_n, optim_bad = finite_audit(ckpt["optimizer_states"])
        b0 = {k: v for k, v in ckpt["state_dict"].items() if k.startswith(B0_PREFIXES)}
        groups = []
        optimizer = ckpt["optimizer_states"][0]
        for group in optimizer["param_groups"]:
            states = [optimizer["state"].get(i, {}) for i in group["params"]]
            steps = [int(state["step"].item()) for state in states if "step" in state]
            groups.append({**{k: v for k, v in group.items() if k != "params"},
                           "parameter_tensor_count": len(group["params"]),
                           "adam_state_count": len(steps),
                           "min_parameter_step": min(steps) if steps else None,
                           "max_parameter_step": max(steps) if steps else None})
        record = {
            "run": str(run.relative_to(ROOT)), "epoch": ckpt["epoch"],
            "global_step": ckpt["global_step"], "epoch_boundary_complete": ckpt["ct_epoch_boundary_complete"],
            "stop_markers": [x for x in log_lines if "max_epochs=60" in x],
            "error_markers": [x for x in log_lines if any(t in x for t in ("Traceback (most recent call last)", "RuntimeError:", "CUDA out of memory"))],
            "state_elements_checked": model_n, "state_nonfinite": model_bad,
            "optimizer_elements_checked": optim_n, "optimizer_nonfinite": optim_bad,
            "optimizer_groups": groups, "module_audit": ckpt["ct_module_audit"],
            "b0_prefix_hashes": ckpt["ct_b0_prefix_hashes"],
            "b0_optimizer_hashes": ckpt["ct_b0_optimizer_state_hashes"],
            "observation_fingerprints": ckpt["ct_observation_batch_fingerprints"],
            "rng_hashes": {k: digest(v) for k, v in ckpt["ct_global_rng_state"].items()},
            "bn_count_values": sorted({int(v) for k, v in b0.items() if k.endswith("num_batches_tracked")}),
            "training_streams": provenance["training_streams"], "datasets": provenance["datasets"],
            "git": provenance["git"], "init_checkpoint_path": provenance["init_checkpoint_path"],
            "checkpoint_path": provenance["checkpoint_path"],
            "lr_schedulers": ckpt["lr_schedulers"],
            "b0_first_scalars": {f"view{i}": scalars(run, f"loss_loss_b0_view{i}") for i in range(4)},
            "late3": [],
        }
        for epoch in (58, 59, 60):
            path = run / f"formal_checkpoints/epoch={epoch:03d}.ckpt"
            checkpoint = ckpt if epoch == 60 else load(path)
            record["late3"].append({"filename": path.name, "bytes": path.stat().st_size,
                "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
                "boundary_complete": checkpoint["ct_epoch_boundary_complete"]})
        if baseline is None:
            baseline = {"state": b0, "record": record}
        else:
            other = baseline["record"]
            cmp = compare_state(baseline["state"], b0)
            cmp["observation_first100_equal"] = record["observation_fingerprints"] == other["observation_fingerprints"]
            cmp["initial_b0_equal"] = record["b0_prefix_hashes"]["initial"] == other["b0_prefix_hashes"]["initial"]
            cmp["first_parameter_hash_difference"] = next((key for key in ("initial", "step_1", "step_100")
                if record["b0_prefix_hashes"][key] != other["b0_prefix_hashes"][key]), None)
            cmp["rng_equal"] = {key: value == other["rng_hashes"][key] for key, value in record["rng_hashes"].items()}
            cmp["first_loss_difference"] = {}
            for name, rows in record["b0_first_scalars"].items():
                cmp["first_loss_difference"][name] = next(({
                    "step": a["step"], "b0": a["value"], "arm": b["value"], "abs_difference": abs(a["value"]-b["value"])}
                    for a,b in zip(other["b0_first_scalars"][name], rows)
                    if a != b), None)
            result["comparisons_to_b0"][arm] = cmp
        record["b0_first_scalars"] = {k: v[:12] for k, v in record["b0_first_scalars"].items()}
        # Keep the baseline's longer series until all comparisons are complete.
        if arm == "b0":
            baseline["record"] = dict(record)
            baseline["record"]["b0_first_scalars"] = {f"view{i}": scalars(run, f"loss_loss_b0_view{i}") for i in range(4)}
        result["runs"][arm] = record
        print(arm, record["epoch"], record["global_step"], groups, flush=True)
    (OUT / "checkpoint_fairness.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["comparisons_to_b0"], indent=2))


if __name__ == "__main__":
    main()
