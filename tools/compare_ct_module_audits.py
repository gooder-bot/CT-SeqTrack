"""Fail if matched-scratch shared module hashes diverge across checkpoints."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument(
        "--modules", nargs="+", default=("b0", "b1", "b2"))
    args = parser.parse_args()
    audits = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        audit = checkpoint.get("ct_module_audit")
        if not isinstance(audit, dict):
            raise SystemExit(f"{path}: missing ct_module_audit")
        audits.append((path, audit, checkpoint))
    for module in args.modules:
        available = [
            (path, audit["parameter_sha256"].get(module))
            for path, audit, _ in audits
            if module in audit["parameter_sha256"]]
        values = {value for _, value in available}
        if len(values) > 1:
            detail = ", ".join(
                f"{path}={value}" for path, value in available)
            raise SystemExit(f"{module} hash mismatch: {detail}")
    safe = [
        str(audit.get("runtime_protocol", "")).strip().lower()
        == "safe_seqtrack_auto_v1"
        for _, audit, _ in audits]
    if any(safe):
        if not all(safe):
            raise SystemExit("cannot compare v24 and v25 checkpoint audits")
        required = ("initial", "step_1", "step_100")
        for key in required:
            available = []
            for path, _, checkpoint in audits:
                value = checkpoint.get("ct_b0_prefix_hashes", {}).get(key)
                if value is None:
                    raise SystemExit(f"{path}: missing B0 prefix hash {key}")
                available.append((path, value))
            if len({value for _, value in available}) != 1:
                detail = ", ".join(
                    f"{path}={value}" for path, value in available)
                raise SystemExit(f"B0 prefix {key} mismatch: {detail}")
            optimizer_available = []
            for path, _, checkpoint in audits:
                value = checkpoint.get(
                    "ct_b0_optimizer_state_hashes", {}).get(key)
                if value is None:
                    raise SystemExit(
                        f"{path}: missing B0 optimizer-state hash {key}")
                optimizer_available.append((path, value))
            if len({value for _, value in optimizer_available}) != 1:
                detail = ", ".join(
                    f"{path}={value}"
                    for path, value in optimizer_available)
                raise SystemExit(
                    f"B0 optimizer state {key} mismatch: {detail}")
        for path, audit, _ in audits:
            if audit.get("active_frozen_parameters"):
                raise SystemExit(
                    f"{path}: active parameters were frozen")
            for module in audit.get("parameter_groups", []):
                maximum = audit.get("max_gradient_norm", {}).get(module)
                if maximum is None or not torch.isfinite(torch.tensor(
                        float(maximum))) or float(maximum) <= 0.0:
                    raise SystemExit(
                        f"{path}: {module} lacks a finite nonzero gradient")
        fingerprints = []
        for path, _, checkpoint in audits:
            rows = checkpoint.get("ct_observation_batch_fingerprints")
            if not isinstance(rows, list) or len(rows) < 100:
                raise SystemExit(
                    f"{path}: missing first-100 observation fingerprints")
            fingerprints.append((path, rows[:100]))
        if len({repr(rows) for _, rows in fingerprints}) != 1:
            raise SystemExit("first-100 observation fingerprints mismatch")
    print("matched module hashes are identical")


if __name__ == "__main__":
    main()
