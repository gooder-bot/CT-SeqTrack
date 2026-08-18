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
        audits.append((path, audit))
    for module in args.modules:
        available = [
            (path, audit["parameter_sha256"].get(module))
            for path, audit in audits
            if module in audit["parameter_sha256"]]
        values = {value for _, value in available}
        if len(values) > 1:
            detail = ", ".join(
                f"{path}={value}" for path, value in available)
            raise SystemExit(f"{module} hash mismatch: {detail}")
    timelines = []
    for path, audit in audits:
        rows = audit.get("b0_hash_timeline")
        if rows:
            event_map = {str(row["event"]): str(row["b0_sha256"])
                         for row in rows}
            timelines.append((path, event_map))
    if timelines:
        if len(timelines) != len(audits):
            raise SystemExit(
                "B0 hash timeline is missing from one or more checkpoints")
        expected_events = set(timelines[0][1])
        required = {"initialization", "step_1", "step_100"}
        if not required.issubset(expected_events):
            raise SystemExit(
                "B0 hash timeline lacks initialization/step_1/step_100")
        for path, event_map in timelines[1:]:
            if set(event_map) != expected_events:
                raise SystemExit(f"{path}: B0 hash timeline events mismatch")
        for event in sorted(expected_events):
            values = {event_map[event] for _, event_map in timelines}
            if len(values) > 1:
                detail = ", ".join(
                    f"{path}={event_map[event]}"
                    for path, event_map in timelines)
                raise SystemExit(
                    f"B0 hash mismatch at {event}: {detail}")
    print("matched module hashes are identical")


if __name__ == "__main__":
    main()
