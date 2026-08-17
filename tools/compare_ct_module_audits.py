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
    print("matched module hashes are identical")


if __name__ == "__main__":
    main()
