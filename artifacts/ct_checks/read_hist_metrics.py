"""Scan selected historical runs for Success/Precision scalars for baseline comparison."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

RUNS = {
    "ct22_b0_train": r"output\20260809-2305-22_ct_joint_repaired_b0-ct22_b0_mini_s42_60ep_bs16",
    "ct22_b0_eval": r"output\20260811-1608-22_ct_joint_repaired_b0-ct22_b0_bestdev_minival_s42",
    "ct24_b0_2x2": r"output\ct24_b0_2x2_60ep_seed42",
    "ct25_b0_a": r"output\20260819-1336-25_b0-ct25_b0_mini_seed42_60ep_bs16",
    "ct25_b0_b": r"output\20260819-2036-25_b0-ct25_b0_mini_seed42_60ep_bs16_20260819-203623",
    "b0_2x2_30ep": r"output\20260821-1732-24_b0_2x2_reseed0_rngshift0-f320_ct24_b0_noreseed_car_30ep_bs16_gpu0_20260821-173206",
}

METRIC_HINTS = ("success", "precision")

for name, run in RUNS.items():
    if not os.path.isdir(run):
        print(f"{name} :: MISSING DIR {run}")
        continue
    found = False
    for d in glob.glob(os.path.join(run, "**", "*uccess*"), recursive=True) + glob.glob(
        os.path.join(run, "**", "*recision*"), recursive=True
    ):
        if not os.path.isdir(d):
            continue
        if not glob.glob(os.path.join(d, "events.out.tfevents.*")):
            continue
        ea = EventAccumulator(d)
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            pts = [(e.step, e.value) for e in ea.Scalars(tag)]
            if not pts:
                continue
            found = True
            vals = ", ".join(f"{v:.2f}" for _, v in pts)
            steps = [s for s, _ in pts]
            print(f"{name} :: {os.path.relpath(d, run)} :: tag={tag}")
            print(f"    steps {steps[0]}..{steps[-1]} n={len(pts)}")
            print(f"    values: {vals}")
    if not found:
        print(f"{name} :: no success/precision scalar dirs with events")
