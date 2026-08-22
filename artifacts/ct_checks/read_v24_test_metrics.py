"""Summarize B1 mechanism metrics from the v24 b1-only run."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

RUN = r"output\20260822-0330-24_b1-ct24_b1_only_car_seed42_60ep_bs16_\lightning_logs\version_0"
TAGS = [
    "loss_motion_v3_prior_rmse",
    "loss_motion_v3_kinematic_rmse",
    "loss_motion_v3_aux_prior_rmse",
    "loss_motion_v3_aux_kinematic_rmse",
    "loss_loss_motion_v3_nll",
    "loss_motion_v3_coverage_50",
    "loss_motion_v3_coverage_80",
    "loss_motion_v3_coverage_95",
    "loss_motion_v3_coverage_ece",
    "loss_motion_v3_sigma_parallel_mean",
    "loss_motion_v3_sigma_perpendicular_mean",
    "loss_motion_v3_prior_valid_rate",
]

for tag in TAGS:
    d = os.path.join(RUN, tag)
    if not os.path.isdir(d):
        print(f"{tag}: MISSING")
        continue
    ea = EventAccumulator(d)
    ea.Reload()
    scalars = ea.Scalars(tag) if tag in ea.Tags()["scalars"] else []
    if not scalars:
        print(f"{tag}: EMPTY")
        continue
    values = [s.value for s in scalars]
    steps = [s.step for s in scalars]
    last10 = values[-10:]
    print(
        f"{tag}:\n"
        f"  first={values[0]:.4f} mid={values[len(values)//2]:.4f} "
        f"last={values[-1]:.4f} mean_last10={sum(last10)/len(last10):.4f}\n"
        f"  tail: {[round(v,4) for v in values[-6:]]} (steps {steps[-6]}..{steps[-1]})"
    )
