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
    "loss_ct_search_expansion_ratio_mean",
]

for tag in TAGS:
    d = os.path.join(RUN, tag)
    if not os.path.isdir(d):
        print(f"{tag}: MISSING DIR")
        continue
    ea = EventAccumulator(d)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        print(f"{tag}: EMPTY")
        continue
    scalars = ea.Scalars(tags[0])
    values = [s.value for s in scalars]
    n = len(values)
    last10 = sum(values[-10:]) / min(10, n)
    print(
        f"{tag}: n={n} first={values[0]:.4f} "
        f"mean_last10={last10:.4f} last={values[-1]:.4f}"
    )
