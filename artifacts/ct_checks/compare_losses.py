"""Compare late-training loss curves: ct22_b0 (contract v2) vs v24_b0 (contract v3)."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

RUNS = {
    "ct22_b0": r"output\20260809-2305-22_ct_joint_repaired_b0-ct22_b0_mini_s42_60ep_bs16\lightning_logs\version_0",
    "v24_b0": r"output\20260822-0330-24_b0-ct24_b0_car_seed42_60ep_bs16_\lightning_logs\version_0",
}
TAGS = [
    "loss_loss_total",
    "loss_loss_seg",
    "loss_loss_center",
    "loss_loss_angle",
    "loss_loss_bc",
    "loss_obs_mean_fg_score",
    "loss_obs_soft_fg_count_mean",
    "loss_obs_num_points_search_mean",
]

for name, run in RUNS.items():
    print(f"== {name} ==")
    for tag in TAGS:
        d = os.path.join(run, tag)
        if not os.path.isdir(d):
            print(f"  {tag}: MISSING")
            continue
        ea = EventAccumulator(d)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if not tags:
            print(f"  {tag}: EMPTY")
            continue
        values = [s.value for s in ea.Scalars(tags[0])]
        n = len(values)
        last10 = sum(values[-10:]) / min(10, n)
        mid = values[n // 2]
        print(f"  {tag}: mid={mid:.4f} mean_last10={last10:.4f} last={values[-1]:.4f}")
