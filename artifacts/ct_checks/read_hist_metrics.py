"""Compare B1 mechanism metrics across all historical b1-only runs."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

RUNS = {
    "v24_b1 (a00935a)": r"output\20260822-0330-24_b1-ct24_b1_only_car_seed42_60ep_bs16_\lightning_logs\version_0",
    "ct21_b1_only": r"output\20260813-0119-02_ct_motion_v3-scratch_ct21_b1_only_car_60ep_bs16_s42\lightning_logs\version_0",
    "ct25_b1_gru": r"output\20260819-2043-b1_gru_mini_seed42-ct25_b1_ra_pmm_gru_mini_seed42_60ep_bs16_fixed_20260819-204347\lightning_logs\version_0",
    "ct25_b1_cfc": r"output\20260819-2043-b1_cfc_mini_seed42-ct25_b1_ra_pmm_cfc_mini_seed42_60ep_bs16_fixed_20260819-204347\lightning_logs\version_0",
}
TAGS = [
    "loss_motion_v3_prior_rmse",
    "loss_motion_v3_kinematic_rmse",
    "loss_motion_v3_aux_prior_rmse_gap2",
    "loss_motion_v3_aux_kinematic_rmse_gap2",
    "loss_motion_v3_aux_prior_rmse_gap4",
    "loss_motion_v3_aux_kinematic_rmse_gap4",
    "loss_loss_motion_v3_nll",
    "loss_motion_v3_coverage_50",
    "loss_motion_v3_coverage_95",
    "loss_motion_v3_coverage_ece",
    "loss_motion_v3_sigma_parallel_mean",
    "loss_motion_v3_sigma_perpendicular_mean",
]

for name, run in RUNS.items():
    print(f"== {name} ==")
    if not os.path.isdir(run):
        print("   MISSING RUN DIR")
        continue
    for tag in TAGS:
        d = os.path.join(run, tag)
        if not os.path.isdir(d):
            continue
        ea = EventAccumulator(d)
        ea.Reload()
        names = ea.Tags()["scalars"]
        if not names:
            continue
        values = [s.value for s in ea.Scalars(names[0])]
        if not values:
            continue
        last10 = sum(values[-10:]) / min(10, len(values))
        short = tag.replace("loss_", "").replace("loss", "")
        print(f"   {short:<42} last10={last10:.4f}")
