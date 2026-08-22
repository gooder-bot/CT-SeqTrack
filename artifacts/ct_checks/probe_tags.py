"""Inspect tags inside one B1 metric event dir, then dump trends."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

RUN = r"output\20260822-0330-24_b1-ct24_b1_only_car_seed42_60ep_bs16_\lightning_logs\version_0"

probe = os.path.join(RUN, "loss_motion_v3_prior_rmse")
ea = EventAccumulator(probe)
ea.Reload()
print("tags:", ea.Tags())
