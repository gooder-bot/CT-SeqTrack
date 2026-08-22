"""Compare canonical vs auxiliary B0 view losses in the v24 b0 run."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

RUN = r"output\20260822-0330-24_b0-ct24_b0_car_seed42_60ep_bs16_\lightning_logs\version_0"
TAGS = [
    "loss_loss_b0_candidate0",
    "loss_loss_b0_view1",
    "loss_loss_b0_view2",
    "loss_loss_b0_view3",
    "loss_ct_canonical_b0_weight",
]

for tag in TAGS:
    d = os.path.join(RUN, tag)
    if not os.path.isdir(d):
        print(f"{tag}: MISSING")
        continue
    ea = EventAccumulator(d)
    ea.Reload()
    names = ea.Tags()["scalars"]
    if not names:
        print(f"{tag}: EMPTY")
        continue
    values = [s.value for s in ea.Scalars(names[0])]
    n = len(values)
    print(
        f"{tag}: n={n} mid={values[n//2]:.4f} "
        f"mean_last50={sum(values[-50:])/min(50,n):.4f} last={values[-1]:.4f}"
    )
