"""Read-only module-design probes; no training and no output/ mutation."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from models.ct_v2.evidence_memory import B2EvidenceAcquirer

torch.set_num_threads(1)
votes = torch.tensor([[[4.0, 0.0], [4.1, 0.0], [9.0, 0.0]]])
weights = torch.tensor([[0.8, 0.7, 0.1]])
valid = torch.ones(1, 3, dtype=torch.bool)
consensus_rows = []
for scale in (1.0, 0.01):
    out = B2EvidenceAcquirer._consensus_vote(
        votes, weights * scale, valid, torch.zeros(1, 2))
    consensus_rows.append({
        "input_weight_scale": scale,
        **{key: out[key].detach().tolist() for key in
           ("center", "consistency", "inlier_ratio", "effective_mass")},
    })

margin_logs = []
for run in sorted((ROOT / "output").glob("20260903*")):
    if not any(key in run.name for key in ("b1_gru-", "b1_cfc-", "26_full-")):
        continue
    record = {"run": run.relative_to(ROOT).as_posix(), "scalars": {}}
    for name in ("loss_ct_acquisition_margin_parallel_mean",
                 "loss_ct_acquisition_margin_perpendicular_mean"):
        directory = run / "lightning_logs/version_0" / name
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        record["scalars"][name] = {}
        for tag in accumulator.Tags()["scalars"]:
            rows = accumulator.Scalars(tag)
            record["scalars"][name][tag] = {
                "first_logged_batch": rows[0].value,
                "last_logged_batch": rows[-1].value,
                "max_logged_batch": max(row.value for row in rows),
                "logged_batch_count": len(rows),
            }
    margin_logs.append(record)

report = {
    "consensus_scale_probe": consensus_rows,
    "margin_training_scalars": margin_logs,
    "interpretation_limits": [
        "These batch means are not epoch means or held-out coverage.",
        "The consensus example illustrates scale sensitivity, not a real-scene error rate.",
        "A trained margin head can change its outputs even when prepass omits the field.",
    ],
}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
