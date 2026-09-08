"""Collect fixed-epoch historical B0 scores; never modify experiment output."""
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    dest = Path(__file__).resolve().parent
    root = dest.parents[3]
    records = json.loads((dest / "historical_b0_recheck.json").read_text(encoding="utf-8"))
    rows = []
    for record in records["runs"][3:]:
        name = record["run"]
        nsteps = 1057 if "27_b0" in name else 1262
        events_root = root / "output" / name / "lightning_logs/version_0"
        prefix = next(key[:-8] for key in record["measures"] if key.endswith("_success"))
        values = {}
        for metric in ("success", "precision"):
            accumulator = EventAccumulator(
                str(events_root / (prefix + "_" + metric)), size_guidance={"scalars": 0}
            ).Reload()
            tags = accumulator.Tags()["scalars"]
            assert len(tags) == 1
            metric_values = {}
            for event in accumulator.Scalars(tags[0]):
                assert event.step % nsteps == 0, (name, metric, event.step)
                epoch = event.step // nsteps
                assert epoch not in metric_values, (name, metric, epoch)
                metric_values[epoch] = event.value
            values[metric] = metric_values
        assert values["success"].keys() == values["precision"].keys()
        rows.append({
            "run": name,
            "population": prefix,
            "eval": [
                {"epoch": epoch, "S": values["success"][epoch], "P": values["precision"][epoch]}
                for epoch in (5, 10, 20, 40, 60) if epoch in values["success"]
            ],
        })
    (dest / "b0_version_curves.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
