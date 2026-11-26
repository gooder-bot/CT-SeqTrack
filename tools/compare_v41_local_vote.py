"""按固定轮次核对四槽局部票的漏搜、持续恢复和轨迹集中度。"""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from statistics import fmean
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.compare_v41_joint_search import EPOCHS, RECEIPTS, load_rows, ordered, read_json, require

FIXED = ("0020", 122)


def sustained_recoveries(rows):
    lost = False
    losses = recovered = streak = 0
    previous = None
    for row in rows:
        if row["initialization"]:
            continue
        if not lost and row["iou"] < 0.1:
            lost = True
            losses += 1
        if previous != row["scan"] - 1 or row["iou"] < 0.5:
            streak = 0
        if lost and row["iou"] >= 0.5:
            streak += 1
            if streak == 3:
                lost = False
                recovered += 1
                streak = 0
        previous = row["scan"]
    return losses, recovered


def track_counts(baseline, trial):
    require(len(baseline) == len(trial) and len(baseline) > 0, "physical track length mismatch")
    counts = Counter(frames=len(baseline), tracks=1, predictions=len(baseline) - 1)
    previous = None
    paired_run = severe_run = 0
    for a, b in zip(baseline, trial):
        require(
            (a["scan"], a["timestamp"], a["initialization"], a.get("raw"))
            == (b["scan"], b["timestamp"], b["initialization"], b.get("raw")),
            "unmatched physical frame",
        )
        for row, name in ((a, "baseline"), (b, "trial")):
            for metric in ("success", "precision"):
                counts[name + "_" + metric] += row[metric]
        if a["initialization"]:
            continue
        misses = [row["raw"] >= 3 and row["union"] == 0 for row in (a, b)]
        counts["baseline_misses"] += misses[0]
        counts["trial_misses"] += misses[1]
        counts["avoided"] += misses[0] and not misses[1]
        counts["added"] += misses[1] and not misses[0]
        severe = a["distance"] > 10
        counts["baseline_over10"] += severe
        counts["trial_over10"] += b["distance"] > 10
        if previous != a["scan"] - 1:
            severe_run = paired_run = 0
        severe_run = severe_run + 1 if severe else 0
        paired_run = paired_run + 1 if severe and b["distance"] <= 2 else 0
        counts["baseline_severe_spells"] += severe_run == 1
        counts["baseline_severe_spells_at_least_three_scans"] += severe_run == 3
        counts["paired_three_scan_runs"] += paired_run == 3
        previous = a["scan"]
    for name, rows in (("baseline", baseline), ("trial", trial)):
        losses, recovered = sustained_recoveries(rows)
        counts[name + "_loss_episodes"] = losses
        counts[name + "_sustained_recovery_episodes"] = recovered
        counts[name + "_unrecovered_at_end"] = losses - recovered
    return counts


def track_keys(counts):
    return {f"{s}/{t}": c["avoided"] - c["added"] for (s, t), c in sorted(counts.items())}


def summarize(epochs):
    """每个排除范围各自重算逐轮分母。共同最大轨由所有给定轮的净减少合计确定。"""
    tracks = sorted(next(iter(epochs.values())))
    require(
        FIXED in tracks and len(tracks) > 1 and all(set(d) == set(tracks) for d in epochs.values()),
        "fixed track missing or epoch track inventories differ",
    )
    totals = {
        t: sum(epoch[t]["avoided"] - epoch[t]["added"] for epoch in epochs.values()) for t in tracks
    }
    largest = max(tracks, key=totals.get)
    net = sum(totals.values())
    scopes = {"all": tuple(), "without_max": (largest,)}
    if largest != FIXED:
        scopes["without_0020_122"] = (FIXED,)
    result = {}
    for name, excluded in scopes.items():
        rows = {}
        for epoch, data in epochs.items():
            counts = Counter()
            for t in tracks:
                if t not in excluded:
                    counts.update(data[t])
            for metric in ("success", "precision"):
                for side in ("baseline", "trial"):
                    counts[side + "_" + metric] *= 100 / counts["frames"]
                counts["delta_" + metric] = counts["trial_" + metric] - counts["baseline_" + metric]
            rows[epoch] = dict(counts)
        result[name] = dict(
            excluded=[f"{s}/{t}" for s, t in excluded],
            epochs=rows,
            epoch_mean={k: fmean(v[k] for v in rows.values()) for k in next(iter(rows.values()))},
        )
    return dict(
        max_track=f"{largest[0]}/{largest[1]}",
        max_net_reduction=totals[largest],
        max_net_share=totals[largest] / net if net > 0 else None,
        total_net_reduction=net,
        fixed_exclusion_scope="without_max" if FIXED == largest else "without_0020_122",
        scopes=result,
    )


def compare(root, receipts):
    frozen = {a["epoch"]: a for a in read_json(receipts)["receipts"] if a["arm"] == "uniform"}
    require(set(frozen) == set(EPOCHS), "missing uniform reference epochs")
    manifest = read_json(root / "run_manifest.json")
    recipe = manifest["actual_recipe"]
    require(
        manifest["run_type"] == "formal"
        and recipe.get("v41_local_vote") is True
        and (recipe["dataset"], recipe["category_name"], recipe["v41_arm"], recipe["epoch"])
        == ("kitti_mf", "Car", "full", 60),
        "expected one formal KITTI Car Full local-vote training run",
    )
    budget = read_json(root / "training_budget.json")
    require(
        budget["completed_epoch"] == 60
        and budget["epoch_complete"] is True
        and budget["auxiliary"]["complete"] is True,
        "E60 main and auxiliary training must be complete",
    )
    epochs, digests = {}, {}
    inventory = None
    for epoch in EPOCHS:
        directory = f"epoch={epoch:03d}"
        report = read_json(root / "epochs" / f"{directory}.json")
        result = report["evaluation"]["test"]
        require(
            report["epoch"] == result["epoch"] == result["checkpoint_epoch"] == epoch
            and result["complete_coverage"]
            and result["diagnostics_level"] == "full"
            and result["split"] == "test",
            f"missing full fixed test evaluation at E{epoch}",
        )
        parts = PurePosixPath(result["frame_file"].replace("\\", "/")).parts[-4:]
        require(
            len(parts) == 4
            and parts[:2] == ("frame_streams", "test")
            and parts[2] not in (".", "..")
            and parts[3] == directory + ".jsonl",
            "unexpected evaluation ledger path",
        )
        old = frozen[epoch]
        a, old_sha = load_rows(receipts.parent / old["archive"], old, joint=False)
        b, new_sha = load_rows(root.joinpath(*parts), result, joint=False)
        require(
            a.keys() == b.keys() and (inventory is None or inventory == a.keys()),
            "frame inventory mismatch",
        )
        inventory = set(a)
        a, b = ordered(a), ordered(b)
        require(len(a) == 120 and sum(map(len, a.values())) == 6424, "expected full KITTI Car test")
        epochs[epoch] = {k: track_counts(rows, b[k]) for k, rows in a.items()}
        digests[epoch] = dict(baseline=old_sha, trial=new_sha)
    return dict(
        run_id=manifest["run_id"],
        paired_ledger_sha256=digests,
        definitions=DEFINITIONS,
        per_track_net_reduction={e: track_keys(v) for e, v in epochs.items()},
        final60=summarize({60: epochs[60]}),
        late3=summarize(epochs),
    )


DEFINITIONS = dict(
    miss=">=3 raw target points and zero target points in actual main+backup ROI union before sampling",
    paired_three_scan_runs="distinct consecutive runs with baseline distance >10 m and trial distance <=2 m for >=3 adjacent raw scans",
    denominator="baseline_severe_spells and baseline_severe_spells_at_least_three_scans; paired runs can recur in one spell",
    sustained_recovery_episodes="per-arm IoU <0.1 starts an episode, three successive adjacent raw scans with IoU >=0.5 end it; denominator is loss_episodes, including unresolved at track end",
    concentration="largest net avoided-minus-added / total net reduction, undefined for nonpositive total; ties choose lowest scene/track",
    late3="same excluded track throughout; scores use physical-frame weighted S/P with first frame, counts exclude initialization",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-receipts", type=Path, default=RECEIPTS)
    args = parser.parse_args()
    print(
        json.dumps(
            compare(args.run, args.baseline_receipts), ensure_ascii=False, indent=2, allow_nan=False
        )
    )
