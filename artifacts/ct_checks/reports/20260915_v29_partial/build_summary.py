"""Reproduce the partial-run comparison; never writes to output/ or model code."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv(name):
    with (HERE / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize():
    supply = read_csv("validation/acquisition_supply.csv")
    events = read_csv("validation/epoch_diagnostic_values.csv")
    sampled = read_csv("validation/sampled_batch_summary.csv")
    provenance = json.loads((HERE / "provenance_profile.json").read_text(encoding="utf-8"))
    result = []
    for arm in ("full_cfc", "full_gru"):
        p = next(p for p in provenance if "29_" + arm + "_perf" in p["run"])
        endpoints = p["streams"]["mechanism_prediction_frames"]
        assert endpoints == p["roles"]["train"]["frames"] - p["roles"]["train"]["tracklets"]
        a = next(r for r in supply if r["arm"] == arm and r["epoch"] == "2")
        es = [r for r in events if r["arm"] == arm and r["epoch"] == "2"]

        def metric(suffix):
            matches = [float(r["value"]) for r in es
                       if r["group"] == "ct_epoch_calibration_sampled_" + suffix]
            assert len(matches) == 1, (arm, suffix, matches)
            return matches[0]

        def batch_metric(suffix):
            matches = [r for r in sampled if r["arm"] == arm and r["epoch"] == "2"
                       and r["group"] == "loss_mechanism_" + suffix]
            assert len(matches) == 1, (arm, suffix)
            return float(matches[0]["arithmetic_logged_mean"]), int(matches[0]["logged_batches"])

        n = metric("help_sampled_rows")
        help_n = metric("sampled_help_positive_count")
        harm_n = metric("sampled_harm_positive_count")
        margin, batches = batch_metric("ct_acquisition_margin_perpendicular_mean")
        row = dict(
            arm=arm, completed_epoch=2, mechanism_endpoints=endpoints,
            prepool_target_rows=int(float(a["eligible_rows"])),
            target_rows_per_endpoint=float(a["eligible_rows"]) / endpoints,
            retained_target_rows=int(float(a["retained_rows"])),
            conditional_768_to_256_row_retention=float(a["row_recall"]),
            raw_extension_target_points=int(float(a["pool_targets"])),
            final256_target_points=int(float(a["sampled_targets"])),
            raw_extension_to_256_point_retention=float(a["point_recall"]),
            sampled_legal_candidate_rows=int(n), sampled_help_rows=int(help_n),
            sampled_harm_rows=int(harm_n), sampled_neutral_rows=int(n - help_n - harm_n),
            sampled_help_rate=help_n / n, sampled_harm_rate=harm_n / n,
            sampled_target_present_rate=metric("sampled_presence_positive_count") / n,
            sampled_h1_utility=metric("sampled_bounded_utility_gain_mean"),
            sampled_presence_ap=metric("sampled_presence_ap"),
            sampled_help_ap=metric("sampled_help_ap"),
            sampled_batch_margin_perpendicular_mean=margin,
            logged_mechanism_batches=batches,
        )
        assert abs(row["raw_extension_to_256_point_retention"] -
                   row["final256_target_points"] / row["raw_extension_target_points"]) < 1e-12
        result.append(row)
    (HERE / "comparison_epoch2.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (HERE / "comparison_epoch2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#236b8e", "#ba692d"]
    labels = ["Full-CfC", "Full-GRU"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), layout="constrained")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
        ax.set_ylabel("Percent")
    values = [100 * r["target_rows_per_endpoint"] for r in result]
    bars = axes[0].bar(labels, values, color=colors, width=.55)
    axes[0].bar_label(bars, fmt="%.2f%%", padding=4)
    axes[0].set_ylim(0, 7)
    axes[0].set_title("768 pool contains a target\nAll 191,288 mechanism endpoints", fontsize=10)
    values = [100 * r["raw_extension_to_256_point_retention"] for r in result]
    bars = axes[1].bar(labels, values, color=colors, width=.55)
    axes[1].bar_label(bars, fmt="%.2f%%", padding=4)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Raw extension pool to final 256\nTarget-point retention, two sampling stages", fontsize=10)
    bottom = [0., 0.]
    for key, name, color in [("sampled_help_rows", "Helpful", "#438c71"),
                             ("sampled_neutral_rows", "Neutral", "#d2d6da"),
                             ("sampled_harm_rows", "Harmful", "#ba5c55")]:
        values = [100 * r[key] / r["sampled_legal_candidate_rows"] for r in result]
        bars = axes[2].bar(labels, values, bottom=bottom, color=color, width=.55, label=name)
        axes[2].bar_label(bars, labels=[f"{v:.1f}%" for v in values], label_type="center", fontsize=9)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Sampled bounded actions: immediate utility\n665 / 682 legal candidates; mixed training policy", fontsize=10)
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=8)
    fig.suptitle("v29 perf | completed epoch 2 | training diagnostics, not validation S/P", fontsize=12)
    fig.savefig(HERE / "epoch2_evidence_and_actions.png", dpi=170)
    plt.close(fig)


def notebook():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# v29 partial-run analysis, 2026-09-15\n",
            "Local synchronized logs only. Epoch 2 is the common completed comparison. No official validation result is available.\n",
            "Sources: `provenance_profile.json`, `validation/*.csv`, and the original run paths recorded in provenance.\n",
            "Run `training/extract_training.py` and `validation/analyze_validation.py` from the repository to refresh source extracts.\n",
            "Acquisition counters cover all mechanism endpoints; action diagnostics cover only sampled legal candidates. Never combine their denominators."
        ]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "from pathlib import Path\nimport runpy\n",
            "here = Path.cwd()\n",
            "if not (here / 'build_summary.py').exists():\n",
            "    here = here / 'artifacts/ct_checks/reports/20260915_v29_partial'\n",
            "analysis = runpy.run_path(str(here / 'build_summary.py'))\n",
            "rows = analysis['summarize']()\nrows"
        ]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
            "analysis['plot'](rows)\n",
            "from IPython.display import Image, display\n",
            "display(Image(filename=str(here / 'epoch2_evidence_and_actions.png')))"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": [
            "Interpretation and verified code failure modes are in `REPORT.md` and `method/REPORT.md`.\n",
            "Target presence means at least one GT target point among final selected points, not correct winning consensus.\n",
            "Help/harm is the sign of the mean immediate normalized Success/Precision gain; these rates are not final tracking scores.\n",
            "Synthetic counterexamples from the September 14 audit establish possible failures, not their prevalence in this dataset."
        ]},
    ]
    doc = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, nbformat=4, nbformat_minor=4)
    (HERE / "analysis.ipynb").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rows = summarize()
    plot(rows)
    notebook()
    print(json.dumps(rows, indent=2))
