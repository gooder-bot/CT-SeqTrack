import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402


FRAME_COLORS = ["#2f6fbe", "#d08c2f", "#7a5195", "#5aa469", "#999999"]


def load_config(path):
    cfg = EasyDict(load_yaml_config(path))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def oriented_box_xy(center, size, theta):
    width, length = float(size[0]), float(size[1])
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.asarray(center[:2], dtype=np.float32)


def draw_box(ax, box4, size, color, label, linestyle="-", linewidth=2.0):
    xy = oriented_box_xy(box4[:3], size, box4[3])
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.scatter([box4[0]], [box4[1]], color=color, s=18)


def set_equal_xy(ax, points_xy):
    finite = np.isfinite(points_xy).all(axis=1)
    if not np.any(finite):
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return
    xy = points_xy[finite]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def plot_sample(data, output_path, title):
    points = to_numpy(data["points"])
    seg_label = to_numpy(data["seg_label"]).astype(bool)
    valid_mask = to_numpy(data["valid_mask"]).astype(int)
    bbox_size = to_numpy(data["bbox_size"]).reshape(-1)
    box_label = to_numpy(data["box_label"]).reshape(-1)
    ref_boxs = to_numpy(data["ref_boxs"])

    hist_num = int(valid_mask.shape[0])
    frame_count = hist_num + 1
    chunk_size = points.shape[0] // frame_count

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_time, ax_seg = axes

    for frame_idx in range(frame_count):
        start = frame_idx * chunk_size
        end = (frame_idx + 1) * chunk_size
        frame_points = points[start:end]
        if frame_idx < hist_num:
            label = f"hist {frame_idx + 1} valid={valid_mask[frame_idx]}"
        else:
            label = "current"
        ax_time.scatter(
            frame_points[:, 0],
            frame_points[:, 1],
            s=2,
            alpha=0.55,
            color=FRAME_COLORS[frame_idx % len(FRAME_COLORS)],
            label=label,
            rasterized=True,
        )

    ax_seg.scatter(
        points[~seg_label, 0],
        points[~seg_label, 1],
        s=2,
        alpha=0.25,
        color="#9a9a9a",
        label="background",
        rasterized=True,
    )
    ax_seg.scatter(
        points[seg_label, 0],
        points[seg_label, 1],
        s=3,
        alpha=0.8,
        color="#d62728",
        label="target label",
        rasterized=True,
    )

    for idx, ref_box in enumerate(ref_boxs):
        draw_box(
            ax_time,
            ref_box,
            bbox_size,
            color="#444444",
            label="ref boxes" if idx == 0 else None,
            linestyle="--",
            linewidth=1.2,
        )
        draw_box(
            ax_seg,
            ref_box,
            bbox_size,
            color="#444444",
            label="ref boxes" if idx == 0 else None,
            linestyle="--",
            linewidth=1.2,
        )

    draw_box(ax_time, box_label, bbox_size, color="#00a676", label="current GT")
    draw_box(ax_seg, box_label, bbox_size, color="#00a676", label="current GT")

    for ax, subtitle in ((ax_time, "colored by frame"), (ax_seg, "foreground labels")):
        ax.set_title(subtitle)
        ax.set_xlabel("x in recent-ref-box frame")
        ax.set_ylabel("y in recent-ref-box frame")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        set_equal_xy(ax, points[:, :2])
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the processed local point cloud sample used by CT-SeqTrack."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--candidate", type=int, default=None)
    parser.add_argument("--output", default="visualizations/pointcloud_sample.png")
    parser.add_argument("--pseudo-time", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.pseudo_time:
        cfg.use_real_time = False
    cfg.use_twc = False

    split = args.split if args.split is not None else cfg.train_split
    dataset = get_dataset(cfg, type=cfg.train_type, split=split)

    index = int(args.index)
    if args.candidate is not None:
        index = (index // int(cfg.num_candidates)) * int(cfg.num_candidates) + int(args.candidate)

    sample = dataset[index]
    if "view_a" in sample:
        sample = sample["view_a"]

    output_path = Path(args.output)
    title = f"{Path(args.cfg).name} | split={split} | index={index}"
    plot_sample(sample, output_path, title)

    points = to_numpy(sample["points"])
    valid_mask = to_numpy(sample["valid_mask"]).astype(int).tolist()
    print(f"saved: {output_path}")
    print(f"points shape: {points.shape}")
    print(f"valid_mask: {valid_mask}")
    print(f"timestamps: {to_numpy(sample.get('timestamps', []))}")
    print(f"delta_t: {to_numpy(sample.get('delta_t', []))}")
    print(f"current_delta_t: {to_numpy(sample.get('current_delta_t', []))}")


if __name__ == "__main__":
    main()
