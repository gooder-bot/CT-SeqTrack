import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from datasets import points_utils  # noqa: E402
from models import get_model  # noqa: E402
from models.ct_variant import configure_ct_variant  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
from utils.metrics import estimateAccuracy, estimateOverlap  # noqa: E402


MODEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


def load_config(path):
    cfg = EasyDict(load_yaml_config(path))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    if str(getattr(cfg, "net_model", "")).strip().lower() == "ctseqtrack":
        configure_ct_variant(cfg)
    return cfg


def apply_dataset_overrides(cfg, args):
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.split is not None:
        cfg.test_split = args.split
    cfg.preloading = bool(args.preloading)
    cfg.workers = 0
    return cfg


def box_bottom_xy(box):
    return box.bottom_corners().T[:, :2]


def draw_box(ax, box, color, label, linestyle="-", linewidth=2.0):
    xy = box_bottom_xy(box)
    xy = np.concatenate([xy, xy[:1]], axis=0)
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    ax.scatter([box.center[0]], [box.center[1]], s=18, color=color)


def set_equal_xy(ax, points_xy, boxes):
    pieces = []
    if points_xy.size > 0:
        finite = np.isfinite(points_xy).all(axis=1)
        if np.any(finite):
            pieces.append(points_xy[finite])
    for box in boxes:
        pieces.append(box_bottom_xy(box))
    if not pieces:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return

    xy = np.concatenate(pieces, axis=0)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def crop_points_for_view(frame, center_box, scale, offset):
    pc = frame["pc"]
    cropped = points_utils.crop_pc_axis_aligned(
        pc,
        center_box,
        scale=scale,
        offset=offset,
    )
    return cropped.points.T[:, :3]


def predict_until(model, sequence, end_frame):
    results_bbs = []
    with torch.no_grad():
        for frame_id in range(min(end_frame + 1, len(sequence))):
            if frame_id == 0:
                results_bbs.append(sequence[frame_id]["3d_bbox"])
                continue

            data_dict, ref_bb = model.build_input_dict(sequence, frame_id, results_bbs)
            if torch.sum(data_dict["points"][:, :, :3]) == 0:
                results_bbs.append(ref_bb)
                continue

            candidate_box, *_ = model.evaluate_one_sample(data_dict, ref_box=ref_bb)
            results_bbs.append(candidate_box)
    return results_bbs


def load_model(label, cfg_path, ckpt_path, backend, args, device):
    cfg = apply_dataset_overrides(load_config(cfg_path), args)
    if backend is not None:
        cfg.motion_v3_temporal_backend = backend
    if str(getattr(cfg, "net_model", "")).strip().lower() == "ctseqtrack":
        configure_ct_variant(cfg)
    model_cls = get_model(cfg.net_model)
    model = model_cls.load_from_checkpoint(
        ckpt_path,
        config=cfg,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return {
        "label": label,
        "cfg": cfg,
        "cfg_path": cfg_path,
        "ckpt_path": ckpt_path,
        "model": model,
    }


def plot_frame(sequence, frame_id, predictions, output_path, crop_scale, crop_offset, iou_space, up_axis):
    frame = sequence[frame_id]
    gt_box = frame["3d_bbox"]
    points = crop_points_for_view(frame, gt_box, crop_scale, crop_offset)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if points.size > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=1,
            color="#9a9a9a",
            alpha=0.35,
            rasterized=True,
            label="points",
        )

    boxes_for_limits = [gt_box]
    draw_box(ax, gt_box, color="#00a676", label="GT", linestyle="-", linewidth=2.4)
    for idx, pred in enumerate(predictions):
        pred_box = pred["boxes"][frame_id]
        boxes_for_limits.append(pred_box)
        iou = estimateOverlap(gt_box, pred_box, dim=iou_space, up_axis=up_axis)
        dist = estimateAccuracy(gt_box, pred_box, dim=3, up_axis=up_axis)
        label = f"{pred['label']} IoU={iou:.2f} D={dist:.2f}m"
        draw_box(
            ax,
            pred_box,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            label=label,
            linestyle="--",
            linewidth=1.8,
        )

    set_equal_xy(ax, points[:, :2] if points.size > 0 else np.zeros((0, 2)), boxes_for_limits)
    ax.set_title(f"tracklet frame {frame_id}")
    ax.set_xlabel("global x")
    ax.set_ylabel("global y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_metrics_csv(sequence, predictions, frame_ids, output_path, iou_space, up_axis):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "model", "iou", "center_distance"],
        )
        writer.writeheader()
        for frame_id in frame_ids:
            gt_box = sequence[frame_id]["3d_bbox"]
            for pred in predictions:
                pred_box = pred["boxes"][frame_id]
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "model": pred["label"],
                        "iou": estimateOverlap(gt_box, pred_box, dim=iou_space, up_axis=up_axis),
                        "center_distance": estimateAccuracy(gt_box, pred_box, dim=3, up_axis=up_axis),
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay GT and multiple CT-SeqTrack checkpoint predictions on the same point-cloud frames."
    )
    parser.add_argument("--cfg", required=True, help="Dataset/default config.")
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tracklet", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-dir", default="visualizations/model_predictions")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--crop-scale", type=float, default=2.0)
    parser.add_argument("--crop-offset", type=float, default=8.0)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("LABEL", "CFG", "CKPT"),
        help="Repeatable: --model A2 cfgs/a2.yaml output/.../last.ckpt",
    )
    parser.add_argument(
        "--model-backend",
        action="append",
        nargs=2,
        metavar=("LABEL", "BACKEND"),
        help=("Optional per-model B1 backend. BACKEND must be gru or cfc; "
              "LABEL must match one --model label."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_cfg = apply_dataset_overrides(load_config(args.cfg), args)
    split = args.split if args.split is not None else dataset_cfg.test_split
    test_dataset = get_dataset(
        dataset_cfg, type="test", split=split, protocol_role="test")
    sequence = test_dataset[args.tracklet]

    end_frame = min(args.end_frame, len(sequence) - 1)
    start_frame = max(args.start_frame, 0)
    frame_ids = list(range(start_frame, end_frame + 1, max(args.frame_stride, 1)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_specs = args.model or []
    model_labels = [label for label, _, _ in model_specs]
    if len(model_labels) != len(set(model_labels)):
        raise ValueError("--model labels must be unique")
    backend_by_label = {}
    for label, backend in args.model_backend or []:
        backend = str(backend).strip().lower()
        if label not in model_labels:
            raise ValueError(
                f"--model-backend label {label!r} has no matching --model")
        if label in backend_by_label:
            raise ValueError(
                f"duplicate --model-backend for label {label!r}")
        if backend not in ("gru", "cfc"):
            raise ValueError("--model-backend BACKEND must be gru or cfc")
        backend_by_label[label] = backend
    predictions = []
    for label, cfg_path, ckpt_path in model_specs:
        item = load_model(
            label,
            cfg_path,
            ckpt_path,
            backend_by_label.get(label),
            args,
            device,
        )
        boxes = predict_until(item["model"], sequence, end_frame)
        predictions.append({"label": label, "boxes": boxes})
        print(f"predicted: {label}")

    output_dir = Path(args.output_dir)
    iou_space = int(getattr(dataset_cfg, "IoU_space", 3))
    up_axis = tuple(getattr(dataset_cfg, "up_axis", [0, 0, 1]))

    for frame_id in frame_ids:
        plot_frame(
            sequence,
            frame_id,
            predictions,
            output_dir / f"track{args.tracklet:03d}_frame{frame_id:04d}.png",
            args.crop_scale,
            args.crop_offset,
            iou_space,
            up_axis,
        )

    write_metrics_csv(
        sequence,
        predictions,
        frame_ids,
        output_dir / f"track{args.tracklet:03d}_metrics.csv",
        iou_space,
        up_axis,
    )

    print(f"saved frames: {len(frame_ids)} -> {output_dir}")
    print(f"saved metrics: {output_dir / f'track{args.tracklet:03d}_metrics.csv'}")


if __name__ == "__main__":
    main()
