import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from models import get_model  # noqa: E402


def load_config(path):
    with open(path, "r") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=False)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def is_paired_batch(batch):
    return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch


def has_full_history(batch, hist_num):
    if is_paired_batch(batch):
        return has_full_history(batch["view_a"], hist_num) and has_full_history(batch["view_b"], hist_num)
    valid_mask = to_numpy(batch["valid_mask"])
    if valid_mask.ndim > 1:
        return bool(valid_mask[0].sum() >= int(hist_num))
    return bool(valid_mask.sum() >= int(hist_num))


def unwrap_optimizer(config_result):
    if isinstance(config_result, torch.optim.Optimizer):
        return config_result, None
    if isinstance(config_result, dict):
        optimizer = config_result["optimizer"]
        scheduler_cfg = config_result.get("lr_scheduler")
        scheduler = None
        if isinstance(scheduler_cfg, dict):
            scheduler = scheduler_cfg.get("scheduler")
        else:
            scheduler = scheduler_cfg
        return optimizer, scheduler
    if isinstance(config_result, (list, tuple)):
        optimizer = config_result[0][0] if isinstance(config_result[0], (list, tuple)) else config_result[0]
        scheduler = None
        if len(config_result) > 1:
            scheduler = config_result[1][0] if isinstance(config_result[1], (list, tuple)) else config_result[1]
        return optimizer, scheduler
    raise TypeError(f"Unsupported optimizer config type: {type(config_result)}")


def grad_stats(model):
    total_sq_norm = 0.0
    max_abs = 0.0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total_sq_norm += float(torch.sum(grad.float() ** 2).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))
    return total_sq_norm ** 0.5, max_abs, finite


def freeze_batchnorm_stats(model):
    frozen = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            module.eval()
            frozen += 1
    return frozen


def save_checkpoint(path, model, optimizer, scheduler, step, cfg, loss_record):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": dict(cfg),
        "loss_record": loss_record,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def build_loader(cfg, args):
    split = args.split if args.split is not None else cfg.train_split
    dataset = get_dataset(cfg, type=cfg.train_type, split=split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=not args.no_shuffle,
        drop_last=True,
        pin_memory=False,
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--pseudo-time", action="store_true")
    parser.add_argument("--twc", action="store_true",
                        help="Temporarily enable P4 paired-view TWC batch mode.")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--train-bn", action="store_true",
                        help="Keep BatchNorm layers in train mode. By default BN stats are frozen so batch_size=1 works.")
    parser.add_argument("--log-file", default="output/check_train_steps_loss.jsonl")
    parser.add_argument("--checkpoint-dir", default="output/check_train_steps_ckpt")
    parser.add_argument("--tag", default="check_train_steps")
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.pseudo_time:
        cfg.use_real_time = False
    if args.twc:
        cfg.use_twc = True
        cfg.twc_candidate_zero_only = True
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    cfg.tag = args.tag

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because compute_loss currently creates CUDA tensors.")

    if args.memory_fraction is not None:
        if not (0.0 < args.memory_fraction <= 1.0):
            raise ValueError("--memory-fraction must be in (0, 1].")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)

    device = torch.device("cuda:0")
    print(f"device: {device}")
    print(f"max_steps: {args.max_steps}")
    print(f"batch_size: {cfg.batch_size}, workers: {cfg.workers}")
    print(f"use_real_time: {getattr(cfg, 'use_real_time', True)}")
    print(f"use_twc: {getattr(cfg, 'use_twc', False)}")
    if args.memory_fraction is not None:
        print(f"cuda memory fraction limit: {args.memory_fraction}")

    loader = build_loader(cfg, args)
    train_dataloader_length = max(len(loader), 1)

    model = get_model(cfg.net_model)(cfg, train_dataloader_length=train_dataloader_length).to(device)
    model.train()
    if not args.train_bn:
        frozen_bn = freeze_batchnorm_stats(model)
        print(f"frozen BatchNorm modules: {frozen_bn}")
    optimizer, scheduler = unwrap_optimizer(model.configure_optimizers())

    log_path = Path(args.log_file)
    ckpt_dir = Path(args.checkpoint_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    completed_steps = 0
    seen_batches = 0

    with log_path.open("a", buffering=1) as log_file:
        for batch_idx, batch in enumerate(loader):
            if batch_idx < args.skip_batches:
                continue

            if args.require_full_history:
                if not has_full_history(batch, cfg.hist_num):
                    continue

            seen_batches += 1
            batch = move_to_device(batch, device)

            if not args.train_bn:
                freeze_batchnorm_stats(model)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss_dict = model.compute_loss(batch, output)
            loss = loss_dict["loss_total"]

            loss_finite = bool(torch.isfinite(loss).all().item())
            finite_by_key = {
                key: bool(torch.isfinite(value).all().item())
                for key, value in loss_dict.items()
                if torch.is_tensor(value)
            }
            if not loss_finite or not all(finite_by_key.values()):
                record = {
                    "step": completed_steps,
                    "batch_idx": batch_idx,
                    "ok": False,
                    "reason": "non_finite_loss",
                    "loss": {
                        key: float(value.detach().cpu().item())
                        for key, value in loss_dict.items()
                        if torch.is_tensor(value) and value.numel() == 1
                    },
                    "finite": finite_by_key,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, completed_steps, cfg, record)
                raise RuntimeError(f"Non-finite loss at batch_idx={batch_idx}: {record}")

            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm, grad_max_abs, grad_finite = grad_stats(model)
            if not grad_finite:
                raise RuntimeError(f"Non-finite gradient at batch_idx={batch_idx}.")

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            completed_steps += 1
            loss_values = {
                key: float(value.detach().cpu().item())
                for key, value in loss_dict.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            record = {
                "step": completed_steps,
                "batch_idx": batch_idx,
                "ok": True,
                "loss": loss_values,
                "finite": finite_by_key,
                "grad_norm": grad_norm,
                "grad_max_abs": grad_max_abs,
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, completed_steps, cfg, record)
            print(
                f"step={completed_steps}/{args.max_steps} "
                f"batch_idx={batch_idx} "
                f"loss_total={loss_values['loss_total']:.6f} "
                f"grad_norm={grad_norm:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if completed_steps >= args.max_steps:
                break

    if completed_steps == 0:
        raise RuntimeError(
            f"No train step was executed. seen_batches={seen_batches}. "
            "Try removing --require-full-history or changing --split/--skip-batches."
        )

    print("finished train-step check")
    print(f"loss log: {log_path}")
    print(f"last checkpoint: {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
