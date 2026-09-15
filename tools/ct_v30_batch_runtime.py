"""v30 单批检查复用正式 observation collate 与当前 host roll-in；不创建 optimizer。"""
from __future__ import annotations

import copy
from itertools import islice

import torch
from torch.utils.data import DataLoader

from utils.v29_rollin import observation_collate, prepare_observation_batch


def prepare_payload(host, payload):
    if not isinstance(payload, dict) or set(payload) != {'ct_v29_observation_items'}:
        raise ValueError('v30 real-batch checks require the production observation collate')
    with torch.no_grad():
        return prepare_observation_batch(host, payload['ct_v29_observation_items'])


def run_real_observation_batch(config, args, *, time_only=False):
    from datasets import get_dataset
    from models import get_model
    from models.ct_variant import configure_ct_variant
    from utils.online_contract import configure_v28_numerics, validate_scratch_training_contract
    from utils.sampling_utils import StatelessObservationBatchSampler
    from tools.check_forward_batch import summarize_time_fields, summarize_output, has_full_history
    from tools.check_time_batch import summarize_batch

    for flag in ('pseudo_time', 'twc', 'obs_gate'):
        if getattr(args, flag, False):
            raise ValueError(f'v30 check cannot change {flag}; use a registered configuration')
    if args.version is not None and args.version != config.version:
        raise ValueError('v30 --version must match the selected configuration')
    if args.split is not None and args.split != config.train_split:
        raise ValueError('v30 observation check uses the registered training split')
    if args.batch_size is not None and int(args.batch_size) != int(config.batch_size):
        raise ValueError('v30 observation check keeps the registered batch size')
    backend = getattr(args, 'b1_backend', None)
    if backend is not None and backend != config.motion_v3_temporal_backend:
        raise ValueError('v30 backend changes require the corresponding registered arm config')
    if args.path is not None:
        config.path = args.path
    if args.workers < 0 or args.skip_batches < 0 or not 1 <= args.max_batches <= 100:
        raise ValueError('v30 checks require nonnegative workers/skip and max-batches in [1,100]')
    if args.skip_batches >= args.max_batches:
        raise ValueError('--skip-batches must be below --max-batches')
    configure_ct_variant(config)
    configure_v28_numerics(config)
    validate_scratch_training_contract(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    host = get_model(config.net_model)(config).to(device).eval()
    observation_config = copy.deepcopy(config)
    observation_config.ct_variant = 'b0'
    configure_ct_variant(observation_config)
    observation_config.ct_online_recursive_training = False
    dataset = get_dataset(observation_config, type=config.train_type,
                          split=config.train_split, protocol_role='train')
    sampler = StatelessObservationBatchSampler(dataset, int(config.batch_size), int(config.seed))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=observation_collate, pin_memory=False)
    batch = None
    for index, payload in enumerate(islice(loader, args.max_batches)):
        if index < args.skip_batches:
            continue
        candidate = prepare_payload(host, payload)
        if args.require_full_history and not has_full_history(candidate, config.hist_num):
            continue
        batch = candidate
        print(f'v30 real observation batch={index}, device={device}, loader_workers={args.workers}, optimizer_steps=0')
        break
    if batch is None:
        raise RuntimeError('No observation batch matched within the bounded scan')
    summarize_time_fields(batch)
    if time_only:
        summarize_batch(batch, config)
        return batch
    with torch.no_grad():
        output = host(batch)
        summarize_output(output)
        if not args.no_loss:
            losses = host.compute_loss(batch, output)
            for name, value in losses.items():
                if torch.is_tensor(value):
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f'Non-finite observation loss: {name}')
                    print(f'{name}: {float(value):.6f}, finite=True')
    return batch
