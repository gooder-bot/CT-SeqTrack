"""在独立进程中运行冻结源码或当前源码；只为工程对照保存张量。"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=('b1', 'full'), default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=('init', 'deterministic', 'masked', 'natural', 'fp64', 'masked_fp64'), default='init')
    parser.add_argument('--common-state', help='仅测试：每步从冻结对照的上一更新后状态开始')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.root).resolve()))
    import torch
    from models.ct_v38.config import normalize_config
    from models.ct_v38.model import JointTracker
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('synthetic', Path(__file__).with_name('synthetic_batch.py'))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    config = normalize_config(dict(v38_arm=args.arm, ct_engineering_check=True,
        point_sample_size=64, batch_size=4, workers=0, epoch=2))
    torch.manual_seed(args.seed)
    model = JointTracker(config)
    report = dict(initial=copy.deepcopy(model.state_dict()), rng=torch.get_rng_state().clone())
    if args.mode == 'init':
        torch.save(report, args.out)
        return

    # 非零出口让 decoder、local reader、prior 和 verifier 的上游差异可见。
    for name, value in model.named_parameters():
        if name.endswith('weight') and value.ndim > 1 and bool(value.detach().eq(0).all()):
            generator = torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
            with torch.no_grad():
                value.copy_(torch.randn(value.shape, generator=generator) * .01)
    double = args.mode in ('fp64', 'masked_fp64')
    if double:
        model.double()
    controlled = args.mode in ('masked', 'masked_fp64')
    calls = {}
    shapes = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Dropout):
            continue
        if args.mode in ('deterministic', 'fp64'):
            module.p = 0.
        elif controlled:
            def forward(value, *, layer=module, key=name):
                if not layer.training or not layer.p:
                    return value
                index = calls.get(key, 0)
                calls[key] = index + 1
                shape = list(value.shape)
                # 共同 query 总是旧序列前32项，B0 K/V总是旧序列前1024项。
                # 只在测试中生成旧尺寸mask，再按当前有效坐标切片。
                if key.startswith('decoder.decoder.'):
                    shape = [56 if dim == 32 else 1316 if dim == 1024 else dim for dim in shape]
                generator = torch.Generator().manual_seed(
                    int(hashlib.sha256(f'{key}/{index}'.encode()).hexdigest()[:8], 16))
                mask = torch.rand(shape, generator=generator) >= layer.p
                mask = mask[tuple(slice(0, dim) for dim in value.shape)].to(value.device)
                shapes.setdefault(key, []).append(list(value.shape))
                return value * mask.to(value.dtype) / (1. - layer.p)
            module.forward = forward

    def batch(case='normal', *, training=False):
        b, n = (4, 64) if training else (2, 16)
        data = fixtures.batch(b,n)
        for key in ('previous_is_propagation', 'consecutive_propagations',
                    'consecutive_untrusted_outputs', 'trusted_velocity_age'):
            data[key] = torch.zeros(b)
        data['trusted_velocity_valid'] = torch.zeros(b, dtype=torch.bool)
        if case == 'sparse':
            data['point_valid'][:, :, 3:] = False
            data['point_ids'][:, :, 1] = data['point_ids'][:, :, 0]
        elif case == 'history_only':
            data['point_valid'][:, -1] = False
        elif case == 'empty':
            data['point_valid'].zero_()
        elif case == 'short_history':
            data['history_valid'][:, :2] = False
            data['history_pair_valid'].zero_()
        if double:
            data = {key: value.double() if value.is_floating_point() else value for key,value in data.items()}
        return data

    def summarize(output):
        decoder = output.decoder
        compact = hasattr(decoder, 'q0_box')
        return dict(q0=decoder.q0_box if compact else decoder.hypothesis_boxes[:, 0],
            history=decoder.history_boxes,
            quality=decoder.q0_quality_logits if compact else decoder.quality_logits[:, 0],
            prior=output.prior.box, sigma=output.prior.log_sigma,
            candidates=output.candidates.boxes, valid=output.candidates.valid,
            selected=output.selected_index, accepted=output.accepted_box,
            eligible=output.commit_decision.eligible, support=output.commit_decision.support_mask,
            verification=None if output.verification is None else output.verification.logits)

    report['eval'] = {}
    model.eval()
    with torch.no_grad():
        for case in ('normal', 'sparse', 'history_only', 'empty', 'short_history'):
            data = batch(case)
            output = model(data)
            report['eval'][case] = dict(output=summarize(output), losses=model.compute_losses(data, output))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(.5, .999), eps=1e-6, foreach=False)
    report['steps'] = []
    reference = torch.load(args.common_state, weights_only=False) if args.common_state else None
    for step in range(1 if controlled else 3):
        if reference is not None and step > 0:
            active = model.state_dict()
            state = {name:value for name,value in reference['steps'][step-1]['state'].items() if name in active}
            model.load_state_dict(state, strict=True)
        optimizer.zero_grad(set_to_none=True)
        data = batch(training=True)
        output = model(data)
        losses = model.compute_losses(data, output)
        losses['loss_total'].backward()
        gradients = {name: None if value.grad is None else value.grad.detach().clone()
                     for name, value in model.named_parameters()}
        optimizer.step()
        moments = {name: copy.deepcopy(optimizer.state[value]) for name, value in model.named_parameters()}
        report['steps'].append(dict(output=summarize(output), losses=losses, gradients=gradients,
            state=copy.deepcopy(model.state_dict()), optimizer=moments))
    report['controlled_dropout_shapes'] = shapes
    torch.save(report, args.out)


if __name__ == '__main__':
    main()
