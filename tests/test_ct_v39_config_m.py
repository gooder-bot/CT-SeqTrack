"""v39 两条调度身份、无点云先验和多步标签边界。"""
from copy import deepcopy
import json

import pytest
import torch

from models.ct_v39.config import (RECIPES, REGISTRATION_SCHEMA, config_identity,
                                  inference_identity, load_config, normalize_config, recipe_identity)
from models.ct_v39.entry import (parse_config, validate_formal_registration, evaluation_plan,
                                _apply_evaluation_policy, source_identity)
from models.ct_v39.multistep import FutureMotionHead, compute_multistep_loss, multistep_loss
from models.ct_v39.prior import PhysicalTimePrior, canonicalize_state


def state_batch(n=2):
    boxes = torch.tensor([[[-2., 0., 0., 0.], [-1., .1, 0., .05], [0., 0., 0., .1]]]).repeat(n, 1, 1)
    return dict(history_boxes=boxes, history_times=torch.tensor([[-1.5, -1., -.5]]).repeat(n, 1),
        history_valid=torch.ones(n, 3, dtype=torch.bool), history_pair_valid=torch.ones(n, 2, dtype=torch.bool),
        current_dt=torch.full((n,), .5), fallback_box=torch.zeros(n, 4),
        box_size=torch.tensor([[4., 2., 1.5]]).repeat(n, 1))


def test_two_formal_recipes_change_only_lr_and_run_labels():
    a, b = [load_config(f'cfgs/ct_seqtrack/39_{name}_mini.yaml') for name in RECIPES]
    allowed = {'cfg', 'experiment_name', 'tag', 'v39_training_recipe', 'lr', 'lr_schedule',
               'lr_milestones', 'lr_decay_rate'}
    assert {k: v for k, v in a.items() if k not in allowed} == {k: v for k, v in b.items() if k not in allowed}
    assert a.v39_attention_backend == b.v39_attention_backend == 'fa2_varlen'
    assert a.v39_feature_bf16 and a.v39_point_packing
    assert config_identity(a) != config_identity(b)
    assert [a.v31_short_window, a.v31_long_window] == [4, 8]
    assert a.v39_m_horizons == [2, 4]


@pytest.mark.parametrize('change', [dict(epoch=59), dict(batch_size=2), dict(lr=1e-4),
    dict(v39_arm='b1'), dict(init_checkpoint='old.ckpt'), dict(v39_attention_backend='auto'),
    dict(v39_feature_bf16=False), dict(v39_m_horizons=[2, 8]), dict(v39_m_loss_weight=.2),
    dict(v39_execution_revision='other'), dict(v39_inference_mode='adaptive'),
    dict(v39_route_threshold=.8), dict(v38_arm='full')])
def test_formal_rejects_unregistered_change(change):
    with pytest.raises(ValueError):
        normalize_config(change)


def test_engineering_cpu_math_and_eval_policy_have_explicit_separate_identity():
    cfg = normalize_config(dict(ct_engineering_check=True, epoch=1, workers=0,
        v39_attention_backend='math_reference', v39_feature_bf16=False, execution_profile='optimized_fp32'))
    assert cfg.v39_attention_backend == 'math_reference'
    full = normalize_config()
    evaluation = normalize_config(dict(full, test=True, checkpoint='same_run.ckpt',
        v39_inference_mode='adaptive', v39_route_threshold=.95))
    assert config_identity(full) == config_identity(evaluation)
    assert inference_identity(full) != inference_identity(evaluation)


def test_cli_requires_explicit_flags_and_allows_evaluation_policy():
    path = 'cfgs/ct_seqtrack/39_full_a_mini.yaml'
    cfg = parse_config(['--cfg', path, '--test', '--checkpoint', 'same_run.ckpt',
                        '--inference_mode', 'adaptive', '--route_threshold', '.95'])
    assert cfg.v39_inference_mode == 'adaptive' and cfg.v39_route_threshold == .95
    with pytest.raises(SystemExit):
        parse_config(['--cfg', path, '--bat', '8'])


def test_fixed_late3_dual_mode_and_final_threshold_curve(tmp_path):
    jobs = evaluation_plan(normalize_config(), tmp_path)
    assert len(jobs) == 8
    for mode in ('accuracy', 'adaptive'):
        main = [job for job in jobs if job['label'] == mode]
        assert [job['epoch'] for job in main] == [58, 59, 60]
        assert all(job['threshold'] == .9 for job in main)
    extra = [job for job in jobs if job['label'].startswith('adaptive_threshold_')]
    assert [(job['epoch'], job['threshold']) for job in extra] == [(60, .8), (60, .95)]
    explicit = normalize_config(dict(test=True, checkpoint='epoch60.ckpt',
        v39_inference_mode='adaptive', v39_route_threshold=.95))
    assert len(evaluation_plan(explicit, tmp_path)) == 1


def test_eval_mode_reaches_model_and_builders_not_only_host_log(tmp_path):
    from types import SimpleNamespace
    owners = [SimpleNamespace(config=dict(normalize_config())) for _ in range(4)]
    host = owners[0]
    host.tracker, host.train_builder, host.evaluation_builder = owners[1:]
    job = evaluation_plan(normalize_config(), tmp_path)[-1]
    _apply_evaluation_policy(host, job)
    assert all(owner.config['test'] and owner.config['v39_inference_mode'] == 'adaptive'
               and owner.config['v39_route_threshold'] == .95 for owner in owners)


def test_source_inventory_excludes_only_explicit_retired_versions(tmp_path):
    from models.ct_v39.identity import RETIRED_SOURCE_DIRECTORIES, RETIRED_SOURCE_FILES
    for name in ('main.py', 'requirement.txt', 'models/ct_v39/model.py',
                 'models/ct_v38/prior.py', 'models/seqtrack_reference/model.py',
                 'datasets/reader.py', 'utils/helper.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original', encoding='utf-8')
    before = source_identity(tmp_path)
    for directory in RETIRED_SOURCE_DIRECTORIES:
        path = tmp_path / directory / 'retired_helper.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('unused remote legacy', encoding='utf-8')
    for name in RETIRED_SOURCE_FILES:
        (tmp_path / name).write_text('unused remote host', encoding='utf-8')
    assert source_identity(tmp_path) == before
    active = tmp_path / 'models/ct_v38/prior.py'
    active.write_text('active dependency changed', encoding='utf-8')
    changed = source_identity(tmp_path)
    assert changed['sha256'] != before['sha256']
    active.write_text('original', encoding='utf-8')
    unknown = tmp_path / 'models/ct_v999/new.py'
    unknown.parent.mkdir()
    unknown.write_text('unknown is protected, not ignored', encoding='utf-8')
    assert source_identity(tmp_path)['sha256'] != before['sha256']
    unknown.unlink()
    # 近似文件名不属于固定退役名单。
    (tmp_path / 'models/ctseqtrackv37_extra.py').write_text('still active inventory', encoding='utf-8')
    assert source_identity(tmp_path)['sha256'] != before['sha256']


def test_registration_requires_both_exact_configs_and_source(tmp_path):
    configs = [load_config(f'cfgs/ct_seqtrack/39_{name}_mini.yaml') for name in RECIPES]
    source = {'sha256': 'f'*64, 'files': {'model.py': 'a'*64}}
    rows = [dict(label=cfg.v39_training_recipe, arm='full', config_sha256=config_identity(cfg),
                 execution_profile=cfg.execution_profile, training_execution_recipe=recipe_identity(cfg)) for cfg in configs]
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(dict(schema=REGISTRATION_SCHEMA, source=source, runs=rows)), encoding='utf-8')
    assert validate_formal_registration(configs[0], source, registration_path=path)['label'] == 'full_a'
    with pytest.raises(ValueError, match='source differs'):
        validate_formal_registration(configs[0], {'sha256': 'changed'}, registration_path=path)
    path.write_text(json.dumps(dict(schema=REGISTRATION_SCHEMA, source=source, runs=rows[:1])), encoding='utf-8')
    with pytest.raises(ValueError, match='exactly two'):
        validate_formal_registration(configs[0], source, registration_path=path)


@pytest.mark.parametrize('recipe,expected', [
    ('full_a', [1.5e-4]*20 + [1.5e-5]*30 + [1.5e-6]*10),
    ('full_b', [1e-4]*20 + [2e-5]*20 + [4e-6]*20)])
def test_sixty_epoch_schedules_resume_at_all_boundaries(recipe, expected):
    cfg = normalize_config(dict(v39_training_recipe=recipe))
    def make():
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([parameter], lr=cfg.lr)
        scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_milestones, cfg.lr_decay_rate)
                     if cfg.lr_schedule == 'multistep'
                     else torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_decay_step, cfg.lr_decay_rate))
        return optimizer, scheduler
    optimizer, scheduler = make()
    rates = []
    for epoch in range(60):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        if epoch + 1 in (20, 40, 50):
            saved_optimizer, saved_scheduler = deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
            optimizer, scheduler = make()
            optimizer.load_state_dict(saved_optimizer)
            scheduler.load_state_dict(saved_scheduler)
            assert scheduler.last_epoch == epoch + 1
    assert rates == pytest.approx(expected)
    if recipe == 'full_b':
        assert optimizer.param_groups[0]['lr'] == pytest.approx(8e-7)


def test_prior_needs_no_points_and_ignores_future_labels():
    torch.manual_seed(2)
    prior = PhysicalTimePrior()
    batch = state_batch()
    result = prior(batch)
    modified = dict(batch, future_dt=torch.full((2, 2), 1e6),
                    future_displacement=torch.full((2, 2, 2), float('nan')),
                    points='must_not_be_read')
    again = prior(modified)
    assert set(canonicalize_state(modified)).isdisjoint({'points', 'future_dt', 'future_displacement'})
    assert not any('acquisition' in name for name in prior.state_dict())
    torch.testing.assert_close(result.feature, again.feature, rtol=0, atol=0)
    torch.testing.assert_close(result.mean_xy, again.mean_xy, rtol=0, atol=0)
    assert result.feature.requires_grad and not result.box.requires_grad


def test_removed_acquisition_preserves_strong_one_step_prior_math():
    from models.ct_v38.prior import PhysicalTimePrior as FrozenPrior
    torch.manual_seed(5)
    old = FrozenPrior()
    new = PhysicalTimePrior()
    new.load_state_dict({k: v for k, v in old.state_dict().items() if not k.startswith('acquisition_head.')})
    batch = state_batch()
    old_batch = dict(batch, points=torch.zeros(2, 4, 6, 5), point_valid=torch.ones(2, 4, 6, dtype=torch.bool))
    before, after = old(old_batch), new(batch)
    for name in ('feature', 'mean_xy', 'log_sigma', 'valid', 'direction_xy', 'kinematic_xy',
                 'envelope', 'unit_residual', 'box'):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)


def test_multistep_endpoint_and_horizon_normalization_masks_tail():
    predicted = torch.tensor([[[2., 2.], [float('nan'), float('nan')]], [[0., 0.], [2., 2.]]], requires_grad=True)
    target = torch.zeros_like(predicted)
    valid = torch.tensor([[True, False], [True, True]])
    loss = multistep_loss(predicted, target, valid, torch.ones(2, 3))
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 1].abs().sum() == 0
    only_second = multistep_loss(predicted.detach(), target, valid, torch.ones(2, 3), torch.tensor([False, True]))
    torch.testing.assert_close(only_second, torch.tensor(.5))


def test_multistep_live_context_gradient_and_empty_tail_are_finite():
    batch = state_batch()
    prior = PhysicalTimePrior()
    head = FutureMotionHead()
    # 首次零初始化的输出层尚未向 context 反传；一次更新后必须可以反传共享表示。
    with torch.no_grad():
        head.head[-1].weight.fill_(.01)
    result = prior(batch)
    result.feature.retain_grad()
    batch.update(future_dt=torch.tensor([[1., 2.], [1., 2.]]),
                 future_displacement=torch.ones(2, 2, 2), future_valid=torch.ones(2, 2, dtype=torch.bool))
    terms = compute_multistep_loss(batch, result, head)
    terms['loss_contribution_m'].backward()
    assert result.feature.grad.abs().sum() > 0
    assert terms['m_h2_valid'] == 2 and terms['m_h4_valid'] == 2
    prior.zero_grad(); head.zero_grad()
    batch['future_valid'].zero_()
    batch['future_dt'].fill_(float('nan'))
    batch['future_displacement'].fill_(float('nan'))
    empty = compute_multistep_loss(batch, prior(batch), head)
    assert empty['loss_m'].requires_grad and empty['loss_m'].item() == 0.
    empty['loss_m'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_multistep_uses_physical_horizon_without_one_step_cap():
    head = FutureMotionHead()
    with torch.no_grad():
        head.head[-1].bias.fill_(20.)
    prediction = head(torch.zeros(1, 128), torch.tensor([.5]), torch.tensor([[1., 2.]]), torch.ones(1, 3))
    assert prediction.shape == (1, 2, 2) and prediction.min() > 12.
