"""探索分支的 CUDA 排序 dtype 回归；CPU 也能截获 bool 排序错误。"""
import pytest
import torch

from utils.v30_policy import choose_mode_action


def _all_validity_patterns(device='cpu'):
    observation = torch.zeros(64, 4, device=device)
    observation[:, 2:] = torch.tensor([.3, .7], device=device)
    actions = observation[:, None].expand(-1, 6, -1).clone()
    actions[:, :, 0] = torch.arange(1, 7, device=device) / 10
    bits = torch.arange(6, device=device)
    valid = ((torch.arange(64, device=device)[:, None] >> bits) & 1).bool()
    # 探索动作由真实几何合法性决定，不要求 B3 q 有效。
    scores = torch.full((64, 6), float('nan'), device=device)
    return observation, actions, valid, scores


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4, 5, 2 ** 32 - 1])
def test_exploration_preserves_slot_order_for_every_validity_pattern(monkeypatch, seed):
    original_argsort = torch.argsort
    sort_dtypes = []

    def cuda_compatible_argsort(values, *args, **kwargs):
        # 原代码在 CPU 可运行，却会在 CUDA 失败；这里直接拦截相同调用。
        assert values.dtype != torch.bool, 'CUDA exploration must never sort bool tensors'
        sort_dtypes.append(values.dtype)
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(torch, 'argsort', cuda_compatible_argsort)
    observation, actions, valid, scores = _all_validity_patterns()
    rng = torch.get_rng_state().clone()
    result = choose_mode_action(observation, actions, valid, scores,
                                {'kind': 'explore', 'action_seed': seed})
    expected = []
    for row in range(64):
        slots = [slot for slot in range(6) if row & (1 << slot)]
        expected.append(slots[(seed + row * 2654435761) % len(slots)] if slots else -1)
    assert torch.equal(result['chosen_action_index'], torch.tensor(expected))
    assert torch.equal(result['applied'], valid.any(1))
    assert torch.equal(result['action_valid'], valid)
    assert torch.equal(result['final_box'][0], observation[0])
    assert torch.equal(result['final_box'][:, 2:], observation[:, 2:])
    rows = torch.arange(1, 64)
    assert torch.equal(result['final_box'][1:, :2], actions[rows, torch.tensor(expected[1:]), :2])
    assert sort_dtypes[-1] == torch.int64
    assert torch.equal(rng, torch.get_rng_state())


def test_exploration_excludes_nonfinite_and_zero_displacements_without_gradients():
    observation, actions, valid, scores = [value[-1:].clone() for value in _all_validity_patterns()]
    actions[0, 0, :2] = observation[0, :2]
    actions[0, 1, 0] = float('nan')
    actions[0, 2, 1] = float('inf')
    observation.requires_grad_()
    actions.requires_grad_()
    scores.requires_grad_()
    for seed in range(6):
        result = choose_mode_action(observation, actions, valid, scores,
                                    {'kind': 'explore', 'action_seed': seed})
        assert result['chosen_action_index'].item() == 3 + seed % 3
        assert torch.equal(result['action_valid'], torch.tensor([[False, False, False, True, True, True]]))
        assert all(not value.requires_grad for value in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_exploration_cuda_matches_cpu_with_strict_determinism():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        cpu = _all_validity_patterns()
        cuda = tuple(value.cuda() for value in cpu)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        for seed in (0, 1, 5, 2 ** 32 - 1):
            policy = {'kind': 'explore', 'action_seed': seed}
            expected = choose_mode_action(*cpu, policy)
            actual = choose_mode_action(*cuda, policy)
            repeated = choose_mode_action(*cuda, policy)
            for key in expected:
                assert torch.equal(actual[key].cpu(), expected[key]), key
                assert torch.equal(actual[key], repeated[key]), key
        torch.cuda.synchronize()
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)
