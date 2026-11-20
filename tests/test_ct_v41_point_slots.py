"""B0 槽位回填的稠密参照、索引来源和可选编译工具链。"""
from types import SimpleNamespace

import pytest
import torch

from models.ct_v41 import packed_points
from models.ct_v41.packed_points import PointSlotLayout


def test_linker_stub_preserves_existing_search_precedence(tmp_path, monkeypatch):
    stub = tmp_path / 'lib64' / 'stubs'
    stub.mkdir(parents=True)
    (stub / 'libcuda.so').touch()
    monkeypatch.setenv('CUDA_PATH', str(tmp_path))
    monkeypatch.setenv('CC', 'cc')
    monkeypatch.setenv('LIBRARY_PATH', '/existing')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/real-driver')
    monkeypatch.setattr(packed_points.subprocess, 'run',
                        lambda *args, **kw: SimpleNamespace(returncode=0, stdout='libcuda.so\n'))
    for _ in range(2):
        packed_points._configure_triton_linker()
    assert packed_points.os.environ['LIBRARY_PATH'].split(packed_points.os.pathsep) == ['/existing', str(stub)]
    assert packed_points.os.environ['LD_LIBRARY_PATH'] == '/real-driver'


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA not available'))])
def test_slot_outputs_and_gradients_match_dense_on_noncontiguous_rows(device):
    valid = torch.tensor([[False, True, False, True], [False]*4, [True, False, True, False]], device=device)
    layout = PointSlotLayout(valid)
    data = torch.tensor([[-torch.inf]*4, [-0., 0., 0., -0.], [float('nan'), 2., 3., float('nan')]],
                        device=device).T
    assert not data.is_contiguous()
    reference = PointSlotLayout(valid, flat_indices=layout.flat_indices)
    results = []
    for impl in (layout, reference):
        rows = data.detach().requires_grad_()
        restored = impl.restore(rows, fill=-2.)
        pooled = impl.max_per_group(rows)
        upstream = torch.arange(pooled.numel(), device=device, dtype=pooled.dtype).view_as(pooled).requires_grad_()
        grad, = torch.autograd.grad((pooled, restored), rows, (upstream, torch.ones_like(restored)), create_graph=True)
        grad_upstream, = torch.autograd.grad(grad.sum(), upstream)
        results.append([x.detach().contiguous().view(torch.int32) for x in (restored, pooled, grad, grad_upstream)])
    for want, got in zip(*results):
        assert torch.equal(want, got)


def test_external_unsorted_indices_keep_original_dense_slot_tie_order():
    valid = torch.tensor([[True, True, False, True]])
    indices = torch.tensor([3, 0, 1])
    rows = torch.tensor([[2.], [2.], [1.]], requires_grad=True)
    layout = PointSlotLayout(valid, flat_indices=indices)
    assert torch.equal(layout.restore(rows).flatten(), torch.tensor([2., 1., 0., 2.]))
    layout.max_per_group(rows).sum().backward()
    assert torch.equal(rows.grad.flatten(), torch.tensor([0., 1., 0.]))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA not available'))])
def test_max_backward_preserves_presence_mask_autograd_checks(device):
    valid = torch.tensor([[True, False, True, False]], device=device)
    rows = torch.randn(2, 3, device=valid.device, requires_grad=True)
    present = valid.any(-1)
    result = PointSlotLayout(valid, present=present).max_per_group(rows)
    present.logical_not_()
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        result.sum().backward()
    with torch.inference_mode():
        inference_mask = valid.any(-1)
    with pytest.raises(RuntimeError, match='Inference tensors cannot be saved'):
        PointSlotLayout(valid, present=inference_mask).max_per_group(rows)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA not available'))])
def test_slot_layout_keeps_forward_ad_and_torch_func_transforms(device):
    valid = torch.tensor([[True, False, True], [False, True, False]], device=device)
    rows = torch.tensor([[1., 3.], [4., 1.], [5., 2.]], device=device)
    tangent = torch.arange(rows.numel(), device=device, dtype=rows.dtype).reshape_as(rows)
    layout = PointSlotLayout(valid)
    external = PointSlotLayout(valid, flat_indices=layout.flat_indices)
    results = []
    for impl in (layout, external):
        def query(x):
            return impl.restore(x).sum(1) + impl.max_per_group(x)
        primal, jvp = torch.func.jvp(query, (rows,), (tangent,))
        gradient = torch.func.grad(lambda x: query(x).sum())(rows)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(rows, tangent)
            dual_result = torch.autograd.forward_ad.unpack_dual(query(dual))
        results.append((primal, jvp, gradient, dual_result.primal, dual_result.tangent))
    for want, got in zip(*results):
        assert torch.equal(want, got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_fast_layout_requires_generated_unmodified_indices_and_compatible_extent(monkeypatch):
    marker = object()
    monkeypatch.setattr(packed_points, '_point_backend', lambda device: marker)
    valid = torch.tensor([[True, False, True, False]], device='cuda')
    rows = torch.randn(2, 3, device=valid.device)
    layout = PointSlotLayout(valid)
    frame = layout.regroup(valid.view(2,2), present=valid.view(2,2).any(-1))
    assert frame._backend(rows) is marker and layout._backend(rows) is marker
    assert PointSlotLayout(valid, flat_indices=layout.flat_indices)._backend(rows) is None
    assert layout._backend(rows[:1]) is None
    assert layout._backend(rows.double()) is None
    assert layout._backend(torch._neg_view(rows)) is None
    assert layout.regroup(valid[:,:2], present=layout.present)._backend(rows) is None
    layout.flat_indices.add_(0)
    assert frame._backend(rows) is None and layout._backend(rows) is None
    layout = PointSlotLayout(valid)
    layout.flat_indices = layout.flat_indices.flip((0,))
    assert layout._backend(rows) is None
    layout = PointSlotLayout(valid)
    layout.slots = 2
    assert layout._backend(rows) is None
    with torch.inference_mode():
        assert PointSlotLayout(valid)._backend(rows) is None
