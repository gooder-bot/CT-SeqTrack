"""无 B2 字段的固定合成 batch。"""
import torch

def make_batch(batch_size=2, point_count=16):
    generator = torch.Generator().manual_seed(713)
    b, n = (batch_size, point_count)
    points = torch.randn((b, 4, n, 5), generator=generator) * 0.3
    points[..., 3] = torch.tensor([-1.5, -1.0, -0.5, 0.0])[None, :, None]
    points[..., 4] = 0.5
    boxes = torch.tensor([[[-0.6, 0.0, 0.0, 0.0], [-0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]).repeat(b, 1, 1)
    target = torch.tensor([[0.5, 0.0, 0.0, 0.1]]).repeat(b, 1)
    size = torch.tensor([[4.0, 2.0, 1.5]]).repeat(b, 1)
    valid = torch.ones((b, 4, n), dtype=torch.bool)
    batch = dict(points=points, point_valid=valid, point_ids=torch.arange(n)[None, None].expand(b, 4, -1).clone(), history_boxes=boxes, history_valid=torch.ones(b, 3, dtype=torch.bool), history_pair_valid=torch.ones(b, 2, dtype=torch.bool), history_times=torch.tensor([[-1.5, -1.0, -0.5]]).repeat(b, 1), frame_times=torch.tensor([[-1.5, -1.0, -0.5, 0.0]]).repeat(b, 1), current_dt=torch.full((b,), 0.5), box_size=size, target_box_size=size.clone(), anchor_box=torch.zeros(b, 4), fallback_box=torch.tensor([[0.3, 0.0, 0.0, 0.0]]).repeat(b, 1), target_box=target, history_target_boxes=boxes.clone(), segmentation_labels=torch.ones(b, 4, n, dtype=torch.long), bc_targets=torch.zeros(b, 4, n, 9), physical_displacement=torch.tensor([[0.4, 0.1]]).repeat(b, 1), physical_valid=torch.ones(b, dtype=torch.bool), acquisition_target=torch.full((b, 2), 0.8), acquisition_valid=torch.ones(b, dtype=torch.bool), acquisition_demand=torch.ones(b, dtype=torch.bool))
    return batch

def batch(b=2,n=16):
    values=make_batch(b,n)
    values["history_is_initial"]=torch.tensor([[True,False,False]]).expand(b,-1).clone()
    return values
