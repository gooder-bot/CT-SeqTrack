import numpy as np
import torch

def resolve_kitti_hv_search_offset(config, frame=None):
    """Return the search crop offset for a KITTI/KITTI-HTV sample.

    Standard SeqTrack3D uses ``bb_offset``.  KITTI-HTV expands the search
    region as the frame interval grows; the role-independent mapping is kept
    in YAML so training and autoregressive evaluation use exactly the same
    protocol.
    """
    default = float(getattr(config, "bb_offset", 2.0))
    mapping = getattr(config, "kitti_hv_search_offsets", None)
    if not mapping or frame is None:
        return default

    interval = frame.get("kitti_hv_interval")
    if interval is None:
        interval = frame.get("meta", {}).get("kitti_hv_interval", 1)
    try:
        interval = int(interval)
    except (TypeError, ValueError):
        return default

    if interval in mapping:
        return float(mapping[interval])
    if str(interval) in mapping:
        return float(mapping[str(interval)])
    return default


def get_history_frame_ids_and_masks(this_frame_id, hist_num):
    history_frame_ids = []
    masks = []
    for i in range(1, hist_num + 1):
        frame_id = this_frame_id - i
        if frame_id < 0:
            frame_id = 0
            masks.append(0)
        else:
            masks.append(1)
        history_frame_ids.append(frame_id)
    return history_frame_ids, masks


def generate_timestamp_prev_list(valid_mask, point_sample_size):
    timestamp_prev_list = []
    valid_time = 0

    for mask in valid_mask:
        if mask == 1:
            valid_time -= 0.1
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time)
        else:
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time)
        timestamp_prev_list.append(timestamp_prev)
    
    return timestamp_prev_list


def get_last_n_bounding_boxes(results_bbs, mask):
    last_n_bbs = []
    last_valid_index = len(results_bbs) - 1
    for m in mask:
        if m == 1 and last_valid_index >= 0:
            last_n_bbs.append(results_bbs[last_valid_index])
            last_valid_index -= 1
        elif len(last_n_bbs) > 0:
            last_n_bbs.append(last_n_bbs[-1])
    return last_n_bbs


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def get_tensor_corners_batch(center, wlh, theta, wlh_factor=1.0): 
    batch_size = center.shape[0]
    dtype = center.dtype
    device = center.device

    wlh = wlh.to(dtype) * wlh_factor
    w, l, h = wlh[:, 0], wlh[:, 1], wlh[:, 2]

    x_corners = l.view(batch_size, 1) / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1], dtype=dtype, device=device)
    y_corners = w.view(batch_size, 1) / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1], dtype=dtype, device=device)
    z_corners = h.view(batch_size, 1) / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1], dtype=dtype, device=device)
    corners = torch.stack((x_corners, y_corners, z_corners), dim=1)

    # Rotate
    rotation_matrices = _axis_angle_rotation("Z", -theta)
    corners = torch.einsum('bij,bjk->bik', rotation_matrices, corners)

    # Translate
    corners += center.view(batch_size, 3, 1)

    return corners.transpose(1, 2)


def create_corner_timestamps(B, H, corner_num=8):
    """
    Generate timestamps for B*N*3 corners: current frame at the end, historical frames at the beginning, e.g., -0.1, -0.2, -0.3, ... current frame +0.1.
    N should be equal to (number of historical frames + 1) * 8.
    The returned tensor can be directly concatenated to the original tensor.
    """
    N = (H + 1) * corner_num
    timestamps = torch.zeros((B, N, 1))

    for i in range(H):
        timestamps[:, (i * corner_num):(i * corner_num) + corner_num] = -(i + 1) * 0.1

    # Set the timestamp of the current box to 0.1
    timestamps[:, -corner_num:] = 0.1

    return timestamps

