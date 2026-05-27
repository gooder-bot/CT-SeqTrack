import numpy as np
import copy
import torch


def generate_virtual_points(box, radius=0.1, num_points=10, ratio=1.0):
    """
    Generate virtual points around the corners of a box.

    Parameters:
    box : The box around which to generate points.
    radius : The radius within which to generate points. Default is 0.1.
    num_points : The number of points to generate for each corner. Default is 10.

    Returns:
    expand_corners : A numpy array of shape (3, num_corners * num_points), where num_corners is the number of corners in the box. Each column is the (x, y, z) coordinates of a virtual point.
    """
    box = copy.deepcopy(box)
    box.wlh *= ratio
    box_corners = box.corners()
    expand_corners = np.zeros((3, box_corners.shape[1]*num_points))

    for i in range(box_corners.shape[1]):
        for j in range(num_points):
            random_point = box_corners[:, i] + np.random.uniform(-radius, radius, size=3)
            expand_corners[:, i*num_points + j] = random_point

    return expand_corners

def get_history_frame_ids_and_masks(this_frame_id, hist_num, offsets=None):
    if offsets is None:
        offsets = list(range(1, hist_num + 1))
    else:
        offsets = [int(offset) for offset in offsets]
        if len(offsets) != int(hist_num):
            raise ValueError(f"offsets length {len(offsets)} must match hist_num {hist_num}.")
        if any(offset <= 0 for offset in offsets):
            raise ValueError("history offsets must be positive frame gaps.")
        if any(offsets[i] >= offsets[i + 1] for i in range(len(offsets) - 1)):
            raise ValueError("history offsets must be strictly increasing from recent to older.")

    history_frame_ids = []
    masks = []
    for offset in offsets:
        frame_id = this_frame_id - offset
        if frame_id < 0:
            frame_id = 0
            masks.append(0)
        else:
            masks.append(1)
        history_frame_ids.append(frame_id)
    return history_frame_ids, masks


def create_history_frame_dict(prev_frames):
    history_frame_dict = {}
    for i, frame in enumerate(prev_frames):
        key = -1 * (i + 1)
        history_frame_dict[str(key)] = frame
    return history_frame_dict

def create_future_frame_dict(next_frames):
    future_frame_dict = {}
    for i, frame in enumerate(next_frames):
        key = i + 1
        future_frame_dict[str(key)] = frame
    return future_frame_dict

def normalize_timestamp(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    abs_value = abs(value)
    if abs_value > 1e14:
        return value * 1e-6
    if abs_value > 1e11:
        return value * 1e-3
    return value


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def compute_history_timestamps(frame_timestamps, current_timestamp, default_step=0.1):
    if frame_timestamps is None:
        return [], []

    current_timestamp = normalize_timestamp(current_timestamp)
    frame_timestamps = [normalize_timestamp(ts) for ts in frame_timestamps]

    relative_timestamps = []
    if current_timestamp is None:
        relative_timestamps = [-(idx + 1) * default_step for idx in range(len(frame_timestamps))]
    else:
        for idx, ts in enumerate(frame_timestamps):
            if ts is None:
                fallback = relative_timestamps[-1] - default_step if relative_timestamps else -default_step
                relative_timestamps.append(fallback)
                continue

            relative_timestamp = ts - current_timestamp
            if relative_timestamp >= 0:
                relative_timestamp = relative_timestamps[-1] - default_step if relative_timestamps else -default_step
            relative_timestamps.append(relative_timestamp)

    delta_t = []
    prev_relative_timestamp = 0.0
    for relative_timestamp in relative_timestamps:
        dt = prev_relative_timestamp - relative_timestamp
        if dt <= 0:
            dt = default_step
        delta_t.append(dt)
        prev_relative_timestamp = relative_timestamp
    return relative_timestamps, delta_t


def build_time_fields(frame_timestamps, current_timestamp, frame_ids=None, current_frame_id=None,
                      use_real_time=True, default_step=0.1, pseudo_step=0.1):
    frame_timestamps = list(frame_timestamps) if frame_timestamps is not None else []
    frame_ids = list(frame_ids) if frame_ids is not None else None
    current_frame_index = safe_float(current_frame_id)

    if not use_real_time:
        relative_timestamps, delta_t = compute_history_timestamps(
            [None for _ in frame_timestamps], None, pseudo_step)
        current_timestamp = current_frame_index * pseudo_step if current_frame_index is not None else 0.0
        local_timestamps = np.array(relative_timestamps + [0.0], dtype=np.float32)
        return relative_timestamps, delta_t, local_timestamps, float(current_timestamp)

    current_timestamp = normalize_timestamp(current_timestamp)
    if current_timestamp is None and current_frame_index is not None:
        current_timestamp = current_frame_index * default_step

    normalized_timestamps = [normalize_timestamp(ts) for ts in frame_timestamps]
    if frame_ids is not None:
        fallback_timestamps = []
        for idx, (timestamp, frame_id) in enumerate(zip(normalized_timestamps, frame_ids)):
            if timestamp is not None:
                fallback_timestamps.append(timestamp)
                continue

            frame_index = safe_float(frame_id)
            if current_timestamp is not None and current_frame_index is not None and frame_index is not None:
                step_count = max(current_frame_index - frame_index, idx + 1)
                fallback_timestamps.append(current_timestamp - step_count * default_step)
            else:
                fallback_timestamps.append(None)
        normalized_timestamps = fallback_timestamps

    relative_timestamps, delta_t = compute_history_timestamps(
        normalized_timestamps, current_timestamp, default_step)
    local_timestamps = np.array(relative_timestamps + [0.0], dtype=np.float32)
    current_timestamp = current_timestamp if current_timestamp is not None else 0.0
    return relative_timestamps, delta_t, local_timestamps, float(current_timestamp)


def generate_timestamp_prev_list(valid_mask_or_timestamps, point_sample_size, current_timestamp=None):
    values = list(valid_mask_or_timestamps)
    if current_timestamp is not None or any(value is None for value in values):
        relative_timestamps, _ = compute_history_timestamps(valid_mask_or_timestamps, current_timestamp)
        return [
            np.full((point_sample_size, 1), fill_value=timestamp, dtype=np.float32)
            for timestamp in relative_timestamps
        ]

    valid_mask = values
    timestamp_prev_list = []
    valid_time = 0

    for mask in valid_mask:
        if mask == 1:
            valid_time -= 0.1
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time, dtype=np.float32)
        else:
            timestamp_prev = np.full((point_sample_size, 1), fill_value=valid_time, dtype=np.float32)
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

def update_results_bbs(results_bbs, valid_mask, new_refboxs):
    update_count = int(sum(valid_mask))
    N = len(new_refboxs)

    if len(results_bbs) >= (N + 1):
        for i in range(N):
            results_bbs[-(i+1)] = new_refboxs[i]
    else:
        for i in range(update_count-1): 
            results_bbs[-(i+1)] = new_refboxs[i]
        
    return results_bbs

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

def get_tensor_corners(center,wlh,theta,wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.

        """
        w, l, h = wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1], dtype=torch.float32, device=center.device)
        y_corners = w / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1], dtype=torch.float32, device=center.device)
        z_corners = h / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1], dtype=torch.float32, device=center.device)
        corners = corners = torch.stack((x_corners, y_corners, z_corners), dim=0)

        # Rotate
        corners = _axis_angle_rotation("Z",-theta)@corners

        # Translate
        x, y, z = center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

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

def sample_box_batch(center, wlh, theta, wlh_factor=1.0):
    """
    Samples the bounding box for corners, edge midpoints, and face centers in batch.

    Args:
        center: Tensor of shape (batch_size, 3).
        wlh: Tensor of shape (batch_size, 3).
        theta: Tensor of shape (batch_size,).
        wlh_factor: float. Multiply w, l, h by a factor to inflate or deflate the box.

    Returns:
        samples: Tensor of shape (batch_size, 26, 3). This includes 8 corners, 
            12 edge midpoints, and 6 face centers.
    """
    # Get the 8 corners
    corners = get_tensor_corners_batch(center, wlh, theta, wlh_factor)

    # Compute the 12 edge midpoints
    edge_midpoints = torch.stack([
        (corners[:, 0] + corners[:, 1]) / 2,
        (corners[:, 1] + corners[:, 2]) / 2,
        (corners[:, 2] + corners[:, 3]) / 2,
        (corners[:, 3] + corners[:, 0]) / 2,
        (corners[:, 4] + corners[:, 5]) / 2,
        (corners[:, 5] + corners[:, 6]) / 2,
        (corners[:, 6] + corners[:, 7]) / 2,
        (corners[:, 7] + corners[:, 4]) / 2,
        (corners[:, 0] + corners[:, 4]) / 2,
        (corners[:, 1] + corners[:, 5]) / 2,
        (corners[:, 2] + corners[:, 6]) / 2,
        (corners[:, 3] + corners[:, 7]) / 2,
    ], dim=1)

    # Compute the 6 face centers
    face_centers = torch.stack([
        torch.mean(corners[:, :4], dim=1),   # Front face
        torch.mean(corners[:, 4:], dim=1),   # Back face
        torch.mean(corners[:, [0, 1, 4, 5]], dim=1),   # Left face
        torch.mean(corners[:, [2, 3, 6, 7]], dim=1),   # Right face
        torch.mean(corners[:, [0, 3, 4, 7]], dim=1),   # Top face
        torch.mean(corners[:, [1, 2, 5, 6]], dim=1),   # Bottom face
    ], dim=1)

    # Combine all the samples
    samples = torch.cat([corners, edge_midpoints, face_centers], dim=1)

    return samples

def create_corner_timestamps(B, H, corner_num=8):
    """
    Deprecated fixed-step timestamp helper. Use create_corner_timestamps_from_deltas for CT-SeqTrack.

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

def create_corner_timestamps_from_deltas(delta_T, corner_num=8):
    if not torch.is_tensor(delta_T):
        delta_T = torch.as_tensor(delta_T, dtype=torch.float32)

    if delta_T.dim() == 1:
        delta_T = delta_T.unsqueeze(0)

    B, H = delta_T.shape
    timestamps = torch.zeros((B, (H + 1) * corner_num, 1), dtype=delta_T.dtype, device=delta_T.device)
    for i in range(H):
        timestamps[:, i * corner_num:(i + 1) * corner_num] = delta_T[:, i].view(B, 1, 1)
    return timestamps


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)
