"""v29 的物理帧布局；仅由显式启用新合同的 B0 host 调用。"""


def frame_aligned_pointnet_input(channel_first_points, frame_count):
    """将 [B,C,L*N] 转为 [B*L,C,N]，保留每个真实帧的通道/点顺序。"""
    if channel_first_points.ndim != 3:
        raise ValueError('v29 point layout requires [B,C,L*N]')
    frame_count = int(frame_count)
    batch, channels, total_points = channel_first_points.shape
    if frame_count <= 0 or total_points == 0 or total_points % frame_count:
        raise ValueError('v29 point slots must divide into nonempty physical frames')
    points_per_frame = total_points // frame_count
    return (channel_first_points.reshape(batch, channels, frame_count, points_per_frame)
            .permute(0, 2, 1, 3).contiguous()
            .reshape(batch * frame_count, channels, points_per_frame))
