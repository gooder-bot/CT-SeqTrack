import numpy as np
from . import points as points_utils
from . import spatial as geometry_utils
from .misc import generate_timestamp_prev_list, resolve_kitti_hv_search_offset

def motion_processing_mf(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frames = data['prev_frames']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    valid_mask = data['valid_mask']
    num_hist = len(valid_mask)
    empty_counter = 0

    prev_pcs  = [prev_frames[key]['pc'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    prev_boxs = [prev_frames[key]['3d_bbox'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']
    bb_offset = resolve_kitti_hv_search_offset(config, this_frame)

    # Check the number of empty boxes
    for prev_box, prev_pc in zip(prev_boxs, prev_pcs):
        num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points[0:3,:]).sum()
        if num_points_in_prev_box < config.limit_num_points_in_prev_box:
            empty_counter += 1
    assert empty_counter < config.empty_box_limit, 'not enough valid box' 

    ref_boxs = []
    for i, prev_box in enumerate(prev_boxs): # Apply a random offset to each box, not uniformly
        if candidate_id == 0:
            sample_offsets = np.zeros(3)
        else:
            sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
            sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
        ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=config.data_limit_box,
                                        degrees=config.degrees)
        ref_boxs.append(ref_box)


    prev_frame_pcs = []
    for i, prev_pc in enumerate(prev_pcs):
        prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
                                                    scale=config.bb_scale,
                                                    offset=bb_offset)
        prev_frame_pcs.append(prev_frame_pc)

    this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(this_pc, ref_boxs[0], ref_boxs[0],
                                                    scale=config.bb_scale,
                                                    offset=bb_offset)

    this_box    = points_utils.transform_box(this_box, ref_boxs[0]) 
    prev_boxs   = [points_utils.transform_box(prev_box, ref_boxs[0]) for prev_box in prev_boxs] 
    ref_boxs    = [points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs]    
    motion_boxs = [points_utils.transform_box(this_box, prev_box) for prev_box in prev_boxs]  

    # Resample each frame of the point cloud to a specific number
    prev_points_list = [points_utils.regularize_pc(prev_frame_pc.points.T, config.point_sample_size)[0] for prev_frame_pc in prev_frame_pcs] 
    this_points = points_utils.regularize_pc(this_frame_pc.points.T, config.point_sample_size)[0] 

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T[:3,:], config.bb_scale).astype(int)
    seg_label_prev_list = [geometry_utils.points_in_box(prev_box, prev_points.T[:3,:], config.bb_scale).astype(int) for prev_box, prev_points in zip(prev_boxs, prev_points_list)] #应当只考虑xyz特征
    seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], config.bb_scale).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征
    if candidate_id != 0:
        for seg_mask_prev in seg_mask_prev_list:
            # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
            # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)


    timestamp_prev_list = generate_timestamp_prev_list(valid_mask,config.point_sample_size)
    timestamp_this = np.full((config.point_sample_size, 1), fill_value=0.1)

    prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
    ]
    this_points = np.concatenate(
        [this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

    stack_points_list = prev_points_list + [this_points]
    stack_points = np.concatenate(stack_points_list, axis=0)

    stack_seg_label_list = seg_label_prev_list + [seg_label_this]
    stack_seg_label = np.hstack(stack_seg_label_list)

    theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
        this_box.orientation.radians * this_box.orientation.axis[-1]
    box_label = np.append(this_box.center, theta_this).astype('float32')
    theta_prev_list = [
        prev_box.orientation.degrees * prev_box.orientation.axis[-1]
        if config.degrees else prev_box.orientation.radians *
        prev_box.orientation.axis[-1] for prev_box in prev_boxs
    ]
    box_label_prev_list = [
        np.append(prev_box.center, theta_prev).astype('float32')
        for prev_box, theta_prev in zip(prev_boxs, theta_prev_list)
    ]

    # Generate a reference box sequence
    theta_ref_list=[
        ref_box.orientation.degrees * ref_box.orientation.axis[-1]
        if config.degrees else ref_box.orientation.radians *
        ref_box.orientation.axis[-1] for ref_box in ref_boxs
    ]
    ref_box_list = [
        np.append(ref_box.center, theta_ref).astype('float32')
        for ref_box, theta_ref in zip(ref_boxs, theta_ref_list)
    ]

    theta_motion_list = [
        motion_box.orientation.degrees * motion_box.orientation.axis[-1]
        if config.degrees else motion_box.orientation.radians *
        motion_box.orientation.axis[-1] for motion_box in motion_boxs
    ]

    motion_label_list = [
        np.append(motion_box.center, theta_motion).astype('float32')
        for motion_box, theta_motion in zip(motion_boxs, theta_motion_list)
    ]
    motion_state_label_list = [ 
        np.sqrt(np.sum((this_box.center - prev_box.center)**2))
        > config.motion_threshold for prev_box in prev_boxs
    ]

    data_dict = {
        'points': stack_points.astype('float32'), # Historical first, then current
        'box_label': box_label, 
        'ref_boxs':np.stack(ref_box_list, axis=0),
        'box_label_prev': np.stack(box_label_prev_list, axis=0),
        'motion_label': np.stack(motion_label_list, axis=0),
        'motion_state_label': np.stack(motion_state_label_list, axis=0).astype('int'),
        'bbox_size': this_box.wlh, 
        'seg_label': stack_seg_label.astype('int'), 
        'valid_mask': np.array(valid_mask).astype('int'), 
    }

    if getattr(config, 'box_aware', False):
        stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
        hist_points_list = stack_points_split[:num_hist] 
        prev_bc_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, prev_boxs)
        ]
        this_points_split = stack_points_split[-1] 
        this_bc = points_utils.get_point_to_box_distance(this_points_split[:,:3], this_box)


        candidate_bc_prev_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, ref_boxs)
        ]

        candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
        candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
        candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

        data_dict.update({'prev_bc': np.stack(prev_bc_list, axis=0).astype('float32'),
                          'this_bc': this_bc.astype('float32'),
                          'candidate_bc': candidate_bc.astype('float32')})

    return data_dict

