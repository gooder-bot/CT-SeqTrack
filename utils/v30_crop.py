"""B1 prepass 与 B0 共用的当前原始裁剪；不访问当前标注。"""
import numpy as np


def prepare_base_crop(point_cloud, anchor, config):
    from datasets.points_utils import generate_subwindow_with_aroundboxs
    from utils.b1_acquisition import build_b0_crop_context
    from utils.point_identity import raw_point_ids
    crop = generate_subwindow_with_aroundboxs(
        point_cloud, anchor, anchor, scale=config.bb_scale, offset=config.bb_offset,
        canonicalize=False,
        copy_selected_only=getattr(config, 'ct_runtime_optimization', '') == 'equivalent_v1')
    context = build_b0_crop_context(len(np.unique(raw_point_ids(crop))), anchor,
                                    scale=config.bb_scale, offset=config.bb_offset)
    return crop, context
