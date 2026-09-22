"""独立参考臂的固定协议；变更任何源/适配文件都会改变身份。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json


CANONICAL = dict(hist_num=3, box_aware=True, use_motion_cls=True,
    point_sample_size=1024, num_candidates=4, bb_scale=1.25, bb_offset=2.,
    degrees=False, data_limit_box=True, use_z=True, limit_box=False,
    motion_threshold=.15, empty_box_limit=3, limit_num_points_in_prev_box=1,
    center_weight=2., angle_weight=10., seg_weight=.1, bc_weight=1.,
    ref_center_weight=.2, ref_angle_weight=1., motion_cls_seg_weight=.1)


def reference_config(config=None):
    # 固定原算法配置，host 预算另由实验 config 管理；不继承 production B0 参数。
    return SimpleNamespace(**CANONICAL)


def protocol_identity():
    root = Path(__file__).parent
    provenance = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    files = {str(path.relative_to(root)).replace('\\', '/'):
             hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob('*')) if path.is_file()
             and (path.suffix in ('.py', '.txt') or path.name == 'LICENSE')}
    return dict(schema='seqtrack_reference.v32.protocol.v1', canonical=CANONICAL.copy(),
                source_provenance=provenance, implementation_sha256=files,
                training='teacher4_nonfirst_nominal_with_original_invalid_history_resampling',
                network_size='train_and_evaluation_first_frame_wlh',
                evaluation_size='first_frame_wlh', coordinates='anchor_local_newest_first',
                evaluation_seed='shared_host_reseeds_python_numpy_torch_and_test_loader_before_each_test',
                partial_batch_bn='running_stats_all_reference_bn',
                random_protocol='original_numpy_and_torch_worker_rng; reproducible_for_fixed_seed_epoch_workers; '
                                'workers0_and_workers2_need_not_match; formal_workers4')
