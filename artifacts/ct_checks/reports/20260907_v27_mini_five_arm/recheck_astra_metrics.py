"""Compare actual pre-a43 metric functions with v27 on identical z-up boxes."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.tracking_metrics_v27 import LocalYawBox, box_metrics, metric_contributions

old_source = subprocess.check_output(['git', 'show', 'a43a8ee^:utils/metrics.py'], cwd=ROOT).decode('utf-8')
names = {'estimateAccuracy', 'fromBoxToPoly', 'estimateOverlap'}
tree = ast.parse(old_source)
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
namespace = dict(np=np, Polygon=Polygon)
exec(compile(functions, '<pre-astra metrics>', 'exec'), namespace)
rng = np.random.default_rng(42)
deltas, contribution_deltas = [], []
for i in range(2000):
    center = rng.normal(0, 10, 3)
    a = LocalYawBox([*center, rng.uniform(-np.pi, np.pi)], rng.uniform([.5, 1., .5], [3., 15., 4.]))
    b = LocalYawBox([*(center + rng.normal(0, 1.5, 3)), rng.uniform(-np.pi, np.pi)], rng.uniform([.5, 1., .5], [3., 15., 4.]))
    if i % 10 == 0:
        b = a
    old = (namespace['estimateOverlap'](a, b, dim=3, up_axis=(0,0,1)),
           namespace['estimateAccuracy'](a, b, dim=3, up_axis=(0,0,1)))
    new = box_metrics(a,b,up_axis=(0,0,1),dim=3,mode='benchmark_compat')
    deltas.append(np.abs(np.asarray(new)-old))
    contribution_deltas.append(np.abs(np.asarray(metric_contributions(*new))-metric_contributions(*old)))
raw = np.max(deltas,axis=0)
contribution = np.max(contribution_deltas,axis=0)
assert raw[0] < 1e-10 and raw[1] < 1e-10
assert float(contribution.max()) == 0
result = dict(samples=2000,up_axis=[0,0,1],dim=3,
    source='a43a8ee^:utils/metrics.py extracted original functions',
    max_iou_difference=float(raw[0]),max_distance_difference=float(raw[1]),
    max_per_frame_success_precision_difference=contribution.tolist(),
    scope='same synthetic boxes; does not assess model outputs or changed evaluation populations')
Path(__file__).with_name('astra_metric_equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
