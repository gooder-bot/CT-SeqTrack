"""Read-only production-code probes; synthetic examples, no dataset claim."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils.tracking_metrics_v27 import metric_contributions
from utils.action_calibration_v27 import shortlist_policies, policy_mask, summarize_rows


def row(frame, score, obs, candidate, applied=False):
    return dict(tracklet_id='synthetic-track', scene_id='synthetic-scene',
                frame_id=frame, is_initial=frame == 0,
                structural_available=frame > 0, action_score=score,
                observation_success=obs, observation_precision=obs,
                candidate_success=candidate, candidate_precision=candidate,
                final_success=candidate if applied else obs,
                final_precision=candidate if applied else obs,
                action_applied=applied)


def rollout(policy):
    rows = [row(0, 0., 1., 1.)]
    accepted_first = accepted_harm = False
    for frame in (1, 2, 3):
        if frame == 1:
            score, obs, candidate = .8, .5, .6
        elif frame == 2:
            score, obs, candidate = (.4, .7, .1) if accepted_first else (.2, .5, .1)
        else:
            score = .2
            obs = .1 if accepted_harm else .7 if accepted_first else .5
            candidate = .1
        applied = bool(policy_mask([score], [True], policy)[0])
        rows.append(row(frame, score, obs, candidate, applied))
        if frame == 1:
            accepted_first = applied
        if frame == 2:
            accepted_harm = applied
    return rows


metric_examples = []
for label, before, after in (
    ('no_overlap_3m_to_2.25m', (0., 3.), (0., 2.25)),
    ('within_same_threshold_bins', (.301, .551), (.349, .599)),
    ('cross_precision_threshold', (0., 2.01), (0., 1.99)),
):
    old, new = metric_contributions(*before), metric_contributions(*after)
    metric_examples.append(dict(name=label, before=before, after=after,
        old_sp=[float(x) for x in old], new_sp=[float(x) for x in new],
        gain_sp=[float(b - a) for a, b in zip(old, new)],
        help=bool(sum(new) - sum(old) > 2e-6),
        harm=bool(sum(new) - sum(old) < -2e-6)))

baseline = rollout({'kind': 'never'})
shortlist, screen = shortlist_policies(baseline, include_controls=True)
policies = [{'kind': 'never'}] + [x['policy'] for x in shortlist]
reports = [dict(policy=p, closed_loop=summarize_rows(rollout(p))) for p in policies]
omitted = {'kind': 'threshold', 'threshold': .5}
result = dict(note='Synthetic counterexamples only; not nuScenes empirical results.',
              metric_examples=metric_examples,
              production_screen_policies=[x['policy'] for x in screen],
              production_shortlist=reports,
              omitted_policy=dict(policy=omitted,
                  closed_loop=summarize_rows(rollout(omitted))),
              same_never_path_mask=(policy_mask([r['action_score'] for r in baseline],
                  [r['structural_available'] for r in baseline], {'kind': 'threshold', 'threshold': .2}).tolist()
                  == policy_mask([r['action_score'] for r in baseline],
                      [r['structural_available'] for r in baseline], omitted).tolist()))
output = Path(__file__).with_name('diagnostic_results.json')
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
