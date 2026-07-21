# Observation Reliability Validation - observation_v1_local_recheck

Decision: **NO_GO_OBSERVATION_RELIABILITY_VALIDATION**

The calibrator, preprocessing statistics, and operating threshold were fitted once on
the standard fitting CSV. Every evaluation protocol used that frozen model unchanged.

## Frozen feature set

- `prev_log_search_points` <- `prev_obs_search_point_count` (log1p)
- `prev_empty_fallback` <- `prev_obs_empty_fallback` (bool)
- `prev_mean_fg_score` <- `prev_obs_mean_fg_score` (identity)
- `prev_fg_margin` <- `prev_obs_fg_margin_mean` (identity)
- `prev_motion_dynamic_probability` <- `prev_obs_motion_dynamic_probability` (identity)

## Metrics

| protocol | N | prevalence | AUROC | AUPRC | AUPRC-prev | Brier | ECE | activation | recall | precision | FPR | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1623 | 0.1109 | 0.7938 | 0.5253 | 0.4144 | 0.0795 | 0.0732 | 0.2742 | 0.7111 | 0.2876 | 0.2197 | report-only |
| gap1124 | 829 | 0.0893 | 0.6804 | 0.3713 | 0.2821 | 0.0859 | 0.0934 | 0.2859 | 0.5676 | 0.1772 | 0.2583 | False |
| burst_drop | 815 | 0.0847 | 0.7120 | 0.4122 | 0.3275 | 0.0780 | 0.0890 | 0.2785 | 0.6087 | 0.1850 | 0.2480 | False |

## Decision boundary

Strong protocols: `gap1124, burst_drop`.
All strong protocols must pass every pre-registered AUROC, AUPRC-margin, ECE,
FPR, and operating-recall check. Evaluation data never refit preprocessing,
weights, or threshold.

This validates only a visible-target next-crop-risk proxy. It does not validate
timestamp causality, complete-occlusion uncertainty, a trajectory anchor, or active tracking.
