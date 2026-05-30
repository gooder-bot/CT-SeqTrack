# SeqTrack vs CT-SeqTrack nuScenes-mini Results

## Run Paths

- **SeqTrack baseline**: `/home/lishengjie/study/lcyu/seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16`
- **CT-SeqTrack**: `/home/lishengjie/study/lcyu/CT-SeqTrack/output/20260528-1718-seqtrack3d_nuscenes_mini_p5_obs_gate-ct_seqtrack_mini_p5_car_60ep_bs16_gpu3`

## Summary

| model | metric | final | final_step | best | best_step |
|---|---|---:|---:|---:|---:|
| SeqTrack baseline | success/test | 50.9858 | 75719 | 52.2834 | 44169 |
| SeqTrack baseline | precision/test | 59.9617 | 75719 | 65.2144 | 12619 |
| CT-SeqTrack | success/test | 31.1937 | 75719 | 44.9836 | 31549 |
| CT-SeqTrack | precision/test | 31.8851 | 75719 | 62.5120 | 6309 |

## SeqTrack baseline

- TensorBoard version: `/home/lishengjie/study/lcyu/seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16/lightning_logs/version_0`

### success/test

- Final: `50.9858` at step `75719`
- Best: `52.2834` at step `44169`

| index | step | value |
|---:|---:|---:|
| 0 | 6309 | 40.6969 |
| 1 | 12619 | 46.0908 |
| 2 | 18929 | 45.7298 |
| 3 | 25239 | 42.8851 |
| 4 | 31549 | 51.7363 |
| 5 | 37859 | 48.9289 |
| 6 | 44169 | 52.2834 |
| 7 | 50479 | 46.4781 |
| 8 | 56789 | 50.4628 |
| 9 | 63099 | 50.2024 |
| 10 | 69409 | 49.1346 |
| 11 | 75719 | 50.9858 |

### precision/test

- Final: `59.9617` at step `75719`
- Best: `65.2144` at step `12619`

| index | step | value |
|---:|---:|---:|
| 0 | 6309 | 41.8698 |
| 1 | 12619 | 65.2144 |
| 2 | 18929 | 50.5416 |
| 3 | 25239 | 49.2309 |
| 4 | 31549 | 60.6324 |
| 5 | 37859 | 53.9092 |
| 6 | 44169 | 62.3392 |
| 7 | 50479 | 51.1740 |
| 8 | 56789 | 59.6105 |
| 9 | 63099 | 57.3085 |
| 10 | 69409 | 59.3184 |
| 11 | 75719 | 59.9617 |

## CT-SeqTrack

- TensorBoard version: `/home/lishengjie/study/lcyu/CT-SeqTrack/output/20260528-1718-seqtrack3d_nuscenes_mini_p5_obs_gate-ct_seqtrack_mini_p5_car_60ep_bs16_gpu3/lightning_logs/version_0`

### success/test

- Final: `31.1937` at step `75719`
- Best: `44.9836` at step `31549`

| index | step | value |
|---:|---:|---:|
| 0 | 6309 | 44.9376 |
| 1 | 12619 | 25.4190 |
| 2 | 18929 | 24.2330 |
| 3 | 25239 | 22.8063 |
| 4 | 31549 | 44.9836 |
| 5 | 37859 | 41.4344 |
| 6 | 44169 | 40.6707 |
| 7 | 50479 | 28.3534 |
| 8 | 56789 | 32.1433 |
| 9 | 63099 | 38.8829 |
| 10 | 69409 | 25.7123 |
| 11 | 75719 | 31.1937 |

### precision/test

- Final: `31.8851` at step `75719`
- Best: `62.5120` at step `6309`

| index | step | value |
|---:|---:|---:|
| 0 | 6309 | 62.5120 |
| 1 | 12619 | 23.3228 |
| 2 | 18929 | 24.2166 |
| 3 | 25239 | 21.5810 |
| 4 | 31549 | 49.4934 |
| 5 | 37859 | 45.3512 |
| 6 | 44169 | 46.4147 |
| 7 | 50479 | 27.2899 |
| 8 | 56789 | 33.2473 |
| 9 | 63099 | 43.9398 |
| 10 | 69409 | 23.8162 |
| 11 | 75719 | 31.8851 |
