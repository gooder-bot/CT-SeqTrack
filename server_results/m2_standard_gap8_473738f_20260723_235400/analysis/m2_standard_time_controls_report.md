# M0 endpoint summary: m2_standard_time_controls

## Run metrics

| run | endpoints | tracklets | Success | Precision | mean error | empty fallback | path gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 2285 | 106 | 55.2834 | 67.2057 | 2.282068 | 217 | NA |
| fixed | 2285 | 106 | 55.2527 | 67.2155 | 2.286322 | 217 | NA |
| shuffled | 2285 | 106 | 55.2155 | 67.1204 | 2.290353 | 217 | NA |

## Paired comparisons

### true:fixed

- Endpoint exact match: `True`
- Success left-right: `0.030635`
- Precision left-right: `-0.009847`
- Mean center error left-right: `-0.004254`
- Empty fallback left-right: `0`

### true:shuffled

- Endpoint exact match: `True`
- Success left-right: `0.067834`
- Precision left-right: `0.085339`
- Mean center error left-right: `-0.008284`
- Empty fallback left-right: `0`

## Interpretation boundary

All deltas are paired by tracklet/endpoint. Epochs and frames are not treated as independent statistical samples. This report is a frozen-checkpoint diagnostic and does not promote a method by itself.
