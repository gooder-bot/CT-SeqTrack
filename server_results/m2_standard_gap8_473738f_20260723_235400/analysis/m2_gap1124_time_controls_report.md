# M0 endpoint summary: m2_gap1124_time_controls

## Run metrics

| run | endpoints | tracklets | Success | Precision | mean error | empty fallback | path gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 1257 | 91 | 59.0632 | 70.8572 | 2.100749 | 113 | NA |
| fixed | 1257 | 91 | 59.1905 | 70.8433 | 2.124841 | 111 | NA |
| shuffled | 1257 | 91 | 59.3815 | 71.0660 | 2.072577 | 111 | NA |

## Paired comparisons

### true:fixed

- Endpoint exact match: `True`
- Success left-right: `-0.127287`
- Precision left-right: `0.013922`
- Mean center error left-right: `-0.024092`
- Empty fallback left-right: `2`

### true:shuffled

- Endpoint exact match: `True`
- Success left-right: `-0.318218`
- Precision left-right: `-0.208831`
- Mean center error left-right: `0.028172`
- Empty fallback left-right: `2`

## Interpretation boundary

All deltas are paired by tracklet/endpoint. Epochs and frames are not treated as independent statistical samples. This report is a frozen-checkpoint diagnostic and does not promote a method by itself.
