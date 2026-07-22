# M2 E6 formal 参数冻结

决定：**`FREEZE_M2_ALPHA_RADIUS`**

本报告只验证进入工程门禁前已经声明的单一规则；未执行 alpha、scale、gate 或半径网格搜索。

## 冻结值

```yaml
dynamics_innovation_alpha: 0.75
dynamics_innovation_radius_base: 0.5
dynamics_innovation_radius_per_second: 0.5
dynamics_innovation_radius_max: 2.0
physical_time_adapter_warmup_epoch: 5
dynamics_innovation_warmup_epoch: 5
```

## mini_train primary 复算

| 指标 | 数值 |
| --- | ---: |
| endpoint | 1311 |
| tracklet | 213 |
| observation error mean | 1.349205 m |
| frozen-rule error mean | 1.060932 m |
| mean gain | 0.288272 m |
| median gain | 0.186809 m |
| gain >= 0.05 m | 73.607933% |
| positive gain | 85.430969% |
| clamp rate | 34.477498% |
| tracklet-equal mean gain | 0.262530 m |
| tracklet bootstrap 95% CI | [0.230185, 0.295833] m |
| long-gap endpoint | 417 |
| long-gap mean gain | 0.288963 m |

## 边界

当前安全半径会牺牲一部分未裁剪 oracle 空间：
unbounded alpha=0.75 error mean 为 0.516155 m，
frozen-rule error mean 为 1.060932 m。
该差异只作为保守性诊断，不用于在同一 oracle 上放大半径。

## 硬检查

- [x] `minimum_samples`
- [x] `minimum_tracklets`
- [x] `mean_gain`
- [x] `useful_gain_rate`
- [x] `tracklet_bootstrap_lower_positive`
- [x] `long_gap_supported`
- [x] `long_gap_mean_gain_positive`

输入 SHA256：

- endpoints: `aa2e890a5fcc3e15964bb89d87dc8c7873b0c97a29f437c220b8cd00e406099b`
- summary: `2ecd6e707ffee6e6551effadb7c896f974988064579174d48ad8e7686ecf367a`
