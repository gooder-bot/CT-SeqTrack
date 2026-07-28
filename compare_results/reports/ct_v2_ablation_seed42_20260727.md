# CT-SeqTrack v2 B0–B3 seed42 消融复核

更新时间：2026-07-27

## Overall Assessment: Needs revision

四组运行均已完整，数据足以完成本轮 seed42 normal-mini 首筛；但结果否定了当前三模块组合。B0 仍是唯一晋级模型，B1 的运动修正大幅退化，B2 只能部分救回，B3 的 learned gate 又退化回 B1 水平。

**结论：当前 B3 不应进入时间负对照、多 seed、full nuScenes 或 Random-20%。后续 Search-only A1 也已失败；当前应先做现有 B0/A1 checkpoint 的 Search 开/关 2×2，不训练 A2。**

## Methodology Review

- 数据：nuScenes v1.0-mini，Car，mini_train 274 tracklets / 5,051 frames，mini_val 106 tracklets / 2,285 frames。
- 协议：normal cadence、seed42、candidate4、batch16、60 epoch，每 5 epoch 验证一次。
- 主结果：固定使用 epoch60 `last.ckpt`；best epoch 和 late mean 只用于稳定性诊断，不用于替代 final。
- 原始来源：TensorBoard scalar events、`run_provenance.json` 和 `last.ckpt` 元数据；未使用服务器控制台汇总数字。
- B2 与 B3 的 resolved config 除配置名、tag 和 `ct_fusion_mode: fixed -> adaptive` 外一致。

## Integrity

| arm | status | commit | clean tracked | train steps | val points | last checkpoint |
|---|---|---|---:|---:|---:|---|
| B0 | COMPLETE | `d86990c` | yes | 75,720/75,720 | 12/12 | epoch 60 |
| B1 | COMPLETE | `d86990c` | yes | 75,720/75,720 | 12/12 | epoch 60 |
| B2 | COMPLETE | `d86990c` | yes | 75,720/75,720 | 12/12 | epoch 60 |
| B3 | COMPLETE | `600bb88` | yes | 75,720/75,720 | 12/12 | epoch 60 |

## Validation Results

| arm | final Success | final Precision | best Success (epoch) | best Precision (epoch) | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 53.360 | 64.382 | 54.135 (50) | 64.382 (60) | 52.905 | 63.104 |
| B1 | 26.021 | 24.972 | 29.115 (5) | 31.211 (20) | 26.080 | 25.299 |
| B2 | 47.973 | 52.088 | 50.080 (25) | 58.499 (25) | 46.437 | 49.818 |
| B3 | 25.537 | 24.707 | 34.458 (5) | 37.724 (5) | 26.321 | 25.104 |

## Module Deltas at Final Checkpoint

| comparison | Success delta | Precision delta | decision |
|---|---:|---:|---|
| B1 − B0 (motion) | -27.339 | -39.410 | reject current fixed-0.75 motion fusion |
| B2 − B1 (search) | 21.952 | 27.116 | positive rescue, but not a search-only proof |
| B2 − B0 (motion + search) | -5.387 | -12.294 | below baseline |
| B3 − B2 (adaptive gate) | -22.435 | -27.381 | reject current adaptive gate |
| B3 − B0 (full v2) | -27.823 | -39.675 | no promotion |
| B3 − B1 | -0.484 | -0.265 | returns to the failed B1 level |

## Learning-Curve Interpretation

- B0 late-3 is 52.905 / 63.104 and final is 53.360 / 64.382; the baseline is the strongest and most stable arm in this screen.
- B2 reaches 50.080 / 58.499 at epoch 25, then finishes at 47.973 / 52.088. Search provides a large recovery relative to B1, but never establishes a gain over B0.
- B3 is best at epoch 5 (34.458 / 37.724), immediately after the five training warmup epochs while the gate is still at its initial nominal alpha 0.25. It falls to 28.542 / 26.042 at epoch 10 and its late-3 is only 26.321 / 25.104. The regression therefore persists across the entire late-training window and is not a bad final checkpoint.

## Mechanism Diagnostics

- B1 post-warmup applied alpha mean is 0.553; innovation is applied to 73.7% of training samples and is radius-clamped on 40.7%. This is an aggressive correction path, consistent with the large B1 regression.
- B2 search is active on only 3.46% of training samples; mean expansion token share is 0.865%. B3 has essentially the same search activation, so its collapse is not explained by search being disabled.
- B3 nominal alpha is 0.250 at epoch 5, rises to 0.707 at epoch 6, and reaches 0.749 at epoch 7. At epoch 60 even the batch-min mean is 0.749998, against a configured maximum of 0.75. The learned gate has saturated into an almost constant maximum-weight gate rather than learning conditional reliability.
- B3 post-warmup applied alpha (0.552), innovation application ratio (73.7%) and clamp ratio (40.8%) are nearly identical to B1/B2. Its epoch-60 mean training loss is 0.206, slightly below B2's 0.213, while validation is far worse. This is evidence of train/recursive-validation mismatch or gate/backbone co-adaptation, not under-training.

## Issues Found

1. **High — adaptive gate collapse.** B3 learns the configured upper bound for virtually every training sample. It neither suppresses unreliable motion nor preserves B2's recovery.
2. **High — current motion correction fails the normal-data guardrail.** B1 best, final and late metrics are all far below B0. Adding a learned gate does not repair it.
3. **Medium — search is not independently isolated.** B2 − B1 is positive, but there is no `B0 + search only` arm. The data prove a rescue interaction, not that search itself beats the baseline.
4. **Medium — shared initialization is not strictly controlled.** `ct_proposal_fusion` is instantiated before `motion_mlp`, `feature_pointnet` and `Transformer`; enabling it consumes RNG before shared layers are initialized. The same issue applies when B1 inserts the motion encoder relative to B0. With only one seed, exact module deltas are therefore partly confounded by initialization.
5. **Medium — no per-tracklet endpoint export is present.** Aggregate Success/Precision cannot show whether B2 recovery is broad or driven by a small subset of sequences.
6. **Low — B3 uses commit `600bb88` while B0–B2 use `d86990c`.** The intervening changes only normalize singleton validation scalar shapes and add tests; no architecture or configured numerical rule changed. This is recorded as a provenance difference, not the leading explanation for the score gap.

## Decision

- Mark the current B0–B3 screen complete and reject the present B3. Do not repeat the same four unchanged configs.
- Before another training cycle, enforce a shared-initialization contract (load one common initialization for shared keys, or isolate optional-module RNG). Add a same-checkpoint inference override for alpha 0 / 0.25 / 0.75 as a cheap sensitivity diagnosis.
- The follow-up `A1 = baseline + time-guided search only` has since completed and failed the normal-mini guardrail. See `ct_search_only_seed42_20260727.md` for the post-screen decision.
- Do not train A2 yet. First evaluate the existing B0 and A1 checkpoints with Search off/on as a no-training 2x2, and add validation endpoint search diagnostics. Do not restore the current unconstrained adaptive gate.
- Seed43/44, `true/fixed/shuffled`, full nuScenes, Random-20%, ChronoTrack consistency and compact memory remain blocked until a final/late-3 model beats its same-initialization baseline.

## Required Caveats

- This is one seed on nuScenes-mini and does not establish statistical stability.
- B2's positive delta is relative to the failed B1 arm; B2 is still below the same-code B0 baseline.
- The current ordering of optional module construction prevents a strictly shared initialization across arms; the screen is sufficient for rejection but not for a paper-level causal effect size.
- No physical-time causal claim is supported until a promoted model passes same-endpoint `true/fixed/shuffled` controls.
