# B0 历史验证频率、RNG 与训练状态审计

日期：2026-09-08。范围为 `049de82`（v25 高分）、`b8222bb`（v25 低分）、`b445ecd`（v26）。只读历史源码与原始 hparams，不修改训练代码，不审计服务器环境。

## 结论

**不能把 v25 的高低分主要归因于 `check_val_every_n_epoch=1` 与 `5`。两次训练在第一次 Adam 更新后已经出现 B0 参数哈希差异，而两版入口都明确关闭训练前 sanity validation。** 验证尚未发生，频率差异不能解释最初分叉。

进一步核对实际启用路径后，没有找到验证通过全局采样 RNG、B0 BN、Dropout、学习率调度或 checkpoint 回灌来改变后续 B0 参数更新的已启用代码路径。验证频率不是已证实的回归根因，也没有证据表明改回每轮验证就能恢复高分。

这不等于已证明不同验证频率下 CUDA 全程逐位一致。源码排除了下面这些具体假说；首步梯度/Adam 增量和同臂重复运行仍需分别定位，不能用“CUDA 非确定性”直接代替该定位。

## 实际运行配置

来源为对应 `output/.../lightning_logs/version_0/hparams.yaml` 中的 `config.dictitems`，不是按当前 YAML 推断历史运行。

| 字段 | v25 高分 `049de82` | v25 低分 `b8222bb` | v26 `b445ecd` |
|---|---|---|---|
| `check_val_every_n_epoch` | 1 | 5 | 2 |
| `workers` | 4 | 4 | 4 |
| `ct_runtime_protocol` | `safe_seqtrack_auto_v1` | 相同 | 相同 |
| `ct_optimizer_topology` | `unified_auto` | 相同 | 相同 |
| `ct_separate_optimizers` | false | false | false |
| `ct_observation_rng_mode` | `stateless_seqtrack` | 相同 | 相同 |
| `ct_validation_rng_mode` | `stateless_tracklet_frame` | 相同 | 相同 |
| `observation_safe_bbox_size` | true | true | true |
| `ct_b0_rng_shift_control` | false | false | false |
| `optimizer / lr` | Adam / 1e-4 | 相同 | 相同 |
| `lr_decay_step / lr_decay_rate` | 20 / 0.1 | 相同 | 相同 |
| `limit_box` | false | false | false |
| `checkpoint / init_checkpoint` | null / null | null / null | null / null |

运行目录分别为：

- `output/20260824-0219-25_b0-ct25_b0_mini_car_60ep_bs16_seed42_retryfix`
- `output/20260825-1931-25_b0-ct25_b0_mini_car_seed42_60ep_bs16_val5`
- `output/20260903-2301-26_b0-ct26_b0_mini_car_seed42_60ep_bs16_20260903_225946`

## 首步分叉发生在任何验证之前

历史入口：

- `049de82:main.py:979-980`：设置 `check_val_every_n_epoch`，紧接着 `num_sanity_val_steps=0`。
- `b8222bb:main.py:993-994`：相同。
- `b445ecd:main.py:1030-1031`：相同。

已有原始 checkpoint 复核见 `historical_b0_recheck.json`：v25 高低分的初始化哈希都是 `798a8def3e82...`，前 100 个 observation 输入指纹相同；第一次更新后的哈希分别为 `7f547e5b4afd...` 与 `cf88c167bc9f...`。第一批的 view0 loss 均为 `19.0524749756`。

因此，即使以后发现验证会影响第 2 轮以后的细微训练差异，也不能拿它解释这里的首次分叉。相同标量 loss 也不能反向证明整个梯度张量完全相同。

## 采样 RNG 路径

三版都给训练与验证 DataLoader 配置了独立 `torch.Generator`：训练 seed 为 `seed+31001`，验证为 `seed+51001`；`worker_init_fn` 使用各自 worker 的 `torch.initial_seed()` 设置 NumPy/Python RNG。

对应源码为：

- `049de82:main.py:795-803, 915-927, 948-954`。
- `b8222bb:main.py:809-817, 929-941, 962-968`。
- `b445ecd:main.py:819-827, 940-952, 974-980`。

验证的递归点采样也不会退回未隔离的 NumPy 全局采样：`observation_safe_bbox_size=true` 使 `build_input_dict` 进入 `use_recursive_contract` 分支。`utils/recursive_state.py:81-97` 在三版中相同，以 tracklet key、frame id、candidate id 生成历史/当前点采样 seed；canonical 的 seed 不依赖 epoch。`regularize_pc(..., seed=seed)` 使用局部 `np.random.default_rng(seed)`。

确实存在会消耗全局 NumPy RNG 的旧代码，但本次三次 B0 运行不触发：

- 不走递归合同的旧评测分支，会给历史采样传 `seed=None`。三次实际 `observation_safe_bbox_size=true`。
- `getOffsetBB` 在 `limit_box=true` 且正向位移超限时随机重置 XY。三次实际 `limit_box=false`，因此不能将这个潜在路径列为已发生原因。
- host 的 `torch.rand((B,), ...)` 控制分支需要 `self.training and ct_b0_rng_shift_control`；三次配置均为 false，验证时也不满足 training。

## 验证不会经已启用路径更新 B0 BN / Dropout

本项目没有覆盖 `on_validation_model_eval` / `on_validation_model_train`。三版 host 的 `train(mode)` 都先调用 `super().train(mode)`；随后两个遗留冻结辅助方法只在各自冻结开关打开时介入，正式 scratch B0 不开启这些冻结路径。它们也不存在将 eval 模式强行切回 train 的逻辑。

Lightning 2.0.2 官方源码中，evaluation loop 在运行前调用 `_on_evaluation_model_eval`，结束后调用 `_on_evaluation_model_train`；其继承自 ModelHooks 的默认 hook 分别执行 `self.trainer.model.eval()` 与 `self.trainer.model.train()`。本项目 B0 attention 使用 `nn.Dropout`，不是写死 `training=True` 的函数式 dropout。故正式 validation 中 BN 使用既有统计量，Dropout 关闭。

这里须区分旧 mechanism 生成 observation 时遗漏完整 eval 的另一个已确认问题。**那个问题是训练 mechanism 子流程的问题，不能据此推导正常 Lightning validation 也在 train 模式运行。** 本次被比较的纯 B0 更没有 B1/B2/B3 mechanism 更新。

官方源码参考：

- [Lightning 2.0.2 evaluation loop](https://github.com/Lightning-AI/lightning/blob/2.0.2/src/lightning/pytorch/loops/evaluation_loop.py)，`on_run_start` / `on_run_end` / `_on_evaluation_model_eval` / `_on_evaluation_model_train`。
- [Lightning 2.0.2 ModelHooks](https://github.com/Lightning-AI/lightning/blob/2.0.2/src/lightning/pytorch/core/hooks.py)，`on_validation_model_eval` / `on_validation_model_train`。

## Scheduler、checkpoint 与递归状态

三次 B0 都是 unified-auto Adam，StepLR 的 interval 为 epoch；不使用验证指标驱动的 ReduceLROnPlateau。`on_train_epoch_end` 中的手动 scheduler 分支只在 `ct_separate_optimizers=true` 时运行，本次为 false。因此验证 1/2/5 轮一次不会把 LR 衰减改成 1/2/5 轮一次。

v25 入口 callback 是 ModelCheckpoint 与 LearningRateMonitor，没有 EarlyStopping，也没有验证后加载 best checkpoint 的回调。保存更多 best checkpoint 会改变文件与 I/O，不会将模型参数回滚到该 checkpoint。本审计比较的是 final60，不能以不同数量的 best checkpoint 解释 final60 分差。

v26 新增 `DataLoaderGeneratorState` 保存/恢复各 loader 私有 generator，`FinalWindowCheckpoint` 保存最后若干 epoch。`b445ecd:utils/lightning_runtime.py` 的前者在 scratch 运行中只读取状态，恢复仅用于 resume；后者调用 `trainer.save_checkpoint`。三次原始 hparams 都没有加载 checkpoint。

`evaluate_one_sequence` 每条轨迹局部创建 `RecursiveTrackState`，将预测提交该局部状态。写到 model 的是评测结果/诊断缓存；验证 epoch hook 清空诊断缓存并更新/重置评测指标，没有向训练 observation sampler 写入验证轨迹的递归框。

## 下一步定位建议

1. 验证频率统一为既定的 5，便于成本和日志对齐；不要把它当作恢复性能的调参手段。
2. 首先做同臂相同代码重复的前 1/10/100 步比较；记录同一批 forward 输出、BN buffers、每个 B0 梯度、Adam step 前后增量。再进行跨 commit/跨臂比较，先分清运行自身漂移与源码差异。
3. 若专门检验验证副作用，在同一训练进程的 epoch 边界记录验证前后 Python/NumPy/CPU torch/CUDA RNG、BN buffers、train flags、optimizer state、训练 loader generator；只插入一次完整验证。允许 metrics/diagnostic caches 变化，其余实际训练状态应保持一致。此项是有针对性的合同检查，不需要再跑两次 60 轮才能定位。

本文件依据源码与已有运行证据得出排除结论；没有声称已经在服务器完成验证前后状态快照，也没有增加生产代码修改。
