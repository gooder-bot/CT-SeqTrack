# `d86990c` 与当前 B0：数据通路和显存根因诊断

## 结论

当前 B0 并不是 `d86990c` 的逐事务复现。显式 DataLoader generator 确实改变了训练轨迹，但现有证据不足以把 `21.945 Success / 33.279 Precision` 的全部差距归因于 shuffle。至少还有三类实质差异：验证时 bbox-size 信息来源、验证/点采样递归合同、Lightning 自动优化改为手动分模块优化。

最重要的新发现是：历史 `d86990c` 在验证的每一帧使用 `this_frame['3d_bbox'].wlh`，即当前帧 GT 尺寸；当前正式配置启用 `observation_safe_bbox_size: true`，改为首帧目标尺寸，并通过 `RecursiveTrackState` 生成历史与点采样 seed。旧高分因此不只是“旧随机轨迹”，还包含不同的验证输入语义。

## 已排除的解释

- 两个 B0 checkpoint 的网络参数和 buffer 体积分别约 `14.183 MiB` 与 `14.188 MiB`。
- 两者 Adam 状态均约 `28.261 MiB`，均有 236 个参数状态；betas=`(0.5, 0.999)`、eps=`1e-6`、末轮 LR=`1e-7`。
- B0 参数 key 和 tensor shape 除新增审计 buffer 外一致。
- 当前 B0 完成 75,720 次 optimizer update。

因此 4.8GB→5.8GB 不是网络参数量增加 1GB，也不是 Adam 多保存了一套参数。

## 数据通路差异

### 1. observation DataLoader RNG 不等价

历史：

```python
DataLoader(..., shuffle=True, drop_last=True, pin_memory=True)
```

当前在 `main.py` 中额外使用 `seed + 31001` 的 `torch.Generator`，并给 worker 设置 NumPy/Python seed。它会同时改变 shuffle、candidate offset、随机点采样和网络 dropout 所遇到的样本序列。

这不是无关紧要的实现细节。四臂 initial B0 hash 一致，但 step 1 全部不同，证明当前 observation 随机流既没有复现历史，也没有跨臂隔离。不过四条新随机轨迹的 observation 性能从约 31 到 50 Success，说明 mini 设置自身对训练轨迹高度敏感；仅凭一个 run 不能估计 generator 对分数的精确贡献。

### 2. 历史与当前验证使用不同 bbox-size 信息

`d86990c` 的 `MotionBaseModelMF.build_input_dict`：

```python
bbox_size = this_frame['3d_bbox'].wlh
```

当前正式 B0：

```python
observation_safe_bbox_size: true
bbox_size = recursive_contract['target_size']  # 首帧尺寸
```

训练 sampler 也由 `this_box.wlh` 切换到首帧 `wlh`。`bbox_size` 会进入 B0 box-aware/decoder 数据通路，不只是最终 IoU 计算，因此也可能影响预测中心和 Precision。

历史逐帧读取当前 GT 尺寸具有目标信息泄漏风险。若论文协议要求严格单目标跟踪，首帧尺寸是更安全、可解释的输入。不能为了复现旧分数直接把它静默改回正式协议；应先做同一 checkpoint 的 paired evaluation，量化 legacy-current-GT-size 与 first-frame-size 的差值，并在论文中明确口径。

### 3. 验证递归状态与点采样发生变化

历史验证直接从 `results_bbs` 取历史框，历史点采样 seed 为 `None`、当前帧 seed 固定为 1。当前验证即使关闭 B1–B3，也因为 `observation_safe_bbox_size=true` 进入 `RecursiveTrackState/build_recursive_input_contract`，使用按 tracklet/frame 派生的确定性 point-sampling seeds。

因此当前验证不仅尺寸不同，同一 crop 中被抽到的 1,024 个点也可能不同。历史与当前 final 指标不是纯粹“相同模型、相同 evaluator”的比较。

### 4. validation cadence 不同

历史 checkpoint 记录 `check_val_every_n_epoch=5`，当前四个 run 均记录为 1。当前 validation loader 没有显式 generator。若 iterator/worker 创建消费全局 RNG，每 epoch 验证会继续改变随后训练的随机状态。

### 5. 配置中仍保留非历史字段，但多数不是 B0 主图

当前 resolved config 的 `ct_history_training_mode=correlated_candidate`、`ct_search_training_history=correlated_candidate` 与历史 canonical 不同。不过 observation stream 在训练 transaction 中关闭 B1/B2/B3，这些字段主要生成未被 B0 核心消费的 CT 辅助数据，现阶段不是首要分数根因。它们仍增加 batch 数据量和审计复杂度，应该在纯 B0 复现配置中移除或证明不进入设备/forward。

## 1GB 显存差异的解释

模型和 optimizer checkpoint 只相差约 5 KiB，故 1GB 差异来自激活、梯度临时张量、batch 搬运或 CUDA caching allocator。

历史 B0 使用 Lightning 自动优化：一次 `forward → loss → backward → step`。当前 B0 设置 `ct_separate_optimizers=true` 和 `automatic_optimization=False`，在 `_ct_isolated_optimizer_step` 中对整组 B0 参数调用 `torch.autograd.grad`，保存 gradients tuple，再赋给 `parameter.grad`，随后计算全参数 gradient norm。这个手动事务与历史执行路径不同，能够提高峰值或 reserved memory，但 checkpoint 数学状态看不出这种运行时差异。

此外，当前 sampler 返回大量 B1/B2/B3 时间、搜索和诊断字段；Lightning 默认会把整个 batch tensor 字典搬到 GPU，即使 B0 forward 最终不消费全部字段。单个字段不大，但会增加 allocated/reserved memory。当前 Transformer 还支持返回 decoder state，不过 B0 的 `use_ct_joint_full=false` 时不应启用该返回路径。

因此显存差异是“当前 B0 仍在 CT 通用训练壳中运行”的证据，但它本身不能证明 B0 数学 forward 已改变。需要服务器逐阶段记录 `memory_allocated` 与 `max_memory_allocated` 才能精确归因。

## 决定性实验

按成本和因果识别能力排序：

1. 对同一个历史 `d86990c last.ckpt` 做两次只评估：A=历史逐帧 GT size/历史采样；B=当前 first-frame size/当前递归采样。不得训练。该实验直接测量 evaluator/data-path 变化造成的分差。
2. 对同一个当前 B0 checkpoint 做相反的 A/B 评估，检查结论是否跨 checkpoint 一致。
3. 建立 `legacy_exact_b0` 100-step smoke：历史 DataLoader、历史 validation cadence、自动优化、历史 batch 字段。与 `git worktree d86990c` 比较 candidate id、offset、采样点、labels、loss、step1/100 hash。
4. 再逐项启用：独立 observation generator、safe bbox size、递归 sampling seed、手动 optimizer。每次只改一个变量并记录 hash。
5. 在服务器对首个训练 step 分段记录 CUDA allocated/peak：batch transfer 后、forward 后、loss 后、autograd 后、optimizer step 后。另记录 `memory_reserved`，避免把 allocator 缓存误认作真实活跃显存。

## 推荐决策

- 如果目标是复现历史数字：先建立只用于审计的 `legacy_exact_b0`，包括历史 RNG 和逐帧 bbox-size；不要把它直接当论文正式基线。
- 如果目标是严谨发论文：保留 first-frame bbox-size，重新建立一个无当前帧 GT 泄漏的 SeqTrack B0，并与同口径四臂比较。历史 53.360/64.382 应标记为 legacy-protocol reference，而不能继续作为 safe-protocol 的硬验收阈值，除非 paired evaluation 证明 size 口径影响可以忽略。
- 无论选择哪种口径，四臂 matched-prefix 必须先通过。否则模块消融仍不可解释。

## 当前证据强度

- 已证实：RNG 轨迹不等价、四臂 step1 分叉、bbox-size 和采样评估通路不同、validation cadence 不同、手动优化路径不同、模型/Adam 体积相同。
- 高概率：显存增加主要来自手动训练事务、额外 batch tensor 和 CUDA reserved-memory 行为。
- 尚未证实：每个差异分别贡献多少 Success/Precision。必须用同 checkpoint paired evaluation 和逐项 100-step hash 实验完成分解。

## 补充：历史帧是否把 GT 框送入验证 forward

审计 `evaluate_one_sequence → build_input_dict` 后，没有发现 `d86990c` 把历史帧 GT 框作为递归历史框送入 B0。除第 0 帧按协议使用 GT 初始化外，历史 crop 和 box tokens 来自 `results_bbs`，即模型上一帧预测。当前实现也把每帧 `candidate_box` append 到 `RecursiveTrackState`，下一帧从该预测状态取历史框。

验证 sequence 中确实包含每帧 `3d_bbox`，但要区分其用途：

- 当前帧 GT 框用于计算 IoU/中心距离，这是离线评估必需，不进入跟踪决策。
- 历史帧字典提供过去点云和 timestamp；点云是传感器观测，不是 GT 框。
- proposal diagnostics 会读取 GT 计算 oracle/gain 标签，但只在 forward 完成后记录，不改写 `candidate_box`。
- 历史 `d86990c` 的实际泄漏点是当前帧 GT `wlh` 被写入 `bbox_size`，不是历史中心/朝向 GT 被递归输入。

所以当前 B0 从 53 降到 31 不能主要解释为“移除了历史 GT 框”，因为历史 B0 本来就没有使用历史 GT 中心/朝向。更强的反证是：B1-only、B1+B2、Full 与当前 B0 使用相同的 safe history/first-frame-size 协议，其 observation Success 仍达到约 48–50。safe history 协议本身不会必然把 B0 压到 31；31 分主要仍指向不一致且高度不稳定的训练随机轨迹。

逐帧 GT size 对历史 53 分贡献多少仍未知。它可能解释历史 53 与当前 safe observation 约 50 的一部分差距，但不太可能在没有 paired evaluation 证据时被认定为 31 分的主要原因。
