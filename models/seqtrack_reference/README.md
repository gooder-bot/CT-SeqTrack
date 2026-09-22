# 独立 SeqTrack 对照

本包从只读 `../seqtrack/` 提取实际使用的网络、原始 loss、teacher 数据处理和几何函数；不调用 `models/ct_v31/observation.py` 或 production B0。来源文件 SHA256、适配说明和 vendored 文件 SHA256 位于 `SOURCE_MANIFEST.json`，MIT 许可位于 `LICENSE`。`protocol_identity()` 将固定原配置、来源与当前实现摘要纳入 checkpoint/运行身份。

`ReferenceTracker` 保留原 SegPointNet/MiniPointNet/FeaturePointNet、hard argmax 前景掩码、moving 分类掩码、Seq2SeqFormer 和原 loss 权重。原 forward 的布局、负 theta 角点旋转、未使用 valid_mask 的 attention、伪时间均原样保留。CPU 适配仅将 loss 中固定 `.cuda()` 改为跟随 logits device；Lightning 与训练日志由共同 host 管理，不引入旧 PointNet++ CUDA 依赖。

严格确定性 CUDA 的必要数值适配在独立 `numerics.py`：正式整除 adaptive max（4096→1、1024→128）改为不重叠固定分桶 max，保留首个最大值梯度；非整除/升采样工程输入仍按原 adaptive start/end 区间选首个最大值，并按桶顺序累加梯度，不静默改变重叠区间。分割 CE 保留原 `[B,2,N]` log-softmax，转为 flat NLL `reduction=none` 后全局求和，分母为整个 batch 的 `sum(weight[label])`。两项均不改变数学目标、不修改全局 deterministic 设置；浮点归约顺序可能产生舍入差，实际 CUDA 仍须服务器只读权限解除后的工程验收。

`build_loaders(config, roles, sources=None)` 输出相同 raw-row host 协议。训练每个合法非首帧登记四个 candidate（0 为零扰动、1–3 为每历史独立扰动），保留原 scale=1.25、历史 hint、newest-first、1024 点有放回补齐，以及少于三个点整帧置零的规则。原“全历史无目标点”断言触发时，均匀重采样合法非首帧 candidate；因此 4777×4=19108 是 nominal rows，batch16 为1195次更新、末批4行，不能把实际曝光也声称为严格每端点四次。`BatchBuilder.exposure_records` 保存 nominal/actual/rejected 明细；`exposure_summary()` 提供计数和明细摘要。

原随机数语义使用 NumPy 点采样/扰动及 Torch 无效样本替换：固定 seed、epoch、worker 数及恢复状态才可重现。workers0 使用主进程 RNG；workers2/4 使用 DataLoader 独立 worker RNG。不同 worker 数不承诺相同点序列，正式 workers4 必须固定。原当前评测点采样固定 seed=1，历史采样保持原 NumPy 行为。

共同 entry 在每次独立 `trainer.test` 前重置 Python/NumPy/Torch 及 test loader generator 的登记 seed，使自动58/59/60与独立 `--test` 的随机起点一致；这不改变历史逐帧采样算法。训练中验证保留已有生命周期。Windows 的原 `np.astype('int')` 会生成 int32，builder 只把 `seg_label/motion_state_label/valid_mask` 转为 int64，连续输入和标签不自动换精度。

训练与评测的网络 `bbox_size` 输入均只从合法首帧取得；这是登记的共同输入适配，原版训练使用当前 GT 尺寸的行为在此明确更改。原 teacher 的历史扰动、裁剪、分割/点框距离标签和 loss 目标保持原处理，后续 GT 尺寸仍可用于训练标签与 evaluator。评测每帧裁剪和局部坐标由 accepted 预测框决定；首帧重复历史、历史 0/1 或 0.2/0.8 hint 及当前 0.5 hint 与原方法一致。最终 local offset 转回世界坐标供统一 `TrackingEvaluation`，每轨迹仅一次完整递归，首帧/单帧轨迹均计入指标。原模型无 quality head；统一接口的 `selected_quality=1` 只是占位，不参与记忆或可信状态，不报告为校准质量诊断。

接口：`plan_prior(batch)` 返回 None；`forward(batch, prior=None)` 返回保留原输出键的 dict 子类，并附 `accepted_box/selected_index/selected_quality/evidence`；`compute_losses(batch, output)` 返回原 `loss_total` 字典。`BatchBuilder.prepare/acquire/commit/reset` 与共同 host 一致。末批 running BN 由共同 host 对整个参考模型应用；禁止额外套用 production 的观测预处理、GT 标签缩放、token 或软前景修改。
