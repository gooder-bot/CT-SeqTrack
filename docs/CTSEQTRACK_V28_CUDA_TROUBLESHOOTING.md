# v28 CUDA 故障与后续工作提醒（2026-09-09）

2026-09-11性能修改提醒：relation AP计算提取到`utils/v29_diagnostics.py::relation_rank_metrics`，
原`int64 cumsum → 原dtype`路径保留，旧binary AP仍在host。不要因减频而认为非确定性算子可以保留，
首次周期日志或epoch尾批flush同样会执行它。`self.log(on_epoch=True)`仍每步更新；H3内的
deterministic异常必须继续报错，不能被影子诊断的missing处理吞掉。新增检查/命令见
[v29性能记录](CTSEQTRACK_V29_PERFORMANCE.md)。

这份记录用于后续排错与重启前复核。**9月10日更新：三组已在保留严格确定性的服务器环境完成60轮/75720次B0更新，两种显式执行阻断已消除。** 但B0/42仅45.200/47.243，B0/52为52.876/64.478，基线尚未稳定恢复；独立重复及epoch边界恢复也不由训练完成自动证明。实际结果与证据见[三组分析](../artifacts/ct_checks/reports/20260910_v28_mini_three_arm/REPORT.md)。以下保留9月9日报错与修复历史；最新排程和命令分别见[正式协议](EXPERIMENT_PROTOCOL.md)与[服务器运行说明](CTSEQTRACK_V28_SERVER_RUNS.md)。

## 事件与定位

| 服务器反馈 | 出错位置与原因 | 修复及版本 | 结论边界 |
| --- | --- | --- | --- |
| B0/42、B0/52、Full/42 都报 `nll_loss2d_forward_out_cuda_template` | B0 分割 logits 为 `[B,2,4096]`；原 CE 被分派至空间 NLL，CUDA mean 归约使用 atomicAdd，strict deterministic 拒绝执行。 | `3b8cc57`：保留三维 log_softmax，只把 log-probability 转成二维 NLL。 | 三组共享 B0 目标，故一起退出；本次异常与两组 seed 不同无关，不构成历史降分的因果证据。 |
| CE 修复后，只有 Full 报 `cumsum_cuda_kernel`，日志指向 `precision_at_k` | B2 relation AP/AUPRC 诊断对浮点形式的0/1标签做前缀累计，触发该 CUDA 路径的确定性限制。B0 没启 B2，不进入此统计。 | `749bc13`：relation AP 及旧 binary rank AP 两处均使用 int64 累计后转回原浮点类型。 | 该指标未加入 B2/B3 总损失；保持数学定义，没有更改网络或训练预算。只有 Full 报错不表示已证明 B3 模型有问题。 |
| 两次主错误后均出现 `tqdm.__del__` / `cannot unpack non-iterable NoneType object` | 前面异常中止训练后，进度条清理再次报错。 | 修复前面的第一个 RuntimeError；不以升级 tqdm 作为本次修复。 | 这是上述日志中的次生异常，不是独立的训练根因。 |

用户最初提供的三组失败日志位于 `artifacts/ct_checks/v28_mini_parallel_20260908_235357/`。随后已要求正式结果改为本地同款日期目录：`output/YYYYMMDD-HHMMSS-28_模块-mini_car_seedXX_60ep_bs16/`；不再沿用旧 `RUN_ROOT` 的实验结果位置。9月9日排错时只有用户粘贴的服务器异常片段；9月10日已读取同步的完整结果与provenance，确认实际PyTorch2.0.1+cu118、Lightning2.0.2、A40。完整训练通过与历史低分根因是不同结论。

## CE：保持原加权目标，换掉非确定性归约

运行入口是 [observation_reference.py](../models/ct_v2/observation_reference.py) 中的 `seqtrack_segmentation_cross_entropy`。实现的关键顺序如下：

```python
log_probabilities = F.log_softmax(logits, dim=1)
rows = log_probabilities.movedim(1, -1).reshape(-1, 2)
loss_seg = F.nll_loss(
    rows, labels.reshape(-1),
    weight=logits.new_tensor([0.5, 2.0]), reduction='mean',
)
```

保留 `[B,2,N]` 上的原 log_softmax；不要先 flatten logits 再计算 softmax，以免额外改变其 CUDA 内核与浮点路径。NLL 的分母仍为全 batch 非忽略标签对应的类别权重之和，不能改成除以点数、逐帧/view 均值或四组平均。BC、moving-row 分母及其他损失不变。

`ct_b0_ce_contract=class_axis_logsoftmax_flat_nll_v1` 已登记到 [28_formal_base.yaml](../cfgs/ct_seqtrack/28_formal_base.yaml)，并由 [online_contract.py](../utils/online_contract.py) 验证及绑定 resume；配置也进入校准身份。因此 CE 修复需要三份运行文件一起同步，不能只改 YAML。B0、B0 seed52、Full 的已有 v28 配置自动继承，不需要新增关闭确定性的 CLI。旧失败 checkpoint 不作为新 run 初始化。

CPU 回归中原 CE 与修复实现的 loss、梯度及完整 B0 更新逐位一致；旧 CUDA 原子归约的求和次序与新路径可能有末位舍入差异，不能据此宣称 CUDA 新旧逐位等同。支持根因判断的官方源码为 [NLL2d 的原子归约与严格拒绝](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/NLLLoss2d.cu#L209-L221)、[CE 分派](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/LossNLL.cpp#L614-L674) 与 [二维 NLL 归约](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/Loss.cu#L200-L242)。这些来源核对的是该实现路径，不代替服务器环境记录。

## AP：二值标签用整数累计

[seqtrack3d.py](../models/seqtrack3d.py) 中两个修改位置分别是 `_compute_ct_contract_v3_loss` 的 `precision_at_k` 和 `compute_loss` 内 `binary_rank_metrics` 的 `precision`：

```python
counts = torch.cumsum(ordered_target, dim=0, dtype=torch.int64).to(dtype)
```

这里的 target 已经阈值化为0/1；整数累计后转回浮点保留原计数、排序与分母，当前 FP32 计数规模可精确表示。`ct_relation_ap` / `ct_relation_auprc` 仅作诊断，不加入 `loss_ct_plugin_total`。不必把模型或指标搬到 CPU，也不必改训练协议、模块优化或冻结状态。AP 修复的增量运行文件只有 `models/seqtrack3d.py`，前提是前面的 CE 修复已同步。

[官方 ScanKernels.cpp](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cuda/ScanKernels.cpp#L81-L88) 仅对该浮点/复数累计路径发出确定性拒绝；[ReduceOps.cpp](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/ReduceOps.cpp#L439-L460) 在分派前按请求的 dtype 转换输入，因此显式 `dtype=torch.int64` 会选择整数累计。不要将这个针对二值计数的做法推广为任意实数累计的整数截断。

## 为什么之前的本地检查没发现

本地是 CPU PyTorch 环境；CPU 上 CE 与浮点 cumsum 可以正常执行，所以即便真实网络与完整 host 事务测试通过，也不能揭示服务器 CUDA 内核的 strict 支持限制。原先已处理 [DeterministicMaxPool1d](../utils/deterministic_pooling.py) 的固定槽池化，但只处理池化没有覆盖损失与日志指标，先后遗漏了这两处。

后续修改时同时考虑前向、反向、优化器和训练日志实际执行的算子。保留 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、FP32、关闭 TF32/benchmark、严格 deterministic、Adam foreach/fused=false；不通过关闭 deterministic、`warn_only=True`、换 seed 或冻结模块跳过异常。固定最大池化仍使用 `max` 的首个最大值梯度语义，不能随意换为 `amax`。

## 已有验证与仍缺的证据

| 范围 | 当时实际结果 | 不能由此推断 |
| --- | --- | --- |
| 9月8日实施全量回归 | 453 passed / 12 skipped | 这是修复前的历史快照，不是当前 HEAD 全量结果。 |
| CE 修复专项 | 47 passed / 1 skipped；修改文件 compileall、diff 通过 | 唯一跳过项为 CE CUDA 重复测试；CPU 等价不等于 CUDA 全流程验收。 |
| AP 修复专项 | 15 passed / 1 skipped；修改文件 compileall、diff 通过 | 非空/空支持、B2/B3 梯度归属、v28 B0/Full host 与 Adam hooks 已覆盖；跳过项仍是 CE CUDA 测试，未执行真实 CUDA AP/Full 训练。 |

两次专项测试覆盖重叠，数量不能相加后称为全量通过。AP 回归在真实 host loss 中强制要求整数 cumsum，并与旧 CPU 二值浮点累计逐位比较；这能防止代码回退，但不能模拟所有 CUDA 行为。详细历史证据见[本地验收](CTSEQTRACK_V28_LOCAL_VALIDATION.md)。需要修改代码并复核同类路径时，从仓库根目录按需运行：

```bash
# CE 专项（本次记录为 47 passed / 1 skipped）
python -m pytest -q tests/test_ct_v28_deterministic_ce.py tests/test_ct_v28_control_plane.py tests/test_ct_v28_b0_reference.py::test_reference_complete_batch_losses_bc_gradient_and_adam_update tests/test_ct_v28_audit_host.py

# AP 专项（本次记录为 15 passed / 1 skipped）
python -m pytest -q tests/test_ct_v27_host.py::test_real_host_loss_selected_presence_counters_and_gradient_ownership tests/test_ct_v28_audit_host.py tests/test_ct_v28_deterministic_ce.py
```

以上是复核入口，不要求在用户仅索取命令或更新文档时反复运行。服务器后续仍需确认修复后的真实完整训练事务（包括正例存在时的 AP）、反向/Adam、逐位重复与 epoch 边界恢复。正式成绩固定 final60 和58/59/60平均；B0 seed52 是已登记的第二组，不能挑较好的 seed 代替失败结果。不得把这两次显式算子报错当作 v25/v26 历史分数崩坏的全部根因。
