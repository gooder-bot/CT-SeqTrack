# v27 正式 observation 的 BC 重复权重：真实路径复核

2026-09-07；只读静态审计。数值复现由同轮checkpoint审计独立完成。

## 结论

五个正式CT臂的观测优化都受BC项重复计算影响，而且**四个candidate全部双计**，不是仅canonical view0双计。B2/Full的插件加法不能在该真实训练路径解除别名，因为observation执行期间插件开关全部临时关闭。

令 `A_i` 为view i除BC外的观测loss、`C_i=bc_weight*loss_bc_i`，实际反传为：

`0.5*(A_0+2*C_0) + sum_i=1..3 (A_i+2*C_i)/6`。

预期是每个括号仅有一个 `C_i`。原reference走 `loss_total`，BC仅加一次；这一点与双方共同继承的FeaturePointNet reshape问题不同，是CT事务loss引入的额外目标偏移。

## 数据与调度

1. `main.py:807–820` 为所有dual_stream臂创建相同observation配置：B0、independent candidate、4个candidate、`ct_observation_payload_mode='seqtrack_core'`。
2. `main.py:940–957` 使用 `StatelessCandidateBatchSampler`，每个16行batch包含四个candidate各4行，进行一个普通batch前向。
3. `datasets/sampler.py:2298–2301` 调用payload prune；`utils/sampling_utils.py:9–16` 的白名单含 `candidate_id`，**不含** `ct_b0_auxiliary_only`。普通observation本来也没有此标记，在线raw路径 `sampler.py:2709` 才创建该标记。
4. `models/seqtrack3d.py:8830–8834` 在递归调用observation `training_step`前，将 `use_ct_joint_full/use_b1motion_v3/ct_enable_b1/ct_enable_b2/ct_enable_b3` 全部置False；直到整次observation `training_step`返回，才于8841–8844恢复。B0、CfC、GRU、B1+B2、Full全部如此。
5. 因此用于损失分支的B2条件也为False，6341行 `loss_total = loss_total + plugin` 不执行。

## 别名与真正反传的值

- `seqtrack3d.py:5809`：`b0_transaction_loss = loss_total` 仅创建同一Tensor对象的别名，没有clone或非原地值构造。
- 普通observation不存在aux标记；`_b0_auxiliary_batch` 第3381–3383行对缺失字段返回False，所以不进入5826的非原地aux分支。
- `seqtrack3d.py:7498`：`loss_total += weighted_BC` 原地改变该Tensor，也改变别名指向的值。
- 第7499–7500行再次执行 `b0_transaction_loss = b0_transaction_loss + weighted_BC`，事务值成为 `A+2C`，而 `loss_total` 仍是 `A+C`。
- `seqtrack3d.py:8921–8925` 对safe_auto普通observation调用 `_ct_candidate_weighted_observation_loss`。
- 该函数在8642–8647逐candidate切片，再次调用 `compute_loss`。切片不会凭空添加aux标记，四个view全部重复上述路径。它以 `loss_b0_transaction` 做0.5/1/6加权，写回 `loss_total` 和事务loss。
- `seqtrack3d.py:8954–8964` 最终又按active module={'b0'}取事务loss，故反传的确是重复BC后的目标，不只是日志显示问题。

## 两个容易混淆的分支

旧在线aux微批：`_ct_b0_auxiliary_microbatch_gradients` 第8588行要求aux标记，并走5826/5827两次非原地加法，各自得到 `A+C`。这不是当前正式dual_stream observation候选分支的路径。

mechanism：B2/Full可执行6341非原地插件加法，从而使其BC事务日志行为不同；但机制流只选择B1/B2/B3事务训练，B0输出本来no_grad。它不能修正已经完成的observation梯度，也不能被用于推断B0正式BC权重。

`cfgs/27_seqtrack_reference.yaml` 不启用safe_seqtrack_auto/unified_auto；普通reference训练取 `loss_total=A+C`。损失修复应重新构造干净的B0 objective，避免总loss与分组loss共享可被原地更新的对象，并用真实四candidate普通observation路径的梯度测试验证。

本问题确定地改变了训练权重，但不能仅凭此断言它解释了全部跟踪低分或第一次跨臂分叉：五个CT臂仍受相同错误影响。量化得分贡献需要匹配训练对照。
