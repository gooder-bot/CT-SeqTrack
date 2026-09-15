# v30 CUDA 排序与 B0 显存修复

基准代码：`8b3a4df`。详见[完整分析](../../../docs/CTSEQTRACK_V30_CUDA_MEMORY_FIX.md)。
原用户日志：`C:/Users/25227/.codex/attachments/c8713893-98cf-4123-a232-b331ad57ed47/pasted-text.txt`。
服务器只读：`lishengjie@10.109.252.89`，项目`/home/lishengjie/study/lcyu/CT-SeqTrack`。
没有远端写文件、上传代码、发起训练或停止已有任务。

## 原始证据

- 两组Full分别在4/1262处因CUDA bool sort失败；排序位置`utils/v30_policy.py`探索分支。
- B0日志最终90/1262，收到SIGTERM15，未记录OOM。只读检查时所给3个PID均已退出。
- B0 TensorBoard唯一已落盘CUDA采样step49：forward allocated6969.0488MiB，peak7059.3389MiB，
  backward后121.7471MiB，step后121.7466MiB，step reserved34672MiB，peak reserved34716MiB。
- 历史v29同batch16/FP32：实际peak3979.6–3983.9MiB、reserved7942MiB。

已只读下载该B0主事件文件到`b0_events_step49.tfevents`，从中提取20个CUDA指标到
`b0_cuda_metrics.json`；未改远端文件。事件采样只有step49，不据此声称拥有90步完整显存曲线。

## 修复

排序int64保持stable及原槽顺序；部分有效BN逐层重算原公式，保留输入和一次running更新；
减少mask副本；启动allocator为`backend:native`、移除64MiB禁止分割限制。
所有训练超参数、三臂共享B0、目标、原始点预算及严格确定性保留。

CPU独立保存量：三PointNet批2，部分无效719.813→400.731MiB（-44.33%），
全有效419.300→400.796MiB。12种BN组合与12种PointNet场景的输出/梯度/BNbuffer直接对照相同。
最终标量where保留旧WhereBackward布局，同时消除zeros_like分配。实际完整B0的旧/新实现对照
输出、所有loss、全部参数梯度、BN、RNG、Adam状态与参数更新逐位一致（独立测试1 passed，8.52秒）。
实际CUDA修复后峰值和吞吐未测，不承诺35GB已恢复到5GB。

## 验证

最终定向回归：**68 passed, 3 skipped，35.47秒，exit0**。三个跳过分别为CUDA BN、CUDA动作排序、
CUDA CE检查。本地PyTorch2.3.0+cpu，CUDA不可用；未执行修复版服务器训练。

```text
python -m pytest -q tests/test_ct_v30_masked_memory.py tests/test_ct_v30_memory_host_equivalence.py tests/test_ct_v30_observation.py tests/test_ct_v30_rollin_loss.py tests/test_ct_v30_host.py tests/test_ct_v30_policy_cuda_sort.py tests/test_ct_v30_modes_actions.py tests/test_ct_v28_deterministic_ce.py --tb=short
```

覆盖旧/新真实host数值、共享三臂B0、空点/少点/历史缺失、八步no-grad加一次梯度端点、
六动作合法探索及历史确定性CE。新增/修改Python文件compileall通过，git diff --check通过；
最新mini文档6个Bash块语法通过，命令生成器GPU0/1/2和native环境值一致。
未重复旧HEAD slimming或整个868项套件；本次结果不可与历史套件数量相加当作新的全量结果。
