# v30 +5GB预算取舍验证

用户目标：允许相对旧B0约增加5GB，优先涨分和实验迭代。详见[依据与决定](../../../docs/CTSEQTRACK_V30_MEMORY_TRADEOFF.md)。
本轮未连接服务器，使用上一轮只读留存的第50步CUDA证据；未改历史output或冻结参考目录。

## 实现

- `ct_b0_masked_bn_recompute:false`：正式共同base默认直接计算部分有效BN。
- true：保留原局部重算；host绑定三PointNet的BN，flag进入v30执行/恢复身份。
- 保留标量where、重复mask优化、原BN统计/少于2有效元素退路、CUDA int64探索排序及native分配器。
- 所有模型参数、训练超参数、点数/历史/roll-in/模式/动作预算不变。

## 定向回归

```text
python -m pytest -q tests/test_ct_v30_masked_memory.py tests/test_ct_v30_memory_host_equivalence.py tests/test_ct_v30_observation.py tests/test_ct_v30_host.py tests/test_ct_v30_contracts.py --tb=short
102 passed, 2 skipped in 35.23s
```

两个跳过是direct/recompute各自的CUDA BN测试。两模式真实完整B0输出、loss、全部梯度、
BN、Adam状态/更新和RNG分别对照首次v30逐位相同；三臂默认更新一致；16份v30配置通过。
开关类型及执行身份检查、修改Python compileall、git diff --check通过。
没有重跑整个测试套件或旧HEAD门禁。

## CPU配对结果（不外推CUDA）

PyTorch2.3.0+cpu，1线程，实际源码AST加载三PointNet；query batch2/history3/1024点每帧。
Seg[2,14,4096]、Mini[2,13,4096]、Feature[8,14,1024]。
同一个测量mask的第一行前1024槽无效，三支各有7168个有效槽。
每路径一次预热后D/R/R/D，各两次。D直接，R重算。

| 顺序 | 路径 | 前向秒 | 反向秒 | 总秒 |
|---|---|---:|---:|---:|
|1|D|0.6617933|1.2129813|1.8747746|
|2|R|0.6249176|1.5362050|2.1611226|
|3|R|0.5857303|1.5386308|2.1243611|
|4|D|0.6390953|1.1829615|1.8220568|

按底层storage去重的保存量（含三支输出平方均值损失）：D702.138929MiB，R401.560804MiB。
均值D1.8484157秒/R2.14274185秒；R约多15.9%时间、少42.81%保存量。
这里只是小型CPU算子对照，未包括完整host、IO、CUDA分配器；不能据此声称已优化CUDA吞吐或降低实测进程占用。

独立审阅初次测量曾缩小Mini/Feature的点数，另一次恰逢共享实现变更；这两次均舍弃，
不使用其25.7%或相同耗时数字。上表为最终同query形状结果。
