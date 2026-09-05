# v27 训练与 checkpoint 二次审计（2026-09-05）

v27 的模块、数据通路与实验配置已接入。二次审计补上了第一轮合同测试没有覆盖的真实模型训练、Lightning 保存/恢复及离线 Full 加载问题。本记录检查代码，不审计服务器环境。

当前可以进入服务器 mini 真实数据短跑验收。尚未运行真实 nuScenes 100-step 和完整60轮，不能将本地通过解释为所有真实数据情况都已验证，更不能保证涨分。最新完整测试结果见 [审计摘要](../artifacts/ct_checks/v27_training_reaudit/summary.json) 和 [pytest输出](../artifacts/ct_checks/v27_training_reaudit/pytest.txt)。

本轮完整回归：**345 passed，2 skipped**，101.90秒；两项跳过为CUDA AMP overflow smoke，v27正式训练使用FP32。compileall与`git diff --check`通过，历史配置与保护输出的tracked diff为空。

## 计划逐项核对

| 范围 | 当前代码状态 |
|---|---|
| 场景与预算 | full为350 train_track训练/150官方val评测；内部17/18子集不扣减训练；mini6/1/1。六运行配置、manifest、缓存和sampler合同已有测试，实际数据覆盖由preflight读取数据后验证。 |
| 公共数据通路 | 原始point ID、valid/unique mask、0/1/2点、无重复FPS、memory轴、空base恢复、唯一effective clock均已接线；真实sampler和共享缓存无原地污染测试通过。 |
| B1 | CfC/GRU共用输入与获取合同；学习输出与实际AcquisitionRecord分离；独立margin分支、81网格novel-target标签、长目标投影几何已实现。 |
| B2 | 768→256、128/96/32选择、局部几何、目标/上下文条件、独立类别计数、targetness-only投票、长车vote范围与134维presence已实现。 |
| B3 | 实际有界动作、H1/H3 S/P收益、独立help/harm、完整闭环阈值选择、每checkpoint独立artifact已实现；缺artifact时训练照常，评测输出observation并标记状态。 |
| 梯度与状态 | 实际六臂模型前向/损失/反传通过；各启用参数组有限非零梯度，未冻结参数。host独占递归提交，训练提交observation，shadow不写回。 |
| 完整训练事务 | Full通过真实Lightning、真实PointNet/Transformer与插件、实际raw sampler→prepare/process→两次观测更新/四个递归tick、TensorBoard、epoch-end与保存测试。 |
| 公平性 | 实际五臂网络两步B0梯度/参数/BN/Adam/Torch RNG逐位一致；多tick反传不改变B0 BN版本或梯度。CUDA同臂重复与跨臂100-step比较仍在服务器完成。 |
| 恢复与校准加载 | 普通/双流 × last/epoch文件的连续训练与恢复对照通过；真实Full完整state_dict与RNG extra_state离线恢复通过。 |
| 结果与论文 | 六运行和完整五类矩阵、late-3、双指标和证据漏斗已有实现；真实分数、后端优劣、耗时/显存、配对区间仍需实验。 |

## 本次发现并修复

1. **稀疏观测仍会因GT历史点数被重抽。** v27包括SeqTrack参考现在保留全空历史样本，依靠已实现的padding与mask处理；避免偷偷改变训练样本分布。
2. **分类标签dtype依赖平台。** v27的segmentation和motion-state标签显式使用int64，满足交叉熵输入合同。
3. **late-3文件保存早于host收尾。** `FinalWindowCheckpoint`改用Lightning的Checkpoint回调基类，确保在模型epoch-end之后保存。v27只有完成epoch收尾才标记可续训，最后batch本身不再冒充完整epoch。
4. **续训shuffle偏移。** Lightning setup的迭代检查会消耗DataLoader generator；恢复时在首个epoch开始再次安装保存的状态。
5. **Full校准/导出加载会对dict访问.shape。** 离线加载器现在区分张量和`_extra_state`，两个插件随机流可以完整恢复。
6. **续训配置合同遗漏部分v27损失/动作设置。** 新增完整规范化配置比较，排除日志路径和运行中生成的统计字段；变更后端、损失或几何设置会在训练开始前明确报出差异。
7. **epoch诊断残留旧标签。** selected-presence改用实际256点集合；help/harm与损失共用有界动作S/P标签，零收益不会因没有目标点而强制记harm。带符号q不再当作概率计算ECE。

## checkpoint 的使用边界

训练到保存点只写出当前运行的模型、Adam、StepLR和随机状态，不会加载其他实验的checkpoint，也不需要冻结或切换训练阶段。`last.ckpt`用于最近完整epoch续训，58/59/60文件用于固定论文比较。

续训沿用相同27配置、类别、seed、训练集合、后端和训练预算；`epoch`继续保持60，不能改成“剩余轮数”。建议保留原日志目录：

```bash
python main.py --cfg cfgs/ct_seqtrack/27_full.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --checkpoint RUN_DIR/lightning_logs/version_0/checkpoints/last.ckpt \
  --log_dir RUN_DIR
```

其中checkpoint路径以实际文件为准。v24/v25/v26、其他臂/类别/后端、mini与full互换、改训练设置的文件不属于同运行续训。不要将工程短跑checkpoint接到正式60轮训练；正式运行仍全部scratch。本轮修复后生成的checkpoint才是当前验证对象。

真实恢复对照使用CPU与`num_workers=0`；Linux多worker仅完成代码时序复核，当前入口没有启用persistent_workers。服务器继续执行原方案的数据预检与100-step检查即可，命令见[实施记录](CTSEQTRACK_V27_IMPLEMENTATION.md#服务器下一步)。同运行恢复还应在真实数据上验证一次完整epoch边界，再开始正式六运行。

本次没有修改历史配置、保护输出或参考仓库，没有引入环境检查或新的训练晋升门槛。

## 本次 mini 启动参数补充

用户随后指定workers=4、preloading、每5轮验证。CT mini共同配置已固化4/5，完整数据配置保留原12/1；预加载在启动命令中显式开启。

针对验证间隔5，main显式在train epoch-end保存last，避免默认validation-end保存未收尾状态。DataLoader callback同时处理恢复后延迟发生的验证loader初始化，记录是否曾初始化，只隔离重复setup的种子消耗；每次真正验证照常推进随机状态。真实Lightning测试覆盖首次验证前第1/4轮恢复，以及已验证后的第58轮恢复到60轮。

每次v27运行新增顶层`resolved_config.yaml`，可直接用于后续校准/同run恢复，保留CLI覆盖后的配置身份。后台命令见[mini启动说明](CTSEQTRACK_V27_MINI_LAUNCH.md)，本次针对性结果见[启动验证摘要](../artifacts/ct_checks/v27_mini_launch_validation/summary.json)。

## 服务器混合轨迹 batch 报错修复

服务器Full在epoch0约21/1057步报`KeyError: motion_margin_global_novel_target_count`。根因是sampler仅在有效搜索区域存在时生成四个margin统计字段：同batch的成熟轨迹和首个query字段不一致，成熟样本为首行时default_collate报错，首个query为首行时则可能静默遗漏成熟样本的统计字段。该错误发生在组批阶段，与checkpoint保存和GPU显存无关；后续tqdm析构异常是退出时的连带现象。

`datasets/sampler.py`现为所有B1样本初始化四个float32标量计数，再在完成获取标签计算后覆盖。无合法支持时仍使用最小margin占位、target_valid=0；实际点云、获取范围、标签算法及参数训练策略不变。预加载缓存保存原始帧，可继续复用。

此前单槽/同阶段模型测试遗漏了异质轨迹合批。新增九类历史/可见性状态、GRU/CfC、两种首行顺序的完整schema检查，以及Full与B1+B2各16条独立轨迹混合frame1/3/8的实际合批/前向/反传测试。结果见[修复验证摘要](../artifacts/ct_checks/v27_mixed_batch_fix/summary.json)。所报告Full运行未完成一个epoch，修复同步后应以新目录重新scratch启动；其他启用B1的臂同样需要此修复，B0不经过该分支。
