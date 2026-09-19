# v31 实验就绪核对（2026-09-19）

## 判断

上次停止时计划尚未完成：仓库有v31模型主体，但 `main.py` 未分流、正式31配置缺失、GRU未接入。本轮已在本地补齐三臂可运行路径，并修复检查发现的接口/训练问题。**上传本轮文件后可以启动三组mini；服务器目前未同步，不能把原目录视为已就绪。**

原计划的真实CUDA约100步、双Full共卡峰值显存、60轮训练及涨分验收尚未完成。本轮没有启动服务器训练、安装依赖、上传文件或改动服务器内容；未改历史output。

## 本轮修复

| 项目 | 当前实现 |
|---|---|
| 入口与配置 | main按net_model分流独立entry；公共31base与三臂配置，旧CLI明确拒绝 |
| 三组差异 | B0不构造B1/B2；Full-CfC与Full-GRU为不同真实cell，共同时间输入/公共头/损失/预算；cell构造不扰动公共参数初始化 |
| B0角度 | 粗框XYZ+sin/cos，decode为统一XYZ/yaw；周期监督 |
| 时序耦合 | context_valid独立于motion pair；弱历史时合法context仍可进入最终定位，sigma隔离保留 |
| 获取几何 | L沿物理运动方向；R尺寸/朝向来自可信终点自身；world XYZ先float64减anchor再float32 |
| CUDA已知兼容点 | bool不排序、整数cumsum、B2显式MHA math、native allocator、CUBLAS工作区与严格确定性 |
| 稳健聚合 | 无效模式不会复活；非有限vote/weight在聚合前排除，防0×NaN；BG vote无主候选定位旁路 |
| Adam与生命周期 | 恢复原betas(.5,.999)/eps1e-6，单次Adam；串行状态提交、每epoch课程loader更新、完整边界checkpoint |
| 评测 | 无旧标定回退；最终58/59/60自动评测；初始化帧与singleton轨迹计入一致分母；完整端点覆盖 |
| 诊断 | 获取原始/novel/可达/实际计数；目标mode按成员FG>=1且纯度>=.5；错误记忆写入/失跟恢复秒数；总耗时与GPU峰值 |

## 验证证据

- **89 passed、1 skipped**：五个v31测试文件，使用本地独立target的Lightning2.0.2。skip是当前已安装Lightning时不适用的“缺Lightning”测试。覆盖稀疏点、全空、singleton、梯度、几何、状态、严格配置、CUDA风险算子的CPU合同。
- 三臂均以真实 `JointTracker + CTSEQTRACKV31 + entry.run` 完成合成原始点3epoch训练、逐checkpoint加载和完整闭环test，均生成1/2/3三轮结果。工程配置N16/batch4/workers0；不是nuScenes成绩，也不是CUDA压力测试。
- 真实Lightning2.0.2的3epoch/epoch1中断恢复测试：最终参数、Adam张量状态、global_step和LR与不中断结果逐位相同；覆盖Dropout/RNG，worker2预取另有专项验证。
- Python3.9语法检查15个v31/host文件、compileall、git diff --check通过。没有重跑固定旧HEAD的slimming门。
- 工程记录：[三臂入口结果](../artifacts/ct_checks/v31_entry_integration_latest.json)、[运行日志](../artifacts/ct_checks/v31_entry_integration_20260919.log)。这些小样本分数只证明流程，不支持模型优劣结论。

## 服务器只读证据

- 项目 `/home/lishengjie/study/lcyu/CT-SeqTrack`，HEAD `1c078a7`；本轮补齐文件尚未上传。
- seqtrack3d Python3.9.19 / Torch2.0.1+cu118 / Lightning2.0.2，CUDA、nuscenes、pointnet2_ops、tensorboard可用。
- GPU0/1为A40 46068MiB；快照占用1716/4245MiB。没有修改或停止他人的进程。
- mini_val实际106条轨迹/2285帧；只读raw→prepare→CPUprior→acquire一行成功，真实dt0.499874秒、当前B0零点/历史1点、extension17点。这验证SDK接口与真实稀疏样本准备，不等同本轮新权重的GPU前后向验证。

## 上传与启动

上传包 `artifacts/ct_checks/ct_v31_mini_20260919_upload.zip` 保留仓库相对路径；上传后在服务器项目根解压，或按包内UPLOAD_MANIFEST.json逐文件同步。不包含数据、checkpoint、历史output或本地测试依赖。

新入口与全部models/ct_v31/文件、host、四份31配置必须一起同步，不能只复制YAML。当前包以服务器已检查的1c078a7为基础；包含复用的attention/CfC/masked算子与数据工具所需原版本由服务器该HEAD提供。

具体三条nohup与新终端tail命令见 [运行说明](CTSEQTRACK_V31_MINI_LAUNCH.md)。不使用旧--preloading；沿用每worker256MiB原始云有界缓存。

## 尚不能作出的结论

已排除旧“无标定→Full强制回退相同B0”的路径，并实测两种时序cell均有梯度、Full可以选择mode。实际S/P是否相同或上涨、两Full共GPU0时真实峰值与吞吐，需正式运行确定；不人为改seed、选epoch或改指标来制造差异。

