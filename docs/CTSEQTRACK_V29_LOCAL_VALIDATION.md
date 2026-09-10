# v29 本地验收（2026-09-10）

后续checkpoint间隔调整仅进行了三臂配置解析、保存调度轻量检查及修改文件语法检查；没有重跑全套pytest或哈希。
下述571项结果是此前v29实施验收记录，不冒充本次重新执行。

本页仅记录本地实际执行。工程产物位于
`artifacts/ct_checks/reports/20260910_v29_implementation/`。
本地 Windows 有CPU PyTorch，没有真实nuScenes和服务器Lightning/CUDA环境。

## 最终检查

`python -m pytest -q -ra`：**571 passed、14 skipped，236.69秒**。
原始日志为该目录的 `pytest.log`，退出码/耗时为 `pytest_result.json`。
跳过项为缺少真实Lightning的4项（含收集期跳过模块）及10项CUDA验证，不能当作服务器通过。

- `python -m compileall -q models/ datasets/ utils/ tools/ main.py`：通过。
- `git diff --check`：通过。恢复了大文件未改行的原始行尾，避免把整文件格式差异混进代码审查。
- 新服务器文档全部5个 Bash 代码块通过 `bash -n`，没有执行其中的训练命令。
- 修改/新增的40个Python文件通过Python3.9语法解析；不据此声称已在服务器Python3.9完整执行。
- 未向`output/`写入产物，git未报告该目录或历史`28_*` YAML的变更。
- `python tools/verify_ct_slimming.py verify`：在固定HEAD门槛失败，期望
  `001951a3aee15fdad6e5d5e32ca02d87bf083f3a`，实际
  `749bc13a0143fcdedba5cee465e2273c639e3b17`。这是已知的历史工具约束，
  没有修改验证器来放行，也不将其写成“全项通过”。

## 覆盖及结论

新增回归覆盖 frame/channel/point 哨兵、local/global/cross与corner query mask、
全无效注意力/空段hold、0/1/2/3点身份、Seg/BC/ref真实GT标签、物理coarse监督、
局部窗口早帧/尾帧、扰动初始化软prior、改变当前/未来GT不改变输入、teacher/roll-in混合collate、
训练次数、所有缓冲区/训练标志/RNG恢复。

真实batch16三臂网络回归比较 B0 前向、全部loss、梯度、BN、Adam状态与更新参数逐位一致；
完整 `training_step` 还加入两次连续Full机制事务，验证accepted状态被下一帧读取，
B0梯度/更新/RNG/观测指纹仍一致。插件的detach与BN隔离不构成冻结。

获取验证覆盖learned/CV多yaw、实际B0 Z包络、raw-ID差集、最大合法可达/不可达、
9×9 margin与真实crop成员、AcquisitionRecord及v5 CSV导出。Full-GRU/CfC真实输入中改变GT
不改变B1/B2前向输入；B3低presence但结构合法的候选仍可执行和学习，H3标签不再改变即时q损失。
新配置、resume及策略身份拒绝旧/错误语义，工具计划测试不伪装CUDA实测。

首轮全量曾出现8项旧host兼容失败：轻量host没有新增`ct_enable_v29`属性，
AcquisitionRecord接线直接读该属性。现改为缺省false，保留旧调用行为，相关41项定向测试通过。
另一个新增prior断言先因float32与Python float64逐位比较失败，现用相同float32期望值；
未通过放宽生产数值确定性解决测试。

## 仍须服务器执行

真实数据preflight、同卡B0重复/两Full各100步逐位比较、各臂完整epoch边界恢复，
以及完整60轮正式实验均尚未在本地执行。CUDA实际检查须包含损失、日志AP、池化和新注意力
的前向/反向，不能使用warn_only。运行[服务器脚本](CTSEQTRACK_V29_SERVER_RUNS.md)后
记录真实吞吐、GPU/RAM峰值、端点覆盖和行为分布。

所有服务器工程checkpoint排除在正式初始化之外。代码通过不是分数恢复证据；最终仍需
58/59/60的正式S/P、两Full逐checkpoint策略拟合及同权重闭环对照。
