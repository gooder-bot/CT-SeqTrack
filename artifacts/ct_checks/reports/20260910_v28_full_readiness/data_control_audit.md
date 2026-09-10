# v28 完整 nuScenes 数据与训练控制面审计

日期：2026-09-10。范围为用户授权的一次 Full / Car / seed42 / 60 epoch 完整数据诊断，不将 mini 尚未稳定恢复解释成禁止该诊断。只读审计生产代码；本文件记录核查结果。

## 结论

训练控制面具备启动单卡完整 nuScenes 的条件；未发现完整配置被 mini 的 1262 步约束拦截、丢弃机制尾部、少用训练场景或漏保存 late-3 的问题。真实数据可读性、实际样本数、服务器 RAM 与当前 CUDA 环境仍需服务器确认。

发现一个正式训练入口之外的工具缺陷：审计时 `tools/preflight_ct_v28.py` 从 YAML 直接构建配置，未补 `preloading`，而完整和 mini 的 28_* 配置都没有该字段。实际 dataset 路径读取 `config.preloading` 会失败；`main.py` 的 argparse 提供此字段，正式训练不受影响。已经向主代理报告，需修复工具默认值后再交付 preflight 命令。

## 已核对内容

| 项目 | 核查结果与依据 |
|---|---|
| 完整配置 | `28_full_nuscenes_full.yaml`、`28_b0_nuscenes_full.yaml` 都存在；本地解析、`configure_ct_variant` 与 `validate_scratch_training_contract` 均通过 |
| 数据根 | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/`；`version=v1.0-trainval`；建议命令显式传 `--path`，本地不能验证服务器文件存在 |
| 场景 | `utils/v28_protocol.py` 恢复全部350个 train_track；150个官方 val 只评测；calibration/dev 为已固定的17/18个训练内场景，明确参数训练重叠 |
| 类别与预算 | Car、seed42、batch16、workers12、60 epoch、每5轮完整验证；Adam lr=1e-4、StepLR每20轮乘0.1；同一GPU完成一组，不支持递归模型多卡DDP |
| B0观察索引 | 4候选完整展开后全局shuffle，`len(dataset)=4*总训练帧数`，自然 `floor(len/16)` 步；drop_last只影响最后不足16个观测索引，不按四候选配平 |
| mini限定 | 完整配置显式 `ct_v28_expected_mini_car_updates_per_epoch: null`，`ct_b0_steps_per_epoch: 0`；`validate_v28_observation_updates` 只对mini Car验证1262 |
| 原始训练对象 | `datasets/__init__.py` train角色 min_points=-1，不以首帧点数过滤训练轨迹；观测重抽只在原三段GT历史全空时触发，允许改变候选，64次上限并留存原/实际索引 |
| 机制覆盖 | `OnlineRecursiveBatchSampler` 看到场景manifest后使用完整轨迹模式，所有长度>1轨迹的frame1到末端各一次；16槽长轨迹均衡，最后不足16槽仍输出，长度取max而不是历史min |
| 双流调度 | `DualStreamLoader` 每轮完整消费一次机制流；按整数累计比例散布于B0更新中，v4支持必要时一个B0事务多个机制tick，host按该事务各tick实际行数加权 |
| 验证集合 | v28在`main.py`传role=val，数据层归一到test，所以每5轮实际为完整150个官方val；旧`precision/mini_val`指标名不改变场景集合 |
| 保存 | `ModelCheckpoint`每个完整epoch结束后保存last；`FinalWindowCheckpoint`保存`formal_checkpoints/epoch=058.ckpt`、059、060，无需对应轮次恰好验证 |
| 恢复 | scratch不接受预训练初始化；同合同完整epoch边界checkpoint绑定数据版本、场景manifest、seed、数值设置与运行环境，恢复RNG、loader generator、Adam和调度状态；不能从mini checkpoint续训到完整集 |

完整数据自然步数用人工1200004行样例进行了控制函数验证，结果为75000步/轮、余4行；这仅证明非1262数量可通过，**不是服务器真实数据统计**。

## Full预加载资源风险

`--preloading` 会让 observation、mechanism、validation 分别构造数据对象。`NuScenesMFDataset._load_data` 逐轨迹逐帧读取整幅点云并保存在嵌套list中；此路径没有对相同sample token去重，也没有使用preload_offset裁剪读入的点云。观测与机制可以复用磁盘cache文件，但各自反序列化到内存。完整集可能造成很大的主机内存和磁盘缓存开销。

本次单次完整数据诊断建议省略 `--preloading`，让 `main.py` 采用默认False；这保留训练采样与优化协议，代价是更多按需I/O。workers12仍按既定配置。不能从mini的内存情况推定完整集预加载安全；若用户坚持预加载，应由服务器实际可用RAM与cache大小决定。

## 本地验证

- `python -m pytest -q tests/test_ct_v28_control_plane.py tests/test_ct_dual_stream.py`：33 passed。
- `python -m pytest -q tests/test_ct_v27_sampler_coverage.py tests/test_ct_v28_resume_tool.py tests/test_ct_lightning_runtime.py`：22 passed，2 skipped。
- 本地未加载真实nuScenes；无服务器GPU/RAM验收。未重新运行历史mini训练。

## 最少服务器预检

工具默认值修复后使用当前v28预检，勿沿用SERVER_PATHS旧v26工具提示：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/preflight_ct_v28.py \
  --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
  --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ \
  --output artifacts/ct_checks/v28_full_preflight.json

CUDA_VISIBLE_DEVICES=3 python tools/check_train_steps.py \
  --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
  --path /home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/ \
  --steps 16 --workers 0 --seed 42 \
  --artifact-dir artifacts/ct_checks/v28_full_smoke_unique_timestamp
```

preflight 默认路径统计场景、索引和完整端点，不等于已读取每个点云文件。后一项复用真实main训练入口，检查短前向/反向、调度和保存，包含有限验证；工程checkpoint不可进入60轮初始化。artifact目录必须新建、不可覆盖已有结果。

## 非阻断但应清楚记录的事项

- preflight的`initial_formal_run`旧输出文案仅标“B0 only”，不是训练gate；单次完整诊断已获用户授权，应更新该说明。
- Mini seed42的末轮退化、seed间差异与B2获取不足仍未解决；full数据可能有帮助，也可能复现，这次运行是诊断证据。
- Full训练时未加载校准策略会按合同回退B0在线输出；模块仍参与训练。完成后应对58/59/60分别在内部17场景拟合B3，再到官方150场景评测；未校准分数不能称Full动作收益。
