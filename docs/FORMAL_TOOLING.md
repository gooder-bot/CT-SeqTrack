# CT-SeqTrack v31 正式工具面

当前训练、工程检查和闭环评测统一从 `main.py` 进入。旧版工具不再保留于工作树，恢复方法见 [历史索引](HISTORY_EVIDENCE_INDEX.md)。

## 正式训练与评测

```bash
python main.py --cfg cfgs/ct_seqtrack/31_full_gru_mini.yaml --path DATA_ROOT --tag exp_name
python main.py --cfg cfgs/ct_seqtrack/31_full_gru_mini.yaml --checkpoint RUN/formal_checkpoints/epoch=060.ckpt --test --log_dir NEW_EVAL_DIR
```

正常训练结束自动评测58/59/60，输出 `results.json`；独立 `--test` 直接闭环评测所给 checkpoint，不经过动作导出或标定。评测输出使用新目录，不能覆盖旧训练/评测结果。续训仅使用相同配置身份的完整 epoch 边界 checkpoint，并遵守 [协议](EXPERIMENT_PROTOCOL.md)。

每个 run 保存 `resolved_config.yaml`、`run_manifest.json`、CSV/TensorBoard、训练checkpoint和逐帧评测结果。mini三臂后台命令见 [运行说明](CTSEQTRACK_V31_MINI_LAUNCH.md)。

## 工程检查

显式 `--ct_engineering_check` 可运行合成/小步检查；指定 `--log_dir` 时必须位于本仓 `artifacts/ct_checks/`。工程checkpoint不能用于正式初始化。实际数据 smoke 需要相应 SDK、数据根与运行环境；本地合成检查不能代替真实CUDA结论。

```bash
python main.py --cfg cfgs/ct_seqtrack/31_b0_mini.yaml --ct_engineering_check --epoch 1 --workers 0 --limit_train_batches 2 --limit_val_batches 1 --no_late3 --log_dir artifacts/ct_checks/NEW_SMOKE
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

其中 `--limit_* 2` 为batch数量，`--limit_* 1.0` 为比例；不要混淆。示例是待用户执行的接口说明，不代表已执行检查。

## 保留能力与证据读取

配置加载统一使用支持 `_base_` 的加载器；v31只接受自身配置白名单。保留nuScenes mini/full、KITTI、四个模块臂、CfC/GRU及true/fixed/shuffled时间控制，不再混用旧正式配置。已有实验与诊断读取 [9/21报告](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md) 及其原始文件。

历史 `verify_ct_slimming.py` 固定要求 `HEAD=001951a`，旧22配置与旧输出清单属于当时快照；不用于当前v31验收。旧可视化和旧checkpoint探针按对应Git版本复跑，不将其旧输入schema当作v31接口。

`output/` 与既有 `artifacts/` 全部受保护。本次工作树收敛的检查见 [精简报告](../artifacts/ct_checks/20260922_v31_slimming/REPORT.md)，历史测试数量不等于本次验证结果。服务器当前仅只读，本页命令不构成自行上传或启动任务的授权。
