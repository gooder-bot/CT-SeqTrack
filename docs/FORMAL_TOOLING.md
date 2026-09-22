# CT-SeqTrack v32 工具面

唯一入口为main.py。以下命令供用户在同步当前 v32 源码和配置后执行；本次本地检查没有代为上传或启动。四条可直接复制的后台命令和新终端 tail 见 [v32 mini 运行说明](CTSEQTRACK_V32_MINI_LAUNCH.md)。

## 四次登记训练

```bash
python main.py --cfg cfgs/ct_seqtrack/32_seqtrack_ref_mini.yaml --path DATA_ROOT --seed 42 --tag ref_seed42
python main.py --cfg cfgs/ct_seqtrack/32_b0_mini.yaml --path DATA_ROOT --seed 42 --tag b0_seed42
python main.py --cfg cfgs/ct_seqtrack/32_full_gru_mini.yaml --path DATA_ROOT --seed 42 --tag full_gru_seed42
python main.py --cfg cfgs/ct_seqtrack/32_full_cfc_mini.yaml --path DATA_ROOT --seed 42 --tag full_cfc_seed42
```

随机初始化，结束自动评测58/59/60并保存results.json。同身份完整epoch恢复可加`--checkpoint RUN/formal_checkpoints/epoch=020.ckpt`；不使用init_checkpoint。环境和数据根见[SERVER_PATHS](SERVER_PATHS.md)。

## 评测和工程检查

```bash
python main.py --cfg cfgs/ct_seqtrack/32_b0_mini.yaml --seed 42 --checkpoint RUN/formal_checkpoints/epoch=060.ckpt --test --log_dir NEW_EVAL_DIR
python main.py --cfg cfgs/ct_seqtrack/32_b0_mini.yaml --ct_engineering_check --epoch 1 --workers 0 --limit_train_batches 2 --limit_val_batches 1 --no_late3 --log_dir artifacts/ct_checks/NEW_SMOKE
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ main.py
git diff --check
```

limit_*整数为batch数、小数为比例。SeqTrack及Full可替换相应32_*配置。真实smoke需要SDK/数据；工程输出放入独立artifacts/ct_checks目录，权重不进入正式初始化。

## 身份与保护

本轮四组各自在 `results.json` 保存 final60 与 late-3；逐 checkpoint 的结果在 `evaluation/epoch=058|059|060/`。先比较 seed42 B0/reference 的 final60 S/P，再报告两个 Full 对 B0 的差异。

下列工具专用于**后续补齐 reference/B0 seed52 后的双 seed 验收**，不是本轮四臂汇总器。它只读固定 final60 差距并从四个基线 run 的全部12份逐帧记录重算指标：

```bash
python tools/compare_v32_baselines.py --b0-42 B0_SEED42_RUN --ref-42 REF_SEED42_RUN --b0-52 B0_SEED52_RUN --ref-52 REF_SEED52_RUN
```

标准输出为JSON：退出码0达标、1未达标、2证据不完整或身份/预算不匹配。不会以late-3替换失败的final60，也不写入训练目录。

run保存resolved config、manifest、CSV/TensorBoard、checkpoint和逐帧结果；参考对照另保存真实重采样曝光。配置和采样身份参与恢复核验。旧31_*仅在Git b1d886e复现，更早版本见[历史索引](HISTORY_EVIDENCE_INDEX.md)。

output/与既有artifacts/受保护；评测使用新目录，不覆盖历史。当前实现和检查边界见[修复说明](B0_V32_REPAIR.md)。
