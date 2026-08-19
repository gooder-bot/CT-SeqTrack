# CT-SeqTrack v25 服务器验收清单

本清单只在服务器执行。脚本不会检测或安装 CUDA、nuScenes、Lightning
或其他大型依赖；缺失项由服务器环境一次性处理。

## 运行前固定项

- 使用当前工作树的 v25 代码和 `cfgs/ct_seqtrack/25*.yaml`。
- 四臂都从随机初始化、epoch 0 开始；不存在 `--init_checkpoint`。
- `--checkpoint` 只用于评估或同一实验在 epoch 边界的严格恢复。
- 保存代码 SHA、resolved config、scene manifest SHA、preflight SHA、promotion
  SHA、checkpoint SHA，以及随机状态/recursive state 恢复记录。

## 顺序

1. 设置 `CT_DATA_ROOT`、`CT_ARTIFACT_ROOT`、`CT_RUN_ROOT` 和 `CT_SEED`。
2. 执行完整 acquisition preflight（不得使用 `--max-batches`）：

   ```bash
   bash tools/run_ct_v25_server_acceptance.sh preflight
   ```

3. 用不携带权重的 passed B2 method-promotion manifest 设置
   `CT_B2_PROMOTION`，再做四臂各 2-batch smoke：

   ```bash
   export CT_B2_PROMOTION=artifacts/v25_acceptance/b2_method_promotion.json
   bash tools/run_ct_v25_server_acceptance.sh smoke
   ```

4. Full 跑至少 20 batches。确认 H3 shadow 有有效行、shadow forward 计数大于
   0、candidate0/c1/c2 数量正确、B0/B1/B2/B3 都产生有限梯度且各参数只属于
   一个 optimizer：

   ```bash
   bash tools/run_ct_v25_server_acceptance.sh full20
   ```

5. 每臂做 5-epoch kill/resume。只在 epoch 边界终止；用原 YAML、原 seed 和
   `last.ckpt` 设置 `CT_RESUME_CONFIG`、`CT_RESUME_CHECKPOINT` 后执行 `resume`。
   验证 optimizer、scheduler、scaler、global RNG、plugin RNG、batch/epoch 步数
   和 module hash 连续。checkpoint 必须携带
   `ct_seqtrack.recursive_state_boundary.v1`：它记录已完成 epoch、活动 state 数量
   和 `next_epoch_reset=true`；恢复端校验后清空 state。未中断运行在同一 epoch
   边界也执行相同清空，因此不允许把活动 tracklet 跨 epoch 偷渡为训练上下文。
6. 执行四臂 scratch training：

   ```bash
   bash tools/run_ct_v25_server_acceptance.sh train
   ```

7. 按 `docs/CTSEQTRACK_V25_PROTOCOL.md` 在互斥的
   `calibration_select` / `calibration_audit` scene 上完成 B3 两阶段校准。
8. 对同一 Full checkpoint 运行 `observation`、`raw_search`、`selective` 三种
   输出；selective 必须绑定通过 audit 的 calibration artifact，否则应
   fail-closed。
9. 运行 scene-paired bootstrap、risk--coverage 和 module hash audit；只有
   artifact identity、分区隔离和三 seed 结论同时通过时才形成论文主张。

## 必须保存的验收证据

- 四臂 2-batch 日志、Full 20-batch 日志及梯度/optimizer 归属；
- 5-epoch uninterrupted 与 kill/resume 的参数哈希和状态连续性对照；
- acquisition preflight、B2 promotion、selection/audit calibration artifacts；
- observation/raw/selective endpoint CSV；
- scene manifest 与 SHA、checkpoint SHA、resolved config、代码 SHA；
- scene bootstrap、risk--coverage、B0/B1/B2/B3 module hash audit。
