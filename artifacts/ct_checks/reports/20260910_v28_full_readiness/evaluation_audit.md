# v28 完整 nuScenes 单组 Full/Car 诊断：评测与校准接口审计

2026-09-10，先完成只读代码审计，再按后续授权实施最小导出元信息修复（文末记录）。用户明确希望进行一次60epoch完整数据诊断，本报告按该意图判断可执行性，不以此前mini得分未达目标否决这次诊断。本报告没有运行服务器nuScenes/CUDA评测，不宣称已有完整数据结果。

## 就绪判断

**现有接口支持单组 `28_full_nuscenes_full.yaml`、Car、seed42、从头60轮，然后逐58/59/60 checkpoint独立校准和官方val评测。未发现阻断该诊断训练的评测接口缺失。** 必须将训练时未校准的observation输出与训练结束后安装策略的Full输出分开报告。`--test`不训练参数；离线校准也是逐轨迹闭环拟合阈值，不冻结后再启动某个模块的训练。

| 项目 | 核查结果 | 是否阻断单组诊断 |
| --- | --- | --- |
| 完整数据配置 | `version=v1.0-trainval`，train_split=train_track，val/test_split=val；mini固定1262期望设为null | 不阻断 |
| 场景 | `utils/v27_protocol.py`与v28包装强制350训练/150官方val且互斥；内部calibration17、dev18均为350训练子集，相互不重叠 | 不阻断；不能称内部场景为未见训练数据 |
| 全端点评测 | main.py --test按test角色构造完整类别轨迹，batch1，不传partition时没有内部子集过滤；host要求每帧包括首帧和空输入fallback导出一次 | 不阻断；实际类别轨迹数/帧数需与服务器provenance核对 |
| 最终checkpoint | `FinalWindowCheckpoint`在模块epoch事务完成后保存058/059/060，与每5轮验证频率独立；另有epoch边界last | 不阻断 |
| 每checkpoint校准 | `calibrate_ct_actions.py --v28`转入真实闭环runner；calibration选策略，dev只做锁定诊断，官方val不进入拟合 | 必须实际完成后才能称Full selective结果 |
| 缺失或错配策略 | observation fail-closed，不报训练失败，action=0可能只是策略未安装 | 不阻断训练；不能据此判断B3收益 |
| 旧metadata | 旧运行的validation用dev标签，endpoint的scene_id未知；full validation还把parameter_training_overlap写true；本次已修导出副本，保留旧monitor标签 | 不改变整体S/P；后续新输出支持按场景和真实角色分析 |
| 运行开销 | 完整诊断导出累计全量endpoint，Full每帧含CUDA同步/GT几何侧信息；校准执行多条真正闭环 | 非逻辑阻断，但需预留时间、CPU内存和磁盘 |

## 正式与内部场景

`main.py:839`已经对v28选val角色；`scene_role`将val映射为test，完整配置的test即150场景官方val。`--test`默认也选test。不要为官方成绩传 `--ct-eval-partition dev` 或 `--test_split train_track`。

`utils/action_calibration_v27.py::validate_scene_manifest`独立核对(350,17,18,150)、test互斥、calibration/dev互斥且包含于train。数据清单选中150场景，不代表Car必定在每个场景都有轨迹；应报告完整Car轨迹/帧数量及manifest场景数，不能强迫没有该类轨迹的场景产生样本。

## 元信息的最小安全修复位置

审计时的 `models/base_model.py::validation_step/on_validation_epoch_end`仅按ct_enable_v27选择dev名，虽然实际v28验证已经是官方val。`utils/v27_evaluation.py:85`从首帧字典取scene_id/scene_name，而 `datasets/nuscenes_lidar_mf.py::get_frames`没有写这两个字段，于是unknown；当时 `test_step`补了tracklet_key却未补scene_id。

建议只在**验证与测试的导出层**，从实际sampler/source dataset的 `virtual_rate_meta[真实源索引]['scene_token']`，通过 `source.nusc.get('scene', token)['name']`解析scene_id；同时写真实tracklet_key、split/partition以及按实际场景交集计算的parameter_training_overlap。分区sampler存在tracklet_indices时先映射源索引，不能直接把batch_idx当原dataset索引。校准runner已经按该方式写了正确scene_id。

**不要为了元信息给传入前向的sequence首帧顺手新增tracklet_key，或者修改RecursiveTrackState的tracklet_key来源。** 现行host用它构造递归输入及采样种子，改变默认eval为真实key可能改变观测采样和成绩。此项修复应证明输出loss/参数/采样流不变，只改导出行。已有mini原始输出保持只读。

## 校准与加载身份

优先使用训练目录的 `resolved_config.yaml`，校准与main评测都读同一份。当前mini原始Full配置与实际run校准identity相同，但若完整诊断另行设置验证频率、workers等，只有保存配置完整体现最终命令。离线加载器require_complete=True会检查保存配置身份，不能不看最终CLI就假定原始YAML总能加载。

v28策略绑定checkpoint文件SHA、canonical resolved config SHA、source内容SHA、场景manifest SHA、metric_mode与score/comparator定义。source文件清单包括base_model、dataset、校准runner等，所以即便只修导出元信息，也应**先固定代码再拟合策略**；后续改相关代码应重新拟合，不能改artifact哈希冒充兼容。CPU/不同卡的评测环境不会成为同run训练恢复；训练resume仍须满足原环境与完整epoch边界要求。

B3的calibrated和阈值buffer设为persistent=False；构造时安装的离线策略不会被训练checkpoint的未校准buffer覆盖。缺策略及加载失败状态记录于endpoint的calibration_status，应确认loaded=true、reason=ok，并报告最终action_policy可能合法选择never。

## 计算与输出规模

校准先对17个场景做never闭环，从41个分位数阈值加always/never做单步筛选，再将最多3个非never候选连同never做实际闭环比较；最多4次不同calibration闭环，dev最多2次不同闭环。同role同policy只复用本进程完成结果，磁盘已有rollout并不会在下一次进程启动自动跳过。每个checkpoint必须各做一次本流程，不能把60的策略用于58/59。

当前mini Full 2285帧的endpoint CSV约3.99MB、candidate CSV约3.39MB，二者合计约3.23KB/端点，仅作量级参考。完整数据eval把这些行作为Python字典先累计于CPU内存；摘要/候选按轨迹遍历等还会在epoch_end花时间。校准也在内存缓存多个完整策略的行，并写出JSONL；它比CSV体积更大。实际开销需按服务器完整类别帧数衡量，不硬报总用时。

`utils/v27_evaluation.py`每预测帧同步CUDA并记录分阶段耗时，sidecar含GT几何与采集诊断；报告器已经标注 `diagnostic_wall_throughput_not_deployment_fps`。本次60轮诊断可保留这些信息；不能把测得速度当论文部署FPS。

## 训练后最少后处理命令

下面使用已经结束训练的新Full目录，`RUN_DIR`替换为真实路径；在服务器仓库根目录运行。每个checkpoint独立生成策略并评完整官方val，任何一步失败就中止脚本，避免误把缺策略fallback当作成功的Full。

```bash
RUN_DIR="output/实际日期-28_full-nuscenes_car_seed42_60ep_bs16"
CFG="$RUN_DIR/resolved_config.yaml"
DATA_ROOT="/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/"
POST_DIR="$RUN_DIR/post_eval_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$POST_DIR"
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

for EPOCH in 058 059 060; do
  CKPT="$RUN_DIR/formal_checkpoints/epoch=${EPOCH}.ckpt"
  POLICY="$POST_DIR/policy_${EPOCH}.json"
  python -u tools/calibrate_ct_actions.py --v28 \
    --config "$CFG" --checkpoint "$CKPT" --path "$DATA_ROOT" \
    --output "$POLICY" > "$POST_DIR/calibration_${EPOCH}.log" 2>&1 || exit 1
  python -u main.py --cfg "$CFG" --checkpoint "$CKPT" --test \
    --path "$DATA_ROOT" --ct_action_calibration_path "$POLICY" \
    --log_dir "$POST_DIR/selective_${EPOCH}" \
    > "$POST_DIR/selective_${EPOCH}.log" 2>&1 || exit 1
done
```

若要回答插件是否有闭环增益，至少再对第60轮同一Full checkpoint跑observation和bounded_always控制，不需要新训练：

```bash
for MODE in observation bounded_always; do
  python -u main.py --cfg "$CFG" \
    --checkpoint "$RUN_DIR/formal_checkpoints/epoch=060.ckpt" --test \
    --path "$DATA_ROOT" --proposal_mode "$MODE" \
    --log_dir "$POST_DIR/${MODE}_060" \
    > "$POST_DIR/${MODE}_060.log" 2>&1 || exit 1
done
```

每个main评测目录保存provenance/config，完整结果默认在 `lightning_logs/version_0/proposal_diagnostics/tracking_endpoints.csv` 及 `v27_endpoint_summary.json`（文件名遗留v27，内部schema可为v28）；proposal_endpoints/proposal_tracklets提供同状态候选诊断。最终分别报告final60和58/59/60算术平均的S/P，不能从raw单步收益或最佳epoch推导闭环增益。

## 最小导出修复与本地验证

按后续授权，本次只修改 `models/base_model.py` 与新测试 `tests/test_ct_v28_export_metadata.py`。helper位于base_model本身，已在既有校准source hash文件清单内，不新增遗漏于身份绑定的模块。仅在真实evaluate_one_sequence返回后给导出副本写scene_id、tracklet_key、source_tracklet_index、真实partition/dataset_split、parameter_training_overlap及metadata_status/error；序列原字典不动，旧TensorBoard/monitor标签和文件名保持兼容。

解析支持单DataLoader或单元素loader列表，逐层处理tracklet_indices与torch Subset.indices；先映射到source index再读取nuScenes场景。无法可靠映射时保留unknown、overlap=None并记录RuntimeWarning和metadata_error，不造场景归属。Full官方val按scene与训练集合的实际交集写false；内部calibration/dev写true。

定向测试命令：

```bash
python -m pytest -q tests/test_ct_v28_export_metadata.py tests/test_ct_v27_evaluation.py tests/test_ct_v28_motion_restore.py
```

结果 **29 passed**。新增测试执行从base_model提取的真实validation/test hook，核对输入对象/字段、观测用RNG、S/P指标输入与原monitor tag保持不变，只有导出元信息改变；并覆盖嵌套索引、mini/full官方val不重叠、内部重叠与缺失/错误映射。它验证导出隔离，不冒充CUDA全网络重新训练。未替用户启动服务器长跑，也未将此前mini门槛变成对本次诊断的禁止条件。
