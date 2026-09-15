# v30 数据协议与服务器运行

本轮实现仅在本地修改 CT-SeqTrack；没有同步、停止或启动服务器任务。以下命令供服务器部署后执行。
正式运行使用新 v30 配置，从 epoch 0 随机初始化；不续接 v29 checkpoint，也不覆盖旧 output。
本地 CPU 测试只验证接口和数值合同，CUDA、吞吐、闭环 S/P 由实际运行验证。

最新四任务后台命令及新终端`tail -f`见[mini启动专页](CTSEQTRACK_V30_MINI_BN_AB_LAUNCH.md)：
GPU0=Full-GRU、GPU2=Full-CfC、GPU1并行B0 BN反向重算false/true；两个Full保持false。
按本轮用户要求直接提供正式命令，下面工程专项按需使用，不增加启动门槛。

## 数据集身份

`utils/dataset_protocol_v30.py` 提供 `build_dataset_manifest(config, scene_splits=None)`、
`select_dataset_protocol(config, requested_role, scene_splits=None)` 和
`validate_dataset_manifest(manifest)`。schema 为 `ct_seqtrack.dataset_protocol.v30`。
manifest 绑定数据集、版本、场景用途、训练重叠、坐标模式、时间来源、frame stride 和完整端点覆盖。

| 数据集 | 参数训练 | 策略拟合 | 锁定诊断 | 最终评测 |
|---|---|---|---|---|
| nuScenes mini | mini_train 全8场景 | 固定内部1场景 | 固定内部1场景 | mini_val 2场景 |
| nuScenes full | train_track 350场景 | 固定内部17场景 | 固定内部18场景 | val 150场景 |
| KITTI | 0000–0016 | 0017 | 0018 | 0019–0020 |

nuScenes拟合/诊断场景与参数训练重叠，沿用v28的场景选择与披露；它们不是独立泛化验证。
KITTI四种用途互不重叠。KITTI训练时的`val`角色映射0018，明确`test/eval`角色才映射0019–0020。
KITTI最后两条序列属于有标签训练集上的SOT研究切分，不是官方29条无标签测试序列。

`ct_enable_v30: true`开启新分支；旧版本保持历史分派。两种dataset都有
`get_frame_metadata(tracklet_id, frame_id)`和`get_frames_metadata(tracklet_id, frame_ids)`，
不读点云且保留框、真实/控制时间、scene/sequence、raw frame身份及manifest哈希。
辅助历史、首帧尺寸可直接调用metadata接口。

## 坐标、时间和稀疏测量

nuScenes点云仍经sensor→ego→global，时间取LiDAR `sample_data.timestamp*1e-6`。
annotation.next路径读取带标注关键帧；设置`key_frame_only=False`不会变成20Hz sweep序列。

KITTI `ct_coordinate_mode: sensor_relative`，`kitti_frame_period: 0.1`。
先组合`R_rect @ Tr_velo_cam`，再构造固定水平参考轴；原点保留在LiDAR、x前/y左/z上。
整云、GT中心和框方向使用同一完整变换，框以纯yaw表示，尺寸由hwl转为wlh，底面中心转几何中心。
这是逐帧传感器相对坐标；没有OXTS时不宣称自车补偿或世界速度。
metadata中`sensor_to_sequence_world=None`明确表示不可用；nuScenes则提供真实变换。

`ct_frame_stride`默认1。大于1时按轨迹均匀保留首帧及后续子序列，保留原始frame编号/token/time，
再运行已有virtual-rate控制；stride写入manifest和tracklet身份。KITTI frame0/3/7的时间仍为0/.3/.7秒。
真实annotation时间必须有限且严格递增；sampler合法首部padding不在该校验总体中。

v30所有数据集、所有角色`preload_offset=-1`。KITTI不再按当前GT预裁剪；裁剪由后续因果状态决定。
缺失KITTI文件默认报错；显式`kitti_allow_missing_pointcloud=true`才允许零真实点缺测。
合法空文件也返回`[3,0]`坐标及空raw IDs，不创建伪原点。零测量仍由v30 mask处理，不跳端点或GT重置。

## 有界IO

nuScenes同进程的相同root/version复用已构造devkit元数据，不重复加载大JSON表；记录视为不可变，
返回metadata做独立拷贝。整云LRU按root/version/坐标/token缓存，默认
`ct_pointcloud_cache_bytes: 268435456`（每进程/worker共256 MiB）；0表示禁用。
预算计入坐标和raw ID数组，超预算单云不缓存；fork首次使用清空继承云缓存。
返回独立可写副本，原点顺序、dtype、raw IDs保持。v30拒绝按轨迹全量preloading。
此预算不含devkit元数据、Python容器、队列或模型内存，也不是整个训练进程的RAM上限。

## 启动三臂

进入服务器CT-SeqTrack目录并激活已有训练环境，确认pointnet2_ops/CUDA依赖可用。
如环境缺nuScenes包，包父目录是`/home/lishengjie/code/SparseFusion-main/nuscenes`，
与完整数据根不同，可按`docs/SERVER_PATHS.md`设置PYTHONPATH。这里不安装任何依赖。

脚本默认只打印命令，不创建文件或启动训练：

```bash
python tools/run_ct_v30_server.py --dataset mini --arm b0 --gpu 1
python tools/run_ct_v30_server.py --dataset mini --arm full_gru --gpu 0
python tools/run_ct_v30_server.py --dataset mini --arm full_cfc --gpu 2
```

在确认所选GPU与训练环境后，同一命令显式添加`--launch`才启动。
`--dataset full`选择`30_*_nuscenes_full.yaml`；`--dataset kitti`选择`30_*_kitti.yaml`。
已核实的数据根默认值为：

| dataset | --path 默认值 |
|---|---|
| mini | `/home/lishengjie/data/nuscenes-mini` |
| full | `/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes` |
| kitti | `/home/lishengjie/data/cxtrack/training` |

可用`--path`显式覆盖。脚本以当前Python启动，可用`--python`指定训练环境解释器。
每次生成新的时间戳/随机后缀目录，或用`--log-dir`指定一个尚不存在的目录；已有目录会拒绝。
环境限制为指定GPU、OMP/MKL各1线程、`PYTORCH_CUDA_ALLOC_CONF=backend:native`；配置batch16/workers4/seed42/scratch60。
9月15日晚按[显存分析](CTSEQTRACK_V30_CUDA_MEMORY_FIX.md)解除旧64MiB分割限制；显式设置可覆盖shell中残留的旧值。
启动后写`train.log`与`train.pid`，不停止任何旧任务。完成情况以训练日志及058/059/060 checkpoint为准。

## 每个checkpoint单独拟合与评测

以下在对应run训练结束后执行；RUN替换成实际训练目录，DATA替换成对应数据根。
DEST必须是新目录，配置使用该run保存的`resolved_config.yaml`。
Full-CfC和Full-GRU各自对058/059/060独立拟合；不跨checkpoint复用policy。

```bash
set -euo pipefail
RUN=/absolute/path/to/new_v30_run
DATA=/home/lishengjie/data/nuscenes-mini
E=058
CKPT="$RUN/formal_checkpoints/epoch=$E.ckpt"
DEST="$RUN/v30_eval/epoch=$E"
mkdir -p "$RUN/v30_eval"
mkdir "$DEST"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/calibrate_ct_actions.py --v30 --config "$RUN/resolved_config.yaml" \
  --checkpoint "$CKPT" --path "$DATA" --device cuda --output "$DEST/policy.json"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python main.py --cfg "$RUN/resolved_config.yaml" --path "$DATA" --checkpoint "$CKPT" \
  --ct_action_calibration_path "$DEST/policy.json" --test --workers 4 --seed 42 \
  --log_dir "$DEST/final_evaluation"
```

v30 CLI也按config自动识别；`--v30`用于清楚标记。
策略拟合使用每行max-action-q产生阈值，最多10个calibration真实闭环；dev仅运行锁定策略/never诊断。
缺失、过期或不匹配policy继续fail closed到observation。B0评测省略校准和policy参数。
final固定epoch60；late-3是58/59/60同口径分数的均值。不要挑选各臂不同最佳epoch。
有需要时可单独导出完整never轨迹：

```bash
python tools/export_ct_action_rows.py --v30 --config "$RUN/resolved_config.yaml" \
  --checkpoint "$CKPT" --path "$DATA" --device cuda --partition calibration \
  --output "$DEST/never_rows.jsonl"
```

## 有针对性的验证

以下只在服务器已有环境和新同步的代码中手动执行。三个臂都检查；CUDA短检查产物不能初始化正式训练。
工具带历史版本名时，传入的`30_*`配置决定实际执行v30通路。

```bash
DATA=/home/lishengjie/data/nuscenes-mini
CFG=cfgs/ct_seqtrack/30_full_cfc_mini.yaml
CHECK=artifacts/ct_checks/v30_cuda_cfc_run1
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native
python tools/check_time_batch.py --cfg "$CFG" --path "$DATA" --workers 4
python tools/check_forward_batch.py --cfg "$CFG" --path "$DATA" --workers 4
python tools/check_train_steps.py --cfg "$CFG" --path "$DATA" --workers 4 \
  --steps 100 --numerical-audit --artifact-dir "$CHECK/numerical"
python tools/check_ct_v28_resume.py --cfg "$CFG" --path "$DATA" --gpu 2 \
  --steps 16 --output "$CHECK/resume"
python tools/benchmark_ct_v29_h3.py --cfg "$CFG" --arm full_cfc --path "$DATA" \
  --gpu 2 --output "$CHECK/legal_h3"
```

数值检查分别对同卡B0两次、Full-CfC和Full-GRU执行；用
`python tools/compare_ct_v28_audits.py LEFT/numerical_audit RIGHT/numerical_audit`比较。
H3工具必须找到合法事件并生成报告才算覆盖，100步内未找到不是通过；不因此改动正式H3抽样比例。
检查0/1/2点需使用真实稀疏帧记录；普通batch有限值不能代替这些分支覆盖。
两Full都完成合法H3、真实策略拟合与评测检查；恢复工具v30使用每轮一次有界验证，不沿用旧两轮不验证排程。

本地数据与旧接口回归：

```bash
python -m pytest -q tests/test_ct_v30_data.py tests/test_ct_v30_action_runtime.py tests/test_ct_v29_perf_data.py tests/test_ct_v28_preflight_data_entry.py tests/test_ct_recursive_state.py tests/test_ct_v27_actions.py
```

服务器CUDA还需覆盖真实0/1/2/3+点、合法H3事件、校准和最终评测路径、同卡重复与epoch边界恢复。
保留3D log_softmax→2D NLL、int64 AP cumsum、strict deterministic；不关闭确定性来绕过故障。
性能检查独立记录loader等待、读取/变换、host处理和GPU时间；调用次数下降不能当成实测加速比。

## mini三臂汇总与下一阶段

每个评测目录的Lightning日志下有`proposal_diagnostics/v27_endpoint_summary.json`；文件名保持旧工具兼容，
内部schema必须为`ct_seqtrack.endpoint_diagnostics.v30`。将三个臂各058/059/060报告路径传入：

```bash
python tools/summarize_ct_v30_mini.py \
  --b0 /eval/b0/058/v27_endpoint_summary.json /eval/b0/059/v27_endpoint_summary.json /eval/b0/060/v27_endpoint_summary.json \
  --full-cfc /eval/cfc/058/v27_endpoint_summary.json /eval/cfc/059/v27_endpoint_summary.json /eval/cfc/060/v27_endpoint_summary.json \
  --full-gru /eval/gru/058/v27_endpoint_summary.json /eval/gru/059/v27_endpoint_summary.json /eval/gru/060/v27_endpoint_summary.json \
  --output artifacts/ct_checks/v30_mini_result/summary.json
```

这里的`/eval/...`是报告占位路径，按实际生成路径替换。工具只做汇总，不启动后续训练。
仅要求至少一个预登记Full在final60的S和P同时严格高于同版本B0；late-3只报告，不额外要求双升，
不增加多seed前置条件。通过后full与KITTI仍先做Car三臂同预算比较，并补mini六项消融。

时间间隔可在新运行用`--ct_frame_stride 2`（nuScenes另有4，KITTI另有5）；
时间模式用`--dynamics_time_mode true|fixed|shuffled`，shuffled同时传`--dynamics_time_manifest`。
stride保留原始时间跨度，写入新的manifest与checkpoint身份；间隔/时间控制不能与另一运行跨配置续训。
