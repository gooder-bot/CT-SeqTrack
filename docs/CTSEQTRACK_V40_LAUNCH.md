# V40：上传后验收、GPU0 两波实验与比较

**当前仅完成实现与静态审阅；本机未执行环境检查、pytest、compileall、训练或 CUDA 检查。未上传服务器，也未启动正式实验。** 静态登记只计算文件字节哈希，不代表任何运行检查通过。设计见 [V40方案](CTSEQTRACK_V40_DESIGN.md)。

## 最新安排：先单独启动 B1 MTP 与 Full

用户后续要求两组独立nohup，同物理GPU0并行，其余参数不变；B0/B1对照之后补齐。下面两段用于**完整上传且下方六阶段验收通过之后**，不是已通过验收的声明。独立nohup彼此不联动，一组报错不会自动停止另一组；如需原有同波监督行为，启动器新增`--wave 2`只选择B1 MTP+Full，默认`--wave all`仍执行原四组。

2026-10-29 20:33只读检查：服务器尚无40配置，torch2.0.1+cu118/Lightning2.0.2/FA2 2.5.8版本匹配；GPU0 A40当时空闲29585MiB，有其他进程，不能据此判定两组峰值容量通过。缺少NVML Python绑定，上传后由用户补齐以下仅用于验收的依赖（代理未执行安装）：

```bash
/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -m pip install 'nvidia-ml-py==12.560.30'
```

这是提供`pynvml`导入接口的[NVIDIA包](https://pypi.org/project/nvidia-ml-py/12.560.30/)，不要另装同名第三方`pynvml`包或升级Torch环境。

本轮静态修复了验证缓存缺口：Lightning 2.0.2默认`inference_mode=True`创建的张量没有版本计数，事务缓存会拒绝复用。V40显式`inference_mode=False`，验证/测试仍在`no_grad`下执行，与工程重放一致；不更改候选数学或训练梯度。此修复及新增测试仍待上传后执行。

B1 MTP：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit 1
mkdir -p output
CT_MTP="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-40_b1_mtp_mini_car_seed42_60ep_bs16-XXXXXX")" || exit 1
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/40_b1_mtp_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_MTP" \
  > "$CT_MTP/train.log" 2>&1 < /dev/null &
echo $! > "$CT_MTP/train.pid"
echo "$CT_MTP"
printf 'tail -n 80 -F %q\n' "$PWD/$CT_MTP/train.log"
```

Full（另一个新目录；可以在上一段返回后立即启动，实现并行）：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit 1
mkdir -p output
CT_FULL="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-40_full_mini_car_seed42_60ep_bs16-XXXXXX")" || exit 1
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/40_full_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --devices 0 --accelerator gpu --trainer_devices 1 --log_dir "$CT_FULL" \
  > "$CT_FULL/train.log" 2>&1 < /dev/null &
echo $! > "$CT_FULL/train.pid"
echo "$CT_FULL"
printf 'tail -n 80 -F %q\n' "$PWD/$CT_FULL/train.log"
```

每段最后会打印含准确绝对路径的`tail -n 80 -F ...`，复制到新SSH终端即可，不依赖旧终端变量。也可在新终端按目录日期定位两组最新日志：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit 1
CT_MTP_LAST="$(find output -maxdepth 1 -type d -name '*-40_b1_mtp_mini_car_seed42_60ep_bs16-*' | sort -r | head -n 1)"
CT_FULL_LAST="$(find output -maxdepth 1 -type d -name '*-40_full_mini_car_seed42_60ep_bs16-*' | sort -r | head -n 1)"
if [ -n "$CT_MTP_LAST" ] && [ -n "$CT_FULL_LAST" ]; then
  tail -n 80 -F "$CT_MTP_LAST/train.log" "$CT_FULL_LAST/train.log"
else
  echo '尚未找到两组V40目录；请使用启动时打印的准确日志路径。'
fi
```

两组都保持seed42/bs16/workers4/60轮/每轮验证/LR1e-4及20、40轮×0.2；不沿用v39的纯`backend:native`启动值，V40正式allocator已登记round2。`reserved>allocated`本身正常；要求重复完整工作负载后的allocated/reserved平台、零重试/OOM及并发余量通过，不能承诺60轮完全没有数值波动。只跑这两组不能单独证明MTP增益，后续仍需B1无MTP对照。

## 上传内容与环境

上传完整当前项目源码、新的四份40配置、`40_registration.json`、新测试和工具。旧v38/v39输出、配置、登记和既有artifacts保留；旧checkpoint仍使用各自封存源码。不要在服务器刷新登记来掩盖漏传/错传文件。

沿用已有路径记录，未假定当前资源仍与历史相同：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
CT40_PY=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
CT40_DATA=/home/lishengjie/data/nuscenes-mini
CT40_STAMP=$(date +%Y%m%d-%H%M%S)
CT40_CHECKS=artifacts/ct_checks/v40_${CT40_STAMP}
```

目标环境沿用Python3.9、torch2.0.1+cu118、Lightning2.0.2、flash-attn2.5.8，不升级公共环境。并发容量检查使用NVML的`pynvml`导入接口（由`nvidia-ml-py`包提供）；缺依赖时检查明确失败，先处理目标环境依赖再重试，不降级为估算通过。数据根必须是含metadata/samples/sweeps的mini目录，不能传nuscenes Python包目录。

## 六个服务器验收阶段

先核验完整部署；此时不要执行`register`或`--refresh-unstarted`：

```bash
"$CT40_PY" tools/verify_ct_v40.py verify
```

各阶段须使用全新输出目录；失败后保留日志并以新时间戳重试。以下命令只在上传后的服务器执行：

```bash
"$CT40_PY" tools/check_v40_server.py --phase contracts --output "$CT40_CHECKS/contracts"
"$CT40_PY" tools/check_v40_server.py --phase kernel --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" --gpu 0 --output "$CT40_CHECKS/kernel"
"$CT40_PY" tools/check_v40_server.py --phase lifecycle --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" --gpu 0 --output "$CT40_CHECKS/lifecycle"
"$CT40_PY" tools/check_v40_server.py --phase memory --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" --gpu 0 --output "$CT40_CHECKS/memory"
"$CT40_PY" tools/check_v40_server.py --phase performance --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" --gpu 0 --output "$CT40_CHECKS/performance"
"$CT40_PY" tools/check_v40_server.py --phase parallel --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" --gpu 0 --output "$CT40_CHECKS/parallel"
```

| 阶段 | 检查内容与解释 |
|---|---|
| contracts | 项目pytest、compileall、登记核验；含公共初始化、B0消融、M标签/梯度隔离、CPU acquisition等价及搜索状态合同 |
| kernel | 实际CUDA FA2前向/反向kernel证据、packed BF16/FP32返回及数值门；正式不静默math |
| lifecycle | 四组各真实mini的8个完整训练事务；另用受控raw-source跑真实entry/Lightning/CUDA完整epoch与独立进程恢复，不冒充完整mini训练 |
| memory | Full在native、round2独立进程，完整成熟训练epoch余段、全量validation、下一轮训练；完整预热后3次同工作负载重放与受控边界输入 |
| performance | 同一FA路径native/round2完整流水线耗时；同权重/固定真实prepared batch的FP32 math与窄BF16 FA2网络训练step对照；另测真实batch=1顺序跟踪帧的p50/p95，分别报告计时范围 |
| parallel | GPU0实际b0+b1、b1_mtp+full两波并发；NVML总用量、其他任务占用、reserved峰值保守校核及安全余量 |

**memory不是短小smoke。** 为覆盖成熟窗口、drain尾批、validation边界和缓存平台，默认每种allocator需一个完整工作负载预热和三次重放；parallel也运行真实成熟负载。不要把这部分工程成本记成正式训练预算。受控空点/稀疏/guard变体会明确标识。全部工程权重弃用。

恢复检查使用完整受控epoch：连续21轮，对照第10轮中断、恢复到20轮再中断、恢复到21轮的独立进程；比对请求顺序、学习率、模型/BN/Adam/调度器。原课程在第10轮已经达到W=8，因此10→11是成熟课程恢复检查，**不是新增窗口突变**；20→21检查LR从1e-4降到2e-5。

显存门包括对应phase的allocated/reserved差、连续增长、实际retries/OOM增量，不仅看nvidia-smi曲线。默认差值上限64/256MiB，持续增长独立失败；两进程预留max(2GiB,总显存10%)并计入非PyTorch和外部占用。normal reserved缓存不能称泄漏，短程通过也不能保证60轮无任何波动，正式训练保留逐批监控。

性能计时不包逐核CPU指纹审计。FP32 math→BF16 FA同时改变kernel与attention精度，分别看数值门与时延；固定prepared输入的网络step不含raw读取/prepare/commit，不能称整体跟踪加速。batch=1计时覆盖实际验证流水线的raw读取、prepare、B1/B0/B3、状态提交与清理，包含该流水线现有的GT标签准备及被动诊断，不是剥离诊断后的部署推理时延；批处理耗时除以16不能作为单帧p50/p95。独占GPU测得的时延才可作独占性能声明；共享状态报告只作共享测量，不宣称提速。

## 四组正式启动

所有组：mini Car/global/true-time、seed42、bs16、workers4、60轮；LR1e-4、完成20/40轮后×0.2，Adam及原窗口课程不变。四组FA与CPU优化相同。GPU0第一波B0+B1，第二波B1M+Full，每波两个进程；OMP/MKL/OPENBLAS各2。allocator在子进程CUDA初始化前固定为`backend:native,roundup_power2_divisions:2`。

先只查看命令计划（不启动训练，也不创建run目录）：

```bash
"$CT40_PY" tools/launch_ct_v40.py --path "$CT40_DATA" --run-root "output/v40_${CT40_STAMP}"
```

只有六份报告成功且source/tool/registration完全匹配后，再由用户显式启动：

```bash
"$CT40_PY" tools/launch_ct_v40.py --path "$CT40_DATA" --run-root "output/v40_${CT40_STAMP}" --execute --acceptance \
  "$CT40_CHECKS/contracts/report.json" \
  "$CT40_CHECKS/kernel/report.json" \
  "$CT40_CHECKS/lifecycle/report.json" \
  "$CT40_CHECKS/memory/report.json" \
  "$CT40_CHECKS/performance/report.json" \
  "$CT40_CHECKS/parallel/report.json"
```

在稳定终端/tmux中运行启动器；它等待第一波两组均退出0才开第二波。每组日志和PID写到对应子目录。若其中一组失败，启动器停止同波自己创建的另一进程并阻止下一波，不操作无关GPU任务。不要混用旧v39 launcher。

```bash
tail -f "output/v40_${CT40_STAMP}/b0/train.log"
tail -f "output/v40_${CT40_STAMP}/b1/train.log"
tail -f "output/v40_${CT40_STAMP}/b1_mtp/train.log"
tail -f "output/v40_${CT40_STAMP}/full/train.log"
```

启动器只处理fresh runs。异常恢复使用该组原目录、原源码、原配置和完整epoch checkpoint，例如：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTORCH_CUDA_ALLOC_CONF=backend:native,roundup_power2_divisions:2 \
"$CT40_PY" main.py --cfg cfgs/ct_seqtrack/40_full_mini.yaml --path "$CT40_DATA" \
  --log_dir "output/v40_${CT40_STAMP}/full" \
  --checkpoint "output/v40_${CT40_STAMP}/full/formal_checkpoints/epoch=020.ckpt"
```

示例中的020只有实际完整保存后才能使用。不能用v38/v39权重初始化V40，不能跨arm恢复；所有启用模块从epoch1参与。中断后的逐帧日志按新会话保存，不覆盖旧流。

## 成绩复核与解释

自动保留58/59/60并评估final60、late3，不挑各组不同best。比较需要项目既有固定R/C原始证据路径；使用真实目录替换下面的`R_DIR`、`C_DIR`：

```bash
"$CT40_PY" tools/compare_ct_v40.py --registered-reference R_DIR --baseline C_DIR \
  --b0 "output/v40_${CT40_STAMP}/b0" --b1 "output/v40_${CT40_STAMP}/b1" \
  --b1-mtp "output/v40_${CT40_STAMP}/b1_mtp" --full "output/v40_${CT40_STAMP}/full" \
  --output "artifacts/ct_checks/v40_${CT40_STAMP}_comparison.json"
```

比较工具核验60份epoch审计、实际重建sampler、每轮预算、公共初始化与逐帧候选/可信/搜索机制。M的未来loss下降不等于跟踪收益；Full−B1M是B3+search联合机制。mini单seed只是方向筛选，late3不是三个独立重复。暂无任何V40分数、显存稳定或速度结论。
