# v37 独立工程降本实验

2026-10-14 后续：用户已另行批准四组 B1 scratch 配方，配置与正式运行入口见
[四组 B1 操作页](CTSEQTRACK_V37_B1_EXECUTION_LAUNCH.md)。它们使用新的独立登记，
不是本页工程权重转正式；原lab归档与旧报告仍保留。下文实验室测速方案尚待服务器执行。

首次实现从 `0662c21ab99a44b64a3be807c3668f66a8ace196` 建立独立分支
`engineering/ct-execution-lab-20261014`；2026-10-14按用户要求已同步到主项目
`CT-SeqTrack` 的 main 工作区。旧五组 YAML、登记文件、checkpoint、输出及恢复规则保留。
工程检查工具不接续正式训练；新的四组B1只使用各自新登记配方从零开始。

## 执行方式与已知结论

原配置的开关默认关闭；实验室由 `execution_scope(..., engineering=True)` 启用，
新四组B1由持久化的 `v37_training_recipe` 自动启用。实验室临时开关不改正式配置；
完整代码 bundle、启用项、环境、课程和真实 Adam 步数写入工程报告。
参数键及形状保持原样，不能据此认为不同执行算法可以共享正式运行身份。

| 开关 | 实现 | 验收政策 |
| --- | --- | --- |
| `labels` | v37 专用基础 loss 适配；四候选共享 XY 几何，分别导出 exact/compat 标签 | 每项 loss、统计分母、fallback、clamp 差异和完整事务逐位一致 |
| `pool` | 当前 forward 共用点池与 raw-ID 排序，decoder live / verifier detach | 保留两路授权、精修前后邻域和梯度边界 |
| `geometry_gather` | 先 gather 选中属性，再相减 | 不改变邻居、排序或投影输入 |
| `unique_mask` | 完整排列 scatter 回填，删除逆排列排序 | 首个有效槽规则和 CUDA strictdet 必须通过 |
| `commit_indices` | 选择索引一次 CPU 批读，诊断复用 | 原合法性检查及状态提交完全保留 |
| `prior_read` | decoder 只读三候选，verifier 仍读四候选 | **已知数值变化，排除严格阶段 A 组合** |
| `factorized_projection` | 原 `1088→512` 权重拆成 local/global 两项，全局每组只投影一次 | 独立阶段 B；FP32 数值改变，正式指标未验证 |

`prior_read` 在 CPU 两步 Adam 的第二步出现
`decoder.local_reader.output.bias` 梯度最大差 `5.820766091346741e-11`，进而参数和
Adam exp_avg 不逐位一致。即使另一个短窗通过，也不移除这条历史失败。
复现测试使用线程1、seed1410、Full、原 batch fixture、Adam lr=1.5e-4。

投影分解保持原 Parameter 的可微切片、mask、分帧 global、bias 和 masked BN 规则。
既有42组隔离探针（20 FP64 + 22 FP32）中，FP64最大输出差约2.36e-14；
FP32有原梯度容差失败及近零 ReLU 门翻转。专属测试保留该失败，不放宽
`atol=1e-5, rtol=1e-4`，也不把容差通过解释为长程训练等价。

静态满槽前向 MAC 代理值预计从6.627G降至4.214G，两层合计减少2.413G。
batch16 FP32可以避免272MiB和34MiB两处显式cat输入；这不是峰值显存降幅。
权重切片可能触发contiguous拷贝，必须以真实训练计时决定净收益。

## 本地检查

从当前项目根目录运行；baseline必须指向迁移前保存的冻结源码：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q
python -m compileall -q models/ datasets/ utils/ tools/ main.py
git diff --check
python tools/check_ct_execution_lab.py --device cpu --synthetic --baseline-root artifacts/ct_checks/20261014-155937_main_sync/baseline_source --output artifacts/ct_checks/local_lab_check/report.json
```

`--synthetic` 明确将 batch/点数设为2/8，使用 TinySource；它只检查合同，不能用于测速
或充当真实服务器验收。输出目录必须全新。原源码重放与关闭开关的工程副本在独立进程中
比较完整事务存储位摘要，包含输出、loss、梯度、BN、Adam、RNG、状态及诊断。
baseline不能指向已更新的main或已修改的工程副本；也可另行检出冻结的0662c21源码。

阶段 A 的每项先单独通过四臂，再加入当前通过组合复验。失败项及证据各有独立JSON；
`prior_read` 固定排除。阶段 B 比较阶段 A 与阶段 A+投影，记录首次分叉、原容差失败、
prepared输入变化以及梯度/Adam覆盖。合同失败不允许进入测速。

## 服务器真实检查与测速（由用户运行）

将工程副本部署到**新的目录**，保留原 v37 源码目录作为只读 baseline。
不要覆盖原训练目录，不加载工程更新后的权重，不修改环境版本或关闭确定性。
以下命令中的 `/path/to/original-v37` 要替换为原0662c21源码位置。

历史工程副本 `CT-SeqTrack-execution-lab-20261014/artifacts/ct_checks/20261014_lab_delivery/`
保留最初工程源码与锁定基线两个zip；它们不是本次四组B1的最新交付。当前最新代码直接
位于主项目main工作区，四组启动以新版B1操作页为准。实验室基线仍需单独保存，不能覆盖正在训练的目录。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native

python tools/check_ct_execution_lab.py \
  --mode check --device cuda \
  --baseline-root /path/to/original-v37 \
  --path /home/lishengjie/data/nuscenes-mini \
  --output artifacts/ct_checks/lab_cuda_check/report.json

python tools/check_ct_execution_lab.py \
  --mode benchmark --device cuda \
  --path /home/lishengjie/data/nuscenes-mini \
  --check-report artifacts/ct_checks/lab_cuda_check/report.json \
  --output artifacts/ct_checks/lab_cuda_benchmark/report.json

python tools/check_ct_execution_lab.py \
  --mode trace --device cuda \
  --path /home/lishengjie/data/nuscenes-mini \
  --check-report artifacts/ct_checks/lab_cuda_check/report.json \
  --output artifacts/ct_checks/lab_cuda_trace/report.json
```

原源码身份校验仅为本工具比较CRLF/LF归一化后的锁定代码内容，正式source identity仍保留
原字节哈希且没有任何放宽。真实check需要四臂、阶段0/10、每阶段至少8个batch，
benchmark/trace会拒绝合成、过期代码、解析配置发生变化、不同设备/环境/数据根或不完整的check报告。

benchmark只测Full FP32，每阶段按AB、BA、AB配对；每次20步预热+100步计时。
阶段 A 比较原路径与通过组合，阶段 B 比较通过组合与投影分解；两个课程合计2880次
丢弃权重的工程更新。原CSV、TensorBoard、Console与训练诊断保留。
短trace另外运行，每个实现/课程5步，总计30步，不进入吞吐统计。

阶段10只代表成熟课程、seed及诊断排程，权重和Adam仍从零开始，Trainer只运行一轮。
报告明确区分课程epoch、真实优化步数、覆盖深度、尾批和drain。不能由epoch10标签
推断短窗已覆盖所有历史状态；专属合成测试另覆盖最大历史深度。

主指标是同步窗口墙钟时间及endpoints/s。CUDA事件间隔包含GPU等待CPU的间隙；
CPU enqueue时间不是GPU延迟。每条run保留中位数/P90、峰值allocated/reserved、日志
核验及进程快照。阶段 A 真Trainer日志或最终参数若改变，测速直接失败。
每次独立运行前在计时窗口之外回收Python对象并清空CUDA allocator缓存，避免上一次
Lightning对象环或缓存污染下一次的显存峰值。
阶段 B 允许数值分叉并如实报告，不因此升级正式配方。

测速应在同一空闲卡上串行执行；工具不停止其他任务。它记录前后进程快照，用户应检查
并发干扰。只有两个课程的全部三次配对均更快，才报告该受控条件下的稳定改善。
累计比率是两对测量的中位数比率之积，明确不是额外第三组直接配对测量。

## 后续采用条件

CPU测试通过不代表CUDA通过，更不代表训练已经加速。真实检查/测速缺失时保持待验证。
工程工具不保存checkpoint，也不更新正式登记。阶段 A通过且有收益后可以用于新的工程版本；
阶段 B若值得正式验证，须另行登记从epoch0开始的scratch实验，按final60/late-3评价。
当前代码不裁有效历史监督、不跨步缓存神经特征、不合并独立权重、不改变attention确定性路径。
