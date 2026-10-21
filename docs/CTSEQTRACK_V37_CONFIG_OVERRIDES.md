# v37 B2 三组：参数覆盖与运行身份

2026-10-21。仅适用于 `b1_b2_mass_detach`、`full_mass_detach`、`full_mass_detach_hold_lr` 三个 `v37_centroid_recipe`；旧五组和旧B1执行配方保持原约束。

配置优先级为 **CLI > 当前子YAML > `_base_` YAML > 代码DEFAULTS**。没传的参数沿用解析后的配置值；参数拼错会报错，例如 `--workerss 2`，不会当作未传参数静默采用默认值。`--lr` 现在也是正式CLI参数。

本轮验证：配置、入口与相关回归测试 **215 passed、1 skipped**，启动器测试 **23 passed**；真实参数解析和登记的启动器整合检查通过，训练启动被测试拦截。跳过项依赖本机未安装的Lightning，未验证真实CUDA训练。本轮未访问或修改服务器。详见 [验收记录](../artifacts/ct_checks/20261021_v37_cli_overrides/REPORT.md)。

## 六项运行与训练参数

| 参数 | 三份默认YAML的值 | 合法范围 | 改为不同值时的分类 |
|---|---:|---|---|
| `--check_val_every_n_epoch` | `1` | 正整数 | `runtime_override`：改变验证频率 |
| `--workers` | `4` | 非负整数 | `runtime_override`：改变数据加载进程数 |
| `--batch_size` | `16` | 正整数，无倍数限制 | `recipe_variant`：改变训练批大小 |
| `--epoch` | `60` | 正整数 | `recipe_variant`：改变训练轮数 |
| `--lr` | `0.00015` | 有限且大于0 | `recipe_variant`：改变初始学习率 |
| `--seed` | `42` | 0至4294967295的整数 | `recipe_variant`：改变随机种子 |

以后修改这六项的合法值无需再改代码或重新登记模板。登记核验源码和基础配方模板，运行仍完整记录实际配置哈希以及覆盖项；这不是跳过登记。结构、FP32/concat、单卡、scratch、物理时间和既有LR衰减位置等约束保持。改变批大小不会自动缩放LR或启用梯度累积。

manifest中的 `run_kind` 为 `registered`、`runtime_override` 或 `recipe_variant`，`run_parameters` 记录实际六项参数；`registration.overrides` 逐项记录 `{registered, effective}`，同时保留实际与模板配置哈希。整数参数和lr均不接受布尔值。epoch为1或2时，最终窗口只含最后 `min(3, epoch)` 轮，不能报告为正式60轮或三轮均值结果。

`runtime_override` 不作为新的结构配方，但必须记录验证频率和实际执行成本。任一 `recipe_variant` 字段偏离模板时，该运行标记为研究变体，不能计入默认三组的180 epoch/215,100更新，也不能替代原主模型结果；预算按实际配置和sampler重新核算。默认三份YAML不改，完整启动命令仍为val1、batch16、60轮、workers4、LR1.5e-4、seed42、物理GPU1，见 [三组启动页](CTSEQTRACK_V37_B2_CENTROID_LAUNCH.md)。

## GPU、YAML和路径

这些操作参数同样允许CLI覆盖：

| 参数 | 含义与未传时的行为 |
|---|---|
| `--devices 1` | 选择物理GPU1，只接受单个非负整数。优先级CLI devices > YAML devices > 现有 `CUDA_VISIBLE_DEVICES`；在CUDA初始化前应用。 |
| `--trainer_devices 1` | 设备数量固定为1；不支持真正多卡训练，不能用它选择物理GPU1。 |
| `--cfg path/to/config.yaml` | 选择YAML；其中仍可用 `_base_` 继承，CLI覆盖该文件解析出的值。 |
| `--path /path/to/data` | 覆盖数据根；未传沿用配置，路径改变不放宽数据与时间协议。 |
| `--log_dir /path/to/new_run` | 指定输出目录；未传沿用配置，配置也未指定时由入口生成。新训练使用新目录；恢复使用原RUN。 |

`devices` 是可选字段，未传且YAML未设置时不会注入默认GPU编号。实际GPU记录在manifest，物理设备编号不参与算法配置哈希。默认命令中的 `CUDA_VISIBLE_DEVICES=1` 仍有效。

启动器也接受额外 `--cfg`、`--path`、`--log_dir`、`--devices`。第一个位置参数提供默认cfg，额外YAML必须属于同一recipe；若改为另一recipe，应同时更换第一个位置参数。第二个位置GPU作为默认CLI `--devices`，优先于YAML；额外 `--devices` 再覆盖它。`CT_DATA_ROOT` 只有显式设置时才提供基础 `--path`，未设置则使用所选YAML的path，额外 `--path` 最后优先。指定 `--log_dir` 时只接受新目录或空目录，否则自动生成输出目录。checkpoint恢复、初始化或test请直接使用 `main.py`，不通过scratch启动器。

## 覆盖例子

在项目根目录、沿用启动页的环境设置，以下CLI只覆盖验证频率和workers，其余值来自YAML：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --check_val_every_n_epoch 5 --workers 2 \
  --accelerator gpu --trainer_devices 1
```

对应后台启动器命令如下；与上面的直接CLI二选一，不重复执行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
bash tools/launch_v37_centroid.sh full_mass_detach 1 --check_val_every_n_epoch 5 --workers 2
```

需要同时指定YAML、数据根、输出目录和GPU时，可写为：

```bash
bash tools/launch_v37_centroid.sh full_mass_detach 1 \
  --cfg cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --log_dir "output/$(date +%Y%m%d-%H%M%S)-37_full_mass_detach-custom" \
  --devices 1 --check_val_every_n_epoch 5 --workers 2
```

启动器透传额外参数，先按实际配置核验，未指定log_dir时再以实际seed/epoch/batch生成日期加随机后缀的输出目录。若使用手写nohup命令并修改这些值，也应同步修改输出目录说明；判定运行身份始终以实际配置和manifest为准。

## 恢复与部署

恢复必须使用**同一实际算法配置、同一源码、原RUN和该RUN的完整epoch checkpoint**。以上六项实际参数均须一致：例如以val5、workers2启动，恢复仍须传 `--check_val_every_n_epoch 5 --workers 2`；省略后会回到YAML的val1、workers4，身份不匹配，恢复应被拒绝。修改batch/epoch/lr/seed同样不能跨实际配置恢复，也不能从其他配方的权重初始化。物理GPU编号不参与算法哈希，选择另一张单卡不等于改变训练配方。

本次需上传更新后的完整源码和新独立登记，三份默认YAML随版本保持一致。旧已运行任务继续使用各自原源码、配置、登记和输出，不覆盖正在运行的部署。修改前版本保存在 `artifacts/ct_checks/20261021_v37_cli_overrides/previous_source/`；旧报告及17:14服务器快照只作历史证据。本轮不访问服务器或自动启动任何变体。
