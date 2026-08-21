# Slimming baseline

本目录由 `python tools/verify_ct_slimming.py snapshot --run-tests` 在
`main@001951a` 上生成，用于证明瘦身前后的配置、Git 可恢复性和本地 `output/`
保护状态。

- `tracked_files.jsonl`：850 个基线文件的路径、大小、SHA256 和恢复对象。
- `active_configs_resolved.json`：22 个正式/B4 入口的完整解析结果。
- `output_protection.json`：只记录路径/大小/mtime 清单哈希，不读取或改写输出内容。
- `model_initialization.json`：当前环境缺依赖时显式标记 blocked；真实环境补齐前不得
  物理删除 dormant source branches。
- `pytest_baseline.txt`：`122 passed, 1 skipped` 的基线结果。

验证配置和受保护输出未改变：

```bash
python tools/verify_ct_slimming.py verify
```
