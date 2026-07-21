# M0-3 / M0-4 portable report

正式交付文件是 `report.html`。它由 `artifact.json`、`report_snapshot.sql` 和项目级复核快照构建，包含同一份 canonical artifact payload、两张原生图、指标卡、来源弹窗和语义 fallback。

在安装 Data Analytics plugin 的 Codex 环境中重新构建：

```bash
node compare_results/reports/m0_m03_m04_report/build_report.mjs
```

脚本默认从 Codex home 下的插件缓存读取 `data-analytics/0.2.8-13ceeea1f599`。若插件位于其他位置，可只为该命令设置 `DATA_ANALYTICS_PLUGIN_ROOT`。构建会提取 light/dark 静态 chart SVG，并验证 1440px、390px、来源交互、无外部请求和 payload 一致性。

`build_report.mjs` 还修复了当前 portable reader 顶栏在 Chromium 垂直滚动条下产生的 8px `100vw` 溢出；该修复只作用于 HTML shell，不修改 artifact payload 或实验数据。
