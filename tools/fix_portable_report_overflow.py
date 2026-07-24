#!/usr/bin/env python3
"""Apply a narrow Windows-scrollbar containment fix to a portable report.

The bundled Data Analytics reader uses ``100vw`` for its sticky top bar. In
Windows headless Chromium, ``100vw`` includes the vertical scrollbar while the
document client width does not, producing an otherwise harmless 8–15 px page
overflow that fails the portable verifier. This post-process keeps the real
reader runtime intact and constrains only that top bar to its report shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STYLE = """<style id="ct-seqtrack-portable-overflow-fix">
.analytics-top-bar {
  width: 100% !important;
  max-width: 100% !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
}
@media (max-width: 760px) {
  .recharts-legend-wrapper {
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
  }
  .chart-legend {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  .chart-legend-item {
    margin: 2px 4px !important;
  }
}
</style>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    html = args.input.read_text(encoding="utf-8")
    if STYLE not in html:
        if "</head>" not in html:
            raise RuntimeError("Portable report has no </head> insertion point.")
        html = html.replace("</head>", f"{STYLE}</head>", 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
