"""v34 真实单批检查入口：不保存工程权重，正式训练另起 scratch 进程。"""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_v33_batch import main as _main


def main(argv=None):
    return _main(argv, version=34)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
