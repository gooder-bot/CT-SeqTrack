"""运行时自动记录当前源码；不依赖预登记，换行转换不改变同run身份。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .identity import retired_source_path

ROOT = Path(__file__).resolve().parents[2]
HASH_POLICY = 'v41_runtime_python_text_lf_v1'


def text_hash(path):
    # Windows/Git上传时CRLF与LF代表同一份Python文本，不strip代码或注释。
    raw = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(raw).hexdigest()


def source_identity(root=ROOT):
    root = Path(root)
    paths = [root / 'main.py', root / 'requirement.txt']
    for folder in ('models', 'datasets', 'utils'):
        paths.extend((root / folder).rglob('*.py'))
    files = {path.relative_to(root).as_posix(): text_hash(path) for path in sorted(paths)
             if not retired_source_path(path.relative_to(root).as_posix())}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(sha256=digest, files=files, inventory_policy=HASH_POLICY)
