"""为工程对照重建冻结 Git 源码；不读取或改写既有实验输出。"""
from pathlib import Path
import subprocess
import tempfile
import zipfile


def prepare_frozen_source():
    root = Path(__file__).resolve().parents[2]
    checks = root/'artifacts/ct_checks'
    checks.mkdir(parents=True, exist_ok=True)
    audit = checks/'20261023_v38_clean1'
    destination = Path(tempfile.mkdtemp(prefix='equivalence_', dir=audit if audit.exists() else checks))
    archive, frozen = destination/'source.zip', destination/'frozen'
    result = subprocess.run(['git','archive','--format=zip','--output='+str(archive),'14c3b2f'],
        cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('冻结对照需要Git提交14c3b2f；请保留对应历史，不能用当前实现替代基准。\n'+result.stderr)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.endswith('/') or name.startswith(('artifacts/','output/')):
                continue
            path = frozen/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read(name))
    return destination, frozen
