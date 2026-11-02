"""V41启动不依赖登记；自动源码记录支持跨换行的同run续训核对。"""
from models.ct_v41.provenance import source_identity


SOURCES = {
    'main.py': 'def main():\n    return 1\n',
    'requirement.txt': 'torch==2.0.1\nnumpy==1.24.4\n',
    'models/head.py': '# 当前模型\nWEIGHT = 1\n',
    'models/nested/layer.py': 'WIDTH = 64\n',
    'datasets/input.py': 'FRAME_PERIOD = 0.1\n',
    'utils/helper.py': 'def identity(value):\n    return value\n',
}


def _source_tree(root, *, newline='\n'):
    for relative, content in SOURCES.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 明确写原始bytes，不能由运行测试的操作系统替换换行。
        destination.write_bytes(content.replace('\n', newline).encode('utf-8'))
    return root


def test_source_identity_is_independent_of_newlines_and_root(tmp_path):
    linux = _source_tree(tmp_path / 'linux', newline='\n')
    windows = _source_tree(tmp_path / 'windows', newline='\r\n')
    assert (linux / 'main.py').read_bytes() != (windows / 'main.py').read_bytes()
    assert source_identity(root=linux) == source_identity(root=windows)


def test_source_identity_changes_for_actual_code_edits(tmp_path):
    root = _source_tree(tmp_path / 'source')
    before = source_identity(root=root)
    (root / 'models/head.py').write_bytes('# 当前模型\nWEIGHT = 2\n'.encode('utf-8'))
    after = source_identity(root=root)
    assert before['sha256'] != after['sha256']


def test_source_identity_does_not_read_configuration_or_registration(tmp_path):
    root = _source_tree(tmp_path / 'source')
    # 无cfg目录、无登记文件也能正常取得源码信息。
    without_registration = source_identity(root=root)
    configuration = root / 'cfgs/ct_seqtrack'
    configuration.mkdir(parents=True)
    (configuration / '41_full_kitti.yaml').write_bytes(b'not valid: [yaml\n')
    registration = configuration / '41_registration.json'
    registration.write_bytes(b'{invalid JSON\xff\n')
    assert source_identity(root=root) == without_registration
    # 已有登记即使内容自相矛盾也不应成为源码信息的输入或启动门。
    registration.write_bytes(b'{"source":{"sha256":"stale"}}\r\n')
    assert source_identity(root=root) == without_registration
