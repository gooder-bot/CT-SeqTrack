"""冻结 uniform E60 在每个 KITTI 训练端点之前的可信测量账本。"""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import zlib

from .q0_replay import decode_q0_state
from .history_augmentation import training_scene


SCHEMA = 'ct_seqtrack.uniform_e60_ledger.v1'


class LocalPrefixBank:
    """已验证覆盖的只读表。按需解压过去状态；每次返回独立的可写副本。"""
    def __init__(self, path, dataset=None):
        path = Path(path).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
        self.sha256 = digest.hexdigest()
        self.uri = path.as_uri() + '?mode=ro&immutable=1'
        self._db = None
        db = self._connection()
        rows = db.execute('SELECT value FROM metadata').fetchall()
        if len(rows) != 1:
            raise ValueError('expected one uniform prefix receipt')
        receipt = json.loads(rows[0][0])
        if (receipt['schema'] != SCHEMA or receipt['split'] != 'train' or receipt['complete'] is not True
                or receipt.get('donor_epoch') != 60 or receipt.get('center_weighting') != 'uniform'
                or not receipt.get('run_id') or any(not re.fullmatch('[0-9a-f]{64}', receipt.get(k, ''))
                    for k in ('checkpoint_sha256', 'config_sha256', 'evaluator_sha256', 'source_sha256'))):
            raise ValueError('local voting requires the complete train-only uniform E60 prefix bank')
        tracks = receipt['tracks']
        self.source_sha256 = receipt['source_sha256']
        self.lengths = dict(tracks)
        if len(self.lengths) != len(tracks):
            raise ValueError('duplicate uniform prefix tracks')
        for key, n in tracks:
            training_scene(key)
            if type(n) is not int or n < 1:
                raise ValueError('invalid track length in uniform prefix bank')
        actual = db.execute('SELECT track,min(frame),max(frame),count(*) FROM states GROUP BY track')
        coverage = {key:(start, end, count) for key, start, end, count in actual}
        expected = {key:(1, n-1, n-1) for key, n in tracks if n > 1}
        if expected != coverage or sum(n - 1 for _, n in tracks) != receipt['endpoints']:
            raise ValueError('missing or unexpected uniform prefix endpoints')
        if dataset is not None:
            self.validate_dataset(dataset)

    def _connection(self):
        if self._db is None:
            self._db = sqlite3.connect(self.uri, uri=True)
        return self._db

    def __getstate__(self):
        return dict(vars(self), _db=None)

    def validate_dataset(self, dataset):
        source = dataset.source
        if (dataset.source_sha256 != self.source_sha256 or not dataset.training or source.v41_data_role != 'train'
                or self.lengths != {source.get_tracklet_key(i): n for i, n in enumerate(dataset.lengths)}):
            raise ValueError('local prefix bank does not match the current training split')

    def before(self, track, frame):
        item = self._connection().execute('SELECT state FROM states WHERE track=? AND frame=?',
                                          (track, frame)).fetchone()
        if item is None:
            raise ValueError(f'missing uniform historical prefix {track} frame {frame}')
        return decode_q0_state(json.loads(zlib.decompress(item[0])), frame)
