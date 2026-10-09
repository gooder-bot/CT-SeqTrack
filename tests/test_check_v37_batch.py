"""Full 检查失败报告完整性；模拟辅助 loss 故障，不执行网络 batch。"""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import check_v37_full_batch as check
from models.ct_v37 import data, entry
from models.ct_v37.config import normalize_config
import models.ctseqtrackv37 as host_module


def test_default_cli_is_new_v37_full():
    from models.ct_v37.config import load_config
    args = check.parse_args([])
    cfg = load_config(args.cfg)
    assert cfg.net_model == 'ctseqtrackv37' and cfg.v37_arm == 'full'
    assert cfg.execution_profile == 'optimized_fp32'


@pytest.mark.parametrize('bad_loss',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_auxiliary_loss_preserves_original_error_and_complete_json(tmp_path,monkeypatch,bad_loss):
    """有限 total、坏辅助日志：应写原始失败原因，不让 JSON 再次报错。"""
    destination=tmp_path/'report.json'
    config=normalize_config(dict(ct_engineering_check=True))
    rows=[dict(tracklet_key=str(i),request=SimpleNamespace(branch=0,frame=1)) for i in range(16)]
    class Loader:
        generator=torch.Generator()
        batch_sampler=SimpleNamespace(set_epoch=lambda epoch:None,state_dict=lambda:{})
        def __iter__(self):
            return iter([rows])
    class Host(torch.nn.Module):
        def __init__(self,config,*,loaders):
            super().__init__()
            self.parameter=torch.nn.Parameter(torch.ones(()))
            self.training_diagnostics=None
            self.train_builder=object()
            self.tracker=SimpleNamespace(compute_losses=lambda batch,output:
                dict(loss_total=self.parameter.square(),loss_aux=torch.tensor(bad_loss)))
        def on_train_epoch_start(self):
            pass
        def configure_optimizers(self):
            return dict(optimizer=torch.optim.Adam(self.parameters()))
        def _forward_raw(self,rows,builder,*,training):
            # 只提供形状合同；不分配点云或运行真实 Full 网络。
            return dict(points=SimpleNamespace(shape=(16,4,1024,5),dtype=torch.float32)),SimpleNamespace(
                accepted_box=torch.zeros(16,4))
    monkeypatch.setattr(check,'report_destination',lambda output,version:destination)
    monkeypatch.setattr(data,'build_loaders',lambda config,roles:dict(train=Loader()))
    monkeypatch.setattr(host_module,'CTSEQTRACKV37',Host)
    from models.ct_v37 import config as config_module
    monkeypatch.setattr(config_module,'load_config',lambda path,overrides:config)
    monkeypatch.setattr(entry,'configure_numerics',lambda:None)
    monkeypatch.setattr(entry,'source_identity',lambda:dict(sha256='test'))
    monkeypatch.setitem(sys.modules,'pytorch_lightning',SimpleNamespace(
        __version__='test',seed_everything=lambda seed,workers:None))
    assert check.main(['--device','cpu'])==1
    def reject_nonfinite(value):
        raise AssertionError('nonstandard JSON number: '+value)
    result=json.loads(destination.read_text(encoding='utf-8'),parse_constant=reject_nonfinite)
    assert result['status']=='failed' and result['stage']=='forward'
    assert result['error_type']=='FloatingPointError' and result['error']=='nonfinite loss/log'
    assert 'losses' not in result
    assert not result['checkpoint_saved']
    assert 'optimizer' not in result and 'commit' not in result
