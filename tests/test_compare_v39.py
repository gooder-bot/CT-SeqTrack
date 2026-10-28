import json
import pytest
from tools.compare_v39 import compare


def write_run(root,recipe,score,*,epoch=60,parameter_hash='same'):
    root.mkdir()
    manifest=dict(source={'sha256':'same'},initial_parameter_hash=parameter_hash,
                  training_execution_recipe={'recipe':recipe})
    result=dict(checkpoint_epochs=[58,59,60],final=dict(checkpoint_epoch=epoch,success=score,precision=score+10),
                late3=dict(success=score-1,precision=score+9))
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    (root/'results.json').write_text(json.dumps(result))


def test_compare_fixed_windows_and_initial_identity(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    write_run(a,'full_a',50);write_run(b,'full_b',52)
    result=compare(a,b)
    assert result['late3_delta_second_minus_first']==dict(success=2,precision=2)
    with pytest.raises(ValueError,match='Full-A then Full-B'): compare(b,a)


@pytest.mark.parametrize('change',[dict(epoch=59),dict(parameter_hash='different')])
def test_compare_rejects_incomplete_or_different_initialization(tmp_path,change):
    a,b=tmp_path/'a',tmp_path/'b'
    write_run(a,'full_a',50);write_run(b,'full_b',52,**change)
    with pytest.raises(ValueError): compare(a,b)
