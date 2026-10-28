"""只比较固定 final60/late3，完整保留两组学习率配方的精度/恢复/计算结果。"""
from pathlib import Path
import argparse
import json


def read_run(path):
    path=Path(path)
    result=json.loads((path/'results.json').read_text(encoding='utf-8'))
    manifest=json.loads((path/'run_manifest.json').read_text(encoding='utf-8'))
    if result.get('checkpoint_epochs') != [58,59,60] or result['final']['checkpoint_epoch']!=60:
        raise ValueError('comparison requires completed final60 and late3=[58,59,60]: '+str(path))
    return dict(path=str(path.resolve()),manifest=manifest,result=result)


def compare(first,second):
    a,b=read_run(first),read_run(second)
    for field in ('source','initial_parameter_hash'):
        if field not in a['manifest'] or field not in b['manifest']:
            raise ValueError('run manifest missing '+field)
        if a['manifest'][field] != b['manifest'][field]:
            raise ValueError('paired runs differ in '+field)
    recipes=[run['manifest'].get('training_execution_recipe',{}).get('recipe') for run in (a,b)]
    if recipes != ['full_a','full_b']:
        raise ValueError('arguments must identify registered Full-A then Full-B')
    rows=[]
    for run in (a,b):
        result=run['result'];final=result['final']
        rows.append(dict(path=run['path'],final60={k:final[k] for k in ('success','precision')},
            late3=result['late3'],motion_slices=final.get('motion_slices'),
            diagnostics=final.get('diagnostics'),trusted=final.get('trusted'),
            computation=final.get('computation'),evaluation_modes=result.get('evaluation_modes'),
            adaptive_threshold_curve=result.get('adaptive_threshold_curve')))
    delta={key:rows[1]['late3'][key]-rows[0]['late3'][key] for key in ('success','precision')}
    return dict(schema='ct_v39_comparison_v1',runs=rows,late3_delta_second_minus_first=delta,
        interpretation='single-seed full LR comparison; no causal E/G/M contribution claim',
        selection_rule='existing overall/motion gates first; then late3 mean(S,P); final60 and severe drift secondary')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('full_a');parser.add_argument('full_b');parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    root=Path(__file__).resolve().parents[1]/'artifacts/ct_checks'
    if root.resolve() not in output.parents or output.exists():
        parser.error('output must be a new file under artifacts/ct_checks')
    result=compare(args.full_a,args.full_b);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(str(output))


if __name__=='__main__': main()
