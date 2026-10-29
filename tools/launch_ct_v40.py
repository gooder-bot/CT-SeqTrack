"""四组 V40 从零训练：验收完整后，GPU0 两波各两个实际进程。默认只生成计划。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_v40.engineering import (ENGINEERING_SCHEMA, acceptance_environment, require,
    write_new_json, analyze_memory_cycles, capacity_gate)
from tools.check_v40_server import PHASES, WAVES, identity


def selected_waves(selection='all'):
    """只选择执行批次，不改变四组科学配方或六阶段验收合同。"""
    require(selection in ('all', '1', '2'), 'wave must be all, 1 or 2')
    return [(index, pair) for index, pair in enumerate(WAVES, 1)
            if selection == 'all' or index == int(selection)]


def validate_acceptance(paths):
    expected = identity()
    reports = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        phase = report.get('phase')
        require(report.get('schema') == ENGINEERING_SCHEMA and report.get('passed') is True,
                'acceptance report is not a successful V40 server check: ' + str(path))
        require(report.get('identity') == expected, 'acceptance source/tools/registration no longer match')
        require(phase in PHASES and phase not in reports, 'duplicate or unknown acceptance phase')
        reports[phase] = report
    require(set(reports) == set(PHASES), 'all six server acceptance phases are required before formal launch')
    commands = reports['contracts'].get('commands', [])
    require(len(commands) == 3 and any('pytest' in command for command in commands)
            and any('compileall' in command for command in commands)
            and any('tools/verify_ct_v40.py' in command for command in commands), 'contract command evidence incomplete')
    def checked_workers(phase, *, count, arms=None):
        workers = reports[phase].get('workers', [])
        require(len(workers) == count and all(worker.get('passed') is True
            and worker.get('identity') == expected and worker.get('phase') == phase for worker in workers),
            phase + ': child worker evidence incomplete or stale')
        if arms is not None:
            require(sorted(worker.get('arm') for worker in workers) == sorted(arms), phase + ': wrong accepted arms')
        return workers
    kernel = checked_workers('kernel', count=1, arms=['full'])[0]
    require(kernel['evidence'].get('kernel_calls', 0) > 0 and kernel['evidence'].get('cuda_kernels')
            and kernel['fa_runtime'].get('kernel_dtype') == 'torch.bfloat16', 'actual Full FA2 kernel evidence missing')
    for worker in checked_workers('lifecycle', count=4, arms=['b0', 'b1', 'b1_mtp', 'full']):
        require(worker['evidence'].get('optimizer_steps') == 8 and worker['evidence'].get('endpoint_rows') == 128,
                'four-arm real lifecycle was not completed')
    resume = reports['lifecycle'].get('entry_epoch_boundary_resume', {})
    require(resume.get('passed') is True and [part.get('stop_epoch') for part in resume.get('resumed_parts', [])] == [10, 20, 21],
            'actual entry epoch-boundary resume evidence missing')
    require(resume['reference']['final'] == resume['resumed_parts'][-1]['final']
            and resume['reference']['order'] == [row for part in resume['resumed_parts'] for row in part['order']],
            'entry resumed terminal state or order differs')
    memory = checked_workers('memory', count=2, arms=['full', 'full'])
    require({worker.get('allocator') for worker in memory} == {'native', 'round2'}, 'independent allocator checks missing')
    for worker in memory:
        cycles = worker['evidence']['cycles']
        require(analyze_memory_cycles(cycles)['passed'] and worker['evidence'].get('gate', {}).get('passed'), 'memory replay gate failed')
        for cycle in cycles:
            require(cycle['train']['count'] >= 256 and cycle['validation_frames'] > 0
                and all(cycle['coverage'].get(key, 0) > 0 for key in
                    ('empty_sequences', 'guard_rows', 'drain_rows', 'partial_batches', 'current_sparse_1_2')),
                'memory workload/branch coverage incomplete')
    require(len({worker['evidence']['cycles'][0]['workload_sha256'] for worker in memory}) == 1,
            'native and round2 workloads differ')
    performance = checked_workers('performance', count=2, arms=['full', 'full'])
    require({worker.get('allocator') for worker in performance} == {'native', 'round2'}
            and all(len(worker['evidence']['cycles']) == 3 for worker in performance), 'full pipeline performance evidence missing')
    fixed = reports['performance'].get('fixed_input_fa_vs_math', {})
    twins = fixed.get('workers', [])
    require(fixed.get('passed') is True and len(twins) == 2
        and {worker.get('backend') for worker in twins} == {'math', 'fa2_varlen'}
        and all(worker.get('passed') is True and worker.get('identity') == expected
                and worker['timing']['count'] >= 8 for worker in twins)
        and len({worker['prepared_sha256'] for worker in twins}) == 1
        and len({worker['model_sha256'] for worker in twins}) == 1,
        'same-input same-weight FA/math network-step performance evidence missing')
    tracking = fixed.get('tracking_batch1', {})
    require(tracking.get('passed') is True and tracking.get('identity') == expected
        and tracking.get('batch_size') == 1 and tracking.get('warmup_frames') == 16
        and tracking.get('timing', {}).get('count', 0) >= 128
        and tracking.get('p50_frame_seconds', 0) > 0 and tracking.get('p95_frame_seconds', 0) > 0,
        'actual batch1 complete-frame latency evidence missing')
    require([pair['labels'] for pair in reports['parallel']['pairs']] == [list(pair) for pair in WAVES],
            'parallel acceptance did not test the two registered pairs')
    for pair in reports['parallel']['pairs']:
        children = pair['children']
        require(len(children) == 2 and all(child.get('passed') is True and child.get('identity') == expected for child in children)
                and [child.get('arm') for child in children] == pair['labels'], 'paired worker evidence differs')
        gate = capacity_gate(pair['nvml_samples'], [child['pid'] for child in children])
        require(gate['passed'] and pair['gate']['passed'] and pair['gate'].get('conservative_margin_passed') is True,
                'actual paired capacity or conservative reserved/context/external margin failed')
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--path', required=True)
    parser.add_argument('--run-root', required=True, help='new parent directory under output/')
    parser.add_argument('--wave', choices=('all', '1', '2'), default='all',
                        help='all: original four-arm order; 1: B0+B1 only; 2: B1 MTP+Full only')
    parser.add_argument('--acceptance', nargs='+', default=[], help='six report.json files; required with --execute')
    parser.add_argument('--execute', action='store_true', help='actually run the selected wave(s) after all acceptance gates')
    args = parser.parse_args(argv)
    from models.ct_v40.config import load_config
    from models.ct_v40.entry import validate_formal_registration
    from models.ct_v40.identity import ALLOCATOR
    from models.ct_v40.engineering import ALLOCATORS
    require(ALLOCATOR == ALLOCATORS['round2'], 'formal allocator changed; rerun registered capacity checks')
    destination = Path(args.run_root).resolve()
    require(destination.is_relative_to((ROOT / 'output').resolve()) and destination != (ROOT / 'output').resolve(),
            '--run-root must be a new child directory of output/')
    require(not destination.exists(), 'formal launch must not reuse or overwrite an existing directory')
    waves = selected_waves(args.wave)
    commands = {}
    for unused, pair in waves:
        for label in pair:
            cfg = ROOT / f'cfgs/ct_seqtrack/40_{label}_mini.yaml'
            config = load_config(cfg, dict(path=str(Path(args.path).resolve()), log_dir=str(destination / label)))
            validate_formal_registration(config)
            commands[label] = [sys.executable, 'main.py', '--cfg', str(cfg), '--path', config.path,
                '--log_dir', config.log_dir, '--tag', 'v40_' + label]
    plan = dict(waves=[list(pair) for unused, pair in waves], selected_wave_ids=[index for index, unused in waves],
                wave_selection=args.wave, physical_gpu=0, commands=commands,
                epochs_per_run=60, total_epochs=60 * len(commands), registered_four_arm_total_epochs=240,
                batch_size=16, workers=4,
                allocator=ALLOCATOR, initialization='scratch_only', execution='not_started')
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    registered = validate_acceptance(args.acceptance)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_json(destination / 'launch_plan.json', dict(plan, identity=registered))
    launched = []
    for wave_index, pair in waves:
        children = []
        try:
            for label in pair:
                directory = destination / label
                directory.mkdir(exist_ok=False)
                log = (directory / 'train.log').open('x', encoding='utf-8')
                process = subprocess.Popen(commands[label], cwd=ROOT, env=acceptance_environment('round2'),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                with (directory / 'train.pid').open('x', encoding='utf-8') as stream:
                    stream.write(str(process.pid) + '\n')
                children.append((label, process, log))
                launched.append(dict(label=label, pid=process.pid, wave=wave_index))
            write_new_json(destination / f'wave{wave_index}_started.json', dict(children=launched[-2:]))
            while any(process.poll() is None for unused, process, log in children):
                require(all(process.poll() in (None, 0) for unused, process, log in children),
                        'one formal run failed; peer stopped and next wave not launched')
                time.sleep(1)
            require(all(process.returncode == 0 for unused, process, log in children), 'formal wave failed')
            write_new_json(destination / f'wave{wave_index}_completed.json', dict(labels=list(pair), returncodes=[0, 0]))
        finally:
            for unused, process, log in children:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                log.close()
    completed_ids = [index for index, unused in waves]
    write_new_json(destination / 'launch_completed.json', dict(children=launched,
        completed_waves=len(waves), completed_wave_ids=completed_ids))
    print(json.dumps(dict(completed_waves=len(waves), completed_wave_ids=completed_ids,
                         directory=str(destination)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
