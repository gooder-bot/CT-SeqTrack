"""CT-SeqTrack v38 clean1 与独立 SeqTrack reference 的单一入口。"""

def main(argv=None):
    """历史实验须用其冻结源码，避免误用当前执行身份恢复。"""
    import argparse
    import sys
    from utils.config import load_yaml_config
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg')
    known, _ = parser.parse_known_args(arguments)
    model = load_yaml_config(known.cfg).get('net_model') if known.cfg else None
    if model in (None, 'ctseqtrackv38'):
        from models.ct_v38.entry import main as entry
    elif model == 'seqtrack_reference':
        from models.seqtrack_reference.entry import main as entry
    else:
        parser.error('retired/unknown model ' + repr(model) +
            '; historical experiments and checkpoints require their original frozen source. '
            'See docs/HISTORY_EVIDENCE_INDEX.md. Current recipes: cfgs/ct_seqtrack/38_*_mini.yaml')
    return entry(arguments)


if __name__ == '__main__':
    main()
