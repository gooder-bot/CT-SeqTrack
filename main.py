"""CT-SeqTrack v33/v34/v35 B0 与独立 SeqTrack 对照的训练和闭环评测入口。"""

def main(argv=None):
    """仅按已解析配置分派；旧入口与配置行为保持独立。"""
    import argparse
    import sys
    from utils.config import load_yaml_config
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg')
    known, _ = parser.parse_known_args(arguments)
    model = load_yaml_config(known.cfg).get('net_model') if known.cfg else None
    if model == 'ctseqtrackv38':
        from models.ct_v38.entry import main as entry
    elif model == 'ctseqtrackv37':
        from models.ct_v37.entry import main as entry
    elif model == 'ctseqtrackv36':
        from models.ct_v36.entry import main as entry
    else:
        from models.ct_v31.entry import main as entry
    return entry(arguments)


if __name__ == '__main__':
    main()
