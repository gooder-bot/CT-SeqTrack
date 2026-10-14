"""2026-10-14 用户登记的 B1 配方；执行算法是配置身份的一部分。"""
from contextlib import nullcontext

from utils.execution_lab import execution_scope

RECIPE_KEY = 'v37_training_recipe'
REGISTRATION_FILE = '37_b1_execution_registration.json'
REGISTRATION_SCHEMA = 'ct_seqtrack.v37.b1_execution_registration.v1'
DEDUP_OPTIONS = ('labels', 'pool', 'geometry_gather', 'unique_mask', 'commit_indices')
RECIPES = {
    'b1_factorized': dict(batch_size=16, lr_milestones=[20, 50],
                          options=DEDUP_OPTIONS + ('factorized_projection',), gpu=0),
    'b1_concat': dict(batch_size=16, lr_milestones=[20, 50],
                      options=DEDUP_OPTIONS, gpu=0),
    'b1_factorized_hold_lr': dict(batch_size=16, lr_milestones=[20],
                                  options=DEDUP_OPTIONS + ('factorized_projection',), gpu=1),
    'b1_factorized_bs128': dict(batch_size=128, lr_milestones=[20, 50],
                                options=DEDUP_OPTIONS + ('factorized_projection',), gpu=1),
    'b1_factorized_bs8': dict(batch_size=8, lr_milestones=[20, 50],
                              options=DEDUP_OPTIONS + ('factorized_projection',), gpu=1),
}


def recipe_spec(config):
    name = config.get(RECIPE_KEY)
    if name is None and RECIPE_KEY not in config:
        return None
    if not isinstance(name, str) or name not in RECIPES:
        raise ValueError('unknown registered B1 training recipe: ' + str(name))
    return RECIPES[name]


def recipe_options(config):
    spec = recipe_spec(config)
    return () if spec is None else tuple(spec['options'])


def recipe_identity(config):
    """原五组不新增默认字段；执行配方显式登记算法与实际生效范围。"""
    spec = recipe_spec(config)
    if spec is None:
        return None
    return dict(recipe=config[RECIPE_KEY], implementation_id='v37_b1_dedup_20261014_v1',
                options=list(spec['options']),
                effective_b1_options=[name for name in spec['options']
                                      if name in ('unique_mask', 'factorized_projection')],
                projection='factorized' if 'factorized_projection' in spec['options'] else 'concat',
                prior_read_applicable=False)


def recipe_scope(config):
    # 没有新配方时，不覆盖外层工程lab开关。新配方只在白名单校验后启用。
    return (nullcontext() if recipe_spec(config) is None else
            execution_scope(recipe_options(config), registered_recipe=True))
