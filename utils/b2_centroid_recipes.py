"""v37 B2 重心梯度三组配方；不改变原五组或 B1 执行配方的缺省身份。"""
from utils.b1_training_recipes import RECIPE_KEY as B1_RECIPE_KEY
from utils.b1_training_recipes import recipe_scope as b1_recipe_scope
from utils.execution_lab import execution_scope


RECIPE_KEY = 'v37_centroid_recipe'
REGISTRATION_FILE = '37_b2_centroid_registration.json'
REGISTRATION_SCHEMA = 'ct_seqtrack.v37.b2_centroid_registration.v1'
RECIPES = {
    'b1_b2_mass_detach': dict(arm='b1_b2', lr_milestones=[20, 50]),
    'full_mass_detach': dict(arm='full', lr_milestones=[20, 50]),
    'full_mass_detach_hold_lr': dict(arm='full', lr_milestones=[20]),
}


def recipe_spec(config):
    if RECIPE_KEY in config and B1_RECIPE_KEY in config:
        raise ValueError('v37 centroid and B1 execution recipes are mutually exclusive')
    if RECIPE_KEY not in config:
        return None
    name = config[RECIPE_KEY]
    if not isinstance(name, str) or name not in RECIPES:
        raise ValueError('unknown registered B2 centroid recipe: ' + str(name))
    return RECIPES[name]


def recipe_identity(config):
    spec = recipe_spec(config)
    if spec is None:
        return None
    return dict(recipe=config[RECIPE_KEY],
                implementation_id='v37_b2_centroid_mass_detach_20261021_v1',
                arm=spec['arm'], centroid_mass_gradient='detached_identity_reliability',
                options=[], projection='concat', execution_profile='optimized_fp32',
                lr_milestones=list(spec['lr_milestones']), check_val_every_n_epoch=5)


def detach_mass_weights(config):
    """只由显式新配方启用；旧配置不注入 False 字段。"""
    return recipe_spec(config) is not None


def recipe_scope(config):
    """新配方屏蔽环境工程开关，旧配置完整保留原 B1/缺省作用域。"""
    if recipe_spec(config) is not None:
        return execution_scope((), registered_recipe=True)
    return b1_recipe_scope(config)
