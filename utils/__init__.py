def register_parent_key_resolver():
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("parent_key", lambda _parent_: _parent_._key())

def register_if_resolver():
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("if", lambda cond, x, y: x if cond else y)

register_parent_key_resolver()
register_if_resolver()
