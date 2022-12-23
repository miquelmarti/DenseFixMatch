def get_ssl_method(cfg, steps_per_epoch):
    import hydra
    from omegaconf import open_dict

    if not cfg.get('ssl'):
        return None
    with open_dict(cfg.ssl):
        warmup_epochs = cfg.ssl.pop('warmup_epochs', 0)
    ssl_method = hydra.utils.instantiate(
        cfg.ssl, warmup_steps=warmup_epochs * steps_per_epoch)
    return ssl_method
