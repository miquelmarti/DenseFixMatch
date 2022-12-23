def get_model(cfg):
    import hydra
    import torch

    from models.custom import MTLSharedEnc


    model = hydra.utils.instantiate(cfg.model.encoder)
    if cfg.model.get('decoder'):
        if hasattr(cfg.dataset, 'tasks'):
            tasks = cfg.get('tasks')
            declared_tasks = cfg.dataset.get('tasks').keys()
            if tasks is not None:
                assert all([t in declared_tasks for t in tasks])
            else:
                tasks = declared_tasks

            decoders = {}
            for t in tasks:
                decoders[t] = hydra.utils.instantiate(cfg.model.decoder.get(t))
            model = MTLSharedEnc(model, decoders)

        else:
            decoder = hydra.utils.instantiate(cfg.model.decoder)
            model = torch.nn.Sequential(model, decoder)

    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
