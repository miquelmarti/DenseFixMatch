def get_criteria(cfg):
    import torch

    from losses.losses import OhemCrossEntropy2D, UpsamplingCELoss, UniformMultiLoss


    def _get_criterion_from_task_def(task_def):
        task_type = task_def.type
        if task_type == 'classification':
            return torch.nn.CrossEntropyLoss(ignore_index=task_def.null_target)
        elif task_type == 'semantic_segmentation':
            if task_def.ohem:
                return OhemCrossEntropy2D(ignore_index=task_def.null_target)
            else:
                return UpsamplingCELoss(ignore_index=task_def.null_target)
        else:
            raise NotImplementedError(f'Criterion for task type {task_type} not defined.')

    if hasattr(cfg.dataset, 'tasks'):
        tasks = cfg.get('tasks')
        declared_tasks = cfg.dataset.tasks.keys()
        if tasks is not None:
            assert all([t in declared_tasks for t in tasks])
        else:
            tasks = declared_tasks

        criteria = {}
        for t in tasks:
            criteria[t] = _get_criterion_from_task_def(cfg.dataset.tasks.get(t))

        criteria = UniformMultiLoss(criteria)
        return criteria

    else:
        return _get_criterion_from_task_def(cfg.dataset.task)
