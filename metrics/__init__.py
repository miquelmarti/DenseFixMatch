def get_metrics(cfg):
    from metrics.metrics import RunningAccuracy, RunningIOU, ArithmeticMean

    def _get_metrics_from_task_def(task_def):
        task_type = task_def.type
        if task_type == 'classification':
            return RunningAccuracy(
                null_target=task_def.null_target, n_classes=task_def.num_classes)
        elif task_type == 'semantic_segmentation':
            return RunningIOU(
                null_target=task_def.null_target, n_classes=task_def.num_classes,
                background_class=task_def.background_class)
        else:
            raise NotImplementedError(f'Metrics for task type {task_type} not defined.')

    joint_metric = None
    metrics = {}
    if hasattr(cfg.dataset, 'tasks'):
        tasks = cfg.get('tasks')
        declared_tasks = cfg.dataset.tasks.keys()
        if tasks is not None:
            assert all([t in declared_tasks for t in tasks])
        else:
            tasks = declared_tasks

        for t in tasks:
            metrics[t] = _get_metrics_from_task_def(cfg.dataset.tasks.get(t))

        joint_metric = ArithmeticMean(metrics, use_ema=cfg.evaluation.use_ema_metrics)

    else:
        metrics[cfg.dataset.task.type] = _get_metrics_from_task_def(cfg.dataset.task)
        joint_metric = ArithmeticMean(metrics, use_ema=cfg.evaluation.use_ema_metrics)

    return metrics, joint_metric
