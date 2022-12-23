import logging

import torch
import kornia
import wandb

from metrics import metrics


def get_logger(name):
    class DistributedAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if torch.distributed.is_initialized():
                return f'Rank: {torch.distributed.get_rank()} - {msg}', kwargs
            else:
                return msg, kwargs

    logger = logging.getLogger(name)
    return DistributedAdapter(logger, {})


def log_stats(running_loss, running_loss_dict, seen_batches, step, epoch, i, summary_writer,
              logger):
    avg_batch_loss = running_loss / seen_batches
    if summary_writer:
        summary_writer.add_scalar("losses/avg_loss", avg_batch_loss, step)
    logger.info('[%d, %5d] loss: %.5f' % (epoch, i, avg_batch_loss))
    if wandb.run:
        wandb.log({"losses/avg_loss": avg_batch_loss}, step=step)
    if len(running_loss_dict):
        for k, v in running_loss_dict.items():
            v = v.item() if isinstance(v, torch.Tensor) else v
            avg_loss = v / seen_batches
            if summary_writer:
                summary_writer.add_scalar("losses/%s" % k, avg_loss, step)
            if wandb.run:
                wandb.log({"losses/%s" % k: avg_loss}, step=step)
        logger.info(" ".join(['%s: %.5f' % (k, v / seen_batches) for k, v in
                              running_loss_dict.items()]))


def log_eval(val_loss, val_loss_dict, running_metrics, global_metric, step, epoch, summary_writer,
             eval_set=None, wandb_images=None):
    """Individual metrics are logged as the raw values regardless of EMA in use or not"""
    prefix_losses = f'eval_losses{"/" + eval_set if eval_set is not None else ""}'
    prefix_metrics = f'eval_metrics{"/" + eval_set if eval_set is not None else ""}'
    if wandb.run:
        wandb.log({"epoch": epoch}, step=step)
    if summary_writer and val_loss is not None:
        summary_writer.add_scalar(f"{prefix_losses}/avg_loss", val_loss, epoch)
    if wandb.run and val_loss is not None:
        wandb.log({f"{prefix_losses}/avg_loss": val_loss}, step=step)

    if len(val_loss_dict) > 0:
        for k, v in val_loss_dict.items():
            if summary_writer:
                summary_writer.add_scalar(f"{prefix_losses}/{k}", v, epoch)
            if wandb.run:
                wandb.log({f"{prefix_losses}/{k}": v}, step=step)

    for t, m in running_metrics.items():
        metric_value = m.get_value()
        if summary_writer:
            summary_writer.add_scalar(f"{prefix_metrics}/{t}", metric_value, epoch)
        if wandb.run:
            wandb.log({f"{prefix_metrics}/{t}": metric_value}, step=step)
        if isinstance(m, (metrics.RunningF1, metrics.RunningIOU)):
            wandb.log({
                f"{prefix_metrics}/{t}/confusion_matrix":
                    _log_confusion_matrix(m.results['confusion_matrix']),
                f"{prefix_metrics}/{t}/micro_acc": m.results['micro_acc'],
                f"{prefix_metrics}/{t}/macro_acc": m.results['macro_acc']
                }, step=step)

    if global_metric is not None:
        if summary_writer:
            summary_writer.add_scalar(f"{prefix_metrics}/global_metric", global_metric, epoch)
        if wandb.run:
            wandb.log({f"{prefix_metrics}/global_metric": global_metric}, step=step)

    if wandb_images:
        wandb.log({'eval/examples': wandb_images}, step=step)


def _log_confusion_matrix(confmatrix, labels=None):
    import plotly.graph_objs as go

    confmatrix = confmatrix.t()
    gt_per_class = confmatrix.sum(dim=1, keepdim=True)
    confmatrix = confmatrix / gt_per_class

    confdiag = torch.diag(confmatrix).clone()

    confmatrix.fill_diagonal_(0.)
    n_confused = torch.sum(confmatrix[~torch.isnan(confmatrix)])
    confmatrix[confmatrix == 0.] = torch.nan
    confmatrix = go.Heatmap(
        {'coloraxis': 'coloraxis1', 'x': labels, 'y': labels, 'z': confmatrix.cpu().numpy(),
         'hoverongaps': False,
         'hovertemplate': 'Predicted %{x}<br>Instead of %{y}<br>(%{z})<extra></extra>'})

    n_right = torch.sum(confdiag[~torch.isnan(confdiag)])
    confdiag = torch.diag_embed(confdiag)
    confdiag[confdiag == 0.] = torch.nan
    confdiag = go.Heatmap(
        {'coloraxis': 'coloraxis2', 'x': labels, 'y': labels, 'z': confdiag.cpu().numpy(),
         'hoverongaps': False,
         'hovertemplate': 'Predicted %{x} just right<br>(%{z})<extra></extra>'})

    fig = go.Figure((confdiag, confmatrix))
    transparent = 'rgba(0, 0, 0, 0)'
    fig.update_layout(
        {'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0)'], [
            1, f'rgba(180, 0, 0, {n_confused})']], 'showscale': True}})
    fig.update_layout(
        {'coloraxis2': {'colorscale': [[0, transparent], [0, 'rgba(0, 180, 0, 0)'], [
            1, f'rgba(0, 180, 0, {n_right})']], 'showscale': True, 'colorbar_xpad': 50}})

    xaxis = {'title': {'text': 'y_pred'}, 'showticklabels': True}
    yaxis = {'title': {'text': 'y_true'}, 'showticklabels': True, 'autorange': 'reversed'}

    fig.update_layout(
        title={'text': 'Confusion matrix', 'x': 0.5},
        paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    return wandb.data_types.Plotly(fig)


def log_best(global_metric, best_metrics, epoch, step, summary_writer, logger):
    logger.info("New best at epoch #%d: %.3f" % (epoch, global_metric))
    if summary_writer is not None:
        summary_writer.add_scalar("best/global_metric", global_metric, epoch)
        summary_writer.add_scalar("best/epoch", epoch, epoch)
        for t, m in best_metrics.items():
            summary_writer.add_scalar("best/%s" % t, m[0], epoch)
    if wandb.run:
        wandb.log({"best/global_metric": global_metric, "best/epoch": epoch}, step=step)
        metrics_to_log = {"best/%s" % t: v[0] for t, v in best_metrics.items()}
        wandb.log(metrics_to_log, step=step)


def log_learning_rate(parameter_groups, step, epoch, summary_writer):
    if wandb.run:
        wandb.log({"epoch": epoch}, step=step)
    for i, param_group in enumerate(parameter_groups):
        if summary_writer is not None:
            summary_writer.add_scalar(
                f'learning_rate/lr_{i}', param_group['lr'], epoch)
        if wandb.run:
            wandb.log({f"learning_rate/lr_{i}": param_group['lr']}, step=step)


def log_train_metrics(running_metrics, step, epoch, summary_writer):
    for t, m in running_metrics.items():
        metric_value = m.get_value()
        if summary_writer:
            summary_writer.add_scalar("train_metrics/%s" % t, metric_value, epoch)
        if wandb.run:
            wandb.log({"train_metrics/%s" % t: metric_value}, step=step)


def log_inputs_and_outputs(inputs, targets, outputs, step, dataset_class, num_samples=1,
                           pre='train/', wandb_images=None, conf_threshold=None):
    """Logs input and output in each batch"""
    if not wandb.run:
        return

    import torch.nn.functional as F

    if isinstance(inputs, dict):
        for k in inputs.keys():
            log_inputs_and_outputs(
                inputs[k], targets[k], outputs[k], step, dataset_class, num_samples=num_samples,
                pre=f"{pre}{k}/", conf_threshold=conf_threshold)
    else:
        input_image_shape = inputs.shape[-2:]
        input = inputs[:num_samples]
        input = kornia.enhance.denormalize(
            input, torch.Tensor(dataset_class.MEAN), torch.Tensor(dataset_class.STD)
        )
        input = input.cpu().numpy().transpose(0, 2, 3, 1)

        targets_ = {}
        for t, target in targets.items():
            target_ = target[:num_samples].clone()
            if t == 'semantic':
                target_[target_ < 0] = dataset_class.semantic_ignore
                targets_[t] = target_.to(dtype=torch.int).cpu().numpy()

        outputs_ = {}
        for t, output in outputs.items():
            if t == 'semantic':
                output = output[:num_samples].clone()
                output = F.interpolate(
                    output, size=input_image_shape, mode='bilinear', align_corners=True)
                output = output.max(dim=1)[1]
                outputs_[t] = output.to(dtype=torch.int).cpu().numpy()
            else:
                raise NotImplementedError(f"No logging for task {t}")

        if wandb_images is None:
            wandb_images = []
        for i in range(len(input)):
            masks = {}
            for t in outputs_.keys():
                masks[f'{t}_predictions'] = {
                    'mask_data': outputs_[t][i].squeeze(),
                    'class_labels': dataset_class.trainid2name
                }
                masks[f'{t}_ground_truth'] = {
                    'mask_data': targets_[t][i].squeeze(),
                    'class_labels': dataset_class.trainid2name
                }
            wandb_image = wandb.Image(input[i], masks=masks)
            wandb_images.append(wandb_image)

        if pre == 'eval/':
            return wandb_images
        else:
            wandb.log({f'{pre}examples': wandb_images}, step=step)


def log_fixmatch(k, matched_pseudo_targets, max_confidences, inputs_s, outputs_s, dataset_class,
                 log_samples, batch_size, unlabeled_masks, image_masks, step):
    matched_pseudo_targets[matched_pseudo_targets < 0] = dataset_class.semantic_ignore

    inputs = kornia.enhance.denormalize(
        inputs_s, torch.Tensor(dataset_class.MEAN), torch.Tensor(dataset_class.STD)
    )
    inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)

    wandb_images_labeled = []
    wandb_pseudo_target_confs_labeled = []
    wandb_images_unlabeled = []
    wandb_pseudo_target_confs_unlabeled = []
    for i in range(log_samples if batch_size > log_samples else batch_size):
        wandb_image = wandb.Image(
            # should be the strongly augmented image that is not available here
            inputs[i],
            masks={
                'predictions': {
                    'mask_data': outputs_s[k][i].max(dim=0)[1].to(
                        dtype=torch.int).cpu().numpy().squeeze(),
                    'class_labels': dataset_class.trainid2name
                },
                'ground_truth': {
                    'mask_data': matched_pseudo_targets[i].numpy().squeeze(),
                    'class_labels': dataset_class.trainid2name
                }
            }
        )
        wandb_pseudo_target_conf = wandb.Image(max_confidences[k][i].cpu().numpy())

        if unlabeled_masks[k][i]:
            wandb_images_unlabeled.append(wandb_image)
            wandb_pseudo_target_confs_unlabeled.append(
                wandb_pseudo_target_conf)
        elif image_masks[k][i]:
            wandb_images_labeled.append(wandb_image)
            wandb_pseudo_target_confs_labeled.append(wandb_pseudo_target_conf)

        wandb.log({
            f'fixmatch/examples/labeled/{k}': wandb_images_labeled,
            f'fixmatch/pseudo_target_confidences/labeled/{k}':
                wandb_pseudo_target_confs_labeled,
            f'fixmatch/examples/unlabeled/{k}': wandb_images_unlabeled,
            f'fixmatch/pseudo_target_confidences/unlabeled/{k}':
                wandb_pseudo_target_confs_unlabeled,
        }, step=step)
