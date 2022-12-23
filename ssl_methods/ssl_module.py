import abc

import torch


class SSLModule(abc.ABC, torch.nn.Module):

    def __init__(self, null_target, consistency_weight_max=None, warmup_steps=None):
        super(SSLModule, self).__init__()
        self.null_target = null_target

        self.ssl_criterion = None
        self.loss = 0.

        self.consistency_weight_max = consistency_weight_max
        self.warmup_steps = warmup_steps
        if consistency_weight_max or warmup_steps:
            assert consistency_weight_max is not None and warmup_steps is not None

    @abc.abstractmethod
    def forward(self, inputs, outputs, labels, model, loss, step=None):
        pass

    def _get_consistency_weight(self, step=None):
        """
        Adapted from https://github.com/brain-research/realistic-ssl-evaluation/

        Multiplier warm-up schedule from Appendix B.1 of
        the Mean Teacher paper (https://arxiv.org/abs/1703.01780)
        "The consistency cost coefficient and the learning rate were ramped up
        from 0 to their maximum values, using a sigmoid-shaped function
        e^{−5(1−x)^2}, where x in [0, 1]."
        """
        if self.consistency_weight_max and self.warmup_steps is not None and step is not None:
            return self.consistency_weight_max * torch.exp(
                -5. * torch.pow(1. - torch.tensor(float(step)) / self.warmup_steps, 2.)
            ) if step < self.warmup_steps else self.consistency_weight_max
        else:
            return 1.
