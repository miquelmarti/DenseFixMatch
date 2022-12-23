import torch
from torch import nn
from torchvision.models import resnet


ARCHITECTURES = resnet.__all__[1:]


class Resnet(nn.Module):
    def __init__(self, arch, pretrained=False, small_images=True, final_pooling=True, fpn=False,
                 multi_grid=False, **kwargs):
        super().__init__()
        assert arch in ARCHITECTURES, f'Resnet architecture name not in: {ARCHITECTURES}'
        self.zero_init_residual = kwargs.get('zero_init_residual')
        self.final_pooling = final_pooling
        self.fpn = fpn

        weights = 'DEFAULT' if pretrained else None
        net = getattr(resnet, arch)(weights=weights, **kwargs)
        if not pretrained and small_images:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = torch.nn.Identity()
        elif pretrained and small_images:
            raise ValueError("Resnets for small images such as cifar are not available pretrained")

        # discard last fc layer
        layers = list(net.named_children())[:-1]

        for name, layer in layers:
            if name == 'fc':
                continue
            elif name == 'avgpool' and not self.final_pooling:
                continue
            elif name == 'layer4' and multi_grid:
                _apply_multi_grid(layer, [2, 2, 4])
            self.add_module(name, layer)

        if not pretrained and small_images:
            self._reinit_all_layers()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.final_pooling:
            x = self.avgpool(x4)
        else:
            x = x4

        if self.fpn:
            return [x1, x2, x3, x4]
        else:
            return x

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, self.net.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, self.net.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


def _apply_multi_grid(bottlenecks, grid):
    for i, bottleneck in enumerate(bottlenecks):
        dilation = (bottleneck.conv2.dilation[0] * grid[i], bottleneck.conv2.dilation[1] * grid[i])
        bottleneck.conv2.dilation = dilation
        bottleneck.conv2.padding = dilation
