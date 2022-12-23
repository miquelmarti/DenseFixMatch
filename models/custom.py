import torch
import torch.nn as nn


class MTLSharedEnc(nn.Module):

    def __init__(self, encoder, decoders: dict):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, input):
        encoder_output = self.encoder(input)
        outputs = {}
        for task, dec in self.decoders.items():
            outputs[task] = dec(encoder_output)
        return outputs

    def reinit_all_layers(self):
        for dec in self.decoders.values():
            dec.reinit_all_layers()


class FlattenLinear(nn.Module):

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, input):
        return self.fc(torch.flatten(input, start_dim=1))
