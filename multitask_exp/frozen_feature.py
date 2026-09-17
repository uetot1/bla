import torch
from torch import nn

from models.experimental import attempt_load


class FrozenYoloFeature(nn.Module):
    """A pretrained YOLOv5 model cut at one layer, never updated.

    Gradients still flow through it to the input image, so the codec is taught
    by exactly the model used at evaluation time.
    """

    def __init__(self, weights, layer, device):
        super().__init__()
        self.model = attempt_load(weights, device=device, fuse=False)
        if not 0 <= layer < len(self.model.model):
            raise ValueError(f"Layer {layer} is outside 0..{len(self.model.model) - 1}")
        self.layer = layer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        # BatchNorm must keep its pretrained running statistics.
        return super().train(False)

    def forward(self, images):
        return self.model(images, cut_model=1, cutting_layer=self.layer)

    @torch.no_grad()
    def target(self, images):
        return self.forward(images).detach()
