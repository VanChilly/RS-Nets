import torchvision.models as models
import torch.nn as nn

class MobileNetv2_DRS(nn.Module):
    def __init__(self, num_classes=3) -> None:
        super().__init__()
        self.model = models.mobilenet_v2(False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

