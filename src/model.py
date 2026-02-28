from typing import List

import torch.nn as nn
from torchvision import models


def build_mobilenet(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def class_names() -> List[str]:
    return ["normal", "smoke_fire", "oil_leak", "conveyor_jam"]

