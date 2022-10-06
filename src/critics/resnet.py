import torchvision
from torch import nn


def ResNet18(out_dim):
    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=(7, 7),
                            stride=(2, 2),
                            padding=(3, 3),
                            bias=False)
    model.fc = nn.Linear(512, out_dim)
    return model
