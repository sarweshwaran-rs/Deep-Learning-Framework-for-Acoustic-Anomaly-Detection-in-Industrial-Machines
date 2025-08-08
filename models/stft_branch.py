import torch.nn as nn
from torchvision.models import resnet18

class STFTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
class STFTFrequencyAdaptiveFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        self.layer3 = self._make_adaptive_layer(resnet.layer3, kernel_size = (1,7))
        self.layer4 = self._make_adaptive_layer(resnet.layer4, kernel_size = (1,15))

    def _make_adaptive_layer(self, layer, kernel_size):
        for block in layer:
            block.conv1 = nn.Conv2d(
                in_channels=block.conv1.in_channels,
                out_channels=block.conv1.out_channels,
                kernel_size=kernel_size,
                stride=block.conv1.stride,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                bias=False 
            )
        return layer
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
