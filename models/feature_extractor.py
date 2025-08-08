import torch.nn as nn
from torchvision.models import resnet18
from timm import create_model
import matplotlib.pyplot as plt 

class STFTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
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

# STFT Feature Extractor with Novalty planned Frequency adaptive convolution
class STFTFrequencyAdaptiveFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
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

# CQT Feature Extractor using the mobilevit_xxs model
class CQTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model('mobilevit_xxs', pretrained=False,num_classes=0, global_pool='')

        self.model.stem.conv = nn.Conv2d( # type: ignore
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
    def forward(self,x):
        return self.model(x)

#Projection + Polling Block to Match the Channels, Height, and Width of the Extractor Features to Match with the Shape(4,16) with 256 Channel Size
class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_channels=256, target_hw=(4,16)):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(target_hw)

    def forward(self, x):
        x = self.proj(x)
        x = self.pool(x)
        return x

# Code to Visualize the Extracted Feature Pairs    
def visualize_feature_maps(feature_tensor, titile_prefix, num_channels=8):
    """
        Feature tesor: torch.Tensor of shape (B,C,H,W)
    """

    feature_tensor = feature_tensor.squeeze(0)
    C = feature_tensor.shape[0]
    num_channels = min(num_channels, C)

    plt.figure(figsize=(12,4))
    for i in range(num_channels):
        fmap = feature_tensor[i].cpu().numpy()

        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        plt.subplot(1, num_channels, i+1)
        plt.imshow(fmap, cmap='viridis', aspect='auto')
        plt.axis('off')
        plt.title(f"{titile_prefix} C{i}")
    plt.tight_layout()
    plt.show()