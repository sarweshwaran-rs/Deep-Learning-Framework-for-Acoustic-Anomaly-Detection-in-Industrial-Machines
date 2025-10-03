import torch
import torch.nn as nn
from torchvision.models import resnet18, efficientnet_b0, densenet121
import torch.nn.functional as F
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

class STFTFeatureExtractor_EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = efficientnet_b0(weights=None)

        efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        self.stem = efficientnet.features[0]   
        self.block1 = efficientnet.features[1]
        self.block2 = efficientnet.features[2]
        self.block3 = efficientnet.features[3]
        self.block4 = efficientnet.features[4]
        self.block5 = efficientnet.features[5]
        self.block6 = efficientnet.features[6]
        self.block7 = efficientnet.features[7]
        self.block8 = efficientnet.features[8]

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)  

        return x

class STFTFeatureExtractor_DenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = densenet121(weights=None)

        densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=8, stride=2, padding=3, bias=False
        )

        self.features = densenet.features
    
    def forward(self, x):
        x = self.features(x)
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

            block.conv2 = nn.Conv2d(
                in_channels=block.conv2.in_channels,
                out_channels=block.conv2.out_channels,
                kernel_size=kernel_size,
                stride=block.conv2.stride,
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

class CQTFeatureExtractor_SwinTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model(
            'swin_tiny_patch4_window7_224', pretrained=False, num_classes=0, global_pool='', img_size=None
        )
        self.model.patch_embed.img_size = None # type: ignore
        self.model.patch_embed.strict_img_size = False # type: ignore

        self.model.patch_embed.proj = nn.Conv2d( # type: ignore
            1, 96, kernel_size=4, stride=4
        )

    def forward(self, x):
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        return self.model(x)

class CQTFeatureExtractor_ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='')
        # Input conv to 1 channel
        self.model.stem[0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)  # type: ignore

    def forward(self, x):
        return self.model(x)   # [B, 768, H/32, W/32]

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


# ===== SpectralPositionalEncoding =====
class SpectralPositionalEncoding(nn.Module):
    """
    Adds frequency-wise positional encoding to a 4D feature tensor (B,C,F,T).
    The encoding is broadcas along the time dimension
    """
    def __init__(self, num_freqs, dim):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1,dim, num_freqs,1)) # (1,C,F,1)

    def forward(self, x):
        # x: (B,C,F,T)
        B,C,F,T = x.shape
        pe = self.positional_encoding.expand(B,C,F,T)
        
        return x + pe