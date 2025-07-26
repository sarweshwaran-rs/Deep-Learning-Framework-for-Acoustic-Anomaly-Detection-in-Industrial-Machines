import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights

class BasicSpectrogramClassifier(nn.Module):

    def __init__(self, encoder_name = 'resnet18', pretrained=False, num_classes=2,in_channels=3):
        super(BasicSpectrogramClassifier, self).__init__()

        if encoder_name == 'resnet18':
            if pretrained:
                self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.encoder = models.resnet18(weights=None)
                
            self.encoder.conv1 = nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
            num_ftrs = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity() # type: ignore
        
        elif encoder_name == 'efficient_b0':
            if pretrained:
                self.encoder = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            else:
                self.encoder = models.efficientnet_b0(weights=None)
            self.encoder.features[0][0] = nn.Conv2d(in_channels,32,kernel_size=3,stride=2, padding=1, bias=False)
            num_ftrs = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity() # type: ignore

        else:
            raise ValueError(f"Encoder {encoder_name} not supported. Choose 'resnet18' or 'efficientnet_b0'")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes) # type: ignore
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output
