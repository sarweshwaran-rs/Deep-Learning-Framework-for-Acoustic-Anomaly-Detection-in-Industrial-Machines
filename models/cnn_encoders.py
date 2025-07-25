import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicSpectrogramClassifier(nn.Module):

    def __init__(self, encoder_name = 'resnet18', pretrained=False, num_classes=2):
        super(BasicSpectrogramClassifier, self).__init__()

        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(pretrained = pretrained)
            self.encoder.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
            num_ftrs = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity() # type: ignore
        
        elif encoder_name == 'efficient_b0':
            self.encoder = models.efficientnet_b0(pretrained=pretrained)
            self.encoder.features[0][0] = nn.Conv2d(1,32,kernel_size=3,stride=2, padding=1, bias=False)
            num_ftrs = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity() # type: ignore

        else:
            raise ValueError(f"Encoder {encoder_name} not supported. Choose 'resnet18' or 'efficientnet_b0'")
        
        self.classifier = nn.Linear(num_ftrs, num_classes) # type: ignore

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output
