import torch.nn as nn
from timm import create_model

class CQTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model('mobilevit_xxs', pretrained=False,num_classes=0, global_pool='')

    def forward(self,x):
        return self.model(x)
