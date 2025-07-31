import torch
import torch.nn as nn
import timm

class TransformerSpectrogramClassifier(nn.Module):
    def __init__(self, model_name = 'swin_tiny_patch4_window7_224', pretrained=True, num_classes=2, dropout_prob=0.3):
        super(TransformerSpectrogramClassifier, self).__init__()

        if model_name == 'swin_tiny':
            self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, in_chans=1)
            
        elif model_name == 'mobilevit_s':
            self.encoder = timm.create_model('mobilevit_s', pretrained=pretrained, in_chans =1)
            
        else:
            raise ValueError("Unsupported transformer model. Choose 'swin_tiny' or 'mobilevit_s'")
        
        self.encoder.head = nn.Identity()  # remove default head
        self.feature_dim = self.encoder.num_features

        self.norm = nn.BatchNorm1d(self.feature_dim) # type: ignore
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.feature_dim, num_classes) # type: ignore

    def forward(self,x):
        features = self.encoder(x)
        # print("Encoder output shape: ", features.shape)

        if features.dim() == 4 and features.shape[-1] == self.feature_dim:
            features = torch.mean(features, dim=[1, 2])
        elif features.dim() == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1,1))
            features = features.view(features.size(0), -1)
        elif features.dim() == 3:
            features = torch.mean(features, dim=1)
        elif features.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected encoder output shape: {features.shape}")
        
        features = self.norm(features)
        features = self.dropout(features)
        
        logits = self.classifier(features)
        return logits
