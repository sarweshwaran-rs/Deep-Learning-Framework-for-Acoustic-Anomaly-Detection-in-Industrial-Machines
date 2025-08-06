import torch
import torch.nn as nn
import timm

class TransformerSpectrogramClassifier(nn.Module):
    def __init__(self, model_name = 'swin_tiny_patch4_window7_224', pretrained=True, num_classes=2, dropout_prob=0.3):
        super(TransformerSpectrogramClassifier, self).__init__()

        self.model_name = model_name
        
        if model_name == 'swin_tiny':
            self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, in_chans=1)
            
        elif model_name == 'mobilevit_s':
            self.encoder = timm.create_model('mobilevit_s', pretrained=pretrained, in_chans =1)
            
        elif model_name == 'mobilevitv2_050':
            self.encoder = timm.create_model('mobilevitv2_050', pretrained=pretrained, in_chans=1)

        elif model_name == 'mobilevit_xs':
            self.encoder = timm.create_model('mobilevit_xs', pretrained=pretrained, in_chans=1)

        elif model_name == 'mobilevit_xxs':
            self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, in_chans =1)
        else:
            raise ValueError("Unsupported transformer model. Choose 'swin_tiny' or 'mobilevit_s'")
        
        self.encoder.head = nn.Identity()  # remove default head

        # ----- Dynamically infer the feature size using a dummy pass -----
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)
            dummy_output = self.encoder(dummy_input)

            if dummy_output.dim() == 4:
                dummy_output = torch.nn.functional.adaptive_avg_pool2d(dummy_output, (1, 1))
                dummy_output = dummy_output.view(dummy_output.size(0), -1)
            elif dummy_output.dim() == 3:
                dummy_output = torch.mean(dummy_output, dim=1)
            elif dummy_output.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected encoder output shape: {dummy_output.shape}")
            
        self.feature_dim = dummy_output.shape[1] # type: ignore

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
