import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, prototype, labels):
        """
        Embeddings: (B, D)
        prototype: (D, ) or (1, D)
        labels: tensor(B, ) with 0=normal, 1=anomaly
        """
        if prototype.dim() == 1:
            prototype = prototype.unsqueeze(0)
        prototype = prototype.expand(embeddings.size(0), prototype.size(1)) #(B,D)
        distances = torch.norm(embeddings - prototype, p=2, dim=1) #(B,)
        labels = labels.float()
        loss = (1.0 - labels) * distances.pow(2) + labels * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return loss.mean()
