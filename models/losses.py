import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        """
        margin: separation in cosine space (0 to 2).
        e.g., marging = 0.5 means anomalies should have similarity <= 1 - margin = 0.5
        """
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, prototype, labels):
        """
        Embeddings: (B, D)
        prototype: (D, ) or (1, D)
        labels: tensor(B, ) with 0=normal, 1=anomaly
        """
        #Normalize for cosine space
        embeddings = F.normalize(embeddings, dim=1)
        prototype = F.normalize(prototype,dim=0)

        if prototype.dim() == 1:
            prototype = prototype.unsqueeze(0)
        prototype = prototype.expand(embeddings.size(0), -1) #(B,D)
        
        cosine_sim = torch.sum(embeddings * prototype, dim=1)

        pos_mask = (labels == 0).float()
        neg_mask = (labels == 1).float()

        # Loss Terms
        pos_loss = ((1 - cosine_sim) * pos_mask).sum() / pos_mask.sum().clamp(min=1) # pull normals toward prototype
        neg_loss = (torch.clamp(cosine_sim - (1 - self.margin), min=0) * neg_mask).sum() / neg_mask.sum().clamp(min=1) # Push anomalies away
        return (pos_loss + neg_loss)
