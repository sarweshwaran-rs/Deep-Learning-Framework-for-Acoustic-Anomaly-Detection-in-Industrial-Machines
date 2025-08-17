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


class FocalLoss(nn.Module): # type: ignore

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
            alpha: Tensor of Shape(C, ) for class weights or scalar for uniform weight
            gamma: Focusing Parameter 
            label_smoothing: Applies smoothing to hard labels
        """
        super(FocalLoss, self).__init__() # type: ignore
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean', clamp_eps=1e-8):
        """
        Binary Focal Loss for imbalanced classification.

        Args:
            alpha (float): Weighting factor for positive class.
            gamma (float): Focusing parameter.
            pos_weight (Tensor): Weight for positive class in BCE.
            reduction (str): 'mean', 'sum', or 'none'.
            clamp_eps (float): Small value to clamp pt for numerical stability.
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.clamp_eps = clamp_eps

    def forward(self, inputs, targets):
        # Ensure shape [B] and float type
        inputs = inputs.view(-1)
        targets = targets.float().view(-1)

        # Move pos_weight to device if needed
        if self.pos_weight is not None and self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)

        # Compute base BCE loss
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Compute pt and clamp for stability
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs).clamp(min=self.clamp_eps, max=1.0)

        # Apply focal weighting
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss