import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyScorer(nn.Module):
    def __init__(self, in_dim=256, dropout = 0.5, mode = 'classifier'):
        super().__init__()

        self.mode = mode
        self.dropout = nn.Dropout(p=dropout)

        if mode == 'classifier':
            self.head = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                self.dropout,
                nn.Linear(128,1), # Binary Classification
            )
        elif mode == 'prototype':
            self.prototype = nn.Parameter(torch.randn(in_dim)) # Learnable normal prototype
        
    def forward(self, x):
        if self.mode == 'classifier':
            return self.head(x) # logits for BCEWithLogitsLoss
        
        elif self.mode == 'prototype':
            x = self.dropout(x)
            return x,self.prototype # Returns the raw embeddgings and prototype for external scoring

# ---------------------------------------------------------
# Simple MLP Anomaly Head (Binary Classifier)
# ---------------------------------------------------------
class SimpleAnomalyMLP(nn.Module):
    def __init__(self, in_dim=256, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x) # logits for BCEWthLogitsLoss

# Embedding Head
class EmbeddingMLP(nn.Module):
    def __init__(self, in_dim=256, hidden=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim)
        )
        self.normal_prototype = nn.Parameter(torch.randn(emb_dim))
    def forward(self, x):
        return self.net(x) # Returns embedding tensor (B, emd_dim)
