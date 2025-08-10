import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

from utils.datasets import PairedSpectrogramDataset
from models.feature_extractor import STFTFeatureExtractor, CQTFeatureExtractor, FeatureProjector, STFTFrequencyAdaptiveFeatureExtractor
from models.cafm import CAFM

class FusedModel(nn.Module):
    def __init__(self, stft_dim=512, cqt_dim=320, fusion_dim=256, head=None, head_mode='classifier'):
        super().__init__()
        self.head_mode = head_mode
        self.head = head

        self.stft_net = STFTFeatureExtractor()
        # self.stft_net = STFTFrequencyAdaptiveFeatureExtractor()
        self.cqt_net = CQTFeatureExtractor()
        self.stft_proj = FeatureProjector(stft_dim)  # should output [B, fusion_dim, H', W']
        self.cqt_proj = FeatureProjector(cqt_dim)
        self.fuser = CAFM(fusion_dim)

    def forward(self, stft, cqt):
        # Extract convolutional features
        stft_raw = self.stft_net(stft)   # [B, C, H, W]
        cqt_raw  = self.cqt_net(cqt)     # [B, C, H, W]

        # Project features while keeping spatial dimensions
        stft_feat = self.stft_proj(stft_raw)  # expected [B, fusion_dim, H', W']
        cqt_feat  = self.cqt_proj(cqt_raw)    # expected [B, fusion_dim, H', W']

        # Final check before CAFM
        if stft_feat.dim() != 4 or cqt_feat.dim() != 4:
            raise RuntimeError(
                f"Expected 4D tensors before CAFM, got stft:{stft_feat.shape}, cqt:{cqt_feat.shape}. "
                "Check FeatureProjector to ensure it does not flatten."
            )

        # Fuse features
        fused = self.fuser(stft_feat, cqt_feat)  # returns [B, fusion_dim]

        # Head
        if self.head is None:
            raise ValueError("Head is None. Please provide a valid head module to FusedModel.")
        
        if self.head_mode == "prototype":
            embeddings, prototype = self.head(fused)
            return embeddings, prototype
        else:
            return self.head(fused)
    
