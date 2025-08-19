import torch
import torch.nn as nn

from models.feature_extractor import STFTFeatureExtractor, CQTFeatureExtractor, FeatureProjector, STFTFrequencyAdaptiveFeatureExtractor, SpectralPositionalEncoding

from models.temporal_smoothing_decoder import TemporalSmoothingDecoder
from models.cafm import CAFM

class FusedModel(nn.Module):
    """
    Dual-branch model with spectral positional encoding and optional temporal smoothing decoder.

    Args:
        stft_dim (int): Output channels for STFT branch projector.
        cqt_dim (int): Output channels for CQT branch projector.
        fusion_dim (int): Output channels for fusion block.
        head (nn.Module, optional): Classification or embedding head module.
        head_mode (str): Head output mode. Default 'classifier-1'.
        use_decoder (bool): If True, expects sequence input and applies temporal smoothing.
        temporal_hidden (int): Hidden dim for GRU decoder.
    """
    def __init__(self,stft_dim=512,cqt_dim=320,fusion_dim=256,head=None,head_mode='classifier-1',use_decoder=False,temporal_hidden=64):
        super().__init__()
        self.head_mode = head_mode
        self.head = head
        self.use_decoder = use_decoder

        # Feature extractors
        # self.stft_net = STFTFeatureExtractor()
        self.stft_net = STFTFrequencyAdaptiveFeatureExtractor()
        self.cqt_net = CQTFeatureExtractor()
        self.stft_proj = FeatureProjector(stft_dim)
        self.cqt_proj = FeatureProjector(cqt_dim)
        self.fuser = CAFM(fusion_dim)

        # SPE modules for both branches
        self.stft_spe = None
        self.cqt_spe = None

        # Temporal smoothing decoder
        self.temporal_decoder = TemporalSmoothingDecoder(input_dim=fusion_dim, hidden_dim=temporal_hidden)

    def forward(self, stft, cqt):
        """
        Args:
            stft: [B, T, C, H, W] or [B, C, H, W]
            cqt:  [B, T, C, H, W] or [B, C, H, W]
        Returns:
            If use_decoder: [B, T] (sequence of scores)
            Else: [B, ...] (batch scores/embeddings)
        """
        if self.use_decoder:
            # Sequence mode: stft/cqt [B, T, C, H, W]
            B, T, C, H, W = stft.shape
            fused_seq = []
            # SPE init on first time step
            stft_raw = self.stft_net(stft[:, 0])
            cqt_raw = self.cqt_net(cqt[:, 0])
            stft_feat_init = self.stft_proj(stft_raw)
            cqt_feat_init = self.cqt_proj(cqt_raw)
            if self.stft_spe is None or self.cqt_spe is None:
                _, C2, F, TT = stft_feat_init.shape
                self.stft_spe = SpectralPositionalEncoding(num_freqs=F, dim=C2).to(stft_feat_init.device)
                self.cqt_spe = SpectralPositionalEncoding(num_freqs=cqt_feat_init.shape[2], dim=cqt_feat_init.shape[1]).to(cqt_feat_init.device)

            for t in range(T):
                stft_raw = self.stft_net(stft[:, t])
                cqt_raw = self.cqt_net(cqt[:, t])
                stft_feat = self.stft_proj(stft_raw)
                cqt_feat = self.cqt_proj(cqt_raw)

                stft_feat = self.stft_spe(stft_feat)
                cqt_feat = self.cqt_spe(cqt_feat)
                fused = self.fuser(stft_feat, cqt_feat)  # [B, fusion_dim]
                fused_seq.append(fused.unsqueeze(1))     # [B, 1, fusion_dim]

            fused_seq = torch.cat(fused_seq, dim=1)      # [B, T, fusion_dim]
            scores = self.temporal_decoder(fused_seq)    # [B, T]
            if self.head is not None:
                scores = self.head(scores)               # pass scores through head, if provided
            return scores
        else:
            # Single frame mode: stft/cqt [B, C, H, W]
            stft_raw = self.stft_net(stft)
            cqt_raw = self.cqt_net(cqt)
            stft_feat = self.stft_proj(stft_raw)
            cqt_feat = self.cqt_proj(cqt_raw)

            if self.stft_spe is None or self.cqt_spe is None:
                B, C, F, T = stft_feat.shape
                self.stft_spe = SpectralPositionalEncoding(num_freqs=F, dim=C).to(stft_feat.device)
                self.cqt_spe = SpectralPositionalEncoding(num_freqs=cqt_feat.shape[2], dim=cqt_feat.shape[1]).to(cqt_feat.device)

            stft_feat = self.stft_spe(stft_feat)
            cqt_feat = self.cqt_spe(cqt_feat)

            if stft_feat.dim() != 4 or cqt_feat.dim() != 4:
                raise RuntimeError(
                    f"Expected 4D tensors before CAFM, got stft:{stft_feat.shape}, cqt:{cqt_feat.shape}. "
                    "Check FeatureProjector to ensure it does not flatten."
                )

            fused = self.fuser(stft_feat, cqt_feat)  # [B, fusion_dim]
            if self.head is None:
                raise ValueError("Head is None. Please provide a valid head module to FusedModel.")
            
            if self.head_mode == "prototype":
                embeddings, prototype = self.head(fused)
                return embeddings, prototype
            else:
                return self.head(fused)
