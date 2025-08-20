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
            # Sequence mode: expect [B, T, C, H, W]
            if stft.dim() != 5 or cqt.dim() != 5:
                raise ValueError("use_decoder=True expects [B, T, C, H, W] inputs")

            B, T, C, H, W = stft.shape
            fused_seq = []

            # Initialize SPE on first frame
            stft_feat_init = self.stft_proj(self.stft_net(stft[:, 0]))
            cqt_feat_init  = self.cqt_proj(self.cqt_net(cqt[:, 0]))

            _, C2, F, TT = stft_feat_init.shape
            if self.stft_spe is None:
                self.stft_spe = SpectralPositionalEncoding(num_freqs=F, dim=C2).to(stft_feat_init.device)
            if self.cqt_spe is None:
                self.cqt_spe = SpectralPositionalEncoding(num_freqs=cqt_feat_init.shape[2],
                                                          dim=cqt_feat_init.shape[1]).to(cqt_feat_init.device)

            # Process each frame
            for t in range(T):
                stft_feat = self.stft_spe(self.stft_proj(self.stft_net(stft[:, t])))
                cqt_feat  = self.cqt_spe(self.cqt_proj(self.cqt_net(cqt[:, t])))
                fused = self.fuser(stft_feat, cqt_feat)      # [B, fusion_dim]
                fused_seq.append(fused.unsqueeze(1))         # [B, 1, fusion_dim]

            fused_seq = torch.cat(fused_seq, dim=1)         # [B, T, fusion_dim]

            # Always compute temporal decoder output
            seq_scores = self.temporal_decoder(fused_seq)   # [B, T]

            # Compute head output (pooled across time)
            pooled = fused_seq.mean(dim=1)                 # [B, fusion_dim]
            head_out = self.head(pooled) if self.head is not None else None

            # Return both head output and sequence scores
            return head_out, seq_scores
        
        else:
            # Single-frame mode (unchanged)
            if stft.dim() != 4 or cqt.dim() != 4:
                raise ValueError("use_decoder=False expects [B, C, H, W] inputs")

            stft_feat = self.stft_proj(self.stft_net(stft))
            cqt_feat  = self.cqt_proj(self.cqt_net(cqt))

            if self.stft_spe is None or self.cqt_spe is None:
                B, C, F, T = stft_feat.shape
                self.stft_spe = SpectralPositionalEncoding(num_freqs=F, dim=C).to(stft_feat.device)
                self.cqt_spe  = SpectralPositionalEncoding(num_freqs=cqt_feat.shape[2], dim=cqt_feat.shape[1]).to(cqt_feat.device)

            stft_feat = self.stft_spe(stft_feat)
            cqt_feat  = self.cqt_spe(cqt_feat)

            if stft_feat.dim() != 4 or cqt_feat.dim() != 4:
                raise RuntimeError(
                    f"Expected 4D tensors before CAFM, got stft:{stft_feat.shape}, cqt:{cqt_feat.shape}. "
                    "Check FeatureProjector to ensure it does not flatten."
                )

            fused = self.fuser(stft_feat, cqt_feat) # [B, fusion_dim]
            if self.head is None:
                return fused
            return self.head(fused)