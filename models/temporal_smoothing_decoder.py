import torch
import torch.nn as nn

class TemporalSmoothingDecoder(nn.Module):
    """
    RNN/TCN for smoothing embeddings over a sliding window.
    Input: Fused embeddings (B,T,D) or (B,D) if single time-step
    Output: Smoothed anomaly scores (B,)
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, smoothing_window=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_dim, 1)
        self.smoothing_window = smoothing_window

    def forward(self, x):
        # x: (B,T,D)
        if x.dim() == 2:
            # If input is (B, X), add time dimension
            x = x.unsqueeze(1) 

        B, T, D = x.shape
        # if T < smoothing_window, pad or repeat
        if T < self.smoothing_window:
            pad = self.smoothing_window - T
            x = torch.cat([x, x[:,:1,:].repeat(1, pad, 1)], dim=1)
        
        out, _ = self.gru(x)
        
        scores = self.out(out)
        smoothed_scores = scores.mean(dim=1)
        return smoothed_scores.squeeze(-1) # (B,T)
        