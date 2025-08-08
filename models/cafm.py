import torch.nn as nn
import torch

class CAFM(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.wq1= nn.Linear(dim, dim)
        self.wq2 = nn.Linear(dim, dim)
        self.wq3 = nn.Linear(dim, dim)

        self.wq4 = nn.Linear(dim, dim)
        self.wq5 = nn.Linear(dim, dim)
        self.wq6 = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, stft_feat, cqt_feat):
        B, C, H, W = stft_feat.shape
        assert stft_feat.shape == cqt_feat.shape, "Shape mismatch between STFT and CQT features"

        stft_seq = stft_feat.flatten(2).transpose(1, 2) # [B, N, C]
        cqt_seq = cqt_feat.flatten(2).transpose(1, 2) # [B, N, C]

        # STFT attends to CQT
        Q1 = self.wq1(stft_seq)
        K1 = self.wq2(cqt_seq)
        V1 = self.wq3(cqt_seq)
        dk = Q1.size(-1)
        attention_1 = self.softmax(torch.bmm(Q1, K1.transpose(1, 2)) / (dk ** 0.5))
        out_1 = torch.bmm(attention_1, V1)

        # CQT attends to STFT
        Q2 = self.wq4(cqt_seq)
        K2 = self.wq5(stft_seq)
        V2 = self.wq6(stft_seq)
        attention_2 = self.softmax(torch.bmm(Q2, K2.transpose(1, 2)) / (dk ** 0.5))
        out_2 = torch.bmm(attention_2, V2)

        # Mean pool and Fuse
        fused = torch.cat([out_1.mean(1), out_2.mean(1)], dim=1)
        # print(f"Fused shape before MLP[Multi Layer Perceptron]: {fused.shape}")
        output = self.out(fused)
        # print(f"Output shape after MLP [Multi Layer Perceptron]: {output.shape}")
        return output
        