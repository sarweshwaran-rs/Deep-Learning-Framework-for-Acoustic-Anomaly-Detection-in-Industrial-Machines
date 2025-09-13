import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.manifold import TSNE
import torch

from scripts.pretrain_pipeline import FusedModel, FusedModelNoSPE
from models.heads import STFTResNetClassifier, CQTMobileViTClassifier, AnomalyScorer

# ==============================
# Feature Extraction
# ==============================
def compute_stft_cqt(audio_path, sr=16000, n_fft=512, hop_length=256, n_bins=84):
    """Compute normalized STFT and CQT spectrograms."""
    y, _ = librosa.load(audio_path, sr=sr)

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))[:n_bins, :n_bins]
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins))[:n_bins, :n_bins]

    # Normalize to [0,1]
    stft = (stft - stft.min()) / (stft.max() - stft.min() + 1e-6)
    cqt = (cqt - cqt.min()) / (cqt.max() - cqt.min() + 1e-6)

    return stft, cqt

def load_audio_files(folder, limit=50, sr=16000, duration=10):
    """Load `limit` audio files padded/truncated to fixed duration."""
    files = sorted(os.listdir(folder))[:limit]
    audio_paths = []
    target_len = int(sr * duration)
    for f in files:
        path = os.path.join(folder, f)
        y, _ = librosa.load(path, sr=sr)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        audio_paths.append(path)
    return audio_paths

# ==============================
# t-SNE Plot
# ==============================
def run_tsne(features, labels, method_name, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_emb = tsne.fit_transform(features)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_emb[labels == 0, 0], X_emb[labels == 0, 1], c="blue", label="Normal", alpha=0.7)
    plt.scatter(X_emb[labels == 1, 0], X_emb[labels == 1, 1], c="red", label="Abnormal", alpha=0.7)
    plt.title(f"t-SNE ({method_name})")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==============================
# Embedding Extraction
# ==============================
def extract_embeddings(model, stft_inputs=None, cqt_inputs=None, use_decoder=False, device="cpu"):
    """
    Extract embeddings suitable for t-SNE from a model.
    Automatically handles:
        - CNN/Transformer classifiers
        - CAFM models with/without decoder
        - Sequence outputs (flattens them)
        - Single-value outputs (expands to 2D)
    """
    model.eval()
    with torch.no_grad():
        # -----------------------------
        # Model-specific feature extraction
        # -----------------------------
        if isinstance(model, STFTResNetClassifier):
            feats = model.extract_features(stft_inputs)  # [batch, feature_dim]
        elif isinstance(model, CQTMobileViTClassifier):
            feats = model.extract_features(cqt_inputs)   # [batch, feature_dim]
        else:  # CAFM family
            if use_decoder:
                head_out, seq_scores = model(stft_inputs, cqt_inputs)
                feats = head_out if head_out is not None else seq_scores
            else:
                feats = model(stft_inputs, cqt_inputs)

        # -----------------------------
        # Move to CPU and numpy
        # -----------------------------
        feats = feats.cpu().numpy()

        # -----------------------------
        # Flatten multidimensional outputs
        # -----------------------------
        if feats.ndim > 2:
            # Sequence or spatial output → flatten to [batch, feature_dim]
            feats = feats.reshape(feats.shape[0], -1)
        elif feats.ndim == 1:
            # Single score per sample → make 2D for t-SNE
            feats = feats[:, np.newaxis]
            feats = np.repeat(feats, 2, axis=1)  # duplicate to at least 2D

    return feats

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset paths
    normal_path = r"F:\CapStone\DFCA\data\raw\-6dB_pump\id_00\normal"
    abnormal_path = r"F:\CapStone\DFCA\data\raw\-6dB_pump\id_00\abnormal"

    normal_audio = load_audio_files(normal_path, limit=50, duration=10)
    abnormal_audio = load_audio_files(abnormal_path, limit=50, duration=10)
    all_audio = normal_audio + abnormal_audio
    labels = np.array([0] * 50 + [1] * 50)

    # Feature preparation
    stft_list, cqt_list, stft_seq_list, cqt_seq_list = [], [], [], []
    T = 10
    for audio_path in all_audio:
        stft, cqt = compute_stft_cqt(audio_path)

        stft_list.append(stft[np.newaxis, :, :])
        cqt_list.append(cqt[np.newaxis, :, :])

        # Sequential version
        stft_slices = np.array_split(stft, T, axis=1)
        cqt_slices = np.array_split(cqt, T, axis=1)
        stft_seq = [np.pad(s[np.newaxis, :, :], ((0, 0), (0, 0), (0, 128 - s.shape[-1])), mode="constant") for s in stft_slices]
        cqt_seq = [np.pad(s[np.newaxis, :, :], ((0, 0), (0, 0), (0, 128 - s.shape[-1])), mode="constant") for s in cqt_slices]
        stft_seq_list.append(np.stack(stft_seq, axis=0))
        cqt_seq_list.append(np.stack(cqt_seq, axis=0))

    stft_tensor = torch.tensor(np.stack(stft_list), dtype=torch.float32).to(device)
    cqt_tensor = torch.tensor(np.stack(cqt_list), dtype=torch.float32).to(device)
    stft_seq_tensor = torch.tensor(np.stack(stft_seq_list), dtype=torch.float32).to(device)
    cqt_seq_tensor = torch.tensor(np.stack(cqt_seq_list), dtype=torch.float32).to(device)

    print("STFT shape:", stft_tensor.shape)
    print("CQT shape:", cqt_tensor.shape)
    print("STFT seq shape:", stft_seq_tensor.shape)
    print("CQT seq shape:", cqt_seq_tensor.shape)

    # ------------------------
    # STFT Model
    # ------------------------
    stft_model = STFTResNetClassifier(head=AnomalyScorer(in_dim=512, mode="classifier-1")).to(device)
    stft_model.load_state_dict(torch.load(r"F:\CapStone\DFCA\checkpoints\Models\STFT_ResNet\best_model.pth", map_location=device), strict=False)
    feats_stft = extract_embeddings(stft_model, stft_tensor, device=device)
    run_tsne(feats_stft, labels, "STFT", "tsne_stft.png")

    # ------------------------
    # CQT Model
    # ------------------------
    cqt_model = CQTMobileViTClassifier(head=AnomalyScorer(in_dim=320, mode="classifier-1")).to(device)
    cqt_model.load_state_dict(torch.load(r"F:\CapStone\DFCA\checkpoints\Models\CQT_MobileViT\best_model.pth", map_location=device), strict=False)
    feats_cqt = extract_embeddings(cqt_model, None, cqt_tensor, device=device)
    run_tsne(feats_cqt, labels, "CQT", "tsne_cqt.png")

    # ------------------------
    # CAFM (no SPE, no decoder)
    # ------------------------
    cafm_model = FusedModelNoSPE().to(device)
    cafm_model.load_state_dict(torch.load(r"F:\CapStone\DFCA\checkpoints\Models\CAFM\best_model.pth", map_location=device), strict=False)
    feats_cafm = extract_embeddings(cafm_model, stft_tensor, cqt_tensor, use_decoder=False, device=device)
    run_tsne(feats_cafm, labels, "CAFM+SPE+TSD", "tsne_cafm_spe_tsd.png")

    # ------------------------
    # CAFM + SPE
    # ------------------------
    cafm_spe_model = FusedModel(use_decoder=False).to(device)
    cafm_spe_model.load_state_dict(torch.load(r"F:\CapStone\DFCA\checkpoints\Models\CAFM_SPE\best_model.pth", map_location=device), strict=False)
    feats_cafm_spe = extract_embeddings(cafm_spe_model, stft_tensor, cqt_tensor, use_decoder=False, device=device)
    run_tsne(feats_cafm_spe, labels, "CAFM+SPE", "tsne_cafm_spe.png")

    # ------------------------
    # CAFM + SPE + TSD
    # ------------------------
    cafm_spe_ts_model = FusedModel(use_decoder=True).to(device)
    cafm_spe_ts_model.load_state_dict(torch.load(r"F:\CapStone\DFCA\checkpoints\DFCAFinalNet\best_model.pth", map_location=device), strict=False)
    feats_cafm_spe_ts = extract_embeddings(cafm_spe_ts_model, stft_seq_tensor, cqt_seq_tensor, use_decoder=True, device=device)
    run_tsne(feats_cafm_spe_ts, labels, "CAFM", "tsne_cafm.png")

    print("✅ t-SNE plots saved for STFT, CQT, CAFM, CAFM+SPE, CAFM+SPE+TSD")
