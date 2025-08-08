import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from utils.datasets import PairedSpectrogramDataset
from models.feature_extractor import STFTFeatureExtractor, CQTFeatureExtractor, FeatureProjector
from models.cafm import CAFM

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE} : {torch.cuda.get_device_name()}")
DATA_DIR = r'data\features'

dataset = PairedSpectrogramDataset(base_dir=DATA_DIR)
print(f"Total samples in dataset: {len(dataset)}")

loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

stft_model = STFTFeatureExtractor().to(DEVICE)
cqt_model = CQTFeatureExtractor().to(DEVICE)

stft_proj = FeatureProjector(in_channels=512).to(DEVICE)
cqt_proj = FeatureProjector(in_channels=320).to(DEVICE)
fusion_model = CAFM(dim=256).to(DEVICE)

all_fused_features, all_metadata = [], []

print("Starting Extraction...")

with torch.no_grad():
    for i, batch in enumerate(loader):
        # print(f"Processing Batch {i+1}/{len(loader)}")

        stft = batch['stft'].to(DEVICE)
        cqt = batch['cqt'].to(DEVICE)
        labels = batch['label']
        machine_ids = batch['machine_id']
        categories = batch['category']

        #Feature extraction
        stft_feat = stft_model(stft)
        cqt_feat = cqt_model(cqt)

        #Projection
        stft_proj_feat = stft_proj(stft_feat)
        cqt_proj_feat = cqt_proj(cqt_feat)

        #Fusion
        fused_feat = fusion_model(stft_proj_feat, cqt_proj_feat)

        # Save features + metadata
        all_fused_features.append(fused_feat.cpu())
        for j in range(fused_feat.size(0)):
            all_metadata.append({
                'label': labels[j].item(),
                'machine_id': machine_ids[j],
                'category': categories[j],
                'feature': fused_feat[j].cpu().numpy()
            })

# ===== Save to .pt ===== #
all_fused_tensor = torch.cat(all_fused_features, dim=0)
torch.save(all_fused_tensor, "fused_features.pt")
print("Saved PyTorch features to fused_features.pt")

# ===== Save to .csv ===== #
csv_rows = []
for entry in all_metadata:
    flat_feat = entry['feature'].flatten()
    row = {
        'label':entry['label'],
        'machine_id':entry['machine_id'],
        'category': entry['category']
    }
    row.update({f'f_{i}': v for i, v in enumerate(flat_feat)})
    csv_rows.append(row)

df = pd.DataFrame(csv_rows)
df.to_csv("fused_features.csv", index=False)
print("Saved CSV to fused_features.csv")
