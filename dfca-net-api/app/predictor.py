import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(current_file_path)))
sys.path.append(project_root)

import torch
import numpy as np

from scripts.pretrain_pipeline import FusedModel
from models.heads import AnomalyScorer

class ModelPredictor:
    def __init__(self, model_path:str, Threshold:float=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} + {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

        head = AnomalyScorer(in_dim=256, dropout=0.4, mode='classifier-1')
        self.model = FusedModel(
            stft_dim=512,
            cqt_dim=320,
            fusion_dim=256,
            head=head,
            use_decoder=False
        )
        self.Threshold = Threshold
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

        self.model.to(self.device)
        print(f"Model Loded from {model_path}")
        print(f"Using the Threshold: {Threshold}")
        self.model.eval()

    def predict(self, stft_spec: np.ndarray, cqt_spec: np.ndarray) -> tuple[str, float]:
        """
            Performs inference on the preprocessed spectrograms.

            Args:
                stft_spec (np.ndarray): The STFT log-mel spectrogram.
                cqt_spec (np.ndarray): The CQT spectrogram.

            Returns:
                A tuple containing the predicted label ('Normal' or 'Abnormal') and the confidence score.
        """
        # Convert numpy arrays to torch tensors and add batch & channel dimensions
        # Shape must be [B, C, H, W] -> [1, 1, H, W]
        stft_tensor = torch.from_numpy(stft_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        cqt_tensor = torch.from_numpy(cqt_spec).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get model output (logits)
            logits = self.model(stft_tensor, cqt_tensor)
            
            # Apply sigmoid to convert logits to probabilities
            probability = torch.sigmoid(logits).squeeze().item()

        
        if probability >= self.Threshold:
            label = "Abnormal"
            confidence = probability
        else:
            label = "Normal"
            confidence = 1 - probability
            
        return label, confidence
    
# --- Instantiate the predictor once when the application starts ---
model_weight_path = r'F:\CapStone\DFCA\checkpoints\DFCAFinalNet\best_model.pth'
predictor = ModelPredictor(model_path=model_weight_path, Threshold=0.65)