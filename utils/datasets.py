import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, category='normal', transform = None, spec_type='stft'):
        """
        Args:
            data_dir(str): Base directory where 'features' folder is located
            category(str): 'normal' or 'abnormal'
            transform (callable, optional): Optional transform to be applied on a sample.
            spec_type(str): 'stft' for Log-Mel Spectrograms or 'cqt' for CQT Spectograms
        """

        self.data_dir = os.path.join(data_dir, category, spec_type)
        self.transform = transform
        self.spec_type = spec_type
        
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        if not self.file_list:
            raise ValueError(f"No .npy files found in {self.data_dir}. Please ensure preprocessing ran correctly and files were saved.")
        
        #Assign labels: 0 for normal, 1 for Abnormal
        self.label = 0 if category == 'normal' else 1

    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spec_name = self.file_list[idx] # type: ignore
        spec_path = os.path.join(self.data_dir, spec_name)

        #Load the Spectrogram (numpy array)
        spectrogram = np.load(spec_path)

        #Ensure 3D shape: (channels, height, width) 
        #Spectrograms are typically (frequency_bins, time_frames)
        spectrogram = np.expand_dims(spectrogram, axis=0)

        #Convert to PyTorch tensor (float32 is common)
        spectrogram = torch.from_numpy(spectrogram).float()

        #Apply transform if any
        if self.transform:
            spectrogram = self.transform(spectrogram)

        sample = {'spectrogram': spectrogram, 'label': self.label}
        return sample
        
class NormalizeSpectrogram:
    def __call__(self, spectrogram):
        #Min-Max normalization
        min_val = spectrogram.min()
        max_val = spectrogram.max()

        if max_val > min_val:
            spectrogram = (spectrogram - min_val) / (max_val - min_val)
        else:
            spectrogram = torch.zeros_like(spectrogram)
        return spectrogram
    

class ZScoreNormalizeSpectrogram:
    def __call__(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = torch.zeros_like(spectrogram)
        
        return spectrogram
        