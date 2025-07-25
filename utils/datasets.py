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

        self.data_dir = data_dir
        self.category = category 
        self.transform = transform
        self.spec_type = spec_type
        self.all_file_paths = []
        self.labels = [] # 0 for normal, 1 for abnormal
        
        #=== Iterating over the folders ===
        for id_folder in os.listdir(self.data_dir):
            id_folder_path = os.path.join(self.data_dir, id_folder)

            if os.path.isdir(id_folder_path) and id_folder.startswith('id_'):
                category_path = os.path.join(id_folder_path, self.category, self.spec_type)

                if os.path.exists(category_path) and os.path.isdir(category_path):
                    for filename in os.listdir(category_path):
                        if filename.endswith('.npy'):
                            self.all_file_paths.append(os.path.join(category_path, filename))
                            self.labels.append(0 if self.category == 'normal' else 1)
                else:
                    print(f"Warning: Category path not found for {os.path.join(id_folder, self.category, self.spec_type)}")
        if not self.all_file_paths:
            raise FileNotFoundError(f"No {self.spec_type} files found for category '{self.category}' under {self.data_dir}.  check your paths and a folder structure")
        
        
    def __len__(self):
        return len(self.all_file_paths)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        spec_path = self.all_file_paths[idx] # type: ignore
        spectrogram = np.load(spec_path).astype(np.float32)

        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)
        
        if self.transform:
            spectrogram = self.transform(torch.from_numpy(spectrogram))
        
        label = self.labels[idx] # type: ignore

        return {'spectrogram': spectrogram, 'label': label, 'path': spec_path}
        
        
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
        