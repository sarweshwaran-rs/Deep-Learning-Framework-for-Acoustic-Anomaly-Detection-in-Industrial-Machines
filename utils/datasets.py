import os
import torch
from torch.utils.data import Dataset
import numpy as np
#Added the torchAudio transforms
import torchaudio.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image

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
        self.category = category.lower() 
        self.transform = transform
        self.spec_type = spec_type.lower()
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
            raise FileNotFoundError(f"[ERROR] No {self.spec_type} files found for category '{self.category}' under {self.data_dir}.\n" 
                                    f"Expected path: <data_dir>/id_xx/{self.category}/{self.spec_type}/"
            )
        
        
    def __len__(self):
        return len(self.all_file_paths)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        spec_path = self.all_file_paths[idx] # type: ignore
        spectrogram = np.load(spec_path).astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)

        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        label = self.labels[idx] # type: ignore

        return {
            'spectrogram': spectrogram, 
            'label': label, 
            'path': spec_path
        }    
        
class NormalizeSpectrogram:
    """
        Min-Max normalization to [0,1]
    """

    def __call__(self, spectrogram):
        min_val = spectrogram.min()
        max_val = spectrogram.max()

        if max_val > min_val:
            spectrogram = (spectrogram - min_val) / (max_val - min_val)
        else:
            spectrogram = torch.zeros_like(spectrogram)
        return spectrogram
    

class ZScoreNormalizeSpectrogram:
    """
    Z-Score normalization: zero mean and unit variance
    """
    def __call__(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = torch.zeros_like(spectrogram)
        
        return spectrogram


class AugmentSpectrogram:
    def __init__(self, time_mask=True, freq_mask=True, time_mask_param=18,freq_mask_param=12):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.time_masker = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, spectrogram):
        if self.freq_mask:
            spectrogram = self.freq_masker(spectrogram)
        if self.time_mask:
            spectrogram = self.time_masker(spectrogram)
        return spectrogram
    

class ResizeSpectroram:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, spec_tensor):
        if isinstance(spec_tensor, torch.Tensor):
            if spec_tensor.dim() == 2:
                spec_tensor = spec_tensor.unsqueeze(0)
            return TF.resize(spec_tensor, self.size)
        elif isinstance(spec_tensor, np.ndarray):
            img = Image.fromarray((spec_tensor * 255).astype(np.uint8))
            img = img.resize(self.size, Image.BILINEAR) # type: ignore
            img = np.asarray(img).astype(np.float32) / 255.0
            return torch.from_numpy(img).unsqueeze(0)
        else:
            raise TypeError("Unsupported type for ResizeSpectrogram")