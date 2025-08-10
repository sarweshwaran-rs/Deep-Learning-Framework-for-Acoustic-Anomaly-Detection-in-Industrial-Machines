import os
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import glob
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


class CQTSpectrogramDataset(Dataset):
    def __init__(self, data_dir: str, label_type: str, transform=None, spec_type: str = 'cqt', file_paths: Optional[List[str]]= None, labels: Optional[List[int]]=None):
        """
            label_type: 'normal' or 'abnormal'
            file_paths and labels: Optional - if provided they override auto-loaded
        """
        self.data_dir = data_dir
        self.label_type = label_type
        self.transform = transform
        self.spec_type = spec_type

        if file_paths is not None and labels is not None:
            self.all_file_paths = file_paths
            self.labels = labels
        else:
            self.all_file_paths, self.labels = self._load_data()

    def _load_data(self):
        pattern = os.path.join(self.data_dir, '**', self.label_type, self.spec_type, '*.npy')
        files = sorted(glob.glob(pattern, recursive=True))
        label = 0 if self.label_type == 'normal' else 1
        return files, [label] * len(files)
    
    def __len__(self):
        return len(self.all_file_paths)
    
    def __getitem__(self, idx):
        spec_path = self.all_file_paths[idx]
        spec = np.load(spec_path)
        spec = torch.from_numpy(spec).float() 

        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
 
        if self.transform:
            spec = self.transform(spec)
        
        label = self.labels[idx]
        return {'spectrogram': spec, 'label': label}


class CQTSpectrogramDataset_1(Dataset):
    def __init__(self, data_dir: str, label_type: str, transform=None, spec_type: str = 'cqt', file_paths: Optional[List[str]]= None, labels: Optional[List[int]]=None, normal_transform=None, abnormal_transform=None):
        """
            label_type: 'normal' or 'abnormal'
            file_paths and labels: Optional - if provided they override auto-loaded
        """
        self.data_dir = data_dir
        self.label_type = label_type
        self.transform = transform
        self.spec_type = spec_type
        self.normal_transform = normal_transform
        self.abnormal_transform = abnormal_transform

        if file_paths is not None and labels is not None:
            self.all_file_paths = file_paths
            self.labels = labels
        else:
            self.all_file_paths, self.labels = self._load_data()

    def _load_data(self):
        pattern = os.path.join(self.data_dir, '**', self.label_type, self.spec_type, '*.npy')
        files = sorted(glob.glob(pattern, recursive=True))
        label = 0 if self.label_type == 'normal' else 1
        return files, [label] * len(files)
    
    def __len__(self):
        return len(self.all_file_paths)
    
    def __getitem__(self, idx):
        spec_path = self.all_file_paths[idx]
        spec = np.load(spec_path)
        spec = torch.from_numpy(spec).float() 
        label = self.labels[idx]
        # if spec.ndim == 2:
        #     spec = spec.unsqueeze(0)
        if self.normal_transform and label == 0:
            spec = self.normal_transform(spec)
        elif self.abnormal_transform and label == 1:
            spec = self.abnormal_transform(spec)
        elif self.transform:
            spec = self.transform(spec)
        
        return {'spectrogram': spec, 'label': label}

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
    

class ResizeSpectrogram:
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
        
class FocalLoss(nn.Module): # type: ignore

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
            alpha: Tensor of Shape(C, ) for class weights or scalar for uniform weight
            gamma: Focusing Parameter 
            label_smoothing: Applies smoothing to hard labels
        """
        super(FocalLoss, self).__init__() # type: ignore
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        print("BinaryFocalLoss initialized with pos_weight =", pos_weight)

        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float().view(-1,1)  # Ensure shape [B, 1]
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
# Dataset to support the Dual-input Version

class PairedSpectrogramDataset(Dataset):
    def __init__(self, base_dir, transform=None): #Added the transfomration option
        self.transform = transform
        self.stft_paths, self.cqt_paths = [], []
        self.labels = []
        self.categories, self.machine_ids = [], []

        for machine in os.listdir(base_dir):
            machine_path = os.path.join(base_dir, machine)
            if not os.path.isdir(machine_path):
                continue
            machine_id = int(machine.split('_')[-1]) #id_00 -> 0
            for category in ['normal', 'abnormal']:
                stft_dir = os.path.join(machine_path, category, 'stft')
                cqt_dir = os.path.join(machine_path, category, 'cqt')

                if not (os.path.isdir(stft_dir) and os.path.isdir(cqt_dir)):
                    continue

                for filename in os.listdir(stft_dir):
                    if filename.endswith('.npy'):
                        stft_path = os.path.join(stft_dir, filename)
                        cqt_path = os.path.join(cqt_dir, filename)

                        self.stft_paths.append(stft_path)
                        self.cqt_paths.append(cqt_path)
                        self.labels.append(0 if category == 'normal' else 1)
                        self.machine_ids.append(machine_id)
                        self.categories.append(category)
    
    def __len__(self):
        return len(self.stft_paths)
    
    def __getitem__(self, idx):
        stft = torch.tensor(np.load(self.stft_paths[idx]), dtype=torch.float32).unsqueeze(0)
        cqt = torch.tensor(np.load(self.cqt_paths[idx]), dtype=torch.float32).unsqueeze(0)
        
        if self.transform is not None:
            stft = self.transform(stft)
            cqt = self.transform(cqt)
                         
        return {
            'stft': stft, 
            'cqt': cqt, 
            'label' :self.labels[idx],
            'machine_id': self.machine_ids[idx],
            'category': self.categories[idx],
            'stft_path': self.stft_paths[idx],
            'cqt_path' : self.cqt_paths[idx]
        }
