import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
class WindowedPairedSpectrogramDataset(Dataset):
    def __init__(self, base_dataset, window_size=5):
        """
        Args:
            base_dataset: PairedSpectrogramDataset
            window_size: number of consecutive frames per sample
        """
        self.base = base_dataset
        self.window_size = window_size
       
    def __len__(self):
        return len(self.base) - self.window_size + 1

    def __getitem__(self, idx):
        stft_seq, cqt_seq, labels_seq = [], [], []

        for i in range(self.window_size):
            item = self.base[idx + i]
            stft_seq.append(item['stft'].unsqueeze(0))  # (1, C, H, W)
            cqt_seq.append(item['cqt'].unsqueeze(0))
            labels_seq.append(item['label'])
        
        stft_seq = torch.cat(stft_seq, dim=0)
        cqt_seq = torch.cat(cqt_seq, dim=0)
        labels = torch.tensor(1 if sum(labels_seq) > len(labels_seq) // 2 else 0, dtype=torch.long)
        
        return {
            'stft': stft_seq,   # (T, C, H, W)
            'cqt': cqt_seq,     # (T, C, H, W)
            'label': labels
        }
    