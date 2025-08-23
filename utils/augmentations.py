import torch
import numpy as np
import random
import torchaudio
import torch.nn as nn
#Added the torchAudio transforms
import torchaudio.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image

class ComposeT:
    """
        Composes several Transformations together
    """
    def __init__(self, transforms):
        self.transforms= transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        
        return x

class ToTensor:
    """
        -Converts a NumPy array or a list-like object to a PyTorch float tensor
    """
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not torch.is_tensor(x):
            x = torch.tensor(x).float()
        return x

class SpecAugment:
    """
        Frequency and Time Masking on Spectrograms.  Accepts (freq, time) or (1, freq, time).
        This is an implementation of the SpecAugment technique for the audio data it masks out random blocks
        in the frequency and time domains of a spectrogram.
        The input can be 2D (freq, time) or 3D (1, freq, time) tensor/array

        Args:
            freq_mask_param (int): Maximum width of the frequency mask. A value 'f' will be chosen 
                from '[0, freq_mask_param]', and a mask of 'f' consecutive frequency channels will be applied
            time_mask_param (int): Maximum width of the time mask. A value 't' will be chosen from 
                '[0, time_mask_param]', and a mask of 't' consecutive time steps will be applied.
            
            n_freq_masks (int): The number of frequency masks to be applied.
            n_time_masks (int): The number of time masks to be applied.
    """
    def __init__(self, freq_mask_param=15, time_mask_param=35, n_freq_masks=1, n_time_masks=1):
        self.fm = freq_mask_param
        self.tm = time_mask_param
        self.FM = torchaudio.transforms.FrequencyMasking(self.fm)
        self.TM = torchaudio.transforms.TimeMasking(self.tm)
        self.nf = n_freq_masks
        self.nt = n_time_masks

    def __call__(self, spec):
        if isinstance(spec, np.ndarray):
            spec = torch.from_numpy(spec).float()
        if spec.ndim == 2: #[F,T]
            spec = spec.unsqueeze(0) # (1,F,T)
        elif spec.ndim == 4: #(B,C,F,T)
            raise ValueError("SpecAugment expects single spectrogram, got batched input")
        
        for _ in range(self.nf): 
            spec = self.FM(spec)
        for _ in range(self.nt):
            spec = self.TM(spec)
        
        return spec

class SpecTimePitchWarp:
    """
    Approximates time-stretching and pitch-shifting by scaling spectrogram axes.

    This transformation warps a spectrogram by scaling its time and frequency axes
    using bilinear interpolation. It's an efficient approximation that works directly
    on spectrograms without needing the original audio waveform. The warped spectrogram
    is then cropped or padded back to its original dimensions.

    Args:
        - max_time_scale (float): The maximum scaling factor for the time axis. The scaling
            factor will be randomly chosen from [1/max_time_scale, max_time_scale].
            Defaults to 1.2.
        - max_freq_scale (float): The maximum scaling factor for the frequency axis. The scaling
            factor will be randomly chosen from [1/max_freq_scale, max_freq_scale].
            Defaults to 1.1.
    """
    def __init__(self, max_time_scale=1.2, max_freq_scale=1.1):
        self.max_time = max_time_scale
        self.max_freq = max_freq_scale

    def _resize_and_crop(self, spec, target_f, target_t):
        _, F, T= spec.shape

        spec = spec.unsqueeze(0) #(1,C,F,T)
        spec = torch.nn.functional.interpolate(spec, size=(target_f, target_t), mode='bilinear', align_corners=False)
        spec = spec.squeeze(0)

        start_f = max(0, (spec.shape[1] - F) // 2)
        start_t = max(0, (spec.shape[2] - T) // 2)
        spec = spec[:, start_f:start_f+F, start_t:start_t+T]
        
        if spec.shape[1] < F or spec.shape[2] < T:
            pad_f = F - spec.shape[1]
            pad_t = T - spec.shape[2]
            spec = torch.nn.functional.pad(spec, (0, pad_t, 0, pad_f))
        
        return spec
    
    def __call__(self, spec):
        if isinstance(spec, np.ndarray):
            spec = torch.from_numpy(spec).float()

        # Handle shape
        if spec.ndim == 2:        # (F, T)
            spec = spec.unsqueeze(0)  # (1, F, T)
        elif spec.ndim == 4:      # (B, C, F, T) -> not supported here
            raise ValueError("SpecTimePitchWarp expects single spectrogram, got batched input")

        _, F, T = spec.shape
        t_scale = random.uniform(1.0 / self.max_time, self.max_time)
        f_scale = random.uniform(1.0 / self.max_freq, self.max_freq)
        newT = max(2, int(T * t_scale))
        newF = max(2, int(F * f_scale))

        spec = self._resize_and_crop(spec, newF, newT)
        return spec  # (C, F, T)

class NormalizeSpectrogram:
    """
        Applies min-max normalization to a spectrogram, scaling it to the range [0, 1].

        This normalization technique scales the tensor values based on the minimum and
        maximum values found in the spectrogram. If the max and min values are equal
        (e.g., a silent spectrogram), it returns a tensor of zeros.
    """

    def __call__(self, spectrogram):
        """
        Normalizes the input spectrogram.

        Args:
            spectrogram (torch.Tensor): The input spectrogram tensor.

        Returns:
            torch.Tensor: The normalized spectrogram in the range [0, 1].
        """
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
        """Normalizes the input spectrogram using Z-score.

        Args:
            spectrogram (torch.Tensor): The input spectrogram tensor.

        Returns:
            torch.Tensor: The Z-score normalized spectrogram.
        """
        mean = spectrogram.mean()
        std = spectrogram.std()
        
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        else:
            spectrogram = torch.zeros_like(spectrogram)
        
        return spectrogram


class AugmentSpectrogram:
    """Applies basic time and frequency masking to a spectrogram.

    A simple wrapper around torchaudio's TimeMasking and FrequencyMasking transforms.
    Allows for easy application of one or both augmentations.

    Args:
        time_mask (bool): If True, applies time masking. Defaults to True.
        freq_mask (bool): If True, applies frequency masking. Defaults to True.
        time_mask_param (int): The maximum number of time steps to mask.
        freq_mask_param (int): The maximum number of frequency bins to mask.
    """
    def __init__(self, time_mask=True, freq_mask=True, time_mask_param=18,freq_mask_param=12):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.time_masker = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, spectrogram):
        """Applies the configured augmentations.

        Args:
            spectrogram (torch.Tensor): The input spectrogram.

        Returns:
            torch.Tensor: The augmented spectrogram.
        """
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
