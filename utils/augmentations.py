import torch
import numpy as np
import random
import torchaudio
import torch.nn as nn

class ComposeT:
    def __init__(self, transforms):
        self.transforms= transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        
        return x

class ToTensor:
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not torch.is_tensor(x):
            x = torch.tensor(x).float()
        return x

class SpecAugment:
    """
    Frequency and Time Masking on Spectrograms.  Accepts (freq, time) or (1, freq, time).
    """
    def __init__(self, freq_mask_param=15, time_mask_param=35, n_freq_masks=1, n_time_masks=1):
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param)
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
            spec = self.fm(spec)
        for _ in range(self.nt):
            spec = self.tm(spec)
        
        return spec

class SpecTimePitchWarp:
    """
    Approximate time-stretch / pitch-shift by scaling time/freq axes of the spectrogram.
    This is an approximation for when you have only spectrograms.
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


    
# =====GradCAM ==========
class GradCAM:
    """
    Light weight Grad-CAM implementation that:
        -takes a model and a target_conv module (nn.Conv2d)
        - captures forward activations and gradients on that conv
        - produces a heatmap upsampled to input spectrogram size
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._registe_hooks()

    def _registe_hooks(self):
        def forward_hook(module, inp, out):
            #out shape: [B,C,H,W]
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.forward_handle = self.target_module.register_forward_hook(forward_hook)
        self.backward_handle = self.target_module.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def heatmap(self, input_tensor, target_logit, device):
        """
        input_tensor: [1,C,H,W]
        target_logit: scalar tesor (outut chosen to backprop)
        returns heatmap upsampled to HXW numpy in [0,1]
        """
        self.model.zero_grad()
        out = target_logit
        out.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM: actions or gradients missing")
        
        # compute weights
        weights = self.gradients.mean(dim=(2,3), keepdim=True) # [B,C,1,1]
        gcam = (weights * self.activations).sum(dim=1, keepdim=True) #[B,1,H,W]
        gcam = torch.relu(gcam)
        # Normalize and upsample to input size
        gcam = gcam.squeeze(0).squeeze(0)
        gcam_np = gcam.cpu().numpy()

        if gcam_np.max() > 0:
            gcam_np = (gcam_np - gcam_np.min()) / (gcam_np.max() - gcam_np.min())
        return gcam_np
