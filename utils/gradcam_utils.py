"""
GradCAM utilities for visualization of model activations
"""
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    """
        Visualizing the convolutional feature importance
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hooks(module, input, output):
            # out shape: [B,C,H,W]
            self.activations = output.detach()

        def backward_hooks(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.forward_handle = self.target_module.register_forward_hook(forward_hooks)
        self.backward_handle = self.target_module.register_full_backward_hook(backward_hooks)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def heatmap(self, input_tensor, target_logit, device):
        """
            input_tensor: [1,C,H,W]
            target_logit:   scalar tensor (outpt chosen to backprop)
            returns heatmap upsampled to H * W numpy [0,1]
        """
        self.model.zero_grad() # type: ignore
        out = target_logit
        out.backward(retain_graph = True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM: activations or gradients missing")
        
        # Compute Weights
        weights = self.gradients.mean(dim=(2,3), keepdim=True) # [B,C,1,1]
        gcam = (weights * self.activations).sum(dim=1, keepdim=True) # [B,1,H,W]
        gcam = torch.relu(gcam)
        # Normalize and upsample to the input size
        gcam = gcam.squeeze(0).squeeze(0)
        gcam_np = gcam.cpu().numpy()

        if gcam_np.max() > 0:
            gcam_np = (gcam_np - gcam_np.min()) / (gcam_np.max() - gcam_np.min())
        
        return gcam_np

def find_last_conv(module, name_contains=None):
    last = (None, None)
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = (m, n)
        
    if name_contains:
        cand = (None, None)
        for n, m in module.named_modules():
            if isinstance(m, nn.Conv2d) and name_contains in n.lower():
                cand = (m, n)

        if cand[0] is not None:
            return cand
            
    return last

def prepare_gradcam_targets(model, device):
    targets = {}
    stft_conv = find_last_conv(model, name_contains='stft')
    cqt_conv = find_last_conv(model, name_contains='cqt')
    
    if stft_conv[0] is None: stft_conv = find_last_conv(model, name_contains=None)

    if cqt_conv[0] is None: cqt_conv = find_last_conv(model, name_contains=None)

    targets['stft'] = stft_conv[0]
    targets['cqt'] = cqt_conv[0]

    return targets

def build_gradcam_for_model(model, device):
    
    targets = prepare_gradcam_targets(model, device)
    cams = {}

    if targets['stft'] is not None:
        cams['stft'] = GradCAM(model, targets['stft'])
    if targets['cqt'] is not None:
        cams['cqt'] = GradCAM(model, targets['cqt'])

    return cams

def run_and_save_gradcams(model, cams, dataset, device, out_dir="gradcam_outputs", n_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    for i in range(len(dataset)):
        item = dataset[i]
        stft = item['stft'].unsqueeze(0).to(device)
        cqt = item['cqt'].unsqueeze(0).to(device)
        label = int(item['label'])

        # forward pass to get logits
        logits = model(stft, cqt)

        # pick scalar to back propagation
        if logits.ndim == 2 and logits.shape[1] == 2:
            target_score = logits[:,1].squeeze()
        else:
            if logits.ndim == 2 and logits.shape[1] == 1:
                target_score = logits.squeeze(1)
            else:
                target_score = logits

        # stft gradcam
        for branch, cam in cams.items():
            try:
                scalar = target_score.sum()
                heat = cam.heatmap(stft if branch =='stft' else cqt, scalar, device)
            except Exception as error:
                print(f"GradCAM failed for sample {i} branch {branch}: {error}")
                heat = None

            # save overlay
            base = (stft.squeeze(0).cpu().numpy() if branch == 'stft' else cqt.squeeze(0).cpu().numpy())
            if base.ndim == 3:
                base_img = base[0]
            else:
                base_img = base

            # Normalize base_img to 0 .. 1
            base_img = base_img - base_img.min()
            if base_img.max() > 0:
                base_img = base_img / base_img.max()

            # Save figure
            plt.figure(figsize=(6,4))
            plt.imshow(base_img, aspect='auto', origin='lower')
            if heat is not None:
                cmap = plt.get_cmap('jet')
                heat_resized = np.flipud(heat)
                plt.imshow(heat_resized, cmap=cmap, alpha=0.5, extent=(0, base_img.shape[1],0,base_img.shape[0]))
            plt.title(f"GradCAM {branch.upper()} - label: {label} idx:{i}")
            fname = os.path.join(out_dir, f"gradcam_{branch}_idx_{i}_label{label}.png")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        saved += 1
        if saved >= n_samples:
            break

    # Remove hooks
    for cam in cams.values():
        cam.remove_hooks()
    print(f"Saved {saved} GradCAM images to {out_dir}")
