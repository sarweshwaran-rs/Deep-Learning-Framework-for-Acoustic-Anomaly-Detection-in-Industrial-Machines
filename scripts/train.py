import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns

from scripts.pretrain_pipeline import FusedModel
from utils.augmentations import ComposeT, ToTensor, SpecTimePitchWarp, SpecAugment, GradCAM
from models.heads import AnomalyScorer, SimpleAnomalyMLP, EmbeddingMLP
from models.losses import ContrastiveLoss
from utils.datasets import PairedSpectrogramDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

def calculate_pAUC(labels, preds, max_fpr = 0.1):
    """
    Calculates Partial AUC (pAUC) for a given FPR range.
    Args:
        labels (array): True binary labels.
        preds (array): Predicted probabilities for the positive class.
        max_fpr (float): Maximum False Positive Rate for pAUC calculation.
    Returns:
        float: pAUC score.
    """
    if len(np.unique(labels)) < 2:
        return float('nan')
    
    fpr, tpr, _ = roc_curve(labels, preds)
    #filter for FPR <= max_fpr
    mask = fpr <= max_fpr
    fpr_filtered, tpr_filtered = fpr[mask], tpr[mask] 
      
    if fpr_filtered.size == 0:
        return 0.0

    if fpr_filtered.max() < max_fpr:
        idx = np.where(fpr <= max_fpr)[0][-1]
        if idx + 1 < len(fpr):
            x1, y1 = fpr[idx], tpr[idx]
            x2, y2 = fpr[idx + 1], tpr[idx + 1]
            tpr_interp = y1 + (y2 - y1) * (max_fpr - x1) / (x2 - x1) if (x2 - x1) > 0 else y1
            fpr_filtered = np.append(fpr_filtered, max_fpr)
            tpr_filtered = np.append(tpr_filtered, tpr_interp)
            sort_idx = np.argsort(fpr_filtered)
            fpr_filtered = fpr_filtered[sort_idx]
            tpr_filtered = tpr_filtered[sort_idx]

    return auc(fpr_filtered, tpr_filtered) / max_fpr if len(fpr_filtered) >= 2 else 0.0

def find_last_conv(module, name_contains=None):
    """
    Returns (module_ref, full_name) of the last nn.Conv2d found in module.
    If name_contains is provided, prefer conv modules whose name includes that substring.
    """
    last = (None, None)
    for n,m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = (m,n)
        
    if name_contains:
        # Try to find last conv with name containing substring
        cand = (None, None)
        for n, m in module.named_modules():
            if isinstance(m,nn.Conv2d) and name_contains in n.lower():
                cand = (m,n)
        
        if cand[0] is not None:
            return cand
    
    return last

def evaluate_model(model, data_loader, criterion, phase="Evaluation", device=device, head_mode='classifier', sample_count=10, threshold=0.5):
    """
    Evaluate a model on a given dataset.
    
    Args:
        model: PyTorch model to evaluate.
        data_loader: DataLoader for the dataset to evaluate on.
        criterion: Loss function.
        phase (str): Label for the evaluation phase (e.g., "Train", "Validation", "Test").
        device: Torch device ('cuda' or 'cpu').
        head_mode (str): Type of model head ('classifier', 'mlp', 'prototype', 'embedding').
        sample_count (int): Number of sample predictions to print for inspection.
        threshold (float): The classification threshold to use for binary predictions.

    Returns:
        avg_loss: Average loss over the dataset.
        auc_score: ROC AUC score.
        acc_score: Accuracy.
        bacc_score: Balanced accuracy.
        f1_score: F1-score.
        all_labels: List of all ground truth labels.
        all_probs: List of all predicted probabilities/scores for the positive class.
        best_threshold: The optimal threshold found, or the provided threshold.
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    
    best_threshold = threshold
    f1 = 0.0
    # [DEBUG]
    class_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=phase):
            stft = batch['stft'].to(device)
            cqt = batch['cqt'].to(device)
            labels = batch['label'].to(device).long()
            
            for lbl in labels.cpu().numpy():
                class_counts[int(lbl)] += 1
            
            loss = None
            if head_mode == "prototype":
                embeddings, prototype = model(stft, cqt)
                embeddings = F.normalize(embeddings, dim=1)
                prototype = F.normalize(prototype, dim=0)
                if prototype.dim() == 1:
                    prototype = prototype.unsqueeze(0)
                prototype = prototype.expand_as(embeddings)
                cos_sim = torch.sum(embeddings * prototype, dim=1)
                probs = 1 - cos_sim
                loss = criterion(embeddings, prototype, labels.float())
            
            elif head_mode in ["classifier", "mlp"]:
                logits = model(stft, cqt)
                if logits.ndim == 2 and logits.shape[1] == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    loss = criterion(logits, labels.long())
                else:
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        logits = logits.squeeze(1)
                    probs = torch.sigmoid(logits)
                    loss = criterion(logits, labels.float())
            
            elif head_mode == "embedding":
                embeddings = model(stft, cqt)
                normal_proto = model.head.normal_prototype
                embeddings = F.normalize(embeddings, dim=1)
                normal_proto = F.normalize(normal_proto, dim=0)
                cos_sim = torch.sum(embeddings * normal_proto.unsqueeze(0).expand_as(embeddings), dim=1)
                probs = 1 - cos_sim
                if isinstance(criterion, ContrastiveLoss):
                    loss = criterion(embeddings, normal_proto, labels)
                else:
                    loss = criterion(probs, labels.float())
            elif head_mode == 'classifier-1':
                logits = model(stft,cqt)
                if logits.ndim == 2 and logits.shape[1] ==1:
                    logits = logits.squeeze(1)
                probs = torch.sigmoid(logits)
                loss = criterion(logits, labels.float())
            
            else:
                raise ValueError(f"Unsupported head_mode:{head_mode}")
            
            running_loss += loss.item() * stft.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"[DEBUG] {phase} label counts: {class_counts}")
    
    # Logic for finding optimal threshold on Validation set
    f1 = 0.0 # Initialize f1
    if phase == "Validation":
        best_f1 = 0
        current_optimal_threshold = 0.5
        for thresh in np.arange(0.01, 1.0, 0.01):
            predictions_thresh = (np.array(all_probs) > thresh).astype(int)
            f1_candidate = f1_score(all_labels, predictions_thresh)
            if f1_candidate > best_f1:
                best_f1 = f1_candidate
                current_optimal_threshold = thresh

        best_threshold = current_optimal_threshold
        f1 = best_f1
        print(f"Optimal Threshold (F1-score): {best_threshold:.2f}")
        print(f"Best F1-score on Validation Set: {best_f1:.4f}")
    
    # Calculate all metrics using the selected or optimal threshold
    all_preds = (np.array(all_probs) > best_threshold).astype(int)
    if phase != "Validation":
        if len(np.unique(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds)
        else:
            f1 = 0.0
    
    avg_loss = running_loss / len(data_loader.dataset)
    auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan')
    acc_score = accuracy_score(all_labels, all_preds)
    bacc_score = balanced_accuracy_score(all_labels, all_preds)
    
    print(f"{phase} Loss: {avg_loss:.4f}, {phase} AUC: {auc_score:.4f}, {phase} ACC: {acc_score:.4f}, {phase} BACC: {bacc_score:.4f}")
    print(f"[DEBUG] {phase} Prediction Distribution: {dict(Counter(all_preds))}")
    print(f"[DEBUG] {phase} Label Distribution: {dict(Counter(all_labels))}")
    # Show some sample predictions
    print("==================================================")
    print("\nSample Predictions vs Labels:")
    for i in range(min(sample_count, len(all_labels))):
        print(f"Sample {i+1}: Pred = {all_preds[i]}, Prob = {all_probs[i]:.4f}, True = {all_labels[i]}")

    # Misclassified samples
    print("==================================================")
    errors = [(i, p, pr, l) for i, (p, pr, l) in enumerate(zip(all_preds, all_probs, all_labels)) if p != l]
    print(f"{phase} Misclassified Samples: {len(errors)} / {len(all_labels)}")
    for idx, pred, prob, label in errors[:10]:
        print(f"Idx {idx}: Pred = {pred}, Prob = {prob:.4f}, True = {label}")
    
    return avg_loss, auc_score, acc_score, bacc_score, f1, all_labels, all_probs, best_threshold

# ---------------------------------
# Training
# ---------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, head_mode, schedular=None, num_epochs=5, model_save_path="best_model.pth", device=device, save_plots=True):
    best_val_auc = -np.inf
    best_val_loss = np.inf
    current_threshold = 0.5
    best_threshold = 0.5

    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accs, val_accs = [], []
    train_baccs, val_baccs = [], []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []

        #DEBUG
        class_counts_train = {0:0, 1:0}
        epoch_stats = defaultdict(list)

        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch in tqdm(train_loader, desc="Train"):
            stft = batch['stft'].to(device)
            cqt = batch['cqt'].to(device)
            labels = batch['label'].to(device).long()

            #DEBUG
            for lbl in labels.cpu().numpy():
                class_counts_train[int(lbl)] +=1

            optimizer.zero_grad()
            outputs = model(stft, cqt)

            if head_mode in ["classifier", "mlp"]:
                # Binary classification
                if outputs.ndim == 2 and outputs.shape[1] == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    preds = torch.argmax(outputs.detach().cpu(), dim=1)
                    loss = criterion(outputs, labels)
                else:
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > current_threshold ).long()
                    loss = criterion(outputs.squeeze(), labels.float())
            elif head_mode == "prototype":
                embeddings, prototype = outputs
                #DEBUG STARTS
                #Cosine SIM LOGGING
                embeddings = F.normalize(embeddings, dim=1)
                prototype = F.normalize(prototype,dim=0)
                if prototype.dim() == 1:
                    prototype = prototype.unsqueeze(0)
                prototype = prototype.expand_as(embeddings)
                cos_sim = torch.sum(embeddings * prototype, dim=1)

                normal_sim = cos_sim[labels == 0].mean().item() if (labels == 0).any() else None
                anomaly_sim = cos_sim[labels ==1].mean().item() if (labels == 1).any() else None
                if normal_sim is not None:
                    epoch_stats['normal_sim'].append(normal_sim)
                if anomaly_sim is not None:
                    epoch_stats['anomaly_sim'].append(anomaly_sim)
                
                #DEBUG ENDS
                anomaly_scores = 1 - cos_sim 
                probs = anomaly_scores
                preds = (anomaly_scores > current_threshold).long()
                loss = criterion(embeddings, prototype, labels.float())

            elif head_mode == "embedding":
                embeddings = outputs
                normal_proto = model.head.normal_prototype
                
                #DEBUG STARTS
                embeddings = F.normalize(embeddings, dim=1)
                normal_proto = F.normalize(normal_proto, dim=0)
                cos_sim = torch.sum(embeddings * normal_proto.unsqueeze(0).expand_as(embeddings), dim=1)

                normal_sim = cos_sim[labels == 0].mean().item() if (labels == 0).any() else None
                anomaly_sim = cos_sim[labels == 1].mean().item() if (labels == 1).any() else None
                if normal_sim is not None:
                    epoch_stats['normal_sim'].append(normal_sim)
                if anomaly_sim is not None:
                    epoch_stats['anomaly_sim'].append(anomaly_sim)
                #DEBUG ENDS
                
                anomaly_scores = 1 - cos_sim 
                probs = anomaly_scores
                preds = (anomaly_scores > current_threshold).long()
                loss = criterion(embeddings,normal_proto, labels)
            
            elif head_mode == 'classifier-1':
                outputs = model(stft, cqt)
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > current_threshold).long()
                loss = criterion(outputs.squeeze(), labels.float())
            else:
                raise ValueError(f"Unsupported head_mode: {head_mode}")
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * stft.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
        #DEBUG
        print(f"[DEBUG] Train label counts (epoch {epoch+1}): {class_counts_train}") 
        if epoch_stats['normal_sim']:
            avg_normal_sim = sum(epoch_stats['normal_sim']) / len(epoch_stats['normal_sim'])
            avg_anomaly_sim = sum(epoch_stats['anomaly_sim']) / len(epoch_stats['anomaly_sim'])
            print(f"[DEBUG] Avg Normal CosSim: {avg_normal_sim:.4f}, Avg Anomaly CosSim: {avg_anomaly_sim:.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels))> 1 else float('nan')
        train_acc = accuracy_score(all_labels, all_preds)
        train_bacc = balanced_accuracy_score(all_labels, all_preds)
        train_aucs.append(train_auc)
        train_accs.append(train_acc)
        train_baccs.append(train_bacc)

        print(f"Train Loss: {epoch_loss:.4f} | Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f}, | Train BAcc: {train_bacc:.4f}")

        # Validation
        val_loss, val_auc, val_acc, val_bacc, _, _, _, current_optimal_threshold = evaluate_model(model, val_loader, criterion, phase="Validation", device=device, head_mode=head_mode)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        val_baccs.append(val_bacc)

        # scheduler step (per epoch)
        if schedular is not None:
            try:
                schedular.step()
            except Exception:
                pass
        print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            loss_path = model_save_path.replace(".pth", "_best_loss.pth")
            torch.save(model.state_dict(), loss_path)
            print(f"Saved Best-Loss model to {loss_path} (val_loss improved to {best_val_loss:.4f})")

        # Save by best AUC
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = current_optimal_threshold
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved Best-AUC model to {model_save_path} (val_auc improved to {best_val_auc:.4f})")
        else:
            print(f"Val AUC {val_auc:.4f} did not improved from best {best_val_auc:.4f}")
    
    if save_plots:
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(18,4))
        plt.subplot(1,4,1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.legend()
        plt.grid(True)
        plt.title("Train/Validation Loss")

        plt.subplot(1, 4, 2)
        plt.plot(epochs, train_aucs, label='Train AUC')
        plt.plot(epochs, val_aucs, label='Val AUC')
        plt.legend()
        plt.grid(True)
        plt.title("Train/Validation AUC")

        plt.subplot(1, 4, 3)
        plt.plot(epochs, train_accs, label='Train Acc')
        plt.plot(epochs, val_accs, label='Val Acc')
        plt.legend()
        plt.grid(True)
        plt.title("Train/Validation Accuracy")

        plt.subplot(1, 4, 4)
        plt.plot(epochs, train_baccs, label='Train BAcc')
        plt.plot(epochs, val_baccs, label='Val BAcc')
        plt.legend()
        plt.grid(True)
        plt.title("Train/Validation Balanced Acc")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "training_summary.png"))
        # plt.show()

    return best_threshold

def plot_confusion_matrix(y_true, y_pred, labels, save_path, title="Consusion Matrix"):
    """
    Plots a comfusion matrix for model evaluation
    Args:
        y_true (list or np.array): Ground truth labels.
        y_pred (list or np.array): Predicted labels.
        labels (list): A list of labels for the matrix axes (['Normal', 'Abnormal'])
        title (str): Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Lable')
    plt.ylabel('True Lable')
    plt.title(title)
    plt.savefig(os.path.join(save_path, "Confusion Matrix.png"))
    # plt.show()

# -------------------------------
# GradCAM Utilities
# -------------------------------
def prepare_gradcam_targets(model, device):
    """
    Heuristic: try to find conv layers for stft and cqt branches by name
    Fallback: the last conv in the model
    Return dict {'stft':module, 'cqt':module}
    """
    targets = {}
    stft_conv = find_last_conv(model, name_contains='stft')
    cqt_conv = find_last_conv(model, name_contains='cqt')

    if stft_conv[0] is None:
        stft_conv = find_last_conv(model, name_contains=None)
    if cqt_conv[0] is None:
        cqt_conv = find_last_conv(model, name_contains=None)
    
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

        #forward pass to get logtis
        logits = model(stft, cqt)
        #Pick scalar to backprop
        if logits.ndim == 2 and logits.shape[1] == 2:
            target_score = logits[:,1].squeeze()
        else:
            if logits.ndim == 2 and logits.shape[1] == 1:
                target_score = logits.squeeze(1)
            else:
                target_score = logits
        
        #stft gradcam
        for branch, cam in cams.items():
            try:
                scalar = target_score.sum()
                heat = cam.heatmap(stft if branch =='stft' else cqt, scalar, device)
            except Exception as e:
                print(f"GradCAM failed for sample {i} branch {branch}: {e}")
                heat = None
            
            # save overlay
            base = (stft.squeeze(0).cpu().numpy() if branch =='stft' else cqt.squeeze(0).cpu().numpy())
            if base.ndim == 3:
                base_img = base[0]
            else:
                base_img = base
            
            # normalize base_img to 0..1
            base_img = base_img - base_img.min()
            if base_img.max() > 0:
                base_img = base_img / base_img.max()
            # Save figure
            plt.figure(figsize=(6,4))
            plt.imshow(base_img, aspect='auto', origin='lower')
            if heat is not None:
                cmap = plt.get_cmap('jet')
                heat_resized = np.flipud(heat)
                plt.imshow(heat_resized, cmap=cmap, alpha=0.5, extent=(0,base_img.shape[1], 0, base_img.shape[0]))
            plt.title(f"GradCAM {branch.upper()} - label:{label} idx:{i}")
            fname = os.path.join(out_dir, f"gradcam_{branch}_idx_{i}_label{label}.png")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        saved +=1
        if saved >= n_samples:
            break

    # remove hooks
    for cam in cams.values():
        cam.remove_hooks()
    print(f"Saved {saved} GradCAM images to {out_dir}")

# ================================
# Main Pipeline 
# ================================

FEATURES_DIR = r'data\features'
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-3
CHECKPOINT_DIR = r'checkpoints'
CONTRASTIVE_MARGIN = 0.5
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
HEAD_MODE = 'mlp'
EMB_DIM = 64

save_path = os.path.join(CHECKPOINT_DIR,'DFCA', '[Anomaly-With-Transformations-dropout=0.4](30)_MLP(5e-5)')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
def main():
    
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_transforms = ComposeT([
        ToTensor(),
        SpecTimePitchWarp(max_time_scale=1.1, max_freq_scale=1.1),
        SpecAugment(freq_mask_param=4, time_mask_param=4, n_freq_masks=2, n_time_masks=2),
    ])

    no_transform = ComposeT([
        ToTensor(),
    ])

    full_dataset = PairedSpectrogramDataset(FEATURES_DIR, transform=None)
    all_labels = [int(x) for x in full_dataset.labels]

    #Stratified Split [train/val/test]
    idxs = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(idxs, test_size=0.3, stratify=all_labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[all_labels[i] for i in temp_idx], random_state=42)
    
    train_set = Subset(PairedSpectrogramDataset(FEATURES_DIR, transform=train_transforms), train_idx)
    val_set = Subset(PairedSpectrogramDataset(FEATURES_DIR, transform=no_transform), val_idx)
    test_set = Subset(PairedSpectrogramDataset(FEATURES_DIR, transform=no_transform), test_idx)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Split sizes => Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print("Label Distribution (Train):",Counter([int(full_dataset[i]['label']) for i in train_idx]))
    print("Label Distribution (Validation):",Counter([int(full_dataset[i]['label']) for i in val_idx]))
    print("Label Distribution (Test):",Counter([int(full_dataset[i]['label']) for i in test_idx]))
    
    head_mode = HEAD_MODE.lower()
    
    if head_mode == 'prototype':
        head = AnomalyScorer(in_dim=256, dropout=0.4, mode='prototype')
        criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
        print(f"Used head:\n {head}")
        print("Used transformations:")
        for transform in train_transforms.transforms:
            # Print the name of the transformation class
            print(f"  - {transform.__class__.__name__}")
    
            # Check for specific transformations and print their parameters
            if isinstance(transform, SpecTimePitchWarp):

                print(f"    - time_scale: {getattr(transform, 'max_time_scale', {transform.max_time})}")
                print(f"    - freq_scale: {getattr(transform, 'max_freq_scale', {transform.max_freq})}")
            if isinstance(transform, SpecAugment):
                print(f"    - freq_mask_param: {getattr(transform,'freq_mask_param',{transform.fm})}")
                print(f"    - time_mask_param: {getattr(transform,'time_mask_param',{transform.tm})}")
                print(f"    - n_freq_masks: {getattr(transform,'n_freq_masks', {transform.nf})}")
                print(f"    - n_time_masks: {getattr(transform,'n_time_masks', {transform.nt})}")
    elif HEAD_MODE == 'mlp':
        head = SimpleAnomalyMLP(in_dim=256, dropout=0.4,hidden=128, out_dim=1)
        pos_count = sum(all_labels)
        neg_count = len(all_labels) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Used head:\n {head}")
        print("Used transformations:")
        for transform in train_transforms.transforms:
            # Print the name of the transformation class
            print(f"  - {transform.__class__.__name__}")
    
            # Check for specific transformations and print their parameters
            if isinstance(transform, SpecTimePitchWarp):

                print(f"    - time_scale: {getattr(transform, 'max_time_scale',{transform.max_time} )}")
                print(f"    - freq_scale: {getattr(transform, 'max_freq_scale', {transform.max_freq})}")
            if isinstance(transform, SpecAugment):
                print(f"    - freq_mask_param: {getattr(transform,'freq_mask_param',{transform.fm})}")
                print(f"    - time_mask_param: {getattr(transform,'time_mask_param',{transform.tm})}")
                print(f"    - n_freq_masks: {getattr(transform,'n_freq_masks',{transform.nf})}")
                print(f"    - n_time_masks: {getattr(transform,'n_time_masks',{transform.nt})}")
    elif HEAD_MODE == 'embedding':
        head = EmbeddingMLP(in_dim=256, hidden=128, dropout=0.4, emb_dim=64)
        criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
        print(f"Used head:\n {head}")
        print("Used transformations:")
        for transform in train_transforms.transforms:
            # Print the name of the transformation class
            print(f"  - {transform.__class__.__name__}")
    
            # Check for specific transformations and print their parameters
            if isinstance(transform, SpecTimePitchWarp):

                print(f"    - time_scale: {getattr(transform, 'max_time_scale', {transform.max_time})}")
                print(f"    - freq_scale: {getattr(transform, 'max_freq_scale', {transform.max_freq})}")
            if isinstance(transform, SpecAugment):
                print(f"    - freq_mask_param: {getattr(transform,'freq_mask_param',{transform.fm})}")
                print(f"    - time_mask_param: {getattr(transform,'time_mask_param',{transform.tm})}")
                print(f"    - n_freq_masks: {getattr(transform,'n_freq_masks',{transform.nf})}")
                print(f"    - n_time_masks: {getattr(transform,'n_time_masks',{transform.nt})}")
    elif HEAD_MODE == 'classifier':
        head = SimpleAnomalyMLP(in_dim=256, dropout=0.4, hidden=128, out_dim=2)
        class_counts = [2624, 319]
        
        alpha = 0.7
        total = sum(class_counts)
        class_weights = [(total / c) ** alpha for c in class_counts]  
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using head: {head}")
        print("Used transformations:")
        for transform in train_transforms.transforms:
            # Print the name of the transformation class
            print(f"  - {transform.__class__.__name__}")
    
            # Check for specific transformations and print their parameters
            if isinstance(transform, SpecTimePitchWarp):

                print(f"    - time_scale: {getattr(transform, 'max_time_scale', {transform.max_time})}")
                print(f"    - freq_scale: {getattr(transform, 'max_freq_scale', {transform.max_freq})}")
            if isinstance(transform, SpecAugment):
                print(f"    - freq_mask_param: {getattr(transform,'freq_mask_param',{transform.fm})}")
                print(f"    - time_mask_param: {getattr(transform,'time_mask_param', {transform.tm})}")
                print(f"    - n_freq_masks: {getattr(transform,'n_freq_masks',{transform.nf})}")
                print(f"    - n_time_masks: {getattr(transform,'n_time_masks',{transform.nt})}")
    elif HEAD_MODE == 'classifier-1':
        head = AnomalyScorer(in_dim=256, dropout=0.4, mode='classifier-1')
        pos_count = sum(all_labels)
        neg_count = len(all_labels) - pos_count
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using head: \n{head}")
        print("Used transformations:")
        for transform in train_transforms.transforms:
            print(f"- {transform.__class__.__name__}")
            if isinstance(transform, SpecTimePitchWarp):

                print(f"    - time_scale: {getattr(transform, 'max_time_scale', {transform.max_time})}")
                print(f"    - freq_scale: {getattr(transform, 'max_freq_scale', {transform.max_freq})}")
            if isinstance(transform, SpecAugment):
                print(f"    - freq_mask_param: {getattr(transform,'freq_mask_param',{transform.fm})}")
                print(f"    - time_mask_param: {getattr(transform,'time_mask_param', {transform.tm})}")
                print(f"    - n_freq_masks: {getattr(transform,'n_freq_masks',{transform.nf})}")
                print(f"    - n_time_masks: {getattr(transform,'n_time_masks',{transform.nt})}")
    else:
        raise ValueError("Invalid Head_Mode")
    
    model = FusedModel(
        stft_dim=512, cqt_dim=320, fusion_dim=256, head=head, head_mode=head_mode,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-5)
    
    model_path = os.path.join(CHECKPOINT_DIR, 'DFCA', '[Anomaly-With-Transformations-dropout=0.4](30)_MLP(5e-5)', "best_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  
    best_threshold = train_model(model, train_loader, val_loader, criterion, optimizer, head_mode, scheduler, num_epochs=NUM_EPOCHS, model_save_path=model_path, device=device, save_plots=True)
    
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load(model_path))
    
    safe_threshold = float(best_threshold) if best_threshold is not None else 0.5

    test_loss, test_auc, test_acc, test_bacc, test_f1, all_labels_test, all_probs_test, _ = evaluate_model(model, test_loader, criterion, "Test", device, head_mode=head_mode, threshold=safe_threshold)
    print(f"\nFinal Test Metrics (with best validation threshold {best_threshold:.2f}):")
    print(f"Loss: {test_loss:.4f} | AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f} | Balanced Accuracy: {test_bacc:.4f} | F1-Score: {test_f1:.4f}")
    if len(np.unique(all_labels_test)) > 1:
        final_pauc = calculate_pAUC(all_labels_test, all_probs_test, max_fpr=0.2)
        print(f"Final Test pAUC (FPR <= 0.2): {final_pauc:.4f}")
    else:
        print("Test set contains only one class; cannot compute AUC/pAUC")


    # Plot ROC
    fpr, tpr, _ = roc_curve(all_labels_test, all_probs_test)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, lw=2, label=f"{test_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle='--', lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Test ROC With Optimal Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path,"roc_test_optimal.png"))
    # plt.show()

    labels = ['Normal', 'Anomaly']
    all_preds_test = (np.array(all_probs_test) > safe_threshold).astype(int)
    plot_confusion_matrix(all_labels_test, all_preds_test,labels,save_path,title='Test Set Confusion Matrix')

    try:
        cams = build_gradcam_for_model(model, device)
        run_and_save_gradcams(model, cams, test_set,device, out_dir=os.path.join(CHECKPOINT_DIR,"DFCA","[Anomaly-With-Transformations-dropout=0.4](30)_MLP(5e-5)","gradcam"),n_samples=8)
    except Exception as error:
        print(f"GradCAM step failed: {error}")

if __name__ == '__main__':
    main()