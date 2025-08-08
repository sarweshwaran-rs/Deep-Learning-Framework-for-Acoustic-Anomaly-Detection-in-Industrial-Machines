import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.datasets import SpectrogramDataset, ZScoreNormalizeSpectrogram, FocalLoss, BinaryFocalLoss
from models.transformers_encoders import TransformerSpectrogramClassifier

BASE_DIR = r'C:\Users\sarwe\raw'
FEATURES_DIR = os.path.join(BASE_DIR,'-6_dB_features')
CHECKPOINTS_DIR = os.path.join(r'F:\CapStone\DFCA\checkpoints','-6_dB_mobilevit_s_(EUpdated-10)_checkpoint')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

SPECTROGRAM_TYPE = 'cqt'
ENCODER_NAME = 'mobilevit_s'
PRETRAINED_ENCODER = True

NUM_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SAVE_PLOTS = True
EARLY_STOPPING_PATIENCE = 5

#Applying the Custom class to apply the Time and Frequency Mask
class ApplyMultipleMasks(nn.Module):
    def __init__(self, num_time_masks=2, num_freq_masks=2, num_masks = 2, freq_param=20, prob=0.3):
        super(ApplyMultipleMasks, self).__init__()
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.num_masks = num_masks
        self.prob = prob


    def forward(self, spec):
        if random.random() > self.prob:
            return spec
        
        for _ in range(self.num_masks):
            spec = FrequencyMasking(freq_mask_param=self.num_freq_masks) (spec)
            spec = TimeMasking(time_mask_param=self.num_time_masks)(spec)
        
        return spec

#Adding the Guassian Noise to Simulate audio variation
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std

#Mixup for avoding the overfitting and underfitting
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device} : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path, scheduler=None):
    best_val_auc = -1
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accs, val_accs = [], []
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_preds, all_classes = [], [], []

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for sample in tqdm(train_loader, desc="Training"):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device) #labels = sample['label'].float().unsqueeze(1).to(device)

            #Adding the Mixup(New)
            #mixed_input, targets_a, targets_b, lam = mixup_data(spectrograms, labels, alpha=0.1)
            
            optimizer.zero_grad()
            outputs = model(spectrograms)
            #outputs= model(mixed_input)

            #loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            #For Binary FocalLoss
            #probs = torch.sigmoid(outputs).squeeze()
            #preds = (probs > 0.2).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())
            all_classes.extend(preds.cpu().numpy())
            
            # For Binary Focal Loss
            # all_labels.extend(labels.cpu().numpy().flatten().tolist())
            # all_preds.extend(probs.detach().cpu().numpy().flatten().tolist())
            # all_classes.extend(preds.cpu().numpy().flatten().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_classes)
        train_losses.append(epoch_loss)
        train_aucs.append(train_auc)
        train_accs.append(train_acc)
        print(f"Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_auc, val_acc, _, _ = evaluate_model(model, val_loader, criterion, "Validation")
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)

        if scheduler:
            scheduler.step(val_loss)
        print(f"[Epoch {epoch+1}] Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_best_loss.pth'))
            print(f"Model saved in {CHECKPOINTS_DIR} with (best val loss): {best_val_loss:.4f}")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved in {CHECKPOINTS_DIR} with improved AUC: {best_val_auc:.4f}")
            early_stop_counter = 0
        else:
            print(f"Val AUC ({val_auc:.4f}) did not improve from best ({best_val_auc:.4f})")
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early Stopping Triggered!")
            break

    if SAVE_PLOTS:
        epochs_ran = len(train_losses)
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(range(1,epochs_ran+1), train_losses, label='Train Loss')
        plt.plot(range(1,epochs_ran+1), val_losses,label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,2)
        plt.plot(range(1,epochs_ran+1), train_aucs, label = 'Train AUC')
        plt.plot(range(1,epochs_ran+1), val_aucs, label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Train/Validation AUC')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,3)
        plt.plot(range(1,num_epochs+1), train_accs, label = 'Train Acc')
        plt.plot(range(1,num_epochs+1), val_accs, label='Validation Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train/Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINTS_DIR,f'{ENCODER_NAME}_metrics_curve_{SPECTROGRAM_TYPE}.png'))
        #plt.show()

def evaluate_model(model, loader, criterion, phase):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_classes = [], [], []

    with torch.no_grad():
        for sample in tqdm(loader, desc=phase):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device) #labels = sample['label'].float().unsqueeze(1).to(device)
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * spectrograms.size(0)
            
            # probs = torch.softmax(outputs, dim=1)[:, 1]
            # preds = torch.argmax(outputs, dim=1)
            #For BinaryFocalLoss
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.2).long()
            
            # all_labels.extend(labels.cpu().numpy())
            # all_preds.extend(probs.cpu().numpy())
            # all_classes.extend(preds.cpu().numpy())
            # For BinaryFocalLoss
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_preds.extend(probs.cpu().numpy().flatten().tolist())
            all_classes.extend(preds.cpu().numpy().flatten().tolist())

    avg_loss = running_loss / len(loader.dataset)
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float('nan')
    acc_score = accuracy_score(all_labels, all_classes)
    
    print(f"{phase} Loss: {avg_loss:.4f}, AUC: {auc_score:.4f}, Accuracy: {acc_score:.4f}") 
    print(f"\n[DEBUG] {phase} Prediction Distribution:")
    pred_counter = Counter(all_classes)
    label_counter = Counter(all_labels)
    print(f"Predicted Classes: {dict(pred_counter)}")
    print(f"True Labels:       {dict(label_counter)}")

    print(f"\nSample Predictions vs Labels:")
    for i in range(min(10, len(all_labels))):
        print(f"Sample {i+1}: Pred = {all_classes[i]}, Prob = {all_preds[i]:.4f}, True = {all_labels[i]}")
    
    return avg_loss, auc_score, acc_score, all_labels, all_preds

def calculate_pAUC(labels, preds, max_fpr=0.1):
    if len(np.unique(labels)) < 2:
        return float('nan')

    fpr, tpr, _ = roc_curve(labels, preds)
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


def check_overlap(dataset, train_idx, val_idx, test_idx):
    """Checks for overlapping file paths between data splits."""
    len_d0 = len(dataset.datasets[0]) 

    def get_paths(indices):
        paths = set()
        for i in indices:
            if i < len_d0:
                paths.add(dataset.datasets[0].all_file_paths[i])
            else:
                paths.add(dataset.datasets[1].all_file_paths[i - len_d0])
        return paths
    
    train_files = get_paths(train_idx)
    val_files = get_paths(val_idx)
    test_files = get_paths(test_idx)
    
    print("\n--- Checking Overlap Between Splits ---")
    print(f"Train ∩ Val: {len(train_files & val_files)}")
    print(f"Train ∩ Test: {len(train_files & test_files)}")
    print(f"Val ∩ Test: {len(val_files & test_files)}")

    if train_files & val_files or train_files & test_files or val_files & test_files:
        print("Warning: Overlapping files detected between splits!")
    else:
        print("No overlap between Train, Validation, and Test splits.")


def main():

    transform = transforms.Compose([
        ZScoreNormalizeSpectrogram(),
        transforms.Resize((224, 224), antialias=True),
        ApplyMultipleMasks(num_time_masks=2, num_freq_masks=2, num_masks=2, freq_param = 15),
        AddGaussianNoise(std=0.01)
    ])

    print(f"Loading CQT data from {FEATURES_DIR}...")
    
    #Load all data into a single pool for splitting
    full_normal_dataset = SpectrogramDataset(FEATURES_DIR,'normal', transform, SPECTROGRAM_TYPE)
    full_abnormal_dataset = SpectrogramDataset(FEATURES_DIR, 'abnormal', transform, SPECTROGRAM_TYPE)

    print(f"Loaded {len(full_normal_dataset)} normal samples.")
    print(f"Loaded {len(full_abnormal_dataset)} abnormal samples.")

    combined_dataset = ConcatDataset([full_normal_dataset, full_abnormal_dataset])
    combined_labels = full_normal_dataset.labels + full_abnormal_dataset.labels
    indices = list(range(len(combined_dataset)))

    # Shuffle indices and labels together to maintain correspondence
    indices, combined_labels = shuffle(indices, combined_labels, random_state=42) # type: ignore

    # === Stratified Split of Train/Val/Test ===
    if len(np.unique(combined_labels)) < 2: # type: ignore
        print("Error: Dataset does not contain both normal and abnormal samples. Cannot perform the split.")
        return

    # Split into 70% train, 15% validation, 15% test
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, combined_labels, test_size=0.3, stratify=combined_labels, random_state=42 # type: ignore
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42 
    )

    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    print(f"\nStratified Split Sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}, Test = {len(test_dataset)}")
    
    # Verify label distribution in splits
    train_labels = [combined_labels[i] for i in train_idx] # type: ignore
    val_labels = [combined_labels[i] for i in val_idx] # type: ignore
    test_labels = [combined_labels[i] for i in test_idx] # type: ignore
    
    print("\n--- Split Label Distribution (Stratified) ---")
    print(f"Train Set:      Normal = {Counter(train_labels)[0]}, Abnormal = {Counter(train_labels)[1]}")
    print(f"Validation Set: Normal = {Counter(val_labels)[0]}, Abnormal = {Counter(val_labels)[1]}")
    print(f"Test Set:       Normal = {Counter(test_labels)[0]}, Abnormal = {Counter(test_labels)[1]}")

    check_overlap(combined_dataset, train_idx, val_idx, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = TransformerSpectrogramClassifier(
        model_name=ENCODER_NAME, 
        pretrained=PRETRAINED_ENCODER,
        num_classes=1,
        dropout_prob=0.3
    ).to(device)
    print(f"Model is using Transformer: {ENCODER_NAME}, Pretrained: {PRETRAINED_ENCODER}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(combined_labels), # type: ignore
        y=combined_labels # type: ignore
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class Weights: {class_weights}")
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    pos_weights = torch.tensor([4.0], dtype=torch.float).to(device)
    criterion = BinaryFocalLoss(alpha=0.1, gamma=2.0,pos_weight=pos_weights, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3, min_lr=1e-6) 

    model_path = os.path.join(CHECKPOINTS_DIR, f'{ENCODER_NAME}_{SPECTROGRAM_TYPE}_best_model.pth')
    
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCH, model_path, scheduler)

    print("\n--- Final Test Evaluation ---")
    
    best_auc_model_path = model_path
    best_loss_model_path = model_path.replace('.pth','_best_loss.pth')

    if os.path.exists(best_auc_model_path):
         print(f"Loading best AUC model from: {best_auc_model_path}")
         model.load_state_dict(torch.load(best_auc_model_path))
    elif os.path.exists(best_loss_model_path):
        print(f"Best AUC model not found. Attempting to load best loss model from: {best_loss_model_path}")
        model.load_state_dict(torch.load(best_loss_model_path))
    else:
        print("Error: No saved model checkpoint found. Cannot perform final evaluation.")
        return 
    
    model.eval()
    _, _, _, all_labels_test, all_preds_test = evaluate_model(model, test_loader, criterion, "Test")

    if len(np.unique(all_labels_test)) > 1:
        final_auc = roc_auc_score(all_labels_test, all_preds_test)
        final_pauc = calculate_pAUC(all_labels_test, all_preds_test, max_fpr=0.1)

        print(f"\nFinal Test AUC-ROC ({SPECTROGRAM_TYPE} branch): {final_auc:.4f}")
        print(f"Final Test pAUC (FPR <= 0.1, {SPECTROGRAM_TYPE} branch): {final_pauc:.4f}")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(all_labels_test, all_preds_test)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic ({SPECTROGRAM_TYPE.upper()} Branch)')
        plt.legend(loc="lower right")
        plt.grid(True)
        if SAVE_PLOTS:
            plt.savefig(os.path.join(CHECKPOINTS_DIR, f'{ENCODER_NAME}_roc_curve_{SPECTROGRAM_TYPE}.png'))
        #plt.show()
    else:
        print("\nFinal Test AUC/pAUC not calculated: Only one class present in the test set.")

if __name__ == "__main__":
    main()