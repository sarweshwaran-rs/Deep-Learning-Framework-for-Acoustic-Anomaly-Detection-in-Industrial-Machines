#----- Standard Libraries -----
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset,Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, balanced_accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.utils import shuffle

#----- Custom Imports -----
from utils.datasets import SpectrogramDataset 
from utils.augmentations import ZScoreNormalizeSpectrogram, AugmentSpectrogram
from models.transformers_encoders import TransformerSpectrogramClassifier

#----- Paths and the Configuration -----
BASE_DIR = r'C:\Users\sarwe\raw'
FEATURES_DIR = os.path.join(BASE_DIR,'-6_dB_features_CQT(new)')
CHECKPOINTS_DIR = os.path.join(BASE_DIR,'CQT_Checkpoint','-6_dB_pump_mobilevit_xs_cqt')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCH = 4
SAVE_PLOTS = True

SPECTROGRAM_TYPE = 'cqt'
ENCODER_NAME = 'mobilevit_xxs'
PRETRAINED_ENCODER = True

#----- Setting the device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device} : {torch.cuda.get_device_name()}")

#----- Training Function -----
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    best_val_auc = -1
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    train_accs, val_accs = [], []
    train_bccs, val_bccs = [], []

    #----- Training Loop -----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_preds, all_classes = [], [], []

        print(f"\nEpoch {epoch+1} / {num_epochs}")

        for sample in tqdm(train_loader, desc="Training"):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device)
            # print(f"\nSectrogram batch shape: {spectrograms.shape}")
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)
            probabilities = torch.softmax(outputs, dim=1)[:,1]
            preds_class = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probabilities.cpu().detach().numpy())
            all_classes.extend(preds_class.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_classes)
        bal_acc = balanced_accuracy_score(all_labels, all_classes)
        train_losses.append(epoch_loss)
        train_aucs.append(train_auc)
        train_accs.append(train_acc)
        train_bccs.append(bal_acc)
        print(f"Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Train Accuracy: {train_acc:.4f}, Balanced Accuracy:{bal_acc:.4f}")

        val_loss, val_auc, val_acc, val_bcc, _, _ = evaluate_model(model, val_loader, criterion, "Validation")
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        val_bccs.append(val_bcc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_best_loss.pth'))
            print(f"Model saved (best val loss: {best_val_loss:.4f})")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved AUC: {best_val_auc:.4f}")
        else:
            print(f"Val AUC ({val_auc:.4f}) did not improve from best ({best_val_auc:.4f})")

    if SAVE_PLOTS:
        plt.figure(figsize=(20,5))

        plt.subplot(1,4,1)
        plt.plot(range(1,num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1,num_epochs+1), val_losses,label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,4,2)
        plt.plot(range(1,num_epochs+1), train_aucs, label = 'Train AUC')
        plt.plot(range(1,num_epochs+1), val_aucs, label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Train/Validation AUC')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,4,3)
        plt.plot(range(1,num_epochs+1), train_accs, label = 'Train Acc')
        plt.plot(range(1,num_epochs+1), val_accs, label='Validation Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train/Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,4,4)
        plt.plot(range(1, num_epochs+1), train_bccs, label="Train Balanced Acc")
        plt.plot(range(1,num_epochs+1), val_bccs, label="Val Balanced ACC")
        plt.xlabel('Epoch')
        plt.ylabel('Train/Validation Balanced Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINTS_DIR,f'{ENCODER_NAME}_metrics_curve_{SPECTROGRAM_TYPE}.png'))
        plt.show()

#----- Evaluation Function -----
def evaluate_model(model, data_loader, criterion, phase="Evaluation"):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_classes = [], [], []

    with torch.no_grad():
        for sample in tqdm(data_loader, desc=phase):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * spectrograms.size(0)

            probabilities = torch.softmax(outputs, dim=1)[:,1]
            preds_class = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probabilities.cpu().numpy())
            all_classes.extend(preds_class.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float('nan')
    acc_score = accuracy_score(all_labels, all_classes)
    val_bcc = balanced_accuracy_score(all_labels, all_classes)
    print(f"{phase} Loss:{avg_loss:.4f}, {phase} AUC: {auc_score:.4f}, {phase} Accuracy: {acc_score:.4f}, Balanced Validation Accuracy Score:{val_bcc:.4f}")
    
    print(f"\n[DEBUG] {phase} Prediction Distribution:")
    pred_counter = Counter(all_classes)
    label_counter = Counter(all_labels)
    print(f"Predicted Classes: {dict(pred_counter)}")
    print(f"True Labels:       {dict(label_counter)}")

    print(f"\nSample Predictions vs Labels:")
    for i in range(min(10, len(all_labels))):
        print(f"Sample {i+1}: Pred = {all_classes[i]}, Prob = {all_preds[i]:.4f}, True = {all_labels[i]}")
    
    return avg_loss, auc_score, acc_score, val_bcc, all_labels, all_preds


#----- Partial AUC Calculation -----    
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


#Added the Logic to Check the overlapping Dataset
def check_overlap(dataset, train_idx, val_idx, test_idx):
    train_files = set(dataset.datasets[0].all_file_paths[i] for i in train_idx if i< len(dataset.datasets[0])) | \
                  set(dataset.datasets[1].all_file_paths[i - len(dataset.datasets[0])] for i in train_idx if i >= len(dataset.datasets[0])) 
    
    val_files = set(dataset.datasets[0].all_file_paths[i] for i in val_idx if i < len(dataset.datasets[0])) | \
                set(dataset.datasets[1].all_file_paths[i - len(dataset.datasets[0])] for i in val_idx if i>= len(dataset.datasets[0]))
    
    test_files = set(dataset.datasets[0].all_file_paths[i] for i in test_idx if i < len(dataset.datasets[0])) | \
                 set(dataset.datasets[1].all_file_paths[i - len(dataset.datasets[0])] for i in test_idx if i >= len(dataset.datasets[0]))
    
    print("\n--- Checking Overlap Between Splits ---")
    print("Train ∩ val:", len(train_files & val_files))
    print("Train ∩ Test:", len(train_files & test_files))
    print("Val ∩ Test:", len(val_files & test_files))

    if train_files & val_files or train_files & test_files or val_files & test_files:
        print("Warning: Overlapping files detected between splits!")
    else:
        print("No overlap between Train, Validation, and Test splits.")



#----- Main Training Pipeline -----
def main():
    train_transform = transforms.Compose([
        ZScoreNormalizeSpectrogram(),
        AugmentSpectrogram() 
    ])

    test_transform = transforms.Compose([
        ZScoreNormalizeSpectrogram(),
        AugmentSpectrogram()
    ])

    # === Load all normal and abnormal data first ===
    print(f"Attempting to load normal data for : '{SPECTROGRAM_TYPE}' from: {FEATURES_DIR}")
    try:
        full_normal_dataset_train = SpectrogramDataset(
            data_dir=FEATURES_DIR, category='normal' , transform=train_transform, spec_type=SPECTROGRAM_TYPE
        )
        full_normal_dataset_test = SpectrogramDataset(
            data_dir=FEATURES_DIR,category='normal',transform=test_transform,spec_type=SPECTROGRAM_TYPE
        )
        print(f"Successfully loaded {len(full_normal_dataset_train)} normal samples.")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error in loading normal dataset: {error}")
        return

    print(f"Attempting to load abnormal data for: '{SPECTROGRAM_TYPE}' from: {FEATURES_DIR}")
    try:
        full_abnormal_dataset_train = SpectrogramDataset(
            data_dir=FEATURES_DIR, category='abnormal', transform=train_transform, spec_type=SPECTROGRAM_TYPE
        )

        full_abnormal_dataset_test = SpectrogramDataset(
            data_dir=FEATURES_DIR, category='abnormal',transform=test_transform,spec_type=SPECTROGRAM_TYPE
        )
        print(f"Successfully loaded {len(full_abnormal_dataset_train)} abnormal samples.")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading abnormal dataset: {error}")
        return

    #=== Combine datasets and create the labels ===
    combined_dataset_train = ConcatDataset([full_normal_dataset_train, full_abnormal_dataset_train])
    combined_dataset_test = ConcatDataset([full_normal_dataset_test, full_abnormal_dataset_test])
    combined_labels = full_normal_dataset_train.labels + full_abnormal_dataset_train.labels
    indices = list(range(len(combined_dataset_train)))
    
    sample_label_pairs = list(zip(range(len(full_normal_dataset_train.labels + full_abnormal_dataset_train.labels)), 
                              full_normal_dataset_train.labels + full_abnormal_dataset_train.labels))
    shuffled_pairs = shuffle(sample_label_pairs, random_state=42)

    # Unzip after shuffle
    indices, combined_labels = zip(*shuffled_pairs)
    indices = list(indices)
    combined_labels = list(combined_labels)
    #=== Stratified Split of Train/Val/Test ===
    if len(np.unique(combined_labels)) < 2:
        print(f"Error: Dataset does not contain both normal and abnormal samples. Cannot perform the split")
        return
    
    train_idx, temp_idx, _,_temp_labels = train_test_split(
        indices, combined_labels, test_size=0.3, stratify= combined_labels, random_state=42
    )

    val_idx, test_idx, _, _  = train_test_split(
        temp_idx, _temp_labels,test_size=0.5, stratify=_temp_labels, random_state=42
    )

    #=== Creating Subsets ===
    train_dataset = Subset(combined_dataset_train, train_idx)
    val_dataset = Subset(combined_dataset_test, val_idx)
    test_dataset = Subset(combined_dataset_test, test_idx)

    print(f"Stratified Split Sizes: Train = {len(train_dataset)}, Val = {len(val_dataset)}, Test = {len(test_dataset)}")

    train_labels = [combined_labels[i] for i in train_idx]
    val_labels = [combined_labels[i] for i in val_idx]
    test_labels = [combined_labels[i] for i in test_idx]

    train_count = Counter(train_labels)
    val_count = Counter(val_labels)
    test_count = Counter(test_labels)

    print("\n--- Split Label Distribution (Stratified) ---")
    print(f"Train Set:     Normal = {train_count[0]}, Abnormal = {train_count[1]}, Total = {len(train_labels)}")
    print(f"Validation Set: Normal = {val_count[0]}, Abnormal = {val_count[1]}, Total = {len(val_labels)}")
    print(f"Test Set:      Normal = {test_count[0]}, Abnormal = {test_count[1]}, Total = {len(test_labels)}")

    check_overlap(combined_dataset_train, train_idx, val_idx, test_idx)
    #=== DataLoaders ===
    train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    #=== Model Setup ===
    model =TransformerSpectrogramClassifier(
        model_name=ENCODER_NAME, 
        pretrained=PRETRAINED_ENCODER, 
        num_classes=2,

    ).to(device)
    
    print(f"Model is using encoder: {ENCODER_NAME}, Pretrained: {PRETRAINED_ENCODER}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-2)

    model_save_path = os.path.join(CHECKPOINTS_DIR, f'{ENCODER_NAME}_{SPECTROGRAM_TYPE}_best_model.pth')
    
    #=== Training ===
    train_model(model, train_loader, val_loader, criterion,optimizer,NUM_EPOCH,model_save_path)

    #=== Final Test Evaluation ===
    print("\n---Final Test Evaluation ---")
    
    if not os.path.exists(model_save_path):
        print(f"Error: Best model not saved at {model_save_path}. Cannot perform final evaluation.")
        best_loss_model_path = model_save_path.replace('.pth','_best_loss.pth')
        if os.path.exists(best_loss_model_path):
            print(f"Attempting to load best loss model from: {best_loss_model_path}")
            model.load_state_dict(torch.load(best_loss_model_path,weights_only=True))
            print("Loaded best loss model for final evaluation")
        else:
            print("No best model (AUC or loss) found. Exiting final evaluation.")
        return 
    else:
        model.load_state_dict(torch.load(model_save_path,weights_only=True))
        model.eval()

    _, _, _, _, all_labels_test, all_preds_test = evaluate_model(model,test_loader,criterion, "Test")

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
        plt.show()
    else:
        print("\nFinal Test AUC/pAUC not calculated: Only one class present in the combined test set.")

if __name__ == "__main__":
    main()
