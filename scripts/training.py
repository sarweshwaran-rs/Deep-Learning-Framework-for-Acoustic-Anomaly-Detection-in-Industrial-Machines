import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from utils.datasets import SpectrogramDataset, ZScoreNormalizeSpectrogram
from models.cnn_encoders import BasicSpectrogramClassifier


BASE_DIR = r'F:\Capstone\DFCA'
FEATURES_DIR = os.path.join(BASE_DIR,'data','features')
CHECKPOINTS_DIR = os.path.join(r'F:\CapStone\DFCA','checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCH = 5

SPECTROGRAM_TYPE = 'stft'
ENCODER_NAME = 'efficient_b0'
PRETRAINED_ENCODER = True


TRAIN_NORMAL_RATIO = 0.7
VAL_NORMAL_RATIO = 0.15
TEST_NORMAL_RATIO = 0.15

TRAIN_ABNORMAL_RATIO = 0.7
VAL_ABNORMAL_RATIO = 0.15
TEST_ABNORMAL_RATIO = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    best_val_auc = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # all_labels = []
        # all_preds = []

        print(f"\nEpoch {epoch+1} / {num_epochs}")

        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)

            #For AUC calculation
            probabilities = torch.softmax(outputs, dim=1)[:,1]
            # all_labels.extend(labels.cpu().numpy())
            # all_preds.extend(probabilities.cpu().detach().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        # train_auc = roc_auc_score(all_labels, all_preds)
        print(f"Train Loss: {epoch_loss:.4f}")#, Train AUC: {train_auc:.4f}")

        val_loss, val_auc, _, _ = evaluate_model(model, val_loader, criterion, "Validation")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved AUC: {best_val_auc:.4f}")
        else:
            print(f"Val AUC ({val_auc:.4f}) did not improve from best ({best_val_auc:.4f})")


def evaluate_model(model, data_loader, criterion, phase="Evaluation"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), total=len(data_loader), desc=phase):
            spectrograms = sample['spectrogram'].to(device)
            labels = sample['label'].to(device)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * spectrograms.size(0)

            probabilities = torch.softmax(outputs, dim=1)[:,1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probabilities.cpu().numpy())

        avg_loss = running_loss / len(data_loader.dataset)

        if len(np.unique(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_preds)
        else:
            auc_score = float('nan')
        
        print(f"{phase} Loss:{avg_loss:.4f}, {phase} AUC: {auc_score:.4f}")
        return avg_loss, auc_score, all_labels, all_preds
    
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
    fpr_filtered = fpr[mask]
    tpr_filtered = tpr[mask]

    #Add(max_fpr, inerpolated_tpr) if max_fpr is not exactly in fpr_filtered
    if fpr_filtered.max() < max_fpr:
        if len(fpr) < 2:
            return 0.0
        
        #Find the two closest points around max_fpr
        idx_less_equal = np.where(fpr <= max_fpr)[0]
        if( len(idx_less_equal) == 0):
            return 0.0
        
        idx = idx_less_equal[-1]

        #Check if we are at the end of curve or need interpolation
        if idx + 1 < len(fpr):
            x1, y1 = fpr[idx], tpr[idx]
            x2, y2 = fpr[idx+1], tpr[idx+1]
            #Linear interpolation for TPR at max_fpr
            if (x2 - x1) > 0:
                tpr_at_max_fpr = y1 + (y2 - y1) * (max_fpr - x1) / (x2-x1)
            else:
                tpr_at_max_fpr = y1

            fpr_filtered = np.append(fpr_filtered, max_fpr)
            tpr_filtered = np.append(tpr_filtered, tpr_at_max_fpr)
        
            #Sort to ensure correct order for AUC calculation
            sort_indices = np.argsort(fpr_filtered)
            fpr_filtered = fpr_filtered[sort_indices]
            tpr_filtered = tpr_filtered[sort_indices]
    elif fpr_filtered.size == 0 and max_fpr > 0:
        return 0.0
    elif fpr_filtered.size == 0 and max_fpr == 0:
        return 0.0
    
    
    #calculate AUC from the filtered points
    if len(fpr_filtered) < 2:
        return 0.0
    
    pauc_score = auc(fpr_filtered, tpr_filtered) / max_fpr
    return pauc_score

def main():
    transform = ZScoreNormalizeSpectrogram()

    # === Load all normal and abnormal data first ===
    print(f"Attempting to load normal data from : {os.path.join(FEATURES_DIR,'normal',SPECTROGRAM_TYPE)}")
    try:
        full_normal_dataset = SpectrogramDataset(
            data_dir=FEATURES_DIR, category='normal' , transform=transform, spec_type=SPECTROGRAM_TYPE
        )
        print(f"Successfully loaded {len(full_normal_dataset)} normal samples.")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error in loading normal dataset: {error}")
        return

    print(f"Attempting to load abnormal data from: {os.path.join(FEATURES_DIR,'abnormal',SPECTROGRAM_TYPE)}")
    try:
        full_abnormal_dataset = SpectrogramDataset(
            data_dir=FEATURES_DIR, category='abnormal', transform=transform, spec_type=SPECTROGRAM_TYPE
        )
        print(f"Successfully loaded {len(full_abnormal_dataset)} abnormal samples.")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading abnormal dataset: {error}")
        return

    #==== Normal Splitting ====
    num_normal = len(full_normal_dataset)
    train_normal_size = int(TRAIN_NORMAL_RATIO * num_normal)
    val_normal_size = int(VAL_NORMAL_RATIO * num_normal)
    test_normal_size = num_normal - train_normal_size - val_normal_size

    train_normal_dataset, val_normal_dataset,test_normal_dataset = random_split(
        full_normal_dataset, [train_normal_size, val_normal_size,test_normal_size],
        generator = torch.Generator().manual_seed(42)
    )

    print(f"Normal data split: Train = {len(train_normal_dataset)}, val = {len(val_normal_dataset)}, Test = {len(test_normal_dataset)}")

    #=== Abnormal Splitting ===
    num_abnormal = len(full_abnormal_dataset)
    train_abnormal_size = int(TRAIN_ABNORMAL_RATIO * num_abnormal)
    val_abnormal_size = int(VAL_ABNORMAL_RATIO * num_abnormal)
    test_abnormal_size = num_abnormal - train_abnormal_size - val_abnormal_size

    train_abnormal_dataset, val_abnormal_dataset, test_abnormal_dataset = random_split(
        full_abnormal_dataset, [train_abnormal_size,val_abnormal_size, test_abnormal_size],
        generator = torch.Generator().manual_seed(42)
    )
    print(f"Abnormal data split: Train = {len(train_abnormal_dataset)}, val = {len(val_abnormal_dataset)}, Test = {len(test_abnormal_dataset)}")


    # === 3. Define Loaders by combining the normal + abnormal ===#
    combined_train_dataset = ConcatDataset([train_normal_dataset, train_abnormal_dataset])
    train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Training Loader Contains {len(train_normal_dataset)} normal + {len(train_abnormal_dataset)} abnormal samples.")

    combined_val_dataset = ConcatDataset([val_normal_dataset, val_abnormal_dataset])
    val_loader = DataLoader(combined_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Validation Loader contains {len(val_normal_dataset)} normal + {len(val_abnormal_dataset)} abnormal samples.")


    model = BasicSpectrogramClassifier(
        encoder_name=ENCODER_NAME, 
        pretrained=PRETRAINED_ENCODER, 
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model_save_path = os.path.join(CHECKPOINTS_DIR, f'{ENCODER_NAME}_{SPECTROGRAM_TYPE}_best_model.pth')
    
    train_model(model, train_loader, val_loader, criterion,optimizer,NUM_EPOCH,model_save_path)

    print("\n---Final Test Evaluation ---")
    
    if not os.path.exists(model_save_path):
        print(f"Error: Best model not saved at {model_save_path}. Cannot perform final evaluation.")
        print("This usually happens if Val AUC never improved (e.g., always NaN or always below initial -1).")
        print("Ensure your validation set is balanced (contains both normal and abnormal samples).")
        return 

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    combined_test_dataset = ConcatDataset([test_normal_dataset, test_abnormal_dataset])
    test_loader_combined = DataLoader(combined_test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    print(f"Combined Test Loader Contains {len(test_normal_dataset)} normal + {len(test_abnormal_dataset)} abnormal samples.")


    _,_,all_labels_test, all_preds_test = evaluate_model(model,test_loader_combined,criterion, "Test")

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
        plt.savefig(os.path.join(CHECKPOINTS_DIR, f'{ENCODER_NAME}_roc_curve_{SPECTROGRAM_TYPE}.png'))
        plt.show()
    else:
        print("\nFinal Test AUC/pAUC not calculated: Only one class present in the combined test set.")

if __name__ == "__main__":
    main()
