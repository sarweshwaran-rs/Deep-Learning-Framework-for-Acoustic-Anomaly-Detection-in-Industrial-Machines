import os
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns 

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


def plot_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    """
        Plots a confusion matrix for model evaluation
    Args:
        y_true (list or np.array): Ground truth labels.
        y_pred (list or np.array): Predicted labels.
        labels (list): A list of labels for the matrix axes (['Normal', 'Abnormal'])
        title (str): Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn} | Precision: {precision:.4f} | Recall: {recall:.4f} | Specificity: {specificity:.4f}")
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Confusion Matrix.png"))
    plt.show()
    plt.close()
