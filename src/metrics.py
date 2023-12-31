import torch
from sklearn.metrics import hamming_loss, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def multi_label_accuracy(y_pred, targets, threshold=0.5, apply_sigmoid=True):
    """
    Compute multi-label accuracy.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    targets: Tensor of shape [batch_size, n_classes] - True multilabels
    threshold: Threshold for converting logits to binary labels. Defaults to 0.5.
    """
    if apply_sigmoid:
        y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()  # Convert logits to 0 or 1
    else:
        y_pred_bin = (y_pred > threshold).float()  # Convert logits to 0 or 1
        
    
    targets = targets['y']
    
    correct_predictions = (y_pred_bin == targets).float()
    accuracy_per_sample = correct_predictions.sum(dim=1) / targets.size(1)  # Average over labels
    
    return accuracy_per_sample.mean()  # Average over samples




def multilabel_hamming_loss(y_pred, targets, threshold=0.5, apply_sigmoid=True):
    if apply_sigmoid:
        y_pred_bin = (torch.sigmoid(y_pred) > threshold).cpu().numpy()
    else:
        y_pred_bin = (y_pred > threshold).cpu().numpy()
    return hamming_loss(targets['y'].cpu().numpy(), y_pred_bin)


def macro_f1_score_multilabel(y_pred, y_true, threshold=0.5, apply_sigmoid=True):
    """
    Compute the macro averaged F1 score for multilabel classification.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    y_true: Tensor of shape [batch_size, n_classes] - True multilabels
    threshold: Threshold for converting logits to binary labels. Defaults to 0.5.
    """
    if apply_sigmoid:
        y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()
    else:
        y_pred_bin = (y_pred > threshold).float()
    
    return f1_score(y_true['y'].cpu().numpy(), y_pred_bin.cpu().numpy(), average='macro', zero_division=1)


def pr_auc_multilabel(y_pred, y_true, apply_sigmoid=True):
    """
    Compute the macro averaged Precision-Recall Area Under Curve (PR AUC) for multilabel classification.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    y_true: Tensor of shape [batch_size, n_classes] - True multilabels
    """
    n_classes = y_pred.size(1)
    
    # Convert logits to probabilities using sigmoid
    if apply_sigmoid:
        y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
    else:
        y_pred_prob = y_pred.cpu().numpy()
    
    y_true_bin = y_true['y'].cpu().numpy()
    
    # Compute PR AUC for each class and then average
    average_precisions = []
    for i in range(n_classes):
        average_precisions.append(average_precision_score(y_true_bin[:, i], y_pred_prob[:, i]))
    
    # Return the macro average PR AUC
    return sum(average_precisions) / n_classes



def auroc_multilabel(y_pred, y_true, apply_sigmoid=True):
    """
    Compute the macro averaged Area Under the Receiver Operating Characteristic curve (AUROC) for multilabel classification.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    y_true: Tensor of shape [batch_size, n_classes] - True multilabels
    """
    # Convert logits to probabilities using sigmoid
    if apply_sigmoid:
        y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
    else:
        y_pred_prob = y_pred.cpu().numpy()
    
    y_true_bin = y_true['y'].cpu().numpy()
    
    # Compute macro-average AUROC
    return roc_auc_score(y_true_bin, y_pred_prob, average="macro")