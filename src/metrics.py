import torch
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score

def multi_label_accuracy(y_pred, targets, threshold=0.5):
    """
    Compute multi-label accuracy.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    targets: Tensor of shape [batch_size, n_classes] - True multilabels
    threshold: Threshold for converting logits to binary labels. Defaults to 0.5.
    """
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()  # Convert logits to 0 or 1
    
    targets = targets['y']
    
    correct_predictions = (y_pred_bin == targets).float()
    accuracy_per_sample = correct_predictions.sum(dim=1) / targets.size(1)  # Average over labels
    
    return accuracy_per_sample.mean()  # Average over samples




def multilabel_hamming_loss(y_pred, targets, threshold=0.5):
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).cpu().numpy()
    return hamming_loss(targets['y'].cpu().numpy(), y_pred_bin)


def macro_f1_score_multilabel(y_pred, y_true, threshold=0.5):
    """
    Compute the macro averaged F1 score for multilabel classification.
    
    y_pred: Tensor of shape [batch_size, n_classes] - Raw logits from the model
    y_true: Tensor of shape [batch_size, n_classes] - True multilabels
    threshold: Threshold for converting logits to binary labels. Defaults to 0.5.
    """
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()  # Convert logits to 0 or 1
    
    return f1_score(y_true['y'].cpu().numpy(), y_pred_bin.cpu().numpy(), average='macro', zero_division=1)