import torch.nn as nn

def binary_cross_entropy(preds, targets):
    return nn.BCEWithLogitsLoss()(preds, targets['y'])
