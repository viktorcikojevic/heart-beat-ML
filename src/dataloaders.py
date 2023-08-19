import numpy as np
import pandas as pd
import ast
import wfdb
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import gc
import os

def encode_label(classes: List, unique_classes: List) -> List:
    """
    Convert a list of classes to a one-hot encoded vector.
    
    Args:
    - classes (List): List of classes.
    - unique_classes (List): List of unique classes.
    
    Returns:
    - List: One-hot encoded vector.
    """
    
    
    
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    encoded = [0] * len(unique_classes)
    for cls in classes:
        if cls in class_to_index:
            encoded[class_to_index[cls]] = 1
    return encoded

def decode_label(encoded_label: List, unique_classes: List) -> List:
    """
    Decode a one-hot vector to its corresponding classes.
    
    Args:
    - encoded_label (list): One-hot encoded vector.
    - unique_classes (list): List of unique classes.
    
    Returns:
    - list: Corresponding classes.
    """
    if sum(encoded_label) == 0:
        return ['']
    return [unique_classes[i] for i, val in enumerate(encoded_label) if val == 1]



class ECGDataset(Dataset):
    """
    ECGDataset is a custom PyTorch Dataset class for ECG data.
    It loads the data from the PTB-XL dataset.
    
    """
    
    def __init__(self, 
                 path: str, 
                 test_folds: List = [9],
                 mode: str = 'train',
                 L: int = 512,
                 ):
        """
        Initialize the ECGDataset object.
        
        Args:
        - path (str): Path to the data directory.
        - test_folds (List): List of test folds to be used for testing.
        - mode (str): Mode of the dataset. Can be either 'train', 'val', or 'test'.
        """
        
        assert mode in ['train', 'val', 'test'], "Invalid mode: it must be either 'train', 'val', or 'test'."
        self.mode = mode
        self.test_folds = test_folds
        self.train_folds = [i for i in range(1,11) if i not in test_folds]
        self.take_folds = self.train_folds if mode == 'train' else test_folds
        self.L = L
        
        print("[INFO] Loading data...")
        files = os.listdir(path)
        files = [os.path.join(path, f) for f in files]
        
        
        X_files = []
        y_files = []
        for fold in self.take_folds:
             X_files.append(os.path.join(path,f"X_fold_{fold}.npy"))
             y_files.append(os.path.join(path,f"superclasses_{fold}.npy"))
        
        
        # Load numpy files
        self.X = np.concatenate([np.load(f) for f in X_files])
        self.super_classes = np.concatenate([np.load(f) for f in y_files])
        self.unique_superclasses = list(set(self.super_classes))
        

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
        - int: Total number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieve a single data point.
        
        Args:
        - idx (int): Index of the data point.
        
        Returns:
        - torch.Tensor: Raw ECG signal.
        - torch.Tensor: Corresponding one-hot encoded label.
        """
        x = self.X[idx]
        
        # normalize the data
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        # x = (x - mean) / std # this scales too much the data
        
        # perform augmentation if in training mode
        if self.mode == 'train':
            x = self.augment(x)
            
        x = self.random_clip(x, self.L)
        
        classes = self.super_classes[idx]
        classes_encoded = encode_label([classes], self.unique_superclasses)
                
        # out =  {
        #     'x': torch.tensor(x, dtype=torch.float32),
        #     'y': torch.tensor(classes_encoded, dtype=torch.float32)
        # }
        
        
        out =  {
            'x': torch.tensor(x, dtype=torch.float32)
        },{ 'y': torch.tensor(classes_encoded, dtype=torch.float32)}
        
        # x = torch.tensor(x, dtype=torch.float32)
        # y = torch.tensor(classes_encoded, dtype=torch.float32)
        # return x, y
        
        return out

    def random_clip(self, x, Lmax):
        
        if Lmax is not None:
            # take random subset of the sequence
            idx_start = torch.randint(0, x.shape[0] - Lmax, (1,)).item()
            idx_end = idx_start + Lmax
            x = x[idx_start:idx_end, :]
    
        return x
    
    def augment(self, x):
        
        return x
    
    
    

def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:
    def __init__(self, dataloader, device="cuda"):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)