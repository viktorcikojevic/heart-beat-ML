import numpy as np
import pandas as pd
import ast
import wfdb
import torch
from torch.utils.data import Dataset



def encode_label(classes: list, unique_classes: list) -> list:
    """
    Convert a list of classes to a one-hot encoded vector.
    
    Args:
    - classes (list): List of classes.
    - unique_classes (list): List of unique classes.
    
    Returns:
    - list: One-hot encoded vector.
    """
    
    
    
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    encoded = [0] * len(unique_classes)
    for cls in classes:
        if cls in class_to_index:
            encoded[class_to_index[cls]] = 1
    return encoded

def decode_label(encoded_label: list, unique_classes: list) -> list:
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
    
    def __init__(self, path: str, sampling_rate: int):
        """
        Initialize the ECGDataset object.
        
        Args:
        - path (str): Path to the data directory.
        - sampling_rate (int): Sampling rate of the ECG signals.
        """
        print("[INFO] Loading data...")
        self.Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        self.agg_df = agg_df[agg_df.diagnostic == 1]

        print("[INFO] Obtaining diagnostic_superclass ...")
        self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(self.aggregate_diagnostic)
        
        self.super_classes = [x[0] if len(x) > 0 else '' for x in self.Y['diagnostic_superclass'].values.tolist()]
        self.unique_superclasses = list(set(self.super_classes))
        
        self.X = self.load_raw_data(self.Y, sampling_rate, path)

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
        x = (x - mean) / std
        
        
        classes = self.super_classes[idx]
        classes_encoded = encode_label([classes], self.unique_superclasses)
                
        out = {
            'x': torch.tensor(x, dtype=torch.float32),
            'classes': [classes],
            'y': torch.tensor(classes_encoded, dtype=torch.long),
        }
        
        return out

    def load_raw_data(self, df: pd.DataFrame, sampling_rate: int, path: str) -> np.array:
        """
        Load raw signal data.
        
        Args:
        - df (pd.DataFrame): DataFrame with ECG data.
        - sampling_rate (int): Sampling rate of the ECG signals.
        - path (str): Path to the data directory.
        
        Returns:
        - np.array: Array of raw ECG signals.
        """
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        return np.array([signal for signal, meta in data])

    def aggregate_diagnostic(self, y_dic: dict) -> list:
        """
        Aggregate diagnostic superclass.
        
        Args:
        - y_dic (dict): Dictionary of scp_codes.
        
        Returns:
        - list: Aggregated diagnostic superclass.
        """
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

