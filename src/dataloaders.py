import numpy as np
import pandas as pd
import ast
import wfdb
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import gc
import os
from scipy.interpolate import interp1d


def encode_label(classes, unique_classes):
    
    """ Convert a list of classes to a one-hot encoded vector.
    
    Args:
    - classes (List[str]): List of classes.
    - unique_classes (List[str]): List of unique classes.
    
    Returns:
    - List: One-hot encoded vector.
    """

    
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    encoded = [0] * len(unique_classes)
    for cls in classes:
        if cls in class_to_index:
            encoded[class_to_index[cls]] = 1
    return encoded

def decode_label(encoded_label, unique_classes):
    
    """ Decode a one-hot vector to its corresponding classes.
    
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
        self.unique_superclasses = sorted(list(set(self.super_classes))) # sorted for reproducibility
            

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
        - int: Total number of samples.
        """
        return len(self.X)

    def get_normalized_signal(self, idx):
        
        x = self.X[idx].copy()
        
        # normalize the data
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        # x = (x - mean) / std # this scales too much the data
        
        
        x = self.random_clip(x, self.L)
        
        return x


    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        Retrieve a single data point.
        
        Args:
        - idx (int): Index of the data point.
        
        Returns:
        - torch.Tensor: Raw ECG signal.
        - torch.Tensor: Corresponding one-hot encoded label.
        """
        x = self.get_normalized_signal(idx)
        
        # perform augmentation if in training mode
        if self.mode == 'train':
            x = self.augment(x)
            
        
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
            if Lmax == x.shape[0]:
                return x
            
            # take random subset of the sequence
            idx_start = torch.randint(0, x.shape[0] - Lmax, (1,)).item()
            idx_end = idx_start + Lmax
            x = x[idx_start:idx_end, :]
    
        return x
    
    def augment(self, x):
        
        
        x = self.resample_augmentation(x)
        x = self.mix_up(x)
        x = self.cut_mix(x)
        x = self.cutout(x)
        x = self.convolve(x)
        x = self.reverse(x)
        x = self.translate(x)

        
        return x
    
    
    
    
    
    def translate(self, x, p=0.5):
        """ Perform translation augmentation on the input signal.
    
        Args:
        - x (np.ndarray): Input signal.
        - p (float): Probability of applying the convolution.
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        x_augmented = np.roll(x, np.random.randint(0, x.shape[0]), axis=0)
        
        return x_augmented
    
    
    def convolve(self, x, p=0.5, window_size=5):
        """ Perform uniform convolution augmentation on the input signal.
    
        Args:
        - x (np.ndarray): Input signal.
        - p (float): Probability of applying the convolution.
        - window_size (int): Size of the convolution window.
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        # Create a uniform window. The sum of all its values should be 1.
        window = np.ones(window_size) / window_size
        
        x_augmented = np.array([np.convolve(xi, window, mode='same') for xi in x.T]).T
        
        return x_augmented
    
    def reverse(self, x, p=0.9):
        
        """ Perform reverse augmentation on the input signal.
        
        Args:
        - x (np.ndarray): Input signal.
        - p (float): Probability of applying reverse
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        # reverse the signal along time dimension
        x_augmented = x[::-1].copy()
        
        return x_augmented
    
    def mix_up(self, x, p=0.9, p_channel=0.7, mixing_factor=0.15):
        
        """ Perform mix_up augmentation on the input signal.
        
        Args:
        - x (np.ndarray): Input signal.
        - p_channel (float): Probability of applying mix_up on a channel.
        - p (float): Probability of applying mix_up on a segment.
        - mixing_factor (float): Mixing factor.
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        x_augmented = x.copy()
        
        # choose another sample randomly
        x_2 = self.get_normalized_signal(np.random.randint(0, self.__len__() - 1))
        
        n_channels = x.shape[1]
        
        for channel in range(n_channels):
            
            if np.random.uniform() < p_channel:
                
                
                
                x_augmented[:, channel] = x_augmented[:, channel] + mixing_factor * x_2[:, channel]
            
        
        return x_augmented
    
    def cut_mix(self, x, p=0.9, p_channel=0.3, cutout_size_range=[0.1, 0.4]):
        
        """ Perform cut_mix augmentation on the input signal.
        
        Args:
        - x (np.ndarray): Input signal.
        - p_channel (float): Probability of applying cutout on a channel.
        - p (float): Probability of applying cutout on a segment.
        - cutout_size_range (List[float]): Range of cutout size.
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        x_augmented = x.copy()
        
        # choose another sample randomly
        x_2 = self.get_normalized_signal(np.random.randint(0, self.__len__() - 1))
        
        n_channels = x.shape[1]
        
        for channel in range(n_channels):
            
            if np.random.uniform() < p_channel:
                

                
                cutout_size = int(np.random.uniform(low=cutout_size_range[0], high=cutout_size_range[1]) * x.shape[0])
                
                start = np.random.randint(0, x.shape[0] - cutout_size)
                end = start + cutout_size
                
                
                x_augmented[start:end, channel] = x_2[start:end, channel]
            
        
        return x_augmented
    
    
    def cutout(self, x, p=0.9, p_channel=0.7, cutout_size_range=[0.1, 0.4]):
        
        """ Perform cutout augmentation on the input signal.
        
        Args:
        - x (np.ndarray): Input signal.
        - p_channel (float): Probability of applying cutout on a channel.
        - p (float): Probability of applying cutout on a segment.
        - cutout_size_range (List[float]): Range of cutout size.
        
        Returns:
        - np.ndarray: Augmented signal.
        """
        
        if np.random.uniform() > p:
            return x
        
        x_augmented = x.copy()
        
        n_channels = x.shape[1]
        
        for channel in range(n_channels):
            
            if np.random.uniform() < p_channel:
                
                cutout_size = int(np.random.uniform(low=cutout_size_range[0], high=cutout_size_range[1]) * x.shape[0])
                
                
                start = np.random.randint(0, x.shape[0] - cutout_size)
                end = start + cutout_size
                
                x_augmented[start:end, channel] = 0
            
        
        return x_augmented
    
    def resample_augmentation(self, x, alpha_range=[0.5, 1.3]):
        """
        Resample the signal.

        Args:
        - x (np.ndarray): Input signal of shape (timesteps, features).
        - alpha_range (float): Resampling factor. Value will be sampled uniformly from this range.
                        Values > 1 speed up, 0 < values < 1 slow down the signal.

        Returns:
        - np.ndarray: Augmented signal.
        """

        # Number of timesteps
        num_timesteps = x.shape[0]

        # Create an interpolating function
        f = interp1d(np.linspace(0, num_timesteps, num_timesteps, endpoint=False),
                    x, axis=0, kind='linear', fill_value='extrapolate')

        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        # Create the new time grid
        stretched_time = np.linspace(0, num_timesteps, int(num_timesteps * alpha), endpoint=False)

        # Apply the function to get the augmented signal
        x_augmented = f(stretched_time)

        # If the signal is shortened, pad it with zeros or truncate if it's longer
        if x_augmented.shape[0] < num_timesteps:
            x_augmented = np.vstack((x_augmented, np.zeros((num_timesteps - x_augmented.shape[0], x.shape[1]))))
        elif x_augmented.shape[0] > num_timesteps:
            x_augmented = x_augmented[:num_timesteps, :]

        return x_augmented
    

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