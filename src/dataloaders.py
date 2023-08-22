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
from tqdm import tqdm


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
                 train_on_ffts: bool = False,
                 nfft_components: int = 192,
                 augment: bool = False,
                 ):
        """
        Initialize the ECGDataset object.
        
        Args:
        - path (str): Path to the data directory.
        - test_folds (List): List of test folds to be used for testing.
        - mode (str): Mode of the dataset. Can be either 'train', 'val', or 'test'.
        - train_on_ffts (bool): If True, train on FFTs instead of raw signals.
        - nfft_components (int): Number of FFT components to use.
        - augment (bool): If True, perform data augmentation.
        """
        
        assert mode in ['train', 'val', 'test'], "Invalid mode: it must be either 'train', 'val', or 'test'."
        self.mode = mode
        self.test_folds = test_folds
        self.train_folds = [i for i in range(1,11) if i not in test_folds]
        self.take_folds = self.train_folds if mode == 'train' else test_folds
        self.L = L
        self.train_on_ffts = train_on_ffts
        self.nfft_components = nfft_components
        self.augment = augment
        
        if self.train_on_ffts:
            self.L = 1000 # take all raw signals and then compute FFTs on the fly
        
        
        print("[INFO] Loading data...")
        
        def aggregate_diagnostic(y_dic) -> list:
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.array:
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            return np.array([signal for signal, meta in data])


        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        print(f"[INFO] Obtaining diagnostic_superclass for {self.mode} mode ...")
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        self.super_classes = [x if len(x) > 0 else '' for x in Y['diagnostic_superclass'].values.tolist()]
        self.unique_superclasses = sorted(list(set(np.concatenate(Y['diagnostic_superclass'].values.tolist()))))
        self.super_classes_encoded = np.array([encode_label(x, self.unique_superclasses) for x in self.super_classes])
        
        # Load X
        self.X = load_raw_data(Y, sampling_rate=100, path=path)
        
        # take "take_folds" folds for X and y
        self.X = self.X[np.where(Y.strat_fold.isin(self.take_folds))]
        self.super_classes_encoded = self.super_classes_encoded[np.where(Y.strat_fold.isin(self.take_folds))]
        
        # remove samples with no superclass
        black_list_indices = np.where(self.super_classes_encoded.sum(axis=1) == 0)[0]
        mask = np.ones(self.super_classes_encoded.shape[0], dtype=bool)
        mask[black_list_indices] = False
        self.X = self.X[mask]
        self.super_classes_encoded = self.super_classes_encoded[mask]
        
        # print some info
        print(f"X.shape: {self.X.shape}, y.shape: {self.super_classes_encoded.shape}")
        print(f"X.min: {self.X.min()}, X.max: {self.X.max()}")
        
        # Calculate sample weights
        if self.mode == 'train':
            self.sample_weights = self.calculate_sample_weights()
            
        # release memory
        del Y, agg_df
        gc.collect()
        
            

    def calculate_sample_weights(self):
        
        # self.super_classes_encoded is (B, 5) shape
        
        # Count occurrences of each superclass
        counts = np.sum(self.super_classes_encoded, axis=0) # (5,)
        class_weights = 1.0 / (counts + 1e-6)  # Compute inverse frequency with a small constant to avoid division by zero. Shape: (5,)

        # get sample weights
        sample_weights = self.super_classes_encoded @ class_weights

        return torch.from_numpy(sample_weights).float()  # Convert to PyTorch tensor

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
        # mean = np.mean(x, axis=0, keepdims=True)
        # std = np.std(x, axis=0, keepdims=True)
        # x = (x - mean) / std # this scales too much the data
        
        n_channels = x.shape[1]
        for ch in range(n_channels):
            x[:, ch] = (x[:, ch] - x[:, ch].min()) / (x[:, ch].max() - x[:, ch].min() + 1e-6) + 0.5
        
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
        
        y = self.super_classes_encoded[idx]
        
        # if len(np.unique(y)) < 2:
        #     # skip this sample and get another one
        #     return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        
        x = self.get_normalized_signal(idx)
        
        # perform augmentation if in training mode
        if self.mode == 'train' and self.augment and not self.train_on_ffts:
            x = self.augment_samples(x)
            
        if self.train_on_ffts:
            num_channels = x.shape[1]
            x_new = []
            
            for channel in range(num_channels):
                x_channel = x[:, channel]

            
                # remove k=0 component
                x_channel = x_channel - np.mean(x_channel) 
                x_fft = np.fft.fftn(x_channel)
                # keep only the first nfft_components
                x_fft_keep = x_fft[:self.nfft_components]
                # normalize
                mean = np.mean(x_fft_keep)
                std = np.std(x_fft_keep)
                x_fft_keep = (x_fft_keep - mean) / std
                x_new.append(x_fft_keep)
        
        
            x = np.array(x_new).T
            
        
        
        out =  {
            'x': torch.tensor(x, dtype=torch.float32)
        },{ 'y': torch.tensor(y, dtype=torch.float32)}
        
        return out

    def random_clip(self, x, Lmax):
        
        if Lmax is not None and Lmax < 1000:
            if Lmax == x.shape[0]:
                return x
            
            # take random subset of the sequence
            idx_start = torch.randint(0, x.shape[0] - Lmax, (1,)).item()
            idx_end = idx_start + Lmax
            x = x[idx_start:idx_end, :]
    
        return x
    
    def augment_samples(self, x):
        
        
        # x = self.resample_augmentation(x)
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
    
    def mix_up(self, x, p=0.9, p_channel=0.3, mixing_factor=0.15):
        
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