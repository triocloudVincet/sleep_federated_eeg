# sleep_federated/data_processing/augmentation.py
import numpy as np
from scipy import signal
from typing import Tuple, List, Dict
import logging
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader

class AugmentationPipeline:
    """Pipeline for EEG signal augmentation."""
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)

    def add_gaussian_noise(self, signal: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to the signal."""
        noise = np.random.normal(0, noise_factor * np.std(signal), signal.shape)
        return signal + noise

    def time_shift(self, signal: np.ndarray, max_shift: float = 0.2) -> np.ndarray:
        """Apply random time shift."""
        shift = int(len(signal) * max_shift * np.random.uniform(-1, 1))
        return np.roll(signal, shift)

    def scaling(self, signal: np.ndarray, scaling_factor: float = 0.2) -> np.ndarray:
        """Apply random amplitude scaling."""
        scale = 1 + np.random.uniform(-scaling_factor, scaling_factor)
        return signal * scale

    def random_crop(self, signal: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """Randomly crop and resize back to original length."""
        orig_len = len(signal)
        crop_len = int(orig_len * crop_ratio)
        start = np.random.randint(0, orig_len - crop_len)
        cropped = signal[start:start + crop_len]
        # Interpolate back to original length
        x = np.linspace(0, 1, crop_len)
        x_new = np.linspace(0, 1, orig_len)
        f = interp1d(x, cropped, kind='cubic')
        return f(x_new)

    def frequency_mask(self, signal: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
        """Apply frequency domain masking."""
        f_signal = np.fft.rfft(signal)
        f_size = len(f_signal)
        num_mask = int(f_size * mask_ratio)
        mask_start = np.random.randint(0, f_size - num_mask)
        f_signal[mask_start:mask_start + num_mask] = 0
        return np.fft.irfft(f_signal, len(signal))

    def augment_signal(self, signal: np.ndarray, augmentation_types: List[str] = None) -> np.ndarray:
        """Apply multiple augmentations to a signal."""
        if augmentation_types is None:
            augmentation_types = ['noise', 'shift', 'scale', 'crop', 'freq_mask']
        
        augmented = signal.copy()
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                augmented = self.add_gaussian_noise(augmented)
            elif aug_type == 'shift':
                augmented = self.time_shift(augmented)
            elif aug_type == 'scale':
                augmented = self.scaling(augmented)
            elif aug_type == 'crop':
                augmented = self.random_crop(augmented)
            elif aug_type == 'freq_mask':
                augmented = self.frequency_mask(augmented)
                
        return augmented

class BalancedSleepDataset(Dataset):
    """Dataset with balanced class distribution through augmentation."""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray, 
                 target_samples_per_class: int = 20):
        self.augmenter = AugmentationPipeline()
        self.signals, self.labels = self._balance_dataset(
            signals, labels, target_samples_per_class
        )
        
    def _balance_dataset(self, signals: np.ndarray, labels: np.ndarray,
                        target_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using augmentation."""
        unique_labels = np.unique(labels)
        balanced_signals = []
        balanced_labels = []
        
        for label in unique_labels:
            # Get original samples for this class
            mask = labels == label
            class_signals = signals[mask]
            n_samples = len(class_signals)
            
            # Add original samples
            balanced_signals.append(class_signals)
            balanced_labels.extend([label] * n_samples)
            
            # Add augmented samples if needed
            if n_samples < target_samples:
                n_augment = target_samples - n_samples
                augmented = []
                
                for i in range(n_augment):
                    # Select random sample to augment
                    idx = np.random.randint(n_samples)
                    
                    # Apply random augmentations
                    augs = np.random.choice(
                        ['noise', 'shift', 'scale', 'crop', 'freq_mask'],
                        size=np.random.randint(1, 4),
                        replace=False
                    )
                    aug_signal = self.augmenter.augment_signal(
                        class_signals[idx], augmentation_types=augs
                    )
                    augmented.append(aug_signal)
                
                balanced_signals.append(np.stack(augmented))
                balanced_labels.extend([label] * n_augment)
        
        # Combine and shuffle
        balanced_signals = np.concatenate(balanced_signals)
        balanced_labels = np.array(balanced_labels)
        
        # Shuffle
        idx = np.random.permutation(len(balanced_labels))
        return balanced_signals[idx], balanced_labels[idx]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = torch.FloatTensor(self.signals[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return signal, label

def create_balanced_dataloaders(signals: np.ndarray, labels: np.ndarray,
                              n_clients: int, batch_size: int = 32,
                              target_samples_per_class: int = 20) -> Dict[int, DataLoader]:
    """Create balanced dataloaders for federated learning."""
    # Create balanced dataset
    dataset = BalancedSleepDataset(signals, labels, target_samples_per_class)
    
    # Split data among clients
    total_samples = len(dataset)
    samples_per_client = total_samples // n_clients
    
    client_dataloaders = {}
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < n_clients - 1 else total_samples
        
        indices = list(range(start_idx, end_idx))
        subset = torch.utils.data.Subset(dataset, indices)
        
        client_dataloaders[i] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
    
    return client_dataloaders