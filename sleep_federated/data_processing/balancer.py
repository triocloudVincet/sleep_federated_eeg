# sleep_federated/data_processing/balancer.py
import numpy as np
from sklearn.utils import resample
from typing import Tuple, List, Dict
import logging

class DataBalancer:
    """Balance data through augmentation and resampling."""
    
    def __init__(self, target_samples_per_class: int = 20):
        self.target_samples = target_samples_per_class
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def sliding_window_augment(self, signal: np.ndarray, window_size: int = 100) -> List[np.ndarray]:
        """Create additional samples using sliding window."""
        augmented = []
        for i in range(0, len(signal) - window_size, window_size // 2):
            window = signal[i:i + window_size]
            if len(window) == window_size:
                augmented.append(window)
        return augmented

    def add_noise(self, signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to signal."""
        noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
        return signal + noise

    def time_warp(self, signal: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping augmentation."""
        length = len(signal)
        warp = np.random.normal(0, sigma, size=length)
        warp = np.cumsum(warp)  # Convert to cumulative sum
        warp = warp / warp[-1]  # Normalize
        
        original_points = np.linspace(0, 1, length)
        warped_points = np.clip(original_points + warp, 0, 1)
        
        warped_signal = np.interp(original_points, warped_points, signal)
        return warped_signal

    def balance_dataset(self, signals: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset through augmentation and resampling."""
        unique_labels = np.unique(labels)
        balanced_signals = []
        balanced_labels = []
        
        # Get count of majority class if it's less than target
        max_samples = max(self.target_samples, max([sum(labels == label) for label in unique_labels]))
        
        for label in unique_labels:
            # Get signals for this class
            class_signals = signals[labels == label]
            n_samples = len(class_signals)
            
            self.logger.info(f"Class {label}: Original samples = {n_samples}")
            
            if n_samples < max_samples:
                # Number of samples to generate
                n_generate = max_samples - n_samples
                
                # Generate new samples through augmentation
                augmented_signals = []
                
                for signal in class_signals:
                    # Apply different augmentation techniques
                    aug1 = self.add_noise(signal)
                    aug2 = self.time_warp(signal)
                    
                    # Sliding window augmentation
                    aug3 = self.sliding_window_augment(signal)
                    
                    augmented_signals.extend([aug1, aug2])
                    augmented_signals.extend(aug3)
                
                # Convert to array and shuffle
                augmented_signals = np.array(augmented_signals)
                np.random.shuffle(augmented_signals)
                
                # Select required number of augmented samples
                if len(augmented_signals) > n_generate:
                    augmented_signals = augmented_signals[:n_generate]
                
                # Combine original and augmented samples
                balanced_signals.append(np.concatenate([class_signals, augmented_signals]))
                balanced_labels.extend([label] * (n_samples + len(augmented_signals)))
                
                self.logger.info(f"Class {label}: Added {len(augmented_signals)} augmented samples")
            else:
                # Downsample majority class
                balanced_signals.append(resample(class_signals, n_samples=max_samples, random_state=42))
                balanced_labels.extend([label] * max_samples)
                
                self.logger.info(f"Class {label}: Downsampled to {max_samples} samples")
        
        # Combine all classes
        balanced_signals = np.concatenate(balanced_signals, axis=0)
        balanced_labels = np.array(balanced_labels)
        
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(balanced_labels))
        balanced_signals = balanced_signals[shuffle_idx]
        balanced_labels = balanced_labels[shuffle_idx]
        
        return balanced_signals, balanced_labels

# Update DataDistributor to use the balancer
class DataDistributor:
    def __init__(self, n_clients: int, target_samples_per_class: int = 20):
        self.n_clients = n_clients
        self.balancer = DataBalancer(target_samples_per_class)
        self.logger = logging.getLogger(self.__class__.__name__)

    def distribute_data(self, signals: np.ndarray, labels: np.ndarray, 
                       batch_size: int = 32) -> Dict[int, torch.utils.data.DataLoader]:
        """Distribute balanced data among clients."""
        # Balance the overall dataset first
        balanced_signals, balanced_labels = self.balancer.balance_dataset(signals, labels)
        
        # Split balanced data among clients
        n_samples = len(balanced_labels)
        client_size = n_samples // self.n_clients
        
        client_data = {}
        
        for i in range(self.n_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < self.n_clients - 1 else n_samples
            
            client_signals = balanced_signals[start_idx:end_idx]
            client_labels = balanced_labels[start_idx:end_idx]
            
            # Create dataset
            dataset = SleepDataset(client_signals, client_labels)
            
            # Create dataloader
            client_data[i] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            
            # Log distribution
            unique, counts = np.unique(client_labels, return_counts=True)
            distribution = {f"class_{label}": count for label, count in zip(unique, counts)}
            self.logger.info(f"Client {i} balanced distribution: {distribution}")
        
        return client_data