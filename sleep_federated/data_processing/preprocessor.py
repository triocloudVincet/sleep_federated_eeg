# sleep_federated/data_processing/preprocessor.py
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
import logging

class EEGPreprocessor:
    def __init__(self, sampling_rate: int = 100, epoch_length: int = 30):
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.logger = logging.getLogger(self.__class__.__name__)

    def bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 30.0) -> np.ndarray:
        nyquist = self.sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """Normalize signal using z-score normalization."""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def create_epochs(self, data: np.ndarray) -> np.ndarray:
        """Create fixed-length epochs from continuous data."""
        samples_per_epoch = self.epoch_length * self.sampling_rate
        n_epochs = len(data) // samples_per_epoch
        epochs = data[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
        
        # Verify shape
        self.logger.info(f"Created epochs with shape: {epochs.shape}")
        return epochs

    def process_signal(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Complete signal processing pipeline."""
        try:
            # Apply bandpass filter
            filtered = self.bandpass_filter(data)
            self.logger.info("Applied bandpass filter")
            
            # Normalize signal
            normalized = self.normalize_signal(filtered)
            self.logger.info("Normalized signal")
            
            # Create epochs
            epochs = self.create_epochs(normalized)
            self.logger.info(f"Created {len(epochs)} epochs")
            
            # Calculate signal statistics
            stats = {
                'mean': float(np.mean(epochs)),
                'std': float(np.std(epochs)),
                'min': float(np.min(epochs)),
                'max': float(np.max(epochs))
            }
            
            return epochs, stats
            
        except Exception as e:
            self.logger.error(f"Error in signal processing: {str(e)}")
            raise