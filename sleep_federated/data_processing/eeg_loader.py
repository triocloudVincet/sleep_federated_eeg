# sleep_federated/data_processing/eeg_loader.py
import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import wget
import os

class EEGDataLoader:
    """Handle loading and initial processing of Sleep-EDF data."""
    
    def __init__(self, data_dir: str, target_channels: List[str] = None):
        self.data_dir = Path(data_dir)
        self.target_channels = target_channels or ['EEG Fpz-Cz', 'EEG Pz-Oz']
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_directories()
        
        # Sleep stage mapping
        self.stage_map = {
            'Sleep stage W': 0,    # Wake
            'Sleep stage 1': 1,    # N1
            'Sleep stage 2': 2,    # N2
            'Sleep stage 3': 3,    # N3
            'Sleep stage 4': 3,    # N4 (combine with N3)
            'Sleep stage R': 4,    # REM
            'Movement time': 0,    # Map movement to wake
            'Sleep stage ?': None  # Invalid/unknown stage
        }
        
    def setup_directories(self):
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_sleep_edf(self, num_files: int = 2):
        """Download Sleep-EDF dataset files."""
        base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
        files_to_download = [
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            "SC4002E0-PSG.edf",
            "SC4002EC-Hypnogram.edf",
        ][:num_files]
        
        for filename in files_to_download:
            file_path = self.data_dir / filename
            if not file_path.exists():
                url = f"{base_url}{filename}"
                self.logger.info(f"Downloading {filename}...")
                wget.download(url, str(file_path))
                self.logger.info(f"Downloaded {filename}")
                
    def load_edf_file(self, file_path: Path) -> Optional[mne.io.Raw]:
        """Load an EDF file using MNE."""
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            self.logger.info(f"Successfully loaded {file_path}")
            return raw
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None
            
    def load_hypnogram(self, hypno_file: Path) -> Optional[np.ndarray]:
        """Load and process hypnogram file."""
        try:
            raw_hypno = mne.read_annotations(hypno_file)
            labels = []
            
            for desc in raw_hypno.description:
                label = self.stage_map.get(desc)
                if label is not None:
                    labels.append(label)
                else:
                    self.logger.warning(f"Unknown sleep stage: {desc}")
                    # Use wake (0) as default for unknown stages
                    labels.append(0)
            
            labels = np.array(labels)
            self.logger.info(f"Loaded {len(labels)} labels with classes: {np.unique(labels)}")
            return labels
            
        except Exception as e:
            self.logger.error(f"Error loading hypnogram {hypno_file}: {str(e)}")
            return None

    def get_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and prepare all data."""
        try:
            # Find all EDF files
            edf_files = sorted(list(self.data_dir.glob("*PSG.edf")))
            hypno_files = sorted(list(self.data_dir.glob("*Hypnogram.edf")))
            
            if not edf_files or not hypno_files:
                self.logger.error("No data files found. Try running download_sleep_edf() first.")
                return None, None
            
            all_signals = []
            all_labels = []
            
            for edf_file, hypno_file in zip(edf_files, hypno_files):
                # Load EEG data
                raw = self.load_edf_file(edf_file)
                labels = self.load_hypnogram(hypno_file)
                
                if raw is not None and labels is not None:
                    # Extract relevant channels
                    signals = raw.get_data(picks=self.target_channels)
                    all_signals.append(signals)
                    all_labels.append(labels)
            
            if all_signals and all_labels:
                # Combine data from all files
                combined_signals = np.concatenate(all_signals, axis=1)
                combined_labels = np.concatenate(all_labels)
                
                # Verify no invalid labels
                valid_mask = combined_labels >= 0
                if not np.all(valid_mask):
                    self.logger.warning(f"Found {np.sum(~valid_mask)} invalid labels, removing them")
                    combined_signals = combined_signals[:, valid_mask]
                    combined_labels = combined_labels[valid_mask]
                
                self.logger.info(f"Final data shapes - Signals: {combined_signals.shape}, Labels: {combined_labels.shape}")
                return combined_signals, combined_labels
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None