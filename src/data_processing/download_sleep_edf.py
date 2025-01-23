import os
import wget
import zipfile
import logging
from pathlib import Path
from typing import List

class SleepEDFDownloader:
    """Handle downloading and extracting Sleep-EDF dataset."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_download.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories."""
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self):
        """Download Sleep-EDF dataset."""
        # URLs for the Sleep-EDF dataset (2018 version)
        urls = [
            "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf",
            "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf",
            # Add more files as needed
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            filepath = self.raw_path / filename
            
            if not filepath.exists():
                self.logger.info(f"Downloading {filename}...")
                try:
                    wget.download(url, str(filepath))
                    self.logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    self.logger.error(f"Error downloading {filename}: {str(e)}")
