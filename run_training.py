# test_pipeline.py
import logging
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from sleep_federated.models.cnn_gru import CNNGRU
from sleep_federated.federated.client import FederatedClient
from sleep_federated.federated.server import FederatedServer
from sleep_federated.data_processing.eeg_loader import EEGDataLoader
from sleep_federated.data_processing.preprocessor import EEGPreprocessor
from sleep_federated.data_processing.augmentation import create_balanced_dataloaders

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("PipelineTest")

def plot_distribution(client_data: Dict, save_path: str):
    """Plot class distribution across clients."""
    plt.figure(figsize=(12, 6))
    
    clients = list(client_data.keys())
    class_names = ['Wake', 'N1', 'N2', 'N3/N4', 'REM']
    x = np.arange(len(class_names))
    width = 0.8 / len(clients)
    
    for i, client_id in enumerate(clients):
        # Count samples per class
        labels = []
        for _, batch_labels in client_data[client_id]:
            labels.extend(batch_labels.numpy())
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        dist = np.zeros(5)
        for label, count in zip(unique_labels, counts):
            dist[label] = count
            
        plt.bar(x + i * width, dist, width, label=f'Client {client_id}')
    
    plt.xlabel('Sleep Stages')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Across Clients (After Balancing)')
    plt.xticks(x + width * (len(clients) - 1) / 2, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_pipeline():
    """Run the complete pipeline test."""
    logger = setup_logging()
    
    try:
        # Setup directories
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('plots').mkdir(exist_ok=True)
        
        # 1. Load data
        logger.info("1. Testing data loading...")
        data_loader = EEGDataLoader(data_dir="data/raw")
        signals, labels = data_loader.get_data()
        
        if signals is None or labels is None:
            raise ValueError("Failed to load data")
            
        logger.info(f"Loaded signals shape: {signals.shape}, labels shape: {labels.shape}")
        
        # 2. Preprocess data
        logger.info("\n2. Testing preprocessing...")
        preprocessor = EEGPreprocessor()
        
        # Process signals
        processed_signals = []
        for channel in range(signals.shape[0]):
            epochs, stats = preprocessor.process_signal(signals[channel])
            processed_signals.append(epochs)
            logger.info(f"Channel {channel} stats: {stats}")
        
        processed_signals = processed_signals[0]  # Use first channel for now
        logger.info(f"Processed signals shape: {processed_signals.shape}")
        
        # 3. Create balanced dataloaders
        logger.info("\n3. Testing data distribution...")
        n_clients = 3
        client_data = create_balanced_dataloaders(
            processed_signals[:len(labels)],
            labels,
            n_clients=n_clients,
            target_samples_per_class=20
        )
        
        # Plot distribution
        plot_distribution(client_data, "plots/class_distribution.png")
        logger.info("Saved class distribution plot")
        
        # Print distribution info
        for client_id, loader in client_data.items():
            labels = []
            for _, batch_labels in loader:
                labels.extend(batch_labels.numpy())
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            distribution = {f"class_{label}": count for label, count in zip(unique_labels, counts)}
            
            logger.info(f"\nClient {client_id}:")
            logger.info(f"Total samples: {len(labels)}")
            logger.info(f"Class distribution: {[int(counts[i]) if i in unique_labels else 0 for i in range(5)]}")
        
        # 4. Setup federated learning
        logger.info("\n4. Testing federated setup...")
        global_model = CNNGRU(input_channels=1, num_classes=5)
        
        clients = [
            FederatedClient(
                model=CNNGRU(input_channels=1, num_classes=5),
                data_loader=client_data[i],
                client_id=i
            )
            for i in range(n_clients)
        ]
        
        server = FederatedServer(global_model=global_model, clients=clients)
        
        # 5. Train
        logger.info("\n5. Testing training round...")
        metrics = server.train_round(local_epochs=2)
        logger.info(f"Training metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pipeline()