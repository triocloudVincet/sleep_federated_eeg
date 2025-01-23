# sleep_federated/data_processing/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

class SleepDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, transform=None):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, self.labels[idx]

class DataDistributor:
    def __init__(self, n_clients: int, balance_strategy: str = 'stratified'):
        """Initialize the data distributor.
        
        Args:
            n_clients: Number of federated learning clients
            balance_strategy: Strategy for data distribution ('stratified' or 'random')
        """
        self.n_clients = n_clients
        self.balance_strategy = balance_strategy
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_balanced_split(self, signals: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """Create balanced data split using stratification."""
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Split data by class
        class_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Calculate samples per class per client
        client_splits = []
        for _ in range(self.n_clients):
            client_indices = []
            
            # Get balanced number of samples from each class
            for label in unique_labels:
                label_indices = class_indices[label]
                n_samples = len(label_indices) // self.n_clients
                
                if n_samples > 0:
                    # Randomly select samples for this client
                    selected_indices = np.random.choice(
                        label_indices, 
                        size=n_samples, 
                        replace=False
                    )
                    client_indices.extend(selected_indices)
                    
                    # Remove selected indices from the pool
                    mask = ~np.isin(label_indices, selected_indices)
                    class_indices[label] = label_indices[mask]
            
            client_splits.append(np.array(client_indices))
            
        # Distribute remaining samples
        remaining_indices = np.concatenate([
            indices for indices in class_indices.values() if len(indices) > 0
        ])
        
        if len(remaining_indices) > 0:
            np.random.shuffle(remaining_indices)
            splits = np.array_split(remaining_indices, self.n_clients)
            for i, split in enumerate(splits):
                client_splits[i] = np.concatenate([client_splits[i], split])
        
        return client_splits

    def distribute_data(self, signals: np.ndarray, labels: np.ndarray, 
                       batch_size: int = 32, shuffle: bool = True,
                       num_workers: int = 2) -> Dict[int, DataLoader]:
        """Distribute data among clients."""
        try:
            # Create balanced split
            if self.balance_strategy == 'stratified':
                split_indices = self._create_balanced_split(signals, labels)
            else:
                # Random split
                indices = np.arange(len(labels))
                np.random.shuffle(indices)
                split_indices = np.array_split(indices, self.n_clients)
            
            client_data = {}
            
            for i, indices in enumerate(split_indices):
                # Create dataset for client
                dataset = SleepDataset(
                    signals=signals[indices],
                    labels=labels[indices]
                )
                
                # Create dataloader
                client_data[i] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers
                )
                
                # Log distribution information
                client_labels = labels[indices]
                unique, counts = np.unique(client_labels, return_counts=True)
                distribution = {f"class_{label}": count for label, count in zip(unique, counts)}
                self.logger.info(f"Client {i} data distribution: {distribution}")
                
            return client_data
            
        except Exception as e:
            self.logger.error(f"Error distributing data: {str(e)}")
            raise

    def get_data_distribution(self, client_data: Dict[int, DataLoader]) -> Dict[int, Dict]:
        """Get statistics about data distribution among clients."""
        stats = {}
        for client_id, loader in client_data.items():
            # Get all labels for this client
            all_labels = []
            for _, batch_labels in loader:
                all_labels.append(batch_labels)
            
            # Concatenate all labels
            all_labels = torch.cat(all_labels)
            
            # Count unique labels
            unique_labels, counts = torch.unique(all_labels, return_counts=True)
            
            # Create distribution dictionary
            class_dist = torch.zeros(5)  # Assuming 5 sleep stages
            for label, count in zip(unique_labels, counts):
                class_dist[label] = count.item()
            
            stats[client_id] = {
                'total_samples': len(loader.dataset),
                'class_distribution': class_dist.tolist(),
                'class_percentages': (class_dist / len(loader.dataset) * 100).tolist()
            }
            
        return stats

    def validate_distribution(self, client_data: Dict[int, DataLoader]) -> bool:
        """Validate the data distribution."""
        try:
            stats = self.get_data_distribution(client_data)
            
            # Check number of clients
            if len(client_data) != self.n_clients:
                self.logger.error(f"Expected {self.n_clients} clients, got {len(client_data)}")
                return False
            
            # Check for empty clients
            for client_id, client_stats in stats.items():
                if client_stats['total_samples'] == 0:
                    self.logger.error(f"Client {client_id} has no samples")
                    return False
            
            # Check class balance
            total_dist = np.zeros(5)
            for client_stats in stats.values():
                total_dist += np.array(client_stats['class_distribution'])
            
            class_percentages = total_dist / total_dist.sum() * 100
            self.logger.info(f"Overall class distribution (%): {class_percentages}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating distribution: {str(e)}")
            return False