# sleep_federated/federated/client.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any
from sleep_federated.models.losses import FocalLoss
from sleep_federated.models.augmentation import MixupAugmentation

class FederatedClient:
    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                 client_id: int, learning_rate: float = 0.0003):
        self.model = model
        self.data_loader = data_loader
        self.client_id = client_id
        self.logger = logging.getLogger(f"Client_{client_id}")
        
        # Calculate class weights with smoothing
        self.class_weights = self._calculate_class_weights(smoothing=0.1)
        
        # Initialize loss and optimizer
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=1.5)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Initialize mixup with lower alpha
        self.mixup = MixupAugmentation(alpha=0.1)
        
        # Initialize metrics tracker
        self.metrics_history = []
        self.best_accuracy = 0.0
        self.best_state = None

    def _calculate_class_weights(self, smoothing: float = 0.1) -> torch.Tensor:
        all_labels = []
        for _, labels in self.data_loader:
            all_labels.extend(labels.numpy())
            
        labels = np.array(all_labels)
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Add smoothing to counts
        smoothed_counts = class_counts + smoothing * total_samples
        
        # Compute balanced weights
        weights = total_samples / (len(class_counts) * smoothed_counts)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Clip weights to prevent extreme values
        weights = np.clip(weights, 0.1, 10.0)
        
        return torch.FloatTensor(weights)

    def train(self, epochs: int) -> Dict[str, Any]:
        self.model.train()
        metrics_history = []
        best_accuracy = 0.0
        patience_counter = 0
        max_patience = 3
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            class_correct = torch.zeros(5)
            class_total = torch.zeros(5)
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Gradient accumulation steps
                accumulation_steps = 4
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target) / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Calculate metrics
                epoch_loss += loss.item() * accumulation_steps
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Per-class accuracy
                for label in range(5):
                    mask = target == label
                    class_correct[label] += predicted[mask].eq(target[mask]).sum().item()
                    class_total[label] += mask.sum().item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(self.data_loader)
            accuracy = 100. * correct / total
            
            # Update learning rate scheduler
            self.scheduler.step(accuracy)
            
            # Calculate per-class accuracy
            per_class_acc = []
            for i in range(5):
                if class_total[i] > 0:
                    class_acc = 100. * class_correct[i] / class_total[i]
                    per_class_acc.append(f"{class_acc:.2f}%")
                else:
                    per_class_acc.append("0.00%")
            
            self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
            self.logger.info(f"Per-class accuracy: {per_class_acc}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_state = {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'accuracy': accuracy
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info("Early stopping triggered")
                break
            
            metrics_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state['model_state'])
        
        return {
            'loss': np.mean([m['loss'] for m in metrics_history]),
            'accuracy': best_accuracy,
            'per_class_accuracy': metrics_history[-1]['per_class_accuracy'],
            'total_samples': total,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for aggregation."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update local model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data = parameters[name].clone()
                    
    def validate(self) -> Dict[str, float]:
        """Validate the model on local data."""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = torch.zeros(5)
        class_total = torch.zeros(5)
        
        with torch.no_grad():
            for data, target in self.data_loader:
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Per-class accuracy
                for label in range(5):
                    mask = target == label
                    class_correct[label] += predicted[mask].eq(target[mask]).sum().item()
                    class_total[label] += mask.sum().item()
        
        accuracy = 100. * correct / total
        per_class_acc = []
        for i in range(5):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                per_class_acc.append(f"{class_acc:.2f}%")
            else:
                per_class_acc.append("0.00%")
                
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }