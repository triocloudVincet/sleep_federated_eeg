# sleep_federated/federated/server.py
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from .base import FederatedComponent
from .client import FederatedClient

class FederatedServer(FederatedComponent):
    def __init__(self, 
                 global_model: nn.Module, 
                 clients: List[FederatedClient]):
        super().__init__()
        self.global_model = global_model
        self.clients = clients
        self.round_metrics = []

    def aggregate_models(self, 
                        client_parameters: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Federated Averaging (FedAvg) algorithm"""
        aggregated_params = {}
        num_clients = len(client_parameters)

        # Initialize with zeros
        for name, param in client_parameters[0].items():
            aggregated_params[name] = torch.zeros_like(param)

        # Sum up parameters
        for client_param in client_parameters:
            for name, param in client_param.items():
                aggregated_params[name] += param

        # Average parameters
        for name in aggregated_params:
            aggregated_params[name] = torch.div(aggregated_params[name], num_clients)

        return aggregated_params

    def train_round(self, local_epochs: int = 5) -> Dict[str, float]:
        """Conduct one round of federated training"""
        self.logger.info("Starting federated training round")
        
        # Collect parameters from all clients
        client_parameters = []
        client_metrics = []

        for client in self.clients:
            # Train client
            metrics = client.train(local_epochs)
            client_metrics.append(metrics)
            
            # Get updated parameters
            parameters = client.get_model_parameters()
            client_parameters.append(parameters)

        # Aggregate parameters
        aggregated_params = self.aggregate_models(client_parameters)

        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.data = aggregated_params[name].clone()

        # Distribute updated model to clients
        self.distribute_model()

        # Compute average metrics
        avg_metrics = {
            'loss': np.mean([m['loss'] for m in client_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in client_metrics])
        }

        self.round_metrics.append(avg_metrics)
        self.logger.info(f"Round completed - Avg Loss: {avg_metrics['loss']:.4f}, "
                        f"Avg Accuracy: {avg_metrics['accuracy']:.2f}%")
        
        return avg_metrics

    def distribute_model(self):
        """Distribute global model to clients"""
        global_parameters = {name: param.data.clone() 
                           for name, param in self.global_model.named_parameters()}
        for client in self.clients:
            client.set_model_parameters(global_parameters)