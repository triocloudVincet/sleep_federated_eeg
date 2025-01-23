# sleep_federated/visualization/plots.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

class VisualizationManager:
    def __init__(self, save_dir: str = 'presentation/plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # Set colors
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_class_distribution(self, client_distributions: Dict[int, List[int]]):
        """Plot class distribution across clients."""
        plt.figure(figsize=(12, 6))
        
        class_names = ['Wake', 'N1', 'N2', 'N3/N4', 'REM']
        x = np.arange(len(class_names))
        width = 0.8 / len(client_distributions)
        
        for i, (client_id, dist) in enumerate(client_distributions.items()):
            plt.bar(x + i * width, dist, width, 
                   color=self.colors[i % len(self.colors)],
                   label=f'Client {client_id}')
        
        plt.xlabel('Sleep Stages', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution Across Clients', fontsize=14, pad=20)
        plt.xticks(x + width * (len(client_distributions) - 1) / 2, 
                  class_names, fontsize=10, rotation=45)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_progress(self, metrics_history: List[Dict]):
        """Plot training metrics over time."""
        epochs = range(len(metrics_history))
        accuracies = [m['accuracy'] for m in metrics_history]
        losses = [m['loss'] for m in metrics_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot accuracy
        ax1.plot(epochs, accuracies, 'o-', color=self.colors[0], 
                linewidth=2, markersize=8, label='Accuracy')
        ax1.set_title('Training Accuracy Over Time', fontsize=14, pad=20)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot loss
        ax2.plot(epochs, losses, 'o-', color=self.colors[1], 
                linewidth=2, markersize=8, label='Loss')
        ax2.set_title('Training Loss Over Time', fontsize=14, pad=20)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_per_class_accuracy(self, client_accuracies: Dict[int, List[float]]):
        """Plot per-class accuracy for each client."""
        plt.figure(figsize=(12, 6))
        
        class_names = ['Wake', 'N1', 'N2', 'N3/N4', 'REM']
        x = np.arange(len(class_names))
        width = 0.8 / len(client_accuracies)
        
        for i, (client_id, accuracies) in enumerate(client_accuracies.items()):
            plt.bar(x + i * width, accuracies, width, 
                   color=self.colors[i % len(self.colors)],
                   label=f'Client {client_id}')
        
        plt.xlabel('Sleep Stages', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy by Client', fontsize=14, pad=20)
        plt.xticks(x + width * (len(client_accuracies) - 1) / 2, 
                  class_names, fontsize=10, rotation=45)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

# create_visualizations.py
def create_visualizations():
    # Create visualization manager
    viz = VisualizationManager(save_dir='presentation/plots')
    
    # Client distributions from our results
    client_distributions = {
        0: [8, 6, 12, 23, 9],  # Client 0 distribution
        1: [6, 9, 16, 23, 4],  # Client 1 distribution
        2: [6, 9, 12, 25, 7]   # Client 2 distribution
    }
    
    # Training metrics
    metrics_history = [
        {'accuracy': 18.97, 'loss': 0.0907},  # Initial
        {'accuracy': 25.86, 'loss': 0.0795},  # Mid-training
        {'accuracy': 27.96, 'loss': 0.0765}   # Final
    ]
    
    # Per-class accuracies
    client_accuracies = {
        0: [25.00, 0.00, 16.67, 43.48, 11.11],  # Client 0 final accuracies
        1: [66.67, 44.44, 18.75, 8.70, 0.00],   # Client 1 final accuracies
        2: [0.00, 11.11, 8.33, 76.00, 0.00]     # Client 2 final accuracies
    }
    
    # Create all visualizations
    print("Creating class distribution plot...")
    viz.plot_class_distribution(client_distributions)
    
    print("Creating training progress plot...")
    viz.plot_training_progress(metrics_history)
    
    print("Creating per-class accuracy plot...")
    viz.plot_per_class_accuracy(client_accuracies)
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    create_visualizations()