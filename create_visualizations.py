# create_visualizations.py
from sleep_federated.visualization.plots import VisualizationManager

def main():
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
    main()