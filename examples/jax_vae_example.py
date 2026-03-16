#!/usr/bin/env python3
"""
Example script demonstrating JAX VAE for time series forecasting.

This script shows how to use the JAX VAE implementation for time series
data, including training, evaluation, and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.data_loader import WeatherDataLoader
from src.models.jax_vae import create_timeseries_vae, initialize_vae_params
from src.utils.jax_training import JAXVAETrainer, create_jax_data_loader, compute_reconstruction_metrics


def prepare_time_series_data(sequence_length: int = 24, num_features: int = 1):
    """
    Prepare time series data for JAX VAE training.
    
    Args:
        sequence_length: Length of input sequences
        num_features: Number of features to use
        
    Returns:
        Prepared data arrays
    """
    print("Loading and preparing time series data...")
    
    # Load weather data
    loader = WeatherDataLoader()
    df = loader.load_data(sample_rate=6)  # Hourly data
    train_df, val_df, test_df = loader.preprocess_data()
    
    # Use temperature as primary feature
    feature_col = 'T (degC)'
    
    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    # Prepare training data
    train_data = train_df[feature_col].values
    val_data = val_df[feature_col].values
    test_data = test_df[feature_col].values
    
    # Create sequences
    train_sequences = create_sequences(train_data, sequence_length)
    val_sequences = create_sequences(val_data, sequence_length)
    test_sequences = create_sequences(test_data, sequence_length)
    
    # Add feature dimension
    train_sequences = train_sequences[:, :, np.newaxis]
    val_sequences = val_sequences[:, :, np.newaxis]
    test_sequences = test_sequences[:, :, np.newaxis]
    
    print(f"Data prepared:")
    print(f"  Train sequences: {train_sequences.shape}")
    print(f"  Val sequences: {val_sequences.shape}")
    print(f"  Test sequences: {test_sequences.shape}")
    
    return train_sequences, val_sequences, test_sequences, loader


def train_jax_vae(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    sequence_length: int = 24,
    num_features: int = 1,
    epochs: int = 50
):
    """
    Train a JAX VAE model on time series data.
    
    Args:
        train_sequences: Training sequences
        val_sequences: Validation sequences
        sequence_length: Length of sequences
        num_features: Number of features
        epochs: Number of training epochs
        
    Returns:
        Trained model and training history
    """
    print("\n" + "=" * 50)
    print("TRAINING JAX VAE MODEL")
    print("=" * 50)
    
    import jax
    
    # Create model
    model = create_timeseries_vae(
        sequence_length=sequence_length,
        num_features=num_features,
        z_channels=8,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.1
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = initialize_vae_params(model, sequence_length, num_features, key)
    
    print(f"Model created with {sum(p.size for p in jax.tree_util.tree_leaves(params))} parameters")
    
    # Create trainer
    trainer = JAXVAETrainer(
        model=model,
        params=params,
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Create data loaders
    train_loader = create_jax_data_loader(train_sequences, batch_size=32, shuffle=True)
    val_loader = create_jax_data_loader(val_sequences, batch_size=32, shuffle=False)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=5,
        verbose=True
    )
    
    return trainer, history


def evaluate_and_visualize(
    trainer: JAXVAETrainer,
    test_sequences: np.ndarray,
    num_examples: int = 5
):
    """
    Evaluate the trained model and create visualizations.
    
    Args:
        trainer: Trained JAX VAE trainer
        test_sequences: Test sequences
        num_examples: Number of examples to visualize
    """
    print("\n" + "=" * 50)
    print("EVALUATING AND VISUALIZING")
    print("=" * 50)
    
    import jax
    
    # Generate reconstructions
    key = jax.random.PRNGKey(123)
    test_subset = test_sequences[:num_examples]
    reconstructions = trainer.generate_reconstructions(test_subset, key)
    
    # Compute metrics
    metrics = compute_reconstruction_metrics(test_subset, reconstructions)
    print(f"Reconstruction Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")
    
    # Visualize reconstructions
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i in range(num_examples):
        original = test_subset[i, :, 0]  # Remove feature dimension
        reconstructed = reconstructions[i, :, 0]
        
        axes[i].plot(original, label='Original', linewidth=2, alpha=0.8)
        axes[i].plot(reconstructed, label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        axes[i].set_title(f'Sequence {i+1} Reconstruction')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Temperature (normalized)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jax_vae_reconstructions.png', dpi=150, bbox_inches='tight')
    print("Reconstructions plot saved to 'jax_vae_reconstructions.png'")
    plt.show()
    
    # Plot training history
    if trainer.train_losses and trainer.val_losses:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses, label='Train Loss')
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(trainer.train_losses[-20:], label='Train Loss (last 20)')
        plt.plot(trainer.val_losses[-20:], label='Validation Loss (last 20)')
        plt.title('Training History (Last 20 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('jax_vae_training_history.png', dpi=150, bbox_inches='tight')
        print("Training history plot saved to 'jax_vae_training_history.png'")
        plt.show()


def demonstrate_latent_interpolation(
    trainer: JAXVAETrainer,
    test_sequences: np.ndarray
):
    """
    Demonstrate interpolation in latent space.
    
    Args:
        trainer: Trained JAX VAE trainer
        test_sequences: Test sequences
    """
    print("\n" + "=" * 50)
    print("LATENT SPACE INTERPOLATION")
    print("=" * 50)
    
    import jax
    
    # Select two different sequences
    seq1 = test_sequences[0]
    seq2 = test_sequences[1]
    
    # Generate interpolations
    key = jax.random.PRNGKey(456)
    interpolations = trainer.interpolate_between_points(seq1, seq2, num_steps=10, rng_key=key)
    
    # Visualize interpolations
    fig, axes = plt.subplots(10, 1, figsize=(12, 20))
    
    for i in range(10):
        axes[i].plot(seq1[:, 0], 'b-', alpha=0.3, linewidth=1, label='Sequence 1')
        axes[i].plot(seq2[:, 0], 'r-', alpha=0.3, linewidth=1, label='Sequence 2')
        axes[i].plot(interpolations[i, :, 0], 'g-', linewidth=2, label=f'Interpolation {i+1}/10')
        axes[i].set_title(f'Latent Interpolation Step {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Temperature (normalized)')
        
        if i == 0:
            axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jax_vae_interpolation.png', dpi=150, bbox_inches='tight')
    print("Latent interpolation plot saved to 'jax_vae_interpolation.png'")
    plt.show()


def main():
    """Main function to run the JAX VAE example."""
    print("JAX VAE TIME SERIES FORECASTING EXAMPLE")
    print("=" * 50)
    
    # Configuration
    sequence_length = 24  # 24 hours
    num_features = 1     # Temperature only
    epochs = 50
    
    # Prepare data
    train_sequences, val_sequences, test_sequences, loader = prepare_time_series_data(
        sequence_length=sequence_length,
        num_features=num_features
    )
    
    # Train model
    trainer, history = train_jax_vae(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        sequence_length=sequence_length,
        num_features=num_features,
        epochs=epochs
    )
    
    # Evaluate and visualize
    evaluate_and_visualize(trainer, test_sequences, num_examples=5)
    
    # Demonstrate latent interpolation
    demonstrate_latent_interpolation(trainer, test_sequences)
    
    print("\n" + "=" * 50)
    print("JAX VAE EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - jax_vae_reconstructions.png")
    print("  - jax_vae_training_history.png")
    print("  - jax_vae_interpolation.png")


if __name__ == "__main__":
    main()
