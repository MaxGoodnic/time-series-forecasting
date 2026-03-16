"""
JAX training utilities for VAE models.

This module provides training functions and utilities for JAX-based VAE models
adapted for time series forecasting.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from functools import partial


class JAXVAETrainer:
    """
    Trainer for JAX-based VAE models for time series forecasting.
    
    This class handles the training loop, optimization, and evaluation
    for JAX VAE models.
    """
    
    def __init__(
        self,
        model,
        params: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """
        Initialize the JAX VAE trainer.
        
        Args:
            model: JAX VAE model instance
            params: Initial model parameters
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.opt_state = self.optimizer.init(params)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def vae_loss_function(
        self,
        params: Dict[str, Any],
        batch: jnp.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            params: Model parameters
            batch: Batch of time series data
            rng_key: Random key for sampling
            
        Returns:
            Total loss value
        """
        # Forward pass through encoder
        latent = self.model.apply(params, batch, method=self.model.encode, training=True)
        
        # For simplicity, we'll use a deterministic VAE (no sampling)
        # In a full implementation, you would sample from the latent distribution
        
        # Decode
        reconstruction = self.model.apply(params, latent, method=self.model.decode, training=True)
        
        # Reconstruction loss (MSE)
        reconstruction_loss = jnp.mean((batch - reconstruction) ** 2)
        
        # KL divergence loss (simplified - assuming standard normal prior)
        # In a full implementation, this would be computed from the latent distribution
        kl_loss = jnp.mean(latent ** 2)  # Simplified KL term
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * kl_loss  # Weight KL term
        
        return total_loss
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: Any,
        batch: jnp.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> Tuple[Dict[str, Any], Any, jnp.ndarray]:
        """
        Perform one training step.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Batch of training data
            rng_key: Random key
            
        Returns:
            Tuple of (updated_params, updated_opt_state, loss)
        """
        loss_fn = lambda p: self.vae_loss_function(p, batch, rng_key)
        
        # Compute gradients and loss
        (loss, grads) = jax.value_and_grad(loss_fn)(params)
        
        # Apply gradients
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_step(
        self,
        params: Dict[str, Any],
        batch: jnp.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Perform one evaluation step.
        
        Args:
            params: Model parameters
            batch: Batch of evaluation data
            rng_key: Random key
            
        Returns:
            Loss value
        """
        loss = self.vae_loss_function(params, batch, rng_key)
        return loss
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        rng_key: jax.random.PRNGKey,
        verbose: bool = True
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            rng_key: Random key
            verbose: Whether to print progress
            
        Returns:
            Average training loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Convert batch to JAX array
            batch_jax = jnp.array(batch)
            
            # Split random key
            rng_key, step_key = jax.random.split(rng_key)
            
            # Perform training step
            self.params, self.opt_state, loss = self.train_step(
                self.params, self.opt_state, batch_jax, step_key
            )
            
            total_loss += loss
            num_batches += 1
            
            if verbose and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(
        self,
        val_loader,
        rng_key: jax.random.PRNGKey
    ) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            rng_key: Random key
            
        Returns:
            Average validation loss
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            batch_jax = jnp.array(batch)
            rng_key, step_key = jax.random.split(rng_key)
            
            loss = self.eval_step(self.params, batch_jax, step_key)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        patience: int = 5,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Initialize random key
        rng_key = jax.random.PRNGKey(42)
        
        print(f"Starting JAX VAE training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 50)
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch, rng_key, verbose)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_loss = self.evaluate(val_loader, rng_key)
            self.val_losses.append(val_loss)
            
            if verbose:
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best parameters
                best_params = self.params
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Restore best parameters
        self.params = best_params
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': training_time,
            'best_val_loss': best_val_loss
        }
    
    def generate_reconstructions(
        self,
        data: np.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> np.ndarray:
        """
        Generate reconstructions for given data.
        
        Args:
            data: Input data to reconstruct
            rng_key: Random key
            
        Returns:
            Reconstructed data
        """
        data_jax = jnp.array(data)
        
        # Encode
        latent = self.model.apply(self.params, data_jax, method=self.model.encode, training=False)
        
        # Decode
        reconstruction = self.model.apply(self.params, latent, method=self.model.decode, training=False)
        
        return np.array(reconstruction)
    
    def interpolate_between_points(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        num_steps: int = 10,
        rng_key: jax.random.PRNGKey
    ) -> np.ndarray:
        """
        Generate interpolations between two data points in latent space.
        
        Args:
            point1: First data point
            point2: Second data point
            num_steps: Number of interpolation steps
            rng_key: Random key
            
        Returns:
            Array of interpolated reconstructions
        """
        point1_jax = jnp.array(point1[None, ...])  # Add batch dimension
        point2_jax = jnp.array(point2[None, ...])
        
        # Encode both points
        latent1 = self.model.apply(self.params, point1_jax, method=self.model.encode, training=False)
        latent2 = self.model.apply(self.params, point2_jax, method=self.model.encode, training=False)
        
        # Generate interpolations
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            latent_interp = (1 - alpha) * latent1 + alpha * latent2
            
            # Decode interpolation
            recon_interp = self.model.apply(self.params, latent_interp, method=self.model.decode, training=False)
            interpolations.append(np.array(recon_interp[0]))  # Remove batch dimension
        
        return np.array(interpolations)


def create_jax_data_loader(
    data: np.ndarray,
    batch_size: int = 32,
    seed: int = 42,
    shuffle: bool = True
):
    """
    Create a simple data loader for JAX training.
    
    Args:
        data: Input data array
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        
    Yields:
        Batches of data
    """
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield data[batch_indices]


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        
    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(original - reconstructed))
    
    # Normalized MSE (divided by variance)
    variance = np.var(original)
    nmse = mse / variance if variance > 0 else float('inf')
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'nmse': float(nmse)
    }
