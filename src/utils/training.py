"""
Training and evaluation utilities for time series forecasting models.

This module provides functions for training, evaluating, and comparing
different time series forecasting models.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import time


class ModelTrainer:
    """
    Utility class for training and evaluating time series models.
    
    This class provides a consistent interface for training different types
    of models and comparing their performance.
    """
    
    def __init__(self, window_generator):
        """
        Initialize the model trainer.
        
        Args:
            window_generator: WindowGenerator instance for data preparation
        """
        self.window_generator = window_generator
        self.models = {}
        self.histories = {}
        self.performance = {}
    
    def add_model(self, name: str, model: tf.keras.Model):
        """
        Add a model to the trainer.
        
        Args:
            name: Model name/identifier
            model: TensorFlow model instance
        """
        self.models[name] = model
        print(f"Added model: {name}")
    
    def compile_model(
        self,
        name: str,
        learning_rate: float = 0.001,
        loss: str = 'mse',
        metrics: List[str] = None
    ):
        """
        Compile a specific model.
        
        Args:
            name: Model name
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        if metrics is None:
            metrics = ['mae']
        
        self.models[name].compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        print(f"Compiled model: {name}")
    
    def train_model(
        self,
        name: str,
        epochs: int = 20,
        patience: int = 2,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train a specific model.
        
        Args:
            name: Model name
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        model = self.models[name]
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True
        )
        
        print(f"Training model: {name}")
        start_time = time.time()
        
        # Train the model
        history = model.fit(
            self.window_generator.train,
            epochs=epochs,
            validation_data=self.window_generator.val,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.histories[name] = history
        return history
    
    def evaluate_model(self, name: str, verbose: int = 1) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            name: Model name
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        model = self.models[name]
        
        # Evaluate on validation set
        val_metrics = model.evaluate(self.window_generator.val, verbose=verbose)
        
        # Evaluate on test set
        test_metrics = model.evaluate(self.window_generator.test, verbose=verbose)
        
        # Get metric names from model
        metric_names = model.metrics_names
        
        # Create performance dictionary
        performance = {}
        for i, metric_name in enumerate(metric_names):
            performance[f'val_{metric_name}'] = val_metrics[i]
            performance[f'test_{metric_name}'] = test_metrics[i]
        
        self.performance[name] = performance
        return performance
    
    def train_all_models(
        self,
        epochs: int = 20,
        patience: int = 2,
        verbose: int = 1
    ) -> Dict[str, tf.keras.callbacks.History]:
        """
        Train all added models.
        
        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Dictionary of training histories
        """
        histories = {}
        
        for name in self.models:
            print(f"\n{'='*50}")
            print(f"Training model: {name}")
            print(f"{'='*50}")
            
            try:
                history = self.train_model(name, epochs, patience, verbose)
                histories[name] = history
            except Exception as e:
                print(f"Error training {name}: {e}")
                histories[name] = None
        
        return histories
    
    def evaluate_all_models(self, verbose: int = 1) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            verbose: Verbosity level
            
        Returns:
            Dictionary of performance metrics for all models
        """
        all_performance = {}
        
        for name in self.models:
            print(f"\n{'='*50}")
            print(f"Evaluating model: {name}")
            print(f"{'='*50}")
            
            try:
                performance = self.evaluate_model(name, verbose)
                all_performance[name] = performance
                
                # Print key metrics
                print(f"Validation Loss: {performance['val_loss']:.4f}")
                print(f"Validation MAE: {performance['val_mae']:.4f}")
                print(f"Test Loss: {performance['test_loss']:.4f}")
                print(f"Test MAE: {performance['test_mae']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                all_performance[name] = {}
        
        return all_performance
    
    def compare_models(self, metric: str = 'test_loss') -> pd.DataFrame:
        """
        Compare all models based on a specific metric.
        
        Args:
            metric: Metric to compare models by
            
        Returns:
            DataFrame with model comparison
        """
        if not self.performance:
            print("No performance data available. Run evaluate_all_models() first.")
            return pd.DataFrame()
        
        comparison_data = []
        for name, performance in self.performance.items():
            if metric in performance:
                comparison_data.append({
                    'Model': name,
                    metric: performance[metric]
                })
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values(metric)
            print(f"\nModel Comparison (sorted by {metric}):")
            print(df.to_string(index=False))
        
        return df
    
    def plot_training_history(
        self,
        model_names: Optional[List[str]] = None,
        metrics: List[str] = ['loss', 'mae'],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot training history for models.
        
        Args:
            model_names: List of model names to plot. If None, plot all
            metrics: List of metrics to plot
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        if model_names is None:
            model_names = list(self.histories.keys())
        
        if not model_names:
            print("No training histories available")
            return
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            for name in model_names:
                if name in self.histories and self.histories[name] is not None:
                    history = self.histories[name]
                    
                    if metric in history.history:
                        axes[i].plot(
                            history.history[metric],
                            label=f'{name} - Train',
                            linestyle='-'
                        )
                    
                    val_metric = f'val_{metric}'
                    if val_metric in history.history:
                        axes[i].plot(
                            history.history[val_metric],
                            label=f'{name} - Val',
                            linestyle='--'
                        )
            
            axes[i].set_title(f'{metric.upper()} over epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(
        self,
        model_names: Optional[List[str]] = None,
        max_subplots: int = 3,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot predictions from models.
        
        Args:
            model_names: List of model names to plot. If None, plot all
            max_subplots: Maximum number of subplots
            figsize: Figure size
            save_path: Optional path to save the plot
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        if not model_names:
            print("No models available")
            return
        
        num_models = len(model_names)
        fig, axes = plt.subplots(num_models, 1, figsize=(figsize[0], figsize[1] * num_models))
        
        if num_models == 1:
            axes = [axes]
        
        for i, name in enumerate(model_names):
            if name in self.models:
                self.window_generator.plot(
                    model=self.models[name],
                    max_subplots=max_subplots,
                    save_path=None  # Don't save individual plots
                )
                axes[i].set_title(f'Predictions - {name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.show()


def create_model_comparison_table(performance_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table from performance metrics.
    
    Args:
        performance_dict: Dictionary of model performance metrics
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, metrics in performance_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    if not df.empty:
        # Sort by test loss if available
        if 'test_loss' in df.columns:
            df = df.sort_values('test_loss')
    
    return df


def save_model_results(
    trainer: ModelTrainer,
    save_dir: str = 'model_results'
):
    """
    Save model results to files.
    
    Args:
        trainer: ModelTrainer instance with results
        save_dir: Directory to save results
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save performance comparison
    if trainer.performance:
        comparison_df = create_model_comparison_table(trainer.performance)
        comparison_path = os.path.join(save_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Model comparison saved to {comparison_path}")
    
    # Save training histories
    for name, history in trainer.histories.items():
        if history is not None:
            history_path = os.path.join(save_dir, f'{name}_history.csv')
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(history_path, index=False)
            print(f"Training history for {name} saved to {history_path}")
    
    print(f"All results saved to {save_dir}")
