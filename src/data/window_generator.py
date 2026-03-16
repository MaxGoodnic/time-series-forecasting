"""
Window generation module for time series forecasting.

This module contains the WindowGenerator class which creates sliding windows
from time series data for training forecasting models.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


class WindowGenerator:
    """
    Generates sliding windows from time series data for forecasting.
    
    This class handles the creation of input/output windows for time series
    forecasting tasks, supporting various configurations for single-step
    and multi-step predictions.
    """
    
    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_columns: Optional[List[str]] = None
    ):
        """
        Initialize the WindowGenerator.
        
        Args:
            input_width: Number of time steps in input window
            label_width: Number of time steps in label window  
            shift: Number of time steps to shift between input and label
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
            label_columns: List of columns to predict. If None, predict all.
        """
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(train_df.columns) if name in label_columns}
        else:
            self.label_columns_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        # Work out the feature column indices
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        # Store window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        # Calculate slice indices
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self) -> str:
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split a window of features into inputs and labels.
        
        Args:
            features: Tensor of shape (batch, window_size, features)
            
        Returns:
            Tuple of (inputs, labels) tensors
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        
        # Slicing doesn't preserve static shape information, so set the shapes
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from the input data.
        
        Args:
            data: DataFrame containing the time series data
            
        Returns:
            tf.data.Dataset yielding (inputs, labels) pairs
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )
        
        ds = ds.map(self.split_window)
        
        return ds
    
    @property
    def train(self) -> tf.data.Dataset:
        """Create training dataset."""
        return self.make_dataset(self.train_df)
    
    @property
    def val(self) -> tf.data.Dataset:
        """Create validation dataset."""
        return self.make_dataset(self.val_df)
    
    @property
    def test(self) -> tf.data.Dataset:
        """Create test dataset."""
        return self.make_dataset(self.test_df)
    
    @property
    def example(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get and cache an example batch of data for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(
        self,
        model: Optional[tf.keras.Model] = None,
        plot_col: str = 'T (degC)',
        max_subplots: int = 3,
        save_path: Optional[str] = None
    ):
        """
        Plot a comparison of true values, predictions, and input features.
        
        Args:
            model: Trained model for making predictions
            plot_col: Column name to plot
            max_subplots: Maximum number of subplots to show
            save_path: Optional path to save the plot
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is not None:
                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                           edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                           marker='X', edgecolors='k', label='Predictions',
                           c='#ff7f0e', s=64)
            
            if n == 0:
                plt.legend()
        
        plt.xlabel('Time [h]')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_window_info(self) -> Dict:
        """Get information about the window configuration."""
        return {
            'input_width': self.input_width,
            'label_width': self.label_width,
            'shift': self.shift,
            'total_window_size': self.total_window_size,
            'input_indices': self.input_indices.tolist(),
            'label_indices': self.label_indices.tolist(),
            'label_columns': self.label_columns,
            'num_features': len(self.column_indices),
            'num_labels': len(self.label_columns_indices) if self.label_columns else len(self.column_indices)
        }


def create_single_step_window(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    label_column: str = 'T (degC)'
) -> WindowGenerator:
    """
    Create a window for single-step prediction.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        label_column: Column to predict
        
    Returns:
        WindowGenerator configured for single-step prediction
    """
    return WindowGenerator(
        input_width=24,
        label_width=1,
        shift=24,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[label_column]
    )


def create_multi_step_window(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = 'T (degC)',
    out_steps: int = 24
) -> WindowGenerator:
    """
    Create a window for multi-step prediction.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        label_column: Column to predict
        out_steps: Number of steps to predict
        
    Returns:
        WindowGenerator configured for multi-step prediction
    """
    return WindowGenerator(
        input_width=24,
        label_width=out_steps,
        shift=out_steps,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=[label_column]
    )
