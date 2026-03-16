"""
Baseline models for time series forecasting.

This module contains simple baseline models that serve as reference points
for evaluating more complex models.
"""

import tensorflow as tf
from typing import Optional


class Baseline(tf.keras.Model):
    """
    Simple baseline model that predicts the last known value.
    
    This model serves as a baseline for single-step predictions by simply
    returning the last input value as the prediction.
    """
    
    def __init__(self, label_index: Optional[int] = None):
        """
        Initialize the baseline model.
        
        Args:
            label_index: Index of the label column to predict.
                        If None, predict all features.
        """
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass - return the last input value.
        
        Args:
            inputs: Input tensor of shape (batch, time, features)
            
        Returns:
            Prediction tensor
        """
        if self.label_index is None:
            return inputs[:, -1:, :]  # Return last time step for all features
        else:
            return inputs[:, -1:, self.label_index:self.label_index+1]


class MultiStepLastBaseline(tf.keras.Model):
    """
    Baseline model for multi-step predictions.
    
    This model predicts that future values will be the same as the last
    known value for all prediction steps.
    """
    
    def __init__(self, label_index: Optional[int] = None, out_steps: int = 1):
        """
        Initialize the multi-step baseline model.
        
        Args:
            label_index: Index of the label column to predict
            out_steps: Number of output steps to predict
        """
        super().__init__()
        self.label_index = label_index
        self.out_steps = out_steps
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass - repeat the last value for all output steps.
        
        Args:
            inputs: Input tensor of shape (batch, time, features)
            
        Returns:
            Prediction tensor of shape (batch, out_steps, features)
        """
        if self.label_index is not None:
            # If predicting specific column, extract it first
            last_value = inputs[:, -1:, self.label_index:self.label_index+1]
        else:
            last_value = inputs[:, -1:, :]
        
        # Repeat the last value for all output steps
        return tf.tile(last_value, [1, self.out_steps, 1])


class RepeatBaseline(tf.keras.Model):
    """
    Baseline model that repeats the input sequence.
    
    This model is useful for multi-step predictions where the output
    sequence length matches the input sequence length.
    """
    
    def __init__(self, label_index: Optional[int] = None, out_steps: int = 1):
        """
        Initialize the repeat baseline model.
        
        Args:
            label_index: Index of the label column to predict
            out_steps: Number of output steps (should match input length)
        """
        super().__init__()
        self.label_index = label_index
        self.out_steps = out_steps
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass - return the input sequence.
        
        Args:
            inputs: Input tensor of shape (batch, time, features)
            
        Returns:
            Input tensor (potentially truncated to out_steps)
        """
        if self.label_index is not None:
            # Extract specific column
            inputs = inputs[:, :, self.label_index:self.label_index+1]
        
        # Truncate or pad to match out_steps
        if inputs.shape[1] > self.out_steps:
            return inputs[:, :self.out_steps, :]
        elif inputs.shape[1] < self.out_steps:
            # Pad by repeating the last value
            padding = self.out_steps - inputs.shape[1]
            last_value = inputs[:, -1:, :]
            padding_tensor = tf.tile(last_value, [1, padding, 1])
            return tf.concat([inputs, padding_tensor], axis=1)
        else:
            return inputs


def create_baseline_model(model_type: str = 'single_step', **kwargs) -> tf.keras.Model:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of baseline model ('single_step', 'multi_step', 'repeat')
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Baseline model instance
    """
    if model_type == 'single_step':
        return Baseline(**kwargs)
    elif model_type == 'multi_step':
        return MultiStepLastBaseline(**kwargs)
    elif model_type == 'repeat':
        return RepeatBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_baseline(
    model: tf.keras.Model,
    window_generator,
    verbose: int = 1
) -> Dict[str, float]:
    """
    Evaluate a baseline model on validation and test sets.
    
    Args:
        model: Trained baseline model
        window_generator: WindowGenerator instance
        verbose: Verbosity level
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Compile model for evaluation
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # Evaluate on validation set
    val_performance = model.evaluate(window_generator.val, verbose=verbose)
    test_performance = model.evaluate(window_generator.test, verbose=verbose)
    
    metrics = {
        'val_loss': val_performance[0],
        'val_mae': val_performance[1],
        'test_loss': test_performance[0],
        'test_mae': test_performance[1]
    }
    
    return metrics
