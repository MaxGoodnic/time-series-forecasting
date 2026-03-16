"""
LSTM models for time series forecasting.

This module contains LSTM-based models for both single-step and multi-step
time series forecasting tasks.
"""

import tensorflow as tf
from typing import Optional


class FeedBack(tf.keras.Model):
    """
    Autoregressive LSTM model for multi-step forecasting.
    
    This model makes predictions one step at a time and feeds the output
    back to the model for the next prediction. This approach is useful for
    multi-step forecasting where predictions depend on previous predictions.
    """
    
    def __init__(self, units: int, out_steps: int, num_features: int):
        """
        Initialize the autoregressive LSTM model.
        
        Args:
            units: Number of LSTM units
            out_steps: Number of output steps to predict
            num_features: Number of input features
        """
        super().__init__()
        self.out_steps = out_steps
        self.num_features = num_features
        self.units = units
        
        # LSTM cell for sequential processing
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        
        # Wrap LSTM cell in RNN for warmup method
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        
        # Dense layer for output prediction
        self.dense = tf.keras.layers.Dense(num_features)
    
    def warmup(self, inputs: tf.Tensor) -> tuple:
        """
        Warmup phase - process initial input sequence.
        
        Args:
            inputs: Input tensor of shape (batch, time, features)
            
        Returns:
            Tuple of (prediction, lstm_state)
        """
        # Process input sequence through LSTM
        x, *state = self.lstm_rnn(inputs)
        
        # Generate prediction for next time step
        prediction = self.dense(x)
        
        return prediction, state
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for autoregressive prediction.
        
        Args:
            inputs: Input tensor of shape (batch, input_width, features)
            training: Whether in training mode
            
        Returns:
            Predictions tensor of shape (batch, out_steps, features)
        """
        # Use TensorArray to capture dynamically unrolled outputs
        predictions = []
        
        # Initialize LSTM state with warmup
        prediction, state = self.warmup(inputs)
        
        # Add first prediction
        predictions.append(prediction)
        
        # Run remaining prediction steps
        for n in range(1, self.out_steps):
            # Use last prediction as input
            x = prediction
            
            # Execute one LSTM step
            x, state = self.lstm_cell(x, states=state, training=training)
            
            # Convert LSTM output to prediction
            prediction = self.dense(x)
            
            # Add prediction to output list
            predictions.append(prediction)
        
        # Stack predictions and convert to proper shape
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        
        return predictions


class LSTMSingleStep(tf.keras.Model):
    """
    LSTM model for single-step time series forecasting.
    
    This model uses LSTM layers to process input sequences and make
    single-step predictions. It's suitable for predicting the next
    time step given a sequence of historical data.
    """
    
    def __init__(
        self,
        units: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_features: int = 1
    ):
        """
        Initialize the single-step LSTM model.
        
        Args:
            units: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_features: Number of output features
        """
        super().__init__()
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        
        # Build LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = i < num_layers - 1  # Only return sequences for non-final layers
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout if i < num_layers - 1 else 0.0
                )
            )
        
        # Dense output layer
        self.dense = tf.keras.layers.Dense(num_features)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for single-step prediction.
        
        Args:
            inputs: Input tensor of shape (batch, time, features)
            training: Whether in training mode
            
        Returns:
            Prediction tensor of shape (batch, 1, features)
        """
        x = inputs
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Generate prediction
        prediction = self.dense(x)
        
        return prediction[:, tf.newaxis, :]  # Add time dimension


class LSTMMultiShot(tf.keras.Model):
    """
    LSTM model for multi-shot multi-step forecasting.
    
    This model makes all predictions at once using a single forward pass,
    as opposed to the autoregressive approach of FeedBack.
    """
    
    def __init__(
        self,
        units: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        out_steps: int = 24,
        num_features: int = 1
    ):
        """
        Initialize the multi-shot LSTM model.
        
        Args:
            units: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            out_steps: Number of output steps to predict
            num_features: Number of output features
        """
        super().__init__()
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_steps = out_steps
        self.num_features = num_features
        
        # Build LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = True  # Return sequences for all layers
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout
                )
            )
        
        # Dense output layer that outputs all steps at once
        self.dense = tf.keras.layers.Dense(out_steps * num_features)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for multi-shot prediction.
        
        Args:
            inputs: Input tensor of shape (batch, input_width, features)
            training: Whether in training mode
            
        Returns:
            Prediction tensor of shape (batch, out_steps, features)
        """
        x = inputs
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Take the last time step output
        x = x[:, -1, :]  # Shape: (batch, units)
        
        # Generate all predictions at once
        predictions = self.dense(x)  # Shape: (batch, out_steps * features)
        
        # Reshape to (batch, out_steps, features)
        predictions = tf.reshape(predictions, [-1, self.out_steps, self.num_features])
        
        return predictions


def create_lstm_model(
    model_type: str = 'single_step',
    units: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    out_steps: int = 24,
    num_features: int = 1
) -> tf.keras.Model:
    """
    Factory function to create LSTM models.
    
    Args:
        model_type: Type of LSTM model ('single_step', 'multi_shot', 'autoregressive')
        units: Number of LSTM units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        out_steps: Number of output steps for multi-step models
        num_features: Number of input/output features
        
    Returns:
        LSTM model instance
    """
    if model_type == 'single_step':
        return LSTMSingleStep(
            units=units,
            num_layers=num_layers,
            dropout=dropout,
            num_features=num_features
        )
    elif model_type == 'multi_shot':
        return LSTMMultiShot(
            units=units,
            num_layers=num_layers,
            dropout=dropout,
            out_steps=out_steps,
            num_features=num_features
        )
    elif model_type == 'autoregressive':
        return FeedBack(
            units=units,
            out_steps=out_steps,
            num_features=num_features
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compile_lstm_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    loss: str = 'mse',
    metrics: list = None
) -> tf.keras.Model:
    """
    Compile an LSTM model with appropriate optimizer and loss.
    
    Args:
        model: LSTM model to compile
        learning_rate: Learning rate for optimizer
        loss: Loss function
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['mae']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model


def train_lstm_model(
    model: tf.keras.Model,
    window_generator,
    epochs: int = 20,
    patience: int = 2,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """
    Train an LSTM model with early stopping.
    
    Args:
        model: LSTM model to train
        window_generator: WindowGenerator instance
        epochs: Maximum number of training epochs
        patience: Patience for early stopping
        verbose: Verbosity level
        
    Returns:
        Training history
    """
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        window_generator.train,
        epochs=epochs,
        validation_data=window_generator.val,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    return history
