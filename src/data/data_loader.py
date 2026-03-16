"""
Data loading and preprocessing module for time series forecasting.

This module handles loading, preprocessing, and windowing of time series data
for weather forecasting using the Jena climate dataset.
"""

import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional


class WeatherDataLoader:
    """
    Loads and preprocesses weather data for time series forecasting.
    
    This class handles downloading, loading, and preprocessing the Jena climate
    dataset for use in time series forecasting models.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Optional path to local CSV file. If None, downloads from web.
        """
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self, sample_rate: int = 6) -> pd.DataFrame:
        """
        Load and preprocess the weather dataset.
        
        Args:
            sample_rate: Sampling rate (default 6 = every 6th row for hourly data)
            
        Returns:
            Preprocessed pandas DataFrame
        """
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from local path: {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            print("Downloading Jena climate dataset...")
            zip_path = tf.keras.utils.get_file(
                origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
                fname='jena_climate_2009_2016.csv.zip',
                extract=True
            )
            csv_path, _ = os.path.splitext(zip_path)
            df = pd.read_csv(csv_path)
        
        # Sample data to get hourly intervals (every 6th row)
        df = df[::sample_rate]
        
        # Convert date time to proper datetime format
        date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        
        # Set the datetime as index
        df.index = date_time
        
        self.df = df
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def preprocess_data(self, split_fraction: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data and split into train/validation/test sets.
        
        Args:
            split_fraction: Fraction of data to use for training
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate split indices
        n = len(self.df)
        train_df = self.df[0:int(n*split_fraction)]
        val_df = self.df[int(n*split_fraction):int(n*0.9)]
        test_df = self.df[int(n*0.9):]
        
        # Normalize the data
        train_mean = train_df.mean()
        train_std = train_df.std()
        
        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        
        # Store normalization parameters
        self.train_mean = train_mean
        self.train_std = train_std
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"Data split and normalized:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples") 
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def get_feature_stats(self) -> Dict:
        """Get statistics of the features."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'mean': self.df.mean(),
            'std': self.df.std(),
            'min': self.df.min(),
            'max': self.df.max()
        }
        return stats
    
    def plot_data_overview(self, save_path: Optional[str] = None):
        """Plot overview of the weather data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Select key features for plotting
        key_features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
        available_features = [f for f in key_features if f in self.df.columns]
        
        if not available_features:
            print("No key features found for plotting")
            return
        
        fig, axes = plt.subplots(len(available_features), 1, figsize=(12, 3*len(available_features)))
        if len(available_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(available_features):
            axes[i].plot(self.df.index[:1000], self.df[feature][:1000])
            axes[i].set_title(f'{feature} over time (first 1000 samples)')
            axes[i].set_ylabel(feature)
            axes[i].grid(True)
        
        plt.xlabel('Time')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def load_weather_data(data_path: Optional[str] = None, sample_rate: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and preprocess weather data.
    
    Args:
        data_path: Optional path to local data file
        sample_rate: Sampling rate for data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = WeatherDataLoader(data_path)
    loader.load_data(sample_rate)
    return loader.preprocess_data()
