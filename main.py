#!/usr/bin/env python3
"""
Main script for time series forecasting project.

This script demonstrates the complete workflow for training and evaluating
time series forecasting models using the Jena climate dataset.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import WeatherDataLoader
from src.data.window_generator import WindowGenerator, create_single_step_window, create_multi_step_window
from src.models.baseline import Baseline, MultiStepLastBaseline, create_baseline_model
from src.models.lstm import create_lstm_model, compile_lstm_model, train_lstm_model
from src.utils.training import ModelTrainer, save_model_results


def setup_matplotlib():
    """Setup matplotlib parameters for consistent plotting."""
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['axes.grid'] = False


def load_and_preprocess_data(data_path: str = None, sample_rate: int = 6):
    """
    Load and preprocess the weather dataset.
    
    Args:
        data_path: Optional path to local data file
        sample_rate: Sampling rate for data
        
    Returns:
        Tuple of (train_df, val_df, test_df) and data loader
    """
    print("=" * 50)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 50)
    
    # Initialize data loader
    loader = WeatherDataLoader(data_path)
    
    # Load data
    df = loader.load_data(sample_rate=sample_rate)
    
    # Plot data overview
    loader.plot_data_overview()
    
    # Preprocess and split data
    train_df, val_df, test_df = loader.preprocess_data()
    
    # Print data statistics
    stats = loader.get_feature_stats()
    print(f"\nDataset Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Features: {stats['columns']}")
    print(f"  Mean temperature: {stats['mean']['T (degC)']:.2f}°C")
    print(f"  Std temperature: {stats['std']['T (degC)']:.2f}°C")
    
    return train_df, val_df, test_df, loader


def create_single_step_models(train_df, val_df, test_df):
    """
    Create and evaluate single-step forecasting models.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        
    Returns:
        ModelTrainer instance with trained models
    """
    print("\n" + "=" * 50)
    print("SINGLE-STEP FORECASTING")
    print("=" * 50)
    
    # Create window generator for single-step prediction
    single_step_window = create_single_step_window(train_df, val_df, test_df)
    print(f"Single-step window configuration:")
    print(single_step_window)
    
    # Initialize trainer
    trainer = ModelTrainer(single_step_window)
    
    # Add baseline model
    baseline_model = create_baseline_model('single_step')
    trainer.add_model('Baseline', baseline_model)
    trainer.compile_model('Baseline')
    
    # Add LSTM model
    lstm_model = create_lstm_model(
        model_type='single_step',
        units=32,
        num_layers=2,
        dropout=0.1,
        num_features=1
    )
    trainer.add_model('LSTM_Single', lstm_model)
    trainer.compile_model('LSTM_Single')
    
    # Train models
    trainer.train_all_models(epochs=20, patience=2)
    
    # Evaluate models
    performance = trainer.evaluate_all_models()
    
    # Compare models
    comparison = trainer.compare_models('test_loss')
    
    # Plot predictions
    trainer.plot_predictions(['Baseline', 'LSTM_Single'])
    
    return trainer


def create_multi_step_models(train_df, val_df, test_df):
    """
    Create and evaluate multi-step forecasting models.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        
    Returns:
        ModelTrainer instance with trained models
    """
    print("\n" + "=" * 50)
    print("MULTI-STEP FORECASTING")
    print("=" * 50)
    
    # Create window generator for multi-step prediction
    multi_step_window = create_multi_step_window(train_df, val_df, test_df, out_steps=24)
    print(f"Multi-step window configuration:")
    print(multi_step_window)
    
    # Initialize trainer
    trainer = ModelTrainer(multi_step_window)
    
    # Add baseline model
    baseline_model = create_baseline_model('multi_step', out_steps=24)
    trainer.add_model('Baseline_Multi', baseline_model)
    trainer.compile_model('Baseline_Multi')
    
    # Add multi-shot LSTM model
    multi_shot_model = create_lstm_model(
        model_type='multi_shot',
        units=32,
        num_layers=2,
        dropout=0.1,
        out_steps=24,
        num_features=1
    )
    trainer.add_model('LSTM_MultiShot', multi_shot_model)
    trainer.compile_model('LSTM_MultiShot')
    
    # Add autoregressive LSTM model
    autoregressive_model = create_lstm_model(
        model_type='autoregressive',
        units=32,
        out_steps=24,
        num_features=1
    )
    trainer.add_model('LSTM_Autoregressive', autoregressive_model)
    trainer.compile_model('LSTM_Autoregressive')
    
    # Train models
    trainer.train_all_models(epochs=20, patience=2)
    
    # Evaluate models
    performance = trainer.evaluate_all_models()
    
    # Compare models
    comparison = trainer.compare_models('test_loss')
    
    # Plot predictions for best model
    if not comparison.empty:
        best_model = comparison.iloc[0]['Model']
        print(f"\nPlotting predictions for best model: {best_model}")
        trainer.plot_predictions([best_model])
    
    return trainer


def main():
    """Main function to run the complete time series forecasting workflow."""
    parser = argparse.ArgumentParser(
        description='Time Series Forecasting with TensorFlow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run complete workflow
  python main.py --single-only            # Single-step forecasting only
  python main.py --multi-only             # Multi-step forecasting only
  python main.py --data-path data.csv     # Use local data file
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to local data file (optional)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=6,
        help='Data sampling rate (default: 6 for hourly data)'
    )
    
    parser.add_argument(
        '--single-only',
        action='store_true',
        help='Run single-step forecasting only'
    )
    
    parser.add_argument(
        '--multi-only',
        action='store_true',
        help='Run multi-step forecasting only'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        default=True,
        help='Save model results to files'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Setup matplotlib
    setup_matplotlib()
    
    print("TIME SERIES FORECASTING WITH TENSORFLOW")
    print("=" * 50)
    
    # Load and preprocess data
    train_df, val_df, test_df, loader = load_and_preprocess_data(
        data_path=args.data_path,
        sample_rate=args.sample_rate
    )
    
    # Initialize results storage
    all_trainers = {}
    
    # Run single-step forecasting
    if not args.multi_only:
        single_trainer = create_single_step_models(train_df, val_df, test_df)
        all_trainers['single_step'] = single_trainer
    
    # Run multi-step forecasting
    if not args.single_only:
        multi_trainer = create_multi_step_models(train_df, val_df, test_df)
        all_trainers['multi_step'] = multi_trainer
    
    # Save results
    if args.save_results:
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        results_dir = Path(args.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        for task_name, trainer in all_trainers.items():
            task_dir = results_dir / task_name
            save_model_results(trainer, str(task_dir))
        
        print(f"\nAll results saved to: {results_dir}")
    
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
