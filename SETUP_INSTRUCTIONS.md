# Setup and GitHub Push Instructions

## Project Overview

This time series forecasting project has been created with two branches:

### Main Branch (`main`)
- Complete TensorFlow-based time series forecasting framework
- Data loading and preprocessing for Jena climate dataset
- Multiple model types (baseline, LSTM variants)
- Training and evaluation utilities
- Comprehensive documentation

### JAX VAE Branch (`jax-vae`)
- JAX/Flax implementation of Variational Autoencoders
- Adapted for time series forecasting
- Training utilities with Optax optimization
- Example scripts with visualizations
- Latent space interpolation demonstrations

## Repository Structure

```
time_series_forecasting/
├── main.py                     # Main TensorFlow training script
├── requirements.txt            # Dependencies (includes JAX)
├── README.md                   # Project documentation
├── SETUP_INSTRUCTIONS.md       # This file
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Data loading and preprocessing
│   │   └── window_generator.py # Time series windowing
│   ├── models/
│   │   ├── baseline.py        # Baseline forecasting models
│   │   ├── lstm.py           # LSTM forecasting models
│   │   └── jax_vae.py        # JAX VAE models (jax-vae branch)
│   └── utils/
│       ├── training.py        # TensorFlow training utilities
│       └── jax_training.py    # JAX training utilities (jax-vae branch)
├── examples/
│   └── jax_vae_example.py    # JAX VAE example (jax-vae branch)
├── tests/                     # Unit tests (placeholder)
└── docs/                      # Documentation (placeholder)
```

## Local Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/maxim/Desktop/programming/time_series_forecasting
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### TensorFlow (Main Branch)
```bash
# Run complete workflow
python main.py

# Run only single-step forecasting
python main.py --single-only

# Run only multi-step forecasting
python main.py --multi-only
```

### JAX VAE (JAX Branch)
```bash
# Switch to JAX branch
git checkout jax-vae

# Run JAX VAE example
python examples/jax_vae_example.py
```

## GitHub Setup Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com and create a new repository
2. Name it something like `time-series-forecasting`
3. Choose "Public" or "Private" as preferred
4. **Do not** initialize with README, .gitignore, or license (we already have these)

### Step 2: Add Remote and Push

```bash
# Add your GitHub repository as remote (replace with your URL)
git remote add origin https://github.com/YOUR_USERNAME/time-series-forecasting.git

# Push main branch
git push -u origin main

# Push JAX branch
git push -u origin jax-vae
```

### Step 3: Verify on GitHub

1. Go to your GitHub repository
2. You should see both branches:
   - `main` - TensorFlow implementation
   - `jax-vae` - JAX VAE implementation
3. Check that all files are present and correctly organized

## Branch Usage

### Main Branch (TensorFlow)
- **Purpose**: Production-ready time series forecasting
- **Models**: Baseline, LSTM (single-step, multi-shot, autoregressive)
- **Use Case**: Traditional forecasting tasks with proven TensorFlow models

### JAX VAE Branch
- **Purpose**: Research and experimentation with VAEs
- **Models**: Variational Autoencoders adapted for time series
- **Use Case**: Unsupervised learning, anomaly detection, data generation

## Switching Between Branches

```bash
# Switch to main branch (TensorFlow)
git checkout main

# Switch to JAX branch
git checkout jax-vae

# See current branch
git branch

# See all branches
git branch -a
```

## Next Steps

1. **Push to GitHub** using the instructions above
2. **Test the installation** by running the example scripts
3. **Explore the code** and modify for your specific use cases
4. **Add your own models** following the established patterns
5. **Extend the documentation** with your findings

## Dependencies

### TensorFlow Branch Requirements
- tensorflow>=2.13.0
- pandas>=1.5.0
- numpy>=1.24.0
- matplotlib>=3.6.0
- scikit-learn>=1.2.0
- seaborn>=0.12.0

### Additional JAX Requirements
- jax>=0.4.0
- flax>=0.7.0
- optax>=0.1.0
- xarray>=2023.1.0

Note: JAX dependencies are included in requirements.txt but are only needed for the jax-vae branch.

## Troubleshooting

### Common Issues

1. **JAX Installation**: JAX may require specific versions for your hardware. Check the [JAX installation guide](https://github.com/google/jax#installation).

2. **TensorFlow Compatibility**: Ensure your Python version is compatible with TensorFlow 2.13+.

3. **Memory Issues**: If you encounter memory errors, reduce batch sizes in the training scripts.

4. **Data Download**: The weather dataset is downloaded automatically on first run. Ensure you have internet access.

### Getting Help

- Check the README.md for detailed usage instructions
- Review the example scripts for implementation patterns
- Examine the docstrings in each module for function details
- Create issues on GitHub for bugs or feature requests
