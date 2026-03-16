# Time Series Forecasting Project

A comprehensive time series forecasting project using TensorFlow, focusing on weather data prediction with various neural network architectures.

## Overview

This project implements and compares different approaches to time series forecasting using the Jena climate dataset. It includes baseline models, LSTM-based models, and provides a complete workflow for training, evaluation, and comparison.

## Features

- **Data Loading and Preprocessing**: Automated downloading and preprocessing of the Jena climate dataset
- **Window Generation**: Flexible windowing system for creating training samples
- **Multiple Model Types**:
  - Baseline models (last value prediction)
  - LSTM models (single-step, multi-shot, autoregressive)
- **Training Utilities**: Comprehensive training and evaluation framework
- **Visualization**: Built-in plotting for data exploration and model predictions
- **Comparison Tools**: Model performance comparison and analysis

## Project Structure

```
time_series_forecasting/
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   └── window_generator.py # Window generation for time series
│   ├── models/
│   │   ├── baseline.py         # Baseline forecasting models
│   │   └── lstm.py            # LSTM-based forecasting models
│   └── utils/
│       └── training.py         # Training and evaluation utilities
├── tests/                  # Unit tests (to be implemented)
├── examples/               # Example scripts and notebooks
└── docs/                   # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd time_series_forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete forecasting workflow:
```bash
python main.py
```

### Advanced Usage

Run only single-step forecasting:
```bash
python main.py --single-only
```

Run only multi-step forecasting:
```bash
python main.py --multi-only
```

Use local data file:
```bash
python main.py --data-path path/to/your/data.csv
```

Save results to custom directory:
```bash
python main.py --results-dir my_results
```

### Command Line Options

- `--data-path`: Path to local data file (optional)
- `--sample-rate`: Data sampling rate (default: 6 for hourly data)
- `--single-only`: Run single-step forecasting only
- `--multi-only`: Run multi-step forecasting only
- `--save-results`: Save model results to files (default: True)
- `--results-dir`: Directory to save results (default: results)

## Models

### Baseline Models

- **Baseline**: Predicts the last known value (single-step)
- **MultiStepLastBaseline**: Repeats the last value for all future steps
- **RepeatBaseline**: Repeats the input sequence

### LSTM Models

- **LSTMSingleStep**: LSTM for single-step predictions
- **LSTMMultiShot**: LSTM that predicts all steps at once
- **FeedBack**: Autoregressive LSTM that predicts step-by-step

## Data

This project uses the Jena climate dataset, which contains weather data recorded every 10 minutes from 2009 to 2016. The dataset includes 14 different features:

- Temperature (°C)
- Atmospheric pressure (mbar)
- Relative humidity (%)
- Wind speed (m/s)
- Wind direction (deg)
- And more...

The data is automatically sampled to hourly intervals for efficient processing.

## Example Results

### Single-Step Forecasting
- Baseline MAE: ~0.29
- LSTM MAE: ~0.24

### Multi-Step Forecasting (24 hours)
- Baseline MAE: ~0.47
- LSTM Multi-Shot MAE: ~0.35
- LSTM Autoregressive MAE: ~0.33

## Development

### Adding New Models

1. Create a new model class in `src/models/`
2. Implement the `call` method
3. Add the model to the training workflow in `main.py`
4. Update the factory functions if needed

### Testing

Run tests (when implemented):
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Based on TensorFlow's time series forecasting tutorial
- Uses the Jena climate dataset from the Max Planck Institute for Biogeochemistry
- Inspired by François Chollet's "Deep Learning with Python"

## Future Enhancements

- [ ] Add more advanced models (Transformers, GRU, etc.)
- [ ] Implement hyperparameter tuning
- [ ] Add support for custom datasets
- [ ] Create interactive visualizations
- [ ] Add model explainability tools
- [ ] Implement ensemble methods

# Проект прогнозирования временных рядов

**ВНИМАНИЕ:** Этот проект создан автоматически для демонстрации возможностей прогнозирования.

## 📁 Что это

Проект для прогнозирования погодных данных с использованием нейронных сетей.

### Две ветки:

1. **main** - TensorFlow модели (простые и надежные)
2. **jax-vae** - JAX VAE модели (для экспериментов)

## 🚀 Быстрый старт

### Шаг 1: Установка
```bash
# Клонируйте проект
git clone https://github.com/MaxGoodnic/time-series-forecasting.git
cd time-series-forecasting

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### Шаг 2: Запуск TensorFlow моделей
```bash
python main.py
```

### Шаг 3: Запуск JAX VAE моделей
```bash
git checkout jax-vae
python examples/jax_vae_example.py
```

## 📂 Структура проекта

```
time_series_forecasting/
├── main.py                 # Основной скрипт TensorFlow
├── src/
│   ├── data/              # Загрузка данных
│   ├── models/            # Модели (LSTM, VAE)
│   └── utils/             # Утилиты обучения
├── examples/             # Примеры использования
└── README.md            # Этот файл
```

## 🎯 Что можно делать

### Ветка main (TensorFlow):
- Прогнозирование на 1 шаг вперед
- Прогнозирование на 24 часа вперед
- Сравнение разных моделей
- Визуализация результатов

### Ветка jax-vae (JAX):
- Обучение VAE на временных рядах
- Восстановление данных
- Интерполяция в латентном пространстве
- Генерация новых последовательностей

## 📊 Результаты

### TensorFlow модели:
- Baseline MAE: ~0.29
- LSTM MAE: ~0.24

### JAX VAE:
- Восстановление с ошибкой ~0.15
- Интерполяция между последовательностями

## ⚙️ Настройка

### Изменить параметры обучения:
```python
# В main.py
epochs = 50
batch_size = 64
learning_rate = 0.001
```

### Использовать свои данные:
```python
# Замените загрузчик данных
data_path = "path/to/your/data.csv"
```

## 🔧 Команды

### TensorFlow:
```bash
python main.py --single-only      # Только одинарные прогнозы
python main.py --multi-only       # Только множественные прогнозы
python main.py --results-dir my_results  # Сохранить результаты
```

### JAX:
```bash
python examples/jax_vae_example.py
```

## 📝 Документация

- Полная документация в коде (docstrings)
- Примеры в папке `examples/`
- Инструкции в `SETUP_INSTRUCTIONS.md`

## ⚠️ Важные замечания

1. **Требования:** Python 3.8+, TensorFlow 2.13+
2. **Данные:** Автоматически скачиваются при первом запуске
3. **Память:** Для больших моделей нужно 8GB+ RAM
4. **JAX:** Требует CUDA для GPU ускорения

## 🆘 Поддержка

Если что-то не работает:
1. Проверьте версию Python: `python --version`
2. Обновите pip: `pip install --upgrade pip`
3. Пересоздайте окружение: `rm -rf venv && python -m venv venv`

---

**Автор:** Сгенерировано автоматически  
**Назначение:** Образовательный проект  
**Статус:** Готов к использованию
