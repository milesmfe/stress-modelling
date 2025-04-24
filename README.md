# WESAD Stress Analysis and Modeling

![Project Banner](https://via.placeholder.com/800x200?text=WESAD+Stress+Analysis)

## Overview

The WESAD (Wearable Stress and Affect Detection) Pipeline is a comprehensive machine learning framework designed to process and analyze physiological signals for stress detection. This project leverages multi-modal sensor data collected from wearable devices to extract meaningful features and build robust machine learning models for stress classification.

## Features

- **Advanced Signal Processing**: Extract time-domain, frequency-domain, and entropy features from physiological signals
- **Multi-Modal Integration**: Process data from both chest and wrist-worn devices, including ECG, EMG, EDA, respiration, temperature, and accelerometer data
- **Parallel Processing**: Optimize feature extraction with multi-threading for improved performance
- **Flexible Model Selection**: Choose from a variety of machine learning models through a simple registry system
- **Cross-Validation**: Evaluate models with rigorous k-fold cross-validation protocols
- **Comprehensive Metrics**: Generate and save detailed performance reports and visualizations

## Requirements

- Python 3.7+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- AntroPy (for entropy-based features)
- imbalanced-learn
- XGBoost
- SKTime/Aeon (for time series classifiers)

## Installation

1. Set up a virtual environment (recommended):

```bash
python -m venv wesad-env
source wesad-env/bin/activate  # On Windows: wesad-env\Scripts\activate
```

2. Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn antropy imbalanced-learn xgboost sktime aeon
```

## Project Structure

```
wesad-pipeline/
├── features/
│   └── feature_extractor.py      # Signal feature extraction utilities
├── models/
│   └── model_registry.py         # Model registry for ease of use
├── utils/
│   ├── arguments.py              # Command-line argument parsing
│   ├── cross_validation.py       # Cross-validation implementation
│   ├── data_loader.py            # Data loading utilities
│   ├── evaluation.py             # Model evaluation and metrics
│   ├── feature_selector.py       # Feature selection utilities
│   └── logger.py                 # Logging configuration
└── main.py                       # Main application entry point
```

## Usage

### Basic Usage

To run the pipeline with default settings:

```bash
python main.py --data_dir /path/to/wesad/data --model_name RandomForest
```

### Command-Line Arguments

The pipeline supports various command-line arguments to customize processing:

```bash
python main.py \
  --data_dir /path/to/wesad/data \
  --results_dir results_folder \
  --model_name RandomForest \
  --n_splits 5 \
  --feature_selection \
  --binary_classification
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Directory containing WESAD .pkl files | Required |
| `--results_dir` | Directory to save results | `results_YYYYMMDD_HHMMSS` |
| `--use_cache` | Path to cache file for preloaded data | None |
| `--drop_non_study` | Drop labels 0, 5, 6, 7 | False |
| `--shorten_non_study` | Shorten non-study labels to 0 | False |
| `--n_splits` | Number of CV folds | 5 |
| `--save_datasets` | Save train/test datasets per fold | False |
| `--model_name` | Model name in registry | RandomForest |
| `--feature_selection` | Enable feature selection | False |
| `--binary_classification` | Use binary classification | False |
| `--imputer` | Imputation strategy | mean |

## Feature Extraction

The pipeline extracts a comprehensive set of features from physiological signals:

### Time Domain Features
- Basic statistics (mean, std, max, min, skewness, kurtosis)
- Peak analysis (number of peaks, average peak distance)
- For ECG signals: HRV metrics (RMSSD, SDNN, pNN50)

### Frequency Domain Features
- Band powers (low, mid, high frequency)
- FFT mean and entropy

### Complexity Features
- Sample entropy
- Higuchi fractal dimension
- Detrended fluctuation analysis

Example of feature extraction from a signal:

```python
# Extract features from ECG signal
features = extract_features(ecg_signal, "chest_ECG", fs=700)

# Example output:
# {
#   'chest_ECG_mean': 0.153,
#   'chest_ECG_std': 0.423,
#   'chest_ECG_max': 1.287,
#   'chest_ECG_min': -0.857,
#   'chest_ECG_skew': 0.374,
#   'chest_ECG_kurtosis': 3.142,
#   'chest_ECG_power_low': 0.035,
#   'chest_ECG_power_mid': 0.021,
#   'chest_ECG_power_high': 0.012,
#   'chest_ECG_fft_mean': 0.023,
#   'chest_ECG_fft_entropy': 0.872,
#   'chest_ECG_num_peaks': 83,
#   'chest_ECG_avg_peak_dist': 8.45
# }
```

## Data Processing Pipeline

The pipeline follows these main steps:

1. **Data Loading**: Load subject data from pickle files
2. **Window Generation**: Create time windows for feature extraction
3. **Feature Extraction**: Extract features from each signal and dimension
4. **Preprocessing**: Handle missing values, scale features
5. **Feature Selection** (optional): Select most important features
6. **Model Training**: Train the selected classifier
7. **Evaluation**: Compute and save performance metrics

## Available Models

The following models are available through the model registry:

- **RandomForest**: Random Forest Classifier
- **SVM**: Support Vector Machine with RBF kernel
- **KNN**: k-Nearest Neighbors
- **XGBoost**: XGBoost Classifier
- **CanonicalIntervalForest**: Canonical Interval Forest (time series)
- **TimeSeriesForest**: Time Series Forest Classifier
- **QUANT**: QUANT Classifier (time series)

To add a new model, simply update the `MODEL_REGISTRY` dictionary in `models/model_registry.py`:

```python
MODEL_REGISTRY = {
    # ... existing models ...
    "MyNewModel": lambda: MyModelClass(param1=value1, param2=value2)
}
```

## Cross-Validation

The pipeline uses k-fold cross-validation at the subject level, ensuring that data from the same subject doesn't appear in both training and testing sets:

```python
# Example cross-validation logic
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(file_keys), start=1):
    train_files = [file_keys[i] for i in train_idx]
    test_files = [file_keys[i] for i in test_idx]
    
    # Process and train on this fold
    # ...
```

## Example Workflow

Here's an example workflow for a complete analysis:

1. **Prepare WESAD Dataset**: Ensure you have the WESAD dataset as pickle files
2. **Run Full Cross-Validation**:

```bash
python main.py \
  --data_dir /path/to/wesad/data \
  --results_dir stress_analysis_results \
  --model_name RandomForest \
  --n_splits 5 \
  --feature_selection \
  --save_datasets
```

3. **Analyze Results**: Examine confusion matrices and classification reports in the results directory

## Extending the Pipeline

### Adding New Features

To add new features, extend the `extract_signal_features` function in `features/feature_extractor.py`:

```python
def extract_my_new_feature(signal):
    # Implement your feature extraction logic
    return {"my_new_feature": calculated_value}

def extract_signal_features(win, signal_name, fs):
    features = {}
    # ... existing code ...
    
    # Add your new feature
    my_features = extract_my_new_feature(signal)
    features.update({f"{signal_name}_{k}": v for k, v in my_features.items()})
    
    return features
```

### Adding New Signals

To support new signal types, update the `SIGNAL_CONFIG` dictionary in `utils/data_loader.py`:

```python
SIGNAL_CONFIG = {
    # ... existing signals ...
    ('new_device', 'NEW_SIGNAL'): {'fs': 256, 'window_s': 1.0},
}
```

## Troubleshooting

### Missing Values

If you encounter issues with missing values, try using a different imputation strategy:

```bash
python main.py --data_dir /path/to/data --imputer median
```

### Memory Issues

For large datasets, consider processing subjects one by one instead of preloading all data:

```bash
python main.py --data_dir /path/to/data --use_cache cache_file.pkl
```

### Library Compatibility

If you encounter issues with library dependencies, ensure you have compatible versions:

```bash
pip install -r requirements.txt  # Create this file with specific version numbers
```

## Contributing

Contributions to improve the WESAD pipeline are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The WESAD dataset creators for providing multimodal physiological signals
- Contributors to the scientific libraries used in this project

---

*For questions or support, please open an issue in the repository.*
