# WESAD Pipeline

An advanced pipeline for processing and analyzing the [WESAD (Wearable Stress and Affect Detection) dataset](https://archive.ics.uci.edu/ml/datasets/WESAD) with machine learning models.

## Description

This project implements an end-to-end machine learning pipeline for processing physiological signals from wearable devices to detect stress and emotional states. It uses advanced feature extraction techniques, parallel processing, and supports multiple classification models. 

The pipeline includes tools for feature extraction, cross-validation, data visualization, and model evaluation, making it suitable for researchers and practitioners working with wearable sensor data.

## Methods & Pipeline Flow

The pipeline follows these key steps:

1. **Data Loading**: Reads physiological signal data from WESAD pickle files
2. **Signal Windowing**: Segments signals into time windows with signal-specific durations
3. **Feature Extraction**: Computes statistical, frequency-domain, and complexity features
4. **Preprocessing**: Applies imputation for missing values and standardization
5. **Feature Selection**: Optional feature importance-based selection using Random Forest
6. **Model Training**: Trains the selected classifier with cross-validation
7. **Evaluation**: Generates classification reports and confusion matrices

Pipeline execution is controlled through [main.py](main.py), which coordinates the entire workflow. The system supports caching preprocessed data to speed up experimentation.

## Technical Highlights

- **Parallel Processing**: Uses [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) for efficient feature extraction and window processing
- **Advanced Signal Processing**: Employs techniques from [`scipy.signal`](https://docs.scipy.org/doc/scipy/reference/signal.html) including Welch's method and peak detection
- **Complexity Measures**: Calculates [Higuchi Fractal Dimension](https://pypi.org/project/antropy/), [Sample Entropy](https://en.wikipedia.org/wiki/Sample_entropy), and [Detrended Fluctuation Analysis](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis) for biosignals
- **Model Registry Pattern**: Implements a clean registry pattern for plug-and-play model integration
- **Cross-Subject Validation**: Uses K-fold cross-validation with subject isolation to ensure robust evaluation
- **Signal-Specific Processing**: Applies tailored window sizes and sampling rates per signal type

## Notable Libraries

- [AntroPy](https://github.com/raphaelvallat/antropy): Advanced entropy and complexity measures for time series
- [scikit-learn](https://scikit-learn.org/): Machine learning tools for classification and evaluation
- [imbalanced-learn](https://imbalanced-learn.org/): Tools for handling class imbalance (RandomUnderSampler)
- [sktime](https://www.sktime.org/en/stable/): Advanced time series classification algorithms
- [aeon](https://github.com/aeon-toolkit/aeon): Time series machine learning toolkit with state-of-the-art classifiers
- [XGBoost](https://xgboost.readthedocs.io/): Gradient boosting implementation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/): Data visualization
- [Tkinter](https://docs.python.org/3/library/tkinter.html): GUI toolkit for the timeframe grapher

## Project Structure

```
wesad-pipeline/
├── features/
│   └── feature_extractor.py
├── models/
│   └── model_registry.py
├── utils/
│   ├── arguments.py
│   ├── cross_validation.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_selector.py
│   └── logger.py
├── timeframe_grapher.py
└── main.py
```

- **features/**: Contains the feature extraction logic for physiological signals
- **models/**: Houses the model registry with various classifiers
- **utils/**: Helper modules for data loading, evaluation, cross-validation, and logging
