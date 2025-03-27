import os
import pickle
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from logging_config import logger

def load_subject_timeframe(filepath, **kwargs):
    window_size_in_seconds = kwargs.get('window_size_in_seconds', 0.25) 
    drop_non_study_labels = kwargs.get('drop_non_study_labels', False)
    clip_outliers = kwargs.get('clip_outliers', True)
    outlier_range = kwargs.get('outlier_range', (0.2, 0.8))
    outlier_threshold = kwargs.get('outlier_threshold', 1.5)

    logger.info(f'Loading subject timeframe from {filepath}')
    with open(filepath, 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        labels = pd.Series(subject['label'])
        data = subject['signal']
    
    signals = [ # (signal_device, signal_name, sampling_freq)
        ('chest', 'ACC', 700),
        ('chest', 'ECG', 700),
        ('chest', 'EMG', 700),
        ('chest', 'EDA', 700),
        ('chest', 'Temp', 700),
        ('chest', 'Resp', 700),
        ('wrist', 'ACC', 32),
        ('wrist', 'BVP', 64),
        ('wrist', 'EDA', 4),
        ('wrist', 'TEMP', 4)
    ]

    dfs = []

    # Custom Features
    def histogram_spread(group):
        hist = group.value_counts(bins=10)
        return hist.max() - hist.min()
    
    def linear_gradient(group):
        x = range(len(group))
        y = group.values
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]
        return 0

    for signal_device, signal_name, sampling_freq in signals:
        if signal_device in data:
            signal_data = data[signal_device][signal_name]
            df = pd.DataFrame(signal_data)
            if len(df.columns) == 1:
                df.columns = [f'{signal_device}_{signal_name}']
            else:
                df.columns = [f'{signal_device}_{signal_name}_{axis}' for axis in ['x', 'y', 'z']]
            if clip_outliers:
                grouped = df.groupby(df.index // (sampling_freq * window_size_in_seconds))
                Q1 = grouped.transform(lambda x: x.quantile(outlier_range[0]))
                Q3 = grouped.transform(lambda x: x.quantile(outlier_range[1]))
                IQR = Q3 - Q1
                df = df.clip(lower=Q1 - outlier_threshold * IQR, upper=Q3 + outlier_threshold * IQR, axis=1)
            df = df.groupby(df.index // sampling_freq * window_size_in_seconds).agg(['mean', 'std', histogram_spread, linear_gradient])
            df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
            dfs.append(df)
        
    label_func = lambda group: group.mode()[0] if not group.mode().empty else group.unique()[0]
    
    X = pd.concat(dfs, axis=1)
    y = labels.groupby(labels.index // 700 * window_size_in_seconds).apply(label_func)

    if drop_non_study_labels:
        discard = y.isin([0, 5, 6, 7])
        X = X[~discard]
        y = y[~discard]

    return X, y

def balance_classes(X, y):
    logger.info('Balancing classes')
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def generate_folds(data_dir, k, random_state):
    logger.info(f'Generating {k} folds with random state {random_state}')
    files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []
    for train_index, test_index in kf.split(files):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]
        folds.append((train_files, test_files))
    return folds

def run_rf(X_train, y_train, X_test, y_test):
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train_min = X_train.min()
    X_train_max = X_train.max()

    X_train = (X_train - X_train_mean) / X_train_std
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)

    X_test = (X_test - X_train_mean) / X_train_std
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

def load_subject_data(data_dir, subject_files, **tfargs):
    subject_data = {}
    logger.info('Loading all subject data')
    for subject_file in subject_files:
        filepath = os.path.join(data_dir, subject_file)
        X, y = load_subject_timeframe(filepath, **tfargs)
        subject_data[subject_file] = (X, y)
    return subject_data

def run_fold(train_files, test_files, subject_data):
    X_train, y_train = pd.DataFrame(), pd.Series()
    for train_file in train_files:
        X, y = subject_data[train_file]
        X_train = pd.concat([X_train, X])
        y_train = pd.concat([y_train, y])

    X_test, y_test = pd.DataFrame(), pd.Series()
    for test_file in test_files:
        X, y = subject_data[test_file]
        X_test = pd.concat([X_test, X])
        y_test = pd.concat([y_test, y])

    X_train, y_train = balance_classes(X_train, y_train)
    report = run_rf(X_train, y_train, X_test, y_test)
    return report

def run_folds(folds, subject_data):
    reports = []
    for i, (train_files, test_files) in enumerate(folds):
        logger.info(f'Running fold {i + 1}...')
        report = run_fold(train_files, test_files, subject_data)
        reports.append(report)
    return reports

if __name__ == '__main__':
    os.makedirs('.out', exist_ok=True)
    subject_files = [file for file in os.listdir('.data') if file.endswith('.pkl')]
    subject_data = load_subject_data(
        '.data', 
        subject_files, 
        window_size_in_seconds=5, 
        drop_non_study_labels=False, 
        clip_outliers=True
    )
    folds = generate_folds('.data', 5, 42)
    logger.info('Running all folds')
    reports = run_folds(folds, subject_data)
    for i, report in enumerate(reports):
        with open(f'.out/fold_{i + 1}_report.txt', 'w') as file:
            file.write(report)
    logger.info('All folds completed')