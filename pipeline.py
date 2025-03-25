import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from logging_config import logger

def load_subject_timeframe(filepath, window_size_in_seconds = 1, drop_non_study_labels = False, remove_outliers = True):
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

    for signal_device, signal_name, sampling_freq in signals:
        if signal_device in data:
            signal_data = data[signal_device][signal_name]
            df = pd.DataFrame(signal_data)
            if len(df.columns) == 1:
                df.columns = [f'{signal_device}_{signal_name}']
            else:
                df.columns = [f'{signal_device}_{signal_name}_{axis}' for axis in ['x', 'y', 'z']]
            if remove_outliers:
                z_scores = (df - df.mean()) / df.std()
                df = df[(z_scores.abs() < 3).all(axis=1)]
            df = df.groupby(df.index // sampling_freq * window_size_in_seconds).agg(['mean', 'std'])
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

def label_correlation_analysis(subject_id, X, y):
    label_map = {
        '0': 'Transient',
        '1': 'Baseline',
        '2': 'Stress',
        '3': 'Amusement',
        '4': 'Meditation',
        '5': 'Ignore',
        '6': 'Ignore',
        '7': 'Ignore'
    }

    data = X.copy()
    data['label'] = y

    grouped = data.groupby('label').mean()
    transposed = grouped.transpose()
    transposed.columns = [label_map[str(col)] for col in transposed.columns]
    normalised = (transposed - transposed.mean()) / transposed.std()
    normalised = (normalised - normalised.min()) / (normalised.max() - normalised.min())

    fig, ax = plt.subplots()
    normalised.plot(kind='bar', ax=ax)
    ax.set_title(f'{subject_id} - Clustered Feature Means (Normalised and Scaled)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    
    fig.set_size_inches(20, 15)
    fig.set_dpi(500)

    return fig

def timeframe_analysis(subject_id, X, y, feature, scatter=False):
    label_map = {
        '0': 'Transient',
        '1': 'Baseline',
        '2': 'Stress',
        '3': 'Amusement',
        '4': 'Meditation',
        '5': 'Ignore',
        '6': 'Ignore',
        '7': 'Ignore'
    }

    X = (X - X.mean()) / X.std()
    X = (X - X.min()) / (X.max() - X.min())

    fig, ax = plt.subplots()
    unique_labels = y.unique()
    for label in unique_labels:
        section = X[y == label]
        if scatter:
            ax.scatter(section.index, section[feature], label=label_map[str(label)])
        else:
            ax.plot(section.index, section[feature], label=label_map[str(label)], drawstyle='steps-post')

    ax.set_title(f'{subject_id} - {feature} Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalised Value')
    ax.legend()

    fig.set_size_inches(20, 15)
    fig.set_dpi(500)

    return fig

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

def analyze_all_subjects(data_dir, subject_files, output_dir):
    for subject_file in subject_files:
        subject_id = subject_file.split('.')[0]
        logger.info(f'Analyzing subject {subject_id}')
        subject_output_dir = f'{output_dir}/{subject_id}'
        os.makedirs(subject_output_dir, exist_ok=True)
        os.makedirs(f'{subject_output_dir}/timeframe', exist_ok=True)
        os.makedirs(f'{subject_output_dir}/balanced_scatter', exist_ok=True)
        X, y = load_subject_timeframe(os.path.join(data_dir, subject_file), remove_outliers=False)
        fig = label_correlation_analysis(subject_id, X, y)
        fig.savefig(f'{subject_output_dir}/{subject_id}_label_correlation_analysis.png')

        for feature in X.columns:
            fig = timeframe_analysis(subject_id, X, y, feature)
            fig.savefig(f'{subject_output_dir}/timeframe/{subject_id}_{feature}.png')

        X, y = balance_classes(X, y)
        for feature in X.columns:
            fig = timeframe_analysis(subject_id, X, y, feature, scatter=True)
            fig.savefig(f'{subject_output_dir}/balanced_scatter/{subject_id}_{feature}.png')

def run_fold(data_dir, train_files, test_files):
    X_train, y_train = pd.DataFrame(), pd.Series()
    for train_file in train_files:
        X, y = load_subject_timeframe(os.path.join(data_dir, train_file))
        X_train = pd.concat([X_train, X])
        y_train = pd.concat([y_train, y])

    X_test, y_test = pd.DataFrame(), pd.Series()
    for test_file in test_files:
        X, y = load_subject_timeframe(os.path.join(data_dir, test_file))
        X_test = pd.concat([X_test, X])
        y_test = pd.concat([y_test, y])

    X_train, y_train = balance_classes(X_train, y_train)
    report = run_rf(X_train, y_train, X_test, y_test)
    return report

def run_folds(data_dir, folds):
    reports = []
    for i, (train_files, test_files) in enumerate(folds):
        logger.info(f'Running fold {i + 1}...')
        report = run_fold(data_dir, train_files, test_files)
        reports.append(report)
    return reports

if __name__ == '__main__':
    # os.makedirs('.analysis', exist_ok=True)
    # subject_files = [file for file in os.listdir('.data') if file.endswith('.pkl')]
    # analyze_all_subjects('.data', subject_files, '.analysis')
    os.makedirs('.out', exist_ok=True)
    logger.info('Running all folds')
    folds = generate_folds('.data', 5, 42)
    reports = run_folds('.data', folds)
    for i, report in enumerate(reports):
        with open(f'.out/fold_{i + 1}_report.txt', 'w') as file:
            file.write(report)
    logger.info('All folds completed')