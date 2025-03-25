import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

def load_subject_timeframe(filepath, window_size = 1):
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
            df = df.groupby(df.index // sampling_freq * window_size).agg(['mean', 'std'])
            df.columns = [f'{col[0]}_{col[1]}' for col in df.columns]
            dfs.append(df)
        
    label_func = lambda group: group.mode()[0] if not group.mode().empty else group.unique()[0]
    
    X = pd.concat(dfs, axis=1)
    y = labels.groupby(labels.index // 700 * window_size).apply(label_func)

    # discard = y.isin([5, 6, 7])

    # X = X[~discard]
    # y = y[~discard]

    return X, y

def balance_classes(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def generate_folds(data_dir, k, random_state):
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

def timeframe_analysis(subject_id, X, y, feature):
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
        X, y = load_subject_timeframe(os.path.join(data_dir, subject_file))
        fig = label_correlation_analysis(subject_id, X, y)
        fig.savefig(f'{output_dir}/{subject_id}_label_correlation_analysis.png')

        for feature in X.columns:
            fig = timeframe_analysis(subject_id, X, y, feature)
            fig.savefig(f'{output_dir}/{subject_id}_{feature}.png')

if __name__ == '__main__':
    # # Test
    # X, y = load_subject_timeframe('.data/S2.pkl')
    # for feature in X.columns:
    #     fig = timeframe_analysis('S2', X, y, feature)
    #     fig.savefig(f'test/S2_{feature}.png')
    os.makedirs('.analysis', exist_ok=True)
    subject_files = [file for file in os.listdir('.data') if file.endswith('.pkl')]
    analyze_all_subjects('.data', subject_files, '.analysis')