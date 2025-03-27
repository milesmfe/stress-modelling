import os

from matplotlib import pyplot as plt
from pipeline import balance_classes, load_subject_timeframe

from logging_config import logger


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

def analyze_all_subjects(data_dir, subject_files, output_dir):
    for subject_file in subject_files:
        subject_id = subject_file.split('.')[0]
        logger.info(f'Analyzing subject {subject_id}')
        subject_output_dir = f'{output_dir}/{subject_id}'
        os.makedirs(subject_output_dir, exist_ok=True)
        os.makedirs(f'{subject_output_dir}/timeframe', exist_ok=True)
        os.makedirs(f'{subject_output_dir}/balanced_scatter', exist_ok=True)
        X, y = load_subject_timeframe(os.path.join(data_dir, subject_file), clip_outliers=False)
        fig = label_correlation_analysis(subject_id, X, y)
        fig.savefig(f'{subject_output_dir}/{subject_id}_label_correlation_analysis.png')

        for feature in X.columns:
            fig = timeframe_analysis(subject_id, X, y, feature)
            fig.savefig(f'{subject_output_dir}/timeframe/{subject_id}_{feature}.png')

        X, y = balance_classes(X, y)
        for feature in X.columns:
            fig = timeframe_analysis(subject_id, X, y, feature, scatter=True)
            fig.savefig(f'{subject_output_dir}/balanced_scatter/{subject_id}_{feature}.png')

if __name__ == '__main__':
    data_dir = '.data'
    output_dir = '.analysis'
    subject_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    analyze_all_subjects(data_dir, subject_files, output_dir)