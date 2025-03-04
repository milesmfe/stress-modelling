import numpy as np
import pandas as pd


def align_signal(base_index, signal_data, collection_seconds):
    # Create time vectors in seconds
    time_base = base_index / base_index[-1] * collection_seconds
    time_signal = np.linspace(0, collection_seconds, len(signal_data))
    
    # Create alignment DataFrame
    df_signal = pd.DataFrame({
        'time': time_signal,
        'value': signal_data
    })
    
    # Use merge_asof for forward-fill alignment
    aligned = pd.merge_asof(
        pd.DataFrame({'time': time_base}),
        df_signal,
        on='time',
        direction='forward'
    )
    return aligned['value'].values