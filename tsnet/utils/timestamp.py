import pandas as pd
import numpy as np

def generate_time_features(timestamps):
    """
    Convert a list/array of timestamps into encoded time features
    used by TemporalEmbedding.

    Args:
        timestamps: List or 1D array of pd.Timestamp or str

    Returns:
        x_mark: np.ndarray of shape (seq_len, 5) with:
                [minute, hour, weekday, day, month]
    """
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps)

    minute = [ts.minute for ts in timestamps]
    hour = [ts.hour for ts in timestamps]
    weekday = [ts.weekday() for ts in timestamps]
    day = [ts.day for ts in timestamps]
    month = [ts.month for ts in timestamps]

    return np.stack([minute, hour, weekday, day, month], axis=1)