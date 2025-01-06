import numpy as np


def signal_to_sliding_window(series: np.ndarray, window_size: int, stride:int) -> np.ndarray:
    """
    Create a 2D array of sliding windows from a temporal series.
    
    Parameters:
        series (np.ndarray): 1D array representing the temporal series.
        window_size (int): Size of each sliding window.
        stride (int): Step size between consecutive windows.
    
    Returns:
        np.ndarray: 2D array where each row is a sliding window.
    """
    
    return np.array([
        series[i : i + window_size]
        for i in range(0, len(series) - window_size + 1, stride)
    ])


def sliding_window_to_signal(windows: np.ndarray, stride: int) -> np.ndarray:
    """
    Reconstruct the original time series from sliding windows using mean over overlaps.
    
    Parameters:
        windows (np.ndarray): 2D array where each row is a sliding window.
        stride (int): Step size between consecutive windows.
    
    Returns:
        np.ndarray: Reconstructed 1D array representing the original time series.
    """
    
    window_size = windows.shape[1]
    total_length = (windows.shape[0] - 1) * stride + window_size
    
    # Initialize a 2D array of nan
    expanded_array = np.full((windows.shape[0], total_length), np.nan)
    
    # Place each window in its shifted position
    for i, window in enumerate(windows):
        start = i * stride
        expanded_array[i, start:start + window_size] = window

    # Compute the mean over the axis 0 to reconstruct the signal
    reconstructed = np.sum(np.nan_to_num(expanded_array, nan=0), axis=0) / np.maximum(1, (~np.isnan(expanded_array)).sum(axis=0))
    
    return reconstructed