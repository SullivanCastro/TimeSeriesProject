import numpy as np


def sliding_window_on_last_axis(series: np.ndarray, window_size: int, stride:int) -> np.ndarray:
    """
    Create a n+1 D array of sliding windows from a temporal series.
    
    Parameters:
        series (np.ndarray): 1D array representing the temporal series. (..., N,)
        window_size (int): Size of each sliding window.
        stride (int): Step size between consecutive windows.
    
    Returns:
        np.ndarray: 2D array where each row is a sliding window. (..., C, window_size)
        where C = (N - window_size) // stride + 1
    """
    
    # Compute the number of sliding windows
    last_dim = series.shape[-1]
    num_windows = (last_dim - window_size) // stride + 1
    
    # Create sliding windows using advanced NumPy slicing
    shape = series.shape[:-1] + (num_windows, window_size)
    strides = series.strides[:-1] + (series.strides[-1] * stride, series.strides[-1])
    sliding_windows = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
    
    return sliding_windows


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