import numpy as np
from math import floor


def sliding_windows_partitions(data, windows_size, step_size):
    """
    Split the data into windows of size windows_size. The windows are overlapping.
    The number of windows is num_windows.
    """
    c_tilde = int(floor((len(data) - windows_size) / step_size)) + 1

    processed_data = np.zeros((windows_size, c_tilde))
    for r in range(c_tilde):
        processed_data[:, r] = data[r * windows_size * r:windows_size * (r + 1)]

    return processed_data

