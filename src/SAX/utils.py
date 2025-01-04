import numpy as np
from math import floor


def sliding_windows_partitions(data, windows_size, step_size):
    """
    Split the data into windows of size windows_size. The windows are overlapping.
    The number of windows is num_windows.
    """
    c_tilde = int(floor((len(data) - windows_size) / step_size)) + 1

    processed_data = np.zeros((c_tilde, windows_size))
    i = windows_size
    s = 0

    while i < len(data) + step_size:
        processed_data[s] = data[i - windows_size:i]
        i += step_size
        s += 1

    return processed_data


def paa_aggregation(data, paa_size):
    """
    Perform Piecewise Aggregate Approximation (PAA) on the data.
    """
    return data.reshape(data.shape[0], data.shape[1]//paa_size, paa_size)


def synthetise_data(data):
    """
    Synthetise the data after the PAA aggregation.
    """

    synthetised_data = np.zeros((data.shape[0], data.shape[1], 3))

    means = np.mean(data, axis=-1)

    argmins = np.argmin(data, axis=-1)
    mins = np.min(data, axis=-1)

    argmaxs = np.argmax(data, axis=-1)
    maxs = np.max(data, axis=-1)


    synthetised_data[:, :, 0] = (argmins < argmaxs) * mins + (argmins > argmaxs) * maxs + (argmins == argmaxs) * means
    synthetised_data[:, :, 1] = (argmins != argmaxs) * means + (argmins == argmaxs) * mins
    synthetised_data[:, :, 2] = (argmins < argmaxs) * maxs + (argmins > argmaxs) * mins + (argmins == argmaxs) * maxs

    return synthetised_data


def dyadic_alphabet(Q):
    # return [format(i, f"0{Q}b") for i in range(2**Q)]
    return np.arange(Q)