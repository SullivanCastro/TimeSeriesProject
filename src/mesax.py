import math
import numpy as np

from src.utils import sliding_window_on_last_axis, sliding_window_to_signal

def dyadic_alphabet(alphabet_size) -> list[str]:
    """Generate a dyadic alphabet of size Q = 2^L, where L is the smallest integer such that 2^L >= alphabet_size.
    
    Parameters:
        alphabet_size: int: The size of the alphabet.

    Returns:
        alphabet: A list of binary strings representing the dyadic alphabet.
    """
    
    level = math.ceil(math.log2(alphabet_size))

    if not (alphabet_size & (alphabet_size - 1)) == 0:
        print("The alphabet size must be a power of 2. The next power of 2 will be used.")
        alphabet_size = 2**level
        print(f"Alphabet size: {2**level}")

    return [format(i, f'0{level}b') for i in range(alphabet_size)]


class meSAX:
    def __init__(self, alphabet_size, rolling_window_size, rolling_window_stride, paa_window_size) -> None:
        """
        The meSAX model for time series representation.

        Parameters:
            alphabet_size: int: The size of the alphabet.
            rolling_window_size: int: The size of the rolling window.
            rolling_window_stride: int: The stride of the rolling window.
            paa_window_size: int: The size of the PAA window.
        """
        self._alphabet_size = alphabet_size # K in the paper
        self._rolling_window_size = rolling_window_size # w in the paper
        self._rolling_window_stride = rolling_window_stride # s in the paper
        self._paa_window_size = paa_window_size # R in the paper
        self._alphabet = dyadic_alphabet(alphabet_size)[:alphabet_size]
        self._breakpoints_numbers = len(self._alphabet) - 1
        self._breakpoints = None

    def _paa(self, rolling_windows: np.ndarray) -> np.ndarray:
        """
        Extract features (min, average, max) from the data using PAA.
        Map a window to a 3*R-dimensional vector.

        Parameters:
            rolling_windows: np.ndarray: A 2D time series data array. (c, self._rolling_window_size)
            where c = (N - self._rolling_window_size) // self._rolling_window_stride + 1

        Returns:
        - paa_data: The PAA representation of the data.
        """
        # when stride = window_size, the sliding windows is not overlapped
        paa_windows = sliding_window_on_last_axis(rolling_windows, self._paa_window_size, self._paa_window_size)

        # retreive the min, mean and max of each last axis and their corresponding index
        features_data = np.zeros((paa_windows.shape[0], paa_windows.shape[1], 3))

        argmins = np.argmin(paa_windows, axis=-1)
        mins = np.min(paa_windows, axis=-1)

        argmaxs = np.argmax(paa_windows, axis=-1)
        maxs = np.max(paa_windows, axis=-1)

        means = np.mean(paa_windows, axis=-1)
        argmeans = (argmins + argmaxs) // 2

        # order the features min, mean, max by appearing time
        features_data[:, :, 0] = np.where(
            (argmins < argmaxs) & (argmins < argmeans),
            mins,
            np.where(
                (argmaxs < argmins) & (argmaxs < argmeans),
                maxs,
                means
            )
        )

        features_data[:, :, 1] = np.where(
            (argmeans < argmins) & (argmins < argmaxs) | (argmaxs < argmins) & (argmins < argmeans),
            mins,
            np.where(
                (argmeans < argmaxs) & (argmaxs < argmins) | (argmins < argmaxs) & (argmaxs < argmeans),
                maxs,
                means
            )
        )

        features_data[:, :, 2] = np.where(
            (argmins > argmaxs) & (argmins > argmeans),
            mins,
            np.where(
                (argmaxs > argmins) & (argmaxs > argmeans),
                maxs,
                means
            )
        )

        return features_data

    def fit(self, X: np.ndarray, fitting_method = "uniform") -> None:
        """
        Fit the meSAX model to the data by learning the breakpoints.

        Parameters:
            data: np.ndarray: A 1D time series data array.
            fitting_method: str: The method used to generate the breakpoints. Default is "uniform".
        """
        if fitting_method == "uniform":
            self._breakpoints = np.linspace(X.min(), X.max(), self._breakpoints_numbers)
        elif fitting_method == "median":
            self._breakpoints = np.quantile(X, np.linspace(0, 1, self._breakpoints_numbers))
        else:
            raise NotImplementedError("Invalid method for breakpoints generation")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data into 

        Parameters:
            data: np.ndarray: A 1D time series data array. (N, )

        Returns:
        - sax_data: The SAX representation of the data.
        """
        rolling_windows = sliding_window_on_last_axis(
            series=data,
            window_size=self._rolling_window_size,
            stride=self._rolling_window_stride
        )

        triplets_data = self._paa(rolling_windows)

        # map the triplets to the alphabet
        encoded_triplets_data = np.digitize(triplets_data, self.breakpoints, right=True)
        
        return encoded_triplets_data
    
    def predict(self, encoded_triplets_data: np.ndarray, sampling_method = "slope") -> np.ndarray:
        """
        Reconstruct the time series data from the SAX representation.

        Parameters:
            data: np.ndarray: A 1D time series data array.

        Returns:
        - reconstructed_data: The reconstructed time series data from the SAX representation.
        """

        extended_breakpoints = np.concatenate(([self.breakpoints[0]], self.breakpoints, [self.breakpoints[-1]]))
        bucket_means = (extended_breakpoints[:-1] + extended_breakpoints[1:]) / 2
        bucket_means[0] = self.breakpoints[0]  # floor the first bucket
        bucket_means[-1] = self.breakpoints[-1]  # Cap the last bucket
        triplets_data = bucket_means[encoded_triplets_data]

        # TODO : implem the random sampling method
        if sampling_method == "slope":
            
            mid_point = self._paa_window_size // 2

            theta_1 = triplets_data[..., 0][..., np.newaxis]
            theta_2 = triplets_data[..., 1][..., np.newaxis]
            theta_3 = triplets_data[..., 2][..., np.newaxis]

            x = np.arange(self._paa_window_size)

            reconstructed_paa_signals = (
                    (theta_1 + (theta_2 - theta_1) / mid_point * x) * (x < mid_point) # slope 1
                    + (theta_2 + (theta_3 - theta_2) / mid_point * x) * (x >= mid_point) # slope 2
                )
            reconstructed_rolling_window = reconstructed_paa_signals.reshape(reconstructed_paa_signals.shape[0], -1)

        else:
            raise NotImplementedError("Invalid method for reconstruction")
            
        return sliding_window_to_signal(reconstructed_rolling_window, self._rolling_window_stride)

    @property
    def breakpoints(self):
        if self._breakpoints is None:
            raise ValueError("The model has not been fitted yet.")
        return self._breakpoints


            
