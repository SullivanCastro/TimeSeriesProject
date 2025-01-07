import math
import numpy as np

from src.utils import sliding_window_on_last_axis, sliding_window_to_signal


def dyadic_alphabet(level) -> list[str]:
    """Generate a dyadic alphabet of size Q = 2^L,

    Parameters:
        level: int: The level of the dyadic alphabet.

    Returns:
        alphabet: list[str]: The dyadic alphabet.
    """
    return [format(i, f"0{level}b") for i in range(2**level)]


class meSAX:
    def __init__(
        self, alphabet_size, rolling_window_size, rolling_window_stride, paa_window_size
    ) -> None:
        """
        The meSAX model for time series representation.

        Parameters:
            alphabet_size: int: The size of the alphabet.
            rolling_window_size: int: The size of the rolling window.
            rolling_window_stride: int: The stride of the rolling window.
            paa_window_size: int: The size of the PAA window.
        """
        self._alphabet_size = alphabet_size  # K in the paper
        self._rolling_window_size = rolling_window_size  # w in the paper
        self._rolling_window_stride = rolling_window_stride  # s in the paper
        self._paa_window_size = paa_window_size  # R in the paper
        self._alphabet_level = math.ceil(math.log2(alphabet_size)) # Q in the paper
        self._alphabet = dyadic_alphabet(self._alphabet_level)[:alphabet_size]
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
        paa_windows = sliding_window_on_last_axis(
            rolling_windows, self._paa_window_size, self._paa_window_size
        )

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
            np.where((argmaxs < argmins) & (argmaxs < argmeans), maxs, means),
        )

        features_data[:, :, 1] = np.where(
            (argmeans < argmins) & (argmins < argmaxs)
            | (argmaxs < argmins) & (argmins < argmeans),
            mins,
            np.where(
                (argmeans < argmaxs) & (argmaxs < argmins)
                | (argmins < argmaxs) & (argmaxs < argmeans),
                maxs,
                means,
            ),
        )

        features_data[:, :, 2] = np.where(
            (argmins > argmaxs) & (argmins > argmeans),
            mins,
            np.where((argmaxs > argmins) & (argmaxs > argmeans), maxs, means),
        )

        return features_data

    def fit(self, X: np.ndarray, fitting_method="uniform") -> None:
        """
        Fit the meSAX model to the data by learning the breakpoints.

        Parameters:
            data: np.ndarray: A 1D time series data array.
            fitting_method: str: The method used to generate the breakpoints. Default is "uniform".
        """
        if fitting_method == "uniform":
            self._breakpoints = np.linspace(X.min(), X.max(), self._breakpoints_numbers)
        elif fitting_method == "median":
            self._breakpoints = np.quantile(
                X, np.linspace(0, 1, self._breakpoints_numbers)
            )
        else:
            raise NotImplementedError("Invalid method for breakpoints generation")

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
            stride=self._rolling_window_stride,
        )

        triplets_data = self._paa(rolling_windows)

        # map the triplets to the alphabet
        encoded_triplets_data = np.digitize(triplets_data, self.breakpoints, right=True)
        embedings = np.array(self._alphabet)[encoded_triplets_data]

        # convert data in binary and compute the compression ratio
        embedings_str = "".join(embedings.reshape(-1))
        compression_ratio = rolling_windows.nbytes / np.array([elt for elt in embedings_str]).nbytes

        return embedings_str, compression_ratio

    def fit_transform(self, data: np.ndarray, fitting_method="uniform") -> np.ndarray:
        """
        Fit the meSAX model to the data and transform it into the SAX representation.

        Parameters:
            data: np.ndarray: A 1D time series data array.
            fitting_method: str: The method used to generate the breakpoints. Default is "uniform".

        Returns:
        - sax_data: The SAX representation of the data.
        """
        self.fit(data, fitting_method)
        return self.transform(data)

    def predict(self, binary_sequence: str, sampling_method="slope") -> np.ndarray:
        """
        Reconstruct the time series data from the SAX representation.

        Parameters:
            data: np.ndarray: A 1D time series data array.

        Returns:
        - reconstructed_data: The reconstructed time series data from the SAX representation.
        """

        # decode each element of the binary sequence 
        binary_sequence_list = [
            binary_sequence[i : i + self._alphabet_level]
            for i in range(0, len(binary_sequence), self._alphabet_level)
        ]

        # decode the binary sequence to triplets by rescising the sequence to the original shape
        c_hat = self._rolling_window_size // self._paa_window_size
        c = len(binary_sequence_list) // (3 * c_hat)
        original_data_shape = (c, c_hat, 3)
        binary_triplets_data = np.array(binary_sequence_list).reshape(
            original_data_shape
        )
        encoded_triplets_data = np.vectorize(lambda x: int(x, 2))(binary_triplets_data)

        # decode the triplets values to beans mean
        extended_breakpoints = np.concatenate(
            ([self.breakpoints[0]], self.breakpoints, [self.breakpoints[-1]])
        )
        bucket_means = (extended_breakpoints[:-1] + extended_breakpoints[1:]) / 2
        bucket_means[0] = self.breakpoints[0]  # floor the first bucket
        bucket_means[-1] = self.breakpoints[-1]  # Cap the last bucket
        triplets_data = bucket_means[encoded_triplets_data]

        # retrieve the data features
        theta_1 = triplets_data[..., 0][..., np.newaxis]
        theta_2 = triplets_data[..., 1][..., np.newaxis]
        theta_3 = triplets_data[..., 2][..., np.newaxis]

        if sampling_method == "slope":
            mid_point = self._paa_window_size // 2
            x = np.arange(self._paa_window_size)
            reconstructed_paa_signals = (
                theta_1 + (theta_2 - theta_1) / mid_point * x
            ) * (
                x < mid_point
            ) + (  # slope 1
                theta_2 + (theta_3 - theta_2) / mid_point * x
            ) * (
                x >= mid_point
            )  # slope 2
            reconstructed_rolling_window = reconstructed_paa_signals.reshape(
                reconstructed_paa_signals.shape[0], -1
            )

        elif sampling_method == "normal":
            reconstructed_paa_signals = np.zeros(
                (triplets_data.shape[0], triplets_data.shape[1], self._paa_window_size)
            )
            for i in range(triplets_data.shape[0]):
                for j in range(triplets_data.shape[1]):
                    mean = theta_2[i, j]
                    min_, max_ = min(theta_1[i, j], theta_3[i, j]), max(
                        theta_1[i, j], theta_3[i, j]
                    )

                    samples = np.random.normal(
                        mean, np.sqrt((max_ - min_) / 2), self._paa_window_size
                    )
                    while np.any(samples < min_) or np.any(samples > max_):
                        wrong_args_min = np.where(samples < min_)[0]
                        wrong_args_max = np.where(samples > max_)[0]
                        wrong_args = np.concatenate((wrong_args_min, wrong_args_max))
                        samples[wrong_args] = np.random.normal(mean, 1, len(wrong_args))

                    np.sort(samples)
                    if theta_1[i, j] > theta_3[i, j]:
                        samples = samples[::-1]

                    reconstructed_paa_signals[i, j] = samples

        else:
            raise NotImplementedError("Invalid method for reconstruction")

        reconstructed_rolling_window = reconstructed_paa_signals.reshape(
            reconstructed_paa_signals.shape[0], -1
        )

        return sliding_window_to_signal(
            reconstructed_rolling_window, self._rolling_window_stride
        )

    @property
    def breakpoints(self):
        if self._breakpoints is None:
            raise ValueError("The model has not been fitted yet.")
        return self._breakpoints
