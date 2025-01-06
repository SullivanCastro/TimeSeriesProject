import numpy as np
from sklearn.decomposition import DictionaryLearning
import scipy.sparse as sp

from src.utils import signal_to_sliding_window, sliding_window_to_signal

class FinancialTimeSeriesSparseModel:
    def __init__(self, n_components: int, n_nonzero_coefs: int, window_size: int, stride:int, max_iter: int = 100):
        """
        Financial Time Series Sparse Model using K-SVD for dictionary learning and Orthogonal Matching Pursuit (OMP) for sparse coding.

        Parameters:
            n_components: int: Number of dictionary atoms to learn.
            n_nonzero_coefs: int: Number of non-zero coefficients to use in the sparse coding.
            window_size: int: Size of the sliding window for segmentation.
            stride: int: Stride for the sliding window.
            max_iter: int: Maximum number of iterations for dictionary learning.
        """
        self._n_components = n_components
        self._n_nonzero_coefs = n_nonzero_coefs
        self._window_size = window_size
        self._stride = stride
        self._max_iter = max_iter
        self._original_data_size = None
        self._train_sparse_codes = None
        self._model = None

    def fit(self, data: np.ndarray) -> None:
        """
        Step 1: Segment time series into sliding windows.
        Step 2: Learn a dictionary from the segmented data using K-SVD.

        Parameters:
            data: np.ndarray: A 1D time series data array.
        """
        print("[*] Segmenting time series into sliding windows...")
        sliding_windows = signal_to_sliding_window(data, self._window_size, self._stride)
        self._original_data_size = sliding_windows.shape[0] * sliding_windows.shape[1]
        print("[*] Learning dictionary from the segmented data...")
        self.model = DictionaryLearning(
            n_components=self._n_components,
            transform_algorithm='omp',
            max_iter=self._max_iter,
            transform_n_nonzero_coefs=self._n_nonzero_coefs,
            transform_max_iter=self._max_iter,
            verbose=True
        )
        self.model.fit(sliding_windows)
        print("\n[*] Transforming the segmented data into sparse codes...")
        self._train_sparse_codes = self.model.transform(sliding_windows)
        return self.model

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Step 3: Transform the segmented time series into sparse representations using the learned dictionary.

        Parameters:
            data: np.ndarray: A 1D time series data array.
        
        Returns:
        - sparse_codes: The sparse representation code of the data, that corresponds to the learned dictionary.
        """
        sliding_windows = signal_to_sliding_window(data, self._window_size, self._stride)
        sparse_codes = self.model.transform(sliding_windows)
        return sparse_codes
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Reconstruct the time series data from the sparse codes.

        Parameters:
            data: np.ndarray: A 1D time series data array.
        
        Returns:
        - reconstructed_data: The reconstructed time series data from the sparse codes.
        """
        sliding_windows = signal_to_sliding_window(data, self._window_size, self._stride)
        sparse_codes = self.model.transform(sliding_windows)
        reconstructed_sliding_windows = sparse_codes @ self.dictionary
        reconstructed_data = sliding_window_to_signal(reconstructed_sliding_windows, self._stride)

        # padd with nan at the end to match the original data size
        if len(reconstructed_data) < len(data):
            reconstructed_data = np.concatenate([reconstructed_data, np.full(len(data) - len(reconstructed_data), np.nan)])


        compression_rattio = sp.csr_matrix(sliding_windows).data.nbytes / sp.csr_matrix(sparse_codes).data.nbytes

        return reconstructed_data, compression_rattio

    @property
    def dictionary(self):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        return self.model.components_
    
    @property
    def train_sparse_codes(self):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        return self._train_sparse_codes
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio of the sparse representation.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please fit the model first.")
        
        compressed_size = (self.train_sparse_codes != 0).sum()
        return self._original_data_size / (self._n_nonzero_coefs)

        