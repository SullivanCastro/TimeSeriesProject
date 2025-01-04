import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from typing import Tuple, List
from numpy.typing import ArrayLike

class FinancialTimeSeriesSparseModel:
    def __init__(self, n_components: int, n_nonzero_coefs: int, window_size: int, max_iter: int = 100):
        """
        Initializes the sparse model with K-SVD algorithm and sliding windows.

        Parameters:
        - n_components: Number of atoms in the dictionary.
        - n_nonzero_coefs: Sparsity level (number of non-zero coefficients).
        - window_size: Size of the sliding window for segmenting time series.
        - max_iter: Maximum number of iterations for dictionary learning.
        """
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.window_size = window_size
        self.max_iter = max_iter
        self.dictionary = None

    def segment_time_series(self, data: ArrayLike) -> ArrayLike:
        """
        Segments the time series data into sliding windows.

        Parameters:
        - data: A 2D numpy array of shape (n_samples, n_features), where each row is a time series.

        Returns:
        - segmented_data: A 2D numpy array where each row corresponds to a window.
        """
        segmented_data = []
        for series in data:
            for i in range(len(series) - self.window_size + 1):
                window = series[i:i + self.window_size]
                segmented_data.append(window)
        return np.array(segmented_data)

    def fit(self, data: ArrayLike) -> None:
        """
        Step 1: Segment time series into sliding windows.
        Step 2: Learn a dictionary from the segmented data using K-SVD.

        Parameters:
        - data: A 2D numpy array of shape (n_samples, n_features), where each row is a time series.
        """
        segmented_data = self.segment_time_series(data)

        # Learn dictionary using DictionaryLearning (K-SVD)
        dl = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.n_nonzero_coefs,
            max_iter=self.max_iter,
            fit_algorithm='lars',
            transform_algorithm='omp',
            transform_n_nonzero_coefs=self.n_nonzero_coefs
        )
        self.dictionary = dl.fit(segmented_data).components_

    def transform(self, data: ArrayLike) -> ArrayLike:
        """
        Step 3: Transform the segmented time series into sparse representations using the learned dictionary.

        Parameters:
        - data: A 2D numpy array of shape (n_samples, n_features), where each row is a time series.

        Returns:
        - sparse_codes: The sparse representation of the data.
        """
        segmented_data = self.segment_time_series(data)

        # Transform data using Orthogonal Matching Pursuit (OMP)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
        omp.fit(self.dictionary.T, segmented_data.T)
        sparse_codes = omp.coef_.T
        return sparse_codes

    def reconstruct(self, sparse_codes: ArrayLike) -> ArrayLike:
        """
        Step 4: Reconstruct the time series from sparse representations.

        Parameters:
        - sparse_codes: The sparse representation of the data.

        Returns:
        - reconstructed_data: The reconstructed time series.
        """
        return sparse_codes.T @ self.dictionary

# Example usage
if __name__ == "__main__":
    # Simulated financial time series data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    data = np.random.randn(n_samples, n_features)

    # Initialize and fit the model
    model = FinancialTimeSeriesSparseModel(n_components=20, n_nonzero_coefs=5, window_size=10, max_iter=50)
    print("Step 1: Fitting the model (dictionary learning)")
    model.fit(data)

    # Transform data into sparse codes
    print("Step 2: Transforming data into sparse codes")
    sparse_codes = model.transform(data)
    
    # Reconstruct the data
    print("Step 3: Reconstructing the data")
    reconstructed_data = model.reconstruct(sparse_codes)

    # Print results
    print("Original Data Shape:", data.shape)
    print("Sparse Codes Shape:", sparse_codes.shape)
    print("Reconstructed Data Shape:", reconstructed_data.shape)
