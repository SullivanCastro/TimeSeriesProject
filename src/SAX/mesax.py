import numpy as np
from tqdm import tqdm
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


class meSAX:

    def __init__(self, K, windows_size, step_size, method="uniform", alphabet_method="classic"):
        self.K = K
        self.method = method
        self.windows_size = windows_size
        self.step_size = step_size
        self.data = None
        self.symbol = []
        self.alphabet_method = alphabet_method
        self.alphabet = self._generate_alphabet()
        self.breakpoints = np.random.rand(2**self.K - 1)


    def _preprocess_data(self, data):
        self.slided_data = sliding_windows_partitions(data, self.windows_size, self.step_size)
        self.aggregated_data = paa_aggregation(self.slided_data, self.windows_size)
        self.synthetised_data = synthetise_data(self.aggregated_data)
        return self.synthetised_data


    def _map_symbols(self, x):
        self.symbol = []
        for i in range(self.synthetised_data.shape[0]):
            for j in range(self.synthetised_data.shape[1]):
                output = []
                for x in self.synthetised_data[i, j]: # triplet element
                    b = 0
                    while b < len(self.breakpoints) and x > self.breakpoints[b]:
                        b += 1
                    output.append(self.alphabet[b])
                self.symbol.append(output)
        self.symbol = np.array(self.symbol)
        return self.symbol
    

    def _generate_alphabet(self):
        if self.alphabet_method == "classic":
            return np.arange(2**self.K)
        elif self.alphabet_method == "dyadic":
            return dyadic_alphabet(2**self.K)
        else:
            raise ValueError("Invalid alphabet method")


    def _generate_breakpoints(self):
        if self.method == "uniform":
            return np.linspace(self.data.min(), self.data.max(), 2**self.K)
        elif self.method == "median":
            return np.quantile(self.data, np.linspace(0, 1, 2**self.K))
        else:
            raise ValueError("Invalid method for breakpoints generation")
        
    def synthesize(self, data):
        self.data = data
        self.breakpoints = self._generate_breakpoints()
        self._preprocess_data(data)
        self._map_symbols(data)
        return self.symbol
    
    def reconstruct(self):
        self.reconstructed_data = np.zeros(self.data.shape)

        for i, triplet in tqdm(enumerate(self.symbol)):
            mean = self.breakpoints[triplet[1]]
            elem_0, elem_1 = self.breakpoints[triplet[0]], self.breakpoints[triplet[2]]
            min_, max_ = min(elem_0, elem_1), max(elem_0, elem_1)

            if mean==min_==max_:
                self.reconstructed_data[i*self.step_size:(i+1)*self.step_size] = mean * np.ones(self.step_size)
            else:
                samples = np.random.normal(mean, 0.5, self.step_size)
                while np.any(samples < min_) or np.any(samples > max_):
                    wrong_args_min = np.where(samples < min_)[0]
                    wrong_args_max = np.where(samples > max_)[0]
                    wrong_args = np.concatenate((wrong_args_min, wrong_args_max))
                    samples[wrong_args] = np.random.normal(mean, 1, len(wrong_args))
                
                np.sort(samples)
                if elem_0 > elem_1:
                    samples = samples[::-1]

                self.reconstructed_data[i*self.step_size:(i+1)*self.step_size] = samples

        self.reconstructed_data = self.reconstructed_data[self.reconstructed_data != 0]
        return self.reconstructed_data
