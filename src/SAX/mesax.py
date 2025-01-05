import numpy as np
from tqdm import tqdm
from math import floor
import sys

def sliding_windows_partitions(data, windows_size):
    """
    Split the data into windows of size windows_size. The windows are overlapping.
    The number of windows is num_windows.
    """
    return data.reshape(data.shape[0]//windows_size, windows_size)


def paa_aggregation(data, paa_size):
    """
    Perform Piecewise Aggregate Approximation (PAA) on the data.
    """
    return data.reshape(data.shape[0], data.shape[1]//paa_size, paa_size)


def synthetise_data(data, method):
    """
    Synthetise the data after the PAA aggregation.
    """

    synthetised_data = np.zeros((data.shape[0], data.shape[1], 3))

    argmins = np.argmin(data, axis=-1)
    mins = np.min(data, axis=-1)

    argmaxs = np.argmax(data, axis=-1)
    maxs = np.max(data, axis=-1)

    means = np.mean(data, axis=-1)
    argmeans = data.shape[-1] // 2

    if method=="slope":

        synthetised_data[:, :, 0] = ((argmins < argmaxs) & (argmins <= argmeans)) * mins + \
                                    ((argmaxs < argmins) & (argmaxs <= argmeans)) * maxs + \
                                    ((argmeans < argmins) & (argmeans < argmaxs)) * means

        synthetised_data[:, :, 1] = ((argmeans <= argmins) & (argmins < argmaxs) | (argmaxs < argmins) & (argmins <= argmeans)) * mins + \
                                    ((argmeans <= argmaxs) & (argmaxs < argmins) | (argmins < argmaxs) & (argmaxs <= argmeans)) * maxs + \
                                    ((argmaxs < argmeans) & (argmeans < argmins) | (argmins < argmeans) & (argmeans < argmaxs)) * means

        synthetised_data[:, :, 2] = ((argmins > argmaxs) & (argmins >= argmeans)) * mins + \
                                    ((argmaxs > argmins) & (argmaxs >= argmeans)) * maxs + \
                                    ((argmeans > argmins) & (argmeans > argmaxs)) * means

    elif method=="normal":
        synthetised_data[:, :, 0] = (argmins <= argmaxs) * mins + (argmins >= argmaxs) * maxs + (argmins == argmaxs) * means
        synthetised_data[:, :, 1] = (argmins != argmaxs) * means + (argmins == argmaxs) * mins
        synthetised_data[:, :, 2] = (argmins <= argmaxs) * maxs + (argmins >= argmaxs) * mins + (argmins == argmaxs) * maxs
    else:
        raise ValueError("Invalid method for data synthetisation")

    return synthetised_data


def dyadic_alphabet(Q):
    return [format(i, f"0{Q}b") for i in range(2**Q)]


class meSAX:

    def __init__(self, K, windows_size, step_size, method="uniform", reconstruction_method="slope"):
        self.K = K
        self.method = method
        self.windows_size = windows_size
        self.step_size = step_size
        self.data = None
        self.symbol = []
        self.alphabet = self._generate_alphabet()
        self.breakpoints = np.random.rand(2**self.K - 1)
        self.reconstruction_method = reconstruction_method


    def _preprocess_data(self, data):
        self.slided_data = sliding_windows_partitions(data, self.windows_size)
        self.aggregated_data = paa_aggregation(self.slided_data, self.step_size)
        self.synthetised_data = synthetise_data(self.aggregated_data, self.reconstruction_method)
        return self.synthetised_data


    def _map_symbols(self, x):
        self.symbol = ""
        for i in range(self.synthetised_data.shape[0]):
            for j in range(self.synthetised_data.shape[1]):
                output = ""
                for x in self.synthetised_data[i, j]: # triplet element
                    b = 0
                    while b < len(self.alphabet) -1  and x > self.breakpoints[b]:
                        b += 1
                    output += str(self.alphabet[b])
                self.symbol += output
        return self.symbol
    

    def _generate_alphabet(self):
        return dyadic_alphabet(self.K)


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
    
    def _group_by_size(self, s):
        split_string = [s[i:i+3*self.K] for i in range(0, len(s), 3*self.K)]
        final_list = []
        for substring in split_string:
            final_list.append([substring[i:i+self.K] for i in range(0, len(substring), self.K)])
        return final_list
            
    
    def reconstruct(self):
        symbol_list = self._group_by_size(self.symbol)
        self.reconstructed_data = np.zeros(len(symbol_list)*self.step_size)
        
        for i, triplet in tqdm(enumerate(symbol_list)):
            elem_0, elem_1, elem_2 = self.breakpoints[int(triplet[0],2)], self.breakpoints[int(triplet[1],2)], self.breakpoints[int(triplet[2],2)]
                

            if self.reconstruction_method == "slope":
                samples = np.zeros(self.step_size)
                samples[:self.step_size//2] = np.linspace(elem_0, elem_1, self.step_size//2)            
                samples[self.step_size//2:] = np.linspace(elem_1, elem_2, len(samples[self.step_size//2:]))
            else:
                mean = elem_1
                min_, max_ = min(elem_0, elem_2), max(elem_0, elem_2)
                
                samples = np.random.normal(mean, np.sqrt((max_-min_)/2), self.step_size)
                while np.any(samples < min_) or np.any(samples > max_):
                    wrong_args_min = np.where(samples < min_)[0]
                    wrong_args_max = np.where(samples > max_)[0]
                    wrong_args = np.concatenate((wrong_args_min, wrong_args_max))
                    samples[wrong_args] = np.random.normal(mean, 1, len(wrong_args))
                
                np.sort(samples)
                if elem_0 > elem_2:
                    samples = samples[::-1]

            self.reconstructed_data[i*self.step_size:(i+1)*self.step_size] = samples

        # self.reconstructed_data = self.reconstructed_data[self.reconstructed_data != 0]
        return self.reconstructed_data
    
    def _get_compression(self):
        return sys.getsizeof(self.symbol)
    
    def get_compression_ratio(self):
        return round(sys.getsizeof(self.data.tolist()) / sys.getsizeof(self.symbol), 3)
