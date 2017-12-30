import numpy as np


class InitParam:
    @staticmethod
    def initialize_params(dim):
        w = np.zeros(shape=(dim, 1))
        # w = np.random.randn(dim) * 0.01
        b = 0
        return w, b