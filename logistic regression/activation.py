import numpy as np


class Activation:
    @staticmethod
    def sigmoid(z):
        s = 1 / (1 + np.exp(-z))
        return s