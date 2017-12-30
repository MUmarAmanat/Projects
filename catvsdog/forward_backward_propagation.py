import numpy as np
from activation import Activation


class Prop:
    @staticmethod
    def propagate(w, b, x, y):
        m = x.shape[1]
        A = Activation.sigmoid(np.dot(w.T, x) + b)
        cost = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A)))
        dz = A - y
        dw = (1 / m) * np.dot(x, dz.T)
        db = (1 / m) * np.sum(dz)
        cost = np.squeeze(cost)
        grads = {"dw": dw,
                 "db": db}
        return grads, cost

