import numpy as np
from activation import Activation


class Prop:
    @staticmethod
    def propagate(w, b, x, y):
        m = x.shape[1]
        A = Activation.sigmoid(np.dot(w.T, x) + b)
        # print('A shape: ', A.shape)
        # print('y shape: ',y.shape)
        cost = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A)))
        dz = A - y
        dw = (1 / m) * np.dot(x, dz.T)
        db = (1 / m) * np.sum(dz)
        # assert (dw.shape == w.shape)
        # assert (db.dtype == float)
        cost = np.squeeze(cost)
        # assert (cost.shape == ())
        grads = {"dw": dw,
                 "db": db}
        return grads, cost

