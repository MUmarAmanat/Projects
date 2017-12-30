from forward_backward_propagation import Prop


class Optimization:
    @staticmethod
    def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(num_iterations):
            grads, cost = Prop.propagate(w, b, x, y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs
