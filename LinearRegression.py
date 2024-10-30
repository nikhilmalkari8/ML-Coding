import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iters=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape()
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(n_iters):
            y_pred = np.dot(self.weights, X) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) + np.sum(y_pred-y)

            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db) 

    def predict(self, X):
        y_pred = np.dot(self.weights, X) + self.bias
        return y_pred