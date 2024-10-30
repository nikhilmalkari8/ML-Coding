
def sigmoid(linear_pred):
    return 1/(1+np.exp(-linear_pred))

class LogisticRegression():
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(n_iters):
            linear_pred = np.dot(self.weights, X) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T -(predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)

    def predict(self, X):
        linear_pred = np.dot(self.weights, X) + self.bias
        y_pred = sigmoid(linear_pred)

        for y in y_pred:
            if y < 0.5:
                return 0
            else:
                return 1
