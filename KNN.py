import numpy as np
from collections import Counter

def euclidean_distance(self, x1, x2):
    distance = np.sqrt(np.sum(x1-x2)**2)
    return distance


class KNN:
    def __init__(self, k=3):
        def fit(self, X, y):
            self.X = X_train
            self.y = y

        def predict(self, x):
            predictions = [self._predict(x) for x in X]
            return predictions

        def _predict(self, x):
            distances = [euclidean_distance(x, x_train) for x_train in X_train]

            k_indices = np.argsort(distance)[:self.k]

            k_neigh = [self.y_train[i] for i in k_indices]

            most_common_labels = Counter(k_neigh).most_common()

            return most_common_labels[0][0]