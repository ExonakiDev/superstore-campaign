import numpy as np
from tqdm import tqdm

def sigmoid_activation(Z):
    return 1 / (1 + np.exp(- Z))

class CustomLogisticRegression:
    """
    Logistic Regression class.

    TODO: Add L1, L2 regularization abilities.
    """
    def __init__(self, lr = 0.001, num_iter = 100):
        self.lr = lr
        self.num_iter = num_iter

        self.w = None
        self.bias = None

    def fit(self, X, y):
        num_rows, num_features = X.shape

        self.w = np.zeros((num_features, ))
        self.bias = 0

        for num in tqdm(range(self.num_iter)):
            Z = np.dot(X, self.w) + self.bias
            y_cap = sigmoid_activation(Z)

            dw = (1 / num_rows) * np.dot(X.T, (y_cap - y))
            db = (1 / num_rows) * np.sum((y_cap - y))

            self.w = self.w - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X) -> np.ndarray:
        Z_pred = np.dot(X, self.w) + self.bias
        preds = sigmoid_activation(Z_pred)
        preds = np.where(preds > 0.5, 1, 0)
        
        return preds 
        
        
        