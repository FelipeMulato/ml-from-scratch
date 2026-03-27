import numpy as np
from src.utils.activation import sigmoid
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr= lr 
        self.n_iters = n_iters 
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(0,self.n_iters):
            x_pred = np.dot(X,self.weights) + self.bias
            y_pred = sigmoid(x_pred)

            dw = (1/n_samples)*(np.dot(X.T,(y_pred-y)))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.weights-= dw*self.lr
            self.bias -= db*self.lr


    def predict(self, X):
        x_pred = np.dot(X,self.weights) + self.bias
        y_pred = sigmoid(x_pred)
        return [0 if y<=0.5 else 1 for y in y_pred]