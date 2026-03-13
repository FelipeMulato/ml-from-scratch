import numpy as np
from src.utils.distance import euclidean_distance
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        #Distance
        distances = [(euclidean_distance(x,x1),y) for x1,y in zip(self.X_train,self.y_train)]
        distances = sorted(distances)
        labels = [y for _,y in distances[:self.k]]

        count = Counter(labels)
        return count.most_common()[0][0]