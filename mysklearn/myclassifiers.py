import random
from mysklearn import myutils

class MyRandomForestClassifier:

    def __init__(self, N, M, F):
        self.N = N
        self.M = M
        self.F = F
        self.X_train = None
        self.y_train = None
        self.tree = []
        self.select_tree = []

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(X_test):
        y_predict = []
        return y_predict

