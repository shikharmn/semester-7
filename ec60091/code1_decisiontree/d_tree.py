import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self,train_path=None,test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        train_pd = pd.read_csv(self.train_path)
        test_pd = pd.read_csv(self.test_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self._np_from_pd(train_pd, test_pd)

    def _np_from_pd(self,train_pd,test_pd):
        train_np = train_pd.to_numpy()
        test_np = test_pd.to_numpy()
        X_train, y_train = train_np[:,:4], train_np[:,4]
        X_test, y_test = test_np[:,:4], test_np[:,4]
        
        ylabel = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
        y_train = np.asarray([ylabel[x] for x in y_train])
        y_test = np.asarray([ylabel[x] for x in y_test])

        return X_train, X_test, y_train, y_test

