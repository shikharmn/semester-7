import numpy as np
import pandas as pd
import argparse
import os
from argparse import RawTextHelpFormatter
from sklearn.metrics import accuracy_score as accuracy

def kNN_trainer(data_train, data_test, ks=None,mode=None):
    """
    Function for training kNN algorithm using different distance metrics.
    Input:
        data_train, data_test = training data and testing data along with labels
        k takes a list when mode = 'hparam', else takes an integer
        mode = l2 -> Use euclidean distance metric
        mode = norm -> Use normalized euclidean distance metric
        mode = cos -> Use cosine similarity distance metric
        mode = hparam -> Hyperparameter search among 3 modes and list of k
    Output:
        if mode is not hparam:
            accuracy: float (resultant accuracy)
        if mode is hparam:
            tuple (accuracy, k, metric)
                accuracy: float (resultant accuracy)
                k: int (best value of k)
                metric: string (best mode)
    """
    X_train, y_train = data_train[:,:8], data_train[:,8]//4     # Transform labels from [2,4] to [0,1]
    X_test, y_test = data_test[:,:8], data_test[:,8]//4

    X_train_norm = X_train/np.linalg.norm(X_train,axis=1,keepdims=True) # Normalize data
    X_test_norm = X_test/np.linalg.norm(X_test,axis=1,keepdims=True)

    cosine_dist = 1 - X_test_norm @ X_train_norm.T                           # Cosine similarity directly calculated by matrix multiplication
    norm_l2_dist = np.sum(X_test_norm ** 2, axis=1, keepdims=True) + \
                np.sum(X_train_norm ** 2, axis=1).T - 2 * (1 - cosine_dist)
    l2_dist = np.sum(X_test ** 2, axis=1, keepdims=True) + \
            np.sum(X_train ** 2, axis=1).T - 2 * X_test @ X_train.T     # Euclidean distance calculated using array broadcasting and..
                                                                        # ..matrix multiplication.
    
    if mode == 'l2':
        idx_l2 = np.argpartition(l2_dist, ks,axis=1)[:,:ks]                 # Obtain the indices of k nearest points
        preds_l2 = (np.mean(y_train[idx_l2],axis=1) > 0.5).astype(np.uint8) # If mean > 0.5, label 1 is in majority, else label 0.
        return accuracy(preds_l2, y_test)

    elif mode == 'norm':
        idx_norml2 = np.argpartition(norm_l2_dist, ks,axis=1)[:,:ks]
        preds_norml2 = (np.mean(y_train[idx_norml2],axis=1) > 0.5).astype(np.uint8)
        return accuracy(preds_norml2, y_test)

    elif mode == 'cos':
        idx_cosine = np.argpartition(cosine_dist, ks,axis=1)[:,:ks]
        preds_cosine = (np.mean(y_train[idx_cosine],axis=1) > 0.5).astype(np.uint8)
        return accuracy(preds_cosine, y_test)

    elif mode == 'hparam':
        results = []                                              # Matrix for grid search results on hyperparameters
        for k in ks:
            idx_cosine = np.argpartition(cosine_dist, k,axis=1)[:,:k]
            idx_norml2 = np.argpartition(norm_l2_dist, k,axis=1)[:,:k]
            idx_l2 = np.argpartition(l2_dist, k,axis=1)[:,:k]
            preds_cosine = (np.mean(y_train[idx_cosine],axis=1) > 0.5).astype(np.uint8)
            preds_norml2 = (np.mean(y_train[idx_norml2],axis=1) > 0.5).astype(np.uint8)
            preds_l2 = (np.mean(y_train[idx_l2],axis=1) > 0.5).astype(np.uint8)
            acc = [0,0,0]
            acc[0] = accuracy(preds_cosine, y_test)
            acc[1] = accuracy(preds_norml2, y_test)
            acc[2] = accuracy(preds_l2, y_test)
            print("For k = %d" % k)
            print("Cosine similarity accuracy: %.4f" % acc[0])
            print("Normalized euclidean distance accuracy: %.4f" % acc[1])
            print("Euclidean distance accuracy: %.4f\n" % acc[2])
            results.append(acc)

        results = np.asarray(results)
        best_k, best_method = np.argmax(results)//3, np.argmax(results) % 3     # Obtain row and column indices of maximum accuracy
        methods = ["cos", "norm", "l2"]
        return results[best_k][best_method], ks[best_k], methods[best_method]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Python implementation of the kNN algorithm.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='./', type=str,
                        help="folder path to train and test data.")
    args = parser.parse_args()

    k_list = [1,3,5,7]
    methods = {'l2': "euclidean distance",
               'norm': "normalized euclidean distance",
               'cos': "cosine similarity"}
    train_path = os.path.join(args.folder, 'train.csv')
    test_path = os.path.join(args.folder, 'test.csv')

    data_train = pd.read_csv(train_path).to_numpy()
    data_test = pd.read_csv(test_path).to_numpy()

    # Find a result before hyperparameter search by uncommenting the next line
    # acc = kNN_trainer(data_train, data_test, ks=3, mode='l2')

    # Hyperparameter search
    acc, kval, best_method = kNN_trainer(data_train, data_test, k_list, mode='hparam')
    print(("Best accuracy is %.3f%%, found with k = %d and " + methods[best_method] + " metric.") % (100*acc, kval))