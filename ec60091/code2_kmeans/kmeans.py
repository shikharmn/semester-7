# Author: Shikhar Mohan
# Roll No.: 18EC10054
# Date: 28/09/2021

import numpy as np
import pandas as pd
from itertools import permutations          # Recursively generates all permutations of a list
from matplotlib import pyplot as plt


class KMeans:
    """
    Implementation Class of K-Means clustering. Takes arguments data_path, the value of K 
    (number of classes) and number of epochs/iterations.
    """
    def __init__(self, data_path=None, k=None, epochs=None):
        self.data_path = data_path
        data_pd = pd.read_csv(self.data_path)
        self.X, self.y = self._np_from_pd(data_pd)
        self.k = k
        self.epochs = epochs
        if k is None:
            self.k = len(np.unique(self.y_train))
        
        self.means = None
        self.preds = None
        self.clusters = None

    def calc_distance(self, p1, p2):
        """
        This function calculates the Euclidean Distance between two ndarray datapoints p1 and p2.
        """
        return(np.linalg.norm(p1-p2))

    def get_preds(self, means=None, data=None):
        """
        This function gets class predictions when given the k means.
        """
        
        if means is None:
            means = self.means
        if data is None:
            data = self.X

        if len(data.shape) == 1: data = [data]
        
        preds = []
        for datum in data:
            dists = np.sum((datum - means)**2, axis=1)
            pred = np.argmin(dists)
            preds.append(pred)

        if len(data.shape) == 1:
            return preds[0]
        else:
            return preds

    def train(self, X_train=None, y_train=None):
        """
        This function trains the K-Means classifier algorithm.
        """
        
        if X_train is None or y_train is None:
            X_train, y_train = self.X, self.y
        
        # 1. Random assignment of means
        mean_idx = np.random.randint(0,120,self.k)
        means = X_train[mean_idx]
        preds = self.get_preds(means)

        # 2. Separate the above into clusters and evaluate means
        clusters = [[] for i in range(self.k)]
        for pred,data in zip(preds,X_train):
            clusters[pred].append(data)

        clusters = [np.asarray(cluster) for cluster in clusters]
        means = np.asarray([cluster.mean(axis=0) for cluster in clusters])

        # 3. Update predictions, clusters then means

        for _ in range(self.epochs):        # Train for self.epochs number of iterations
            preds = self.get_preds(means)

            # Update clusters for next epoch using new predictions
            clusters = [[] for i in range(self.k)]
            for pred,data in zip(preds,X_train):
                clusters[pred].append(data)
            clusters = [np.asarray(cluster) for cluster in clusters]

            # Update means
            means = np.zeros((means.shape))
            for idx,cluster in enumerate(clusters):
                if len(cluster.shape) == 0:
                    rand_idx = np.random.randint(0,120)
                    print(idx, rand_idx)
                    means[idx] = X_train[rand_idx]
                else:
                    means[idx] = cluster.mean(axis=0)

        self.means = means
        self.preds = preds

    def predict_one(self, x):
        """
        This function provides classification for one sample.
        """
        dists = np.sum((self.means - x)**2, axis=1)
        pred = np.argmin(dists)
        
        return pred

    def evaluate(self):
        """
        This function does multiple things.
        1. Finds out the permuation mapping between k-means clusters and ground truth clusters
        2. Prints the corresponding means
        3. Evaluates and prints the jaccard score for each cluster.
        """
        
        perms = permutations(range(self.k))
        max_total = 0
        best_perm = []

        for perm in perms:
            
            # Find the best permutation of classes using accuracy as a heuristic
            total = 0
            for idx in range(self.y.shape[0]):
                total += (self.y[idx] == perm[self.preds[idx]])
            
            if total > max_total:
                max_total = total
                best_perm = perm

        # Obtain prediction clusters and ground truth clusters
        pred_clusters = [[] for i in range(self.k)]
        gt_clusters = [[] for i in range(self.k)]
        self.preds = [best_perm[pred] for pred in self.preds]

        for idx,(real,pred) in enumerate(zip(self.y,self.preds)):
            pred_clusters[best_perm[pred]].append(idx)                  # K-means labels are now mapped to real labels
            gt_clusters[real].append(idx)

        self.clusters = (pred_clusters,gt_clusters)

        # Calculate Jaccard Score for each cluster
        for idx in range(self.k):
            s1 = len(pred_clusters[best_perm[idx]])
            s2 = len(gt_clusters[idx])
            total = pred_clusters[best_perm[idx]] + gt_clusters[idx]
            u12 = len(set(total))

            jaccard_score = (s1 + s2 - u12)/u12                         # Intersection over union
            print("The mean for cluster %d is: (%.4f, %.4f, %.4f, %.4f)" % (idx, *self.means[best_perm[idx]]))
            print("The Jaccard Score for cluster %d is %.4f\n" % (idx, 1-jaccard_score))

        return (1.0*max_total/self.y.shape[0])

    def scatter_plot(self):
        """
        This function creates two scatter plots: One with ground truth clusters, one with prediction clusters.
        """
        x = self.X[:,1]                         # Features 2 and 4 are chosen for scatter plot
        y = self.X[:,3]
        color = ['blue', 'red', 'green']
        
        colors = [color[pred] for pred in self.preds]
        plt.scatter(x, y, color=colors)
        plt.xlabel("Sepal Width (cm)")
        plt.ylabel("Petal Width (cm)")
        plt.title("Scatter Plot with Predicted Clusters")
        plt.savefig('pred_scatter.png')

        colors = [color[pred] for pred in model.y]
        plt.scatter(x, y, color=colors)
        plt.xlabel("Sepal Width (cm)")
        plt.ylabel("Petal Width (cm)")
        plt.title("Scatter Plot with Ground Truth Clusters")
        plt.savefig('gt_scatter.png')

    def _np_from_pd(self, data_pd):
        """
        This function takes a dataset as a pandas dataframe with string class labels as input
        and returns numpy arrays (ndarrays) for testing and training data and labels.
        """
        data_np = data_pd.to_numpy()
        np.random.shuffle(data_np)

        # Splitting into data and class labels
        X, y = data_np[:,:data_np.shape[1]-1], data_np[:,data_np.shape[1]-1]
        
        string_labels = np.unique(y)
        class_idx = range(len(string_labels))
        ylabel = dict(zip(string_labels, class_idx))

        y = np.asarray([ylabel[x] for x in y])

        return X, y


if __name__ == '__main__':

    np.random.seed(4)                                   # Fixing seed for reproducibility
    kwargs = {'data_path': './iris_plant.csv',          # CSV file path
              'k': 3,
              'epochs': 10}                             # Number of iterations

    model = KMeans(**kwargs)
    model.train()
    acc = model.evaluate()
    print("Accuracy: %.5f" % (acc))
    model.scatter_plot()                                # Print scatter plot
