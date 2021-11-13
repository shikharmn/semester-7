# Author: Shikhar Mohan
# Roll No.: 18EC10054
# Date: 17/09/2021

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    # breakpoint()
    z = 1/(1+np.exp(-x))
    return z

def dsigmoid(x):
    z = sigmoid(x)
    return z*(1-z)

def data_normalize(X):
    for i in range(9):
        X[:,i] = (X[:,i] - X[:,i].mean())/X[:,i].std()
    return X

def data_onehot(x,classes):
    y = [0,0,0]
    y[x] = 1
    return y

def data_helper(path):
    data = pd.read_csv(path).to_numpy()
    X = data[:,:9].astype(np.float64)
    y = data[:,9]
    classes = len(np.unique(y))
    label_dict = dict(zip(np.unique(y), range(classes)))
    for idx in range(len(y)): y[idx] = label_dict[y[idx]]
    out = [0 for i in range(len(y))]
    for idx in range(len(y)): out[idx] = data_onehot(y[idx],classes)
    X = data_normalize(X)
    return X,np.asarray(out)

def data_helper(path, visualize=False):
    data = pd.read_csv(path).to_numpy()
    X = data[:,:9].astype(np.float64)
    y = data[:,9]
    classes = len(np.unique(y))
    label_dict = dict(zip(np.unique(y), range(classes)))
    for idx in range(len(y)): y[idx] = label_dict[y[idx]]
    out = [0 for i in range(len(y))]
    for idx in range(len(y)): out[idx] = data_onehot(y[idx],classes)
    X_n = data_normalize(X)
    if visualize == True:
        return X, X_n, np.asarray(out)
    return X_n,np.asarray(out)

def data_scatter_plot(X, y, filename, idx_1 = 2, idx_2 = 3):
    
    colors = [[] for i in range(3)]
    for d,l in zip(X, y):
        label = np.argmax(l)
        colors[label].append(d)
        
    for idx,val in enumerate(colors):
        plt.scatter(np.array(val)[:,idx_1],np.array(val)[:,idx_2], label = idx)

    plt.xlabel("Feature %d" % (idx_1))
    plt.ylabel("Feature %d" % (idx_2))
    plt.legend()
    plt.savefig(filename + ".jpg")
    plt.show()

class NN(object):
    """
    Two layer neural network (one hidden, one final) with sigmoid activations for both layers.
    """
    def __init__(self, config, std=3e-4):
        # Parse the config file
        in_size = config['in_size']
        hid_size = config['hid_size']
        out_size = config['out_size']
        self.batch_size = config['batch_size']
        self.num_iters = config['epochs']
        self.lr = config['learning_rate']
        self.path = config['path']

        # Create the weight and bias vectors
        self.W1 = std * np.random.randn(in_size, hid_size)
        self.b1 = np.zeros(hid_size)
        self.W2 = std * np.random.randn(hid_size, out_size)
        self.b2 = np.zeros(out_size)

    def forward(self, X, y = None):
        """
        This function takes as input the data and outputs activations is y is not given
        If y is given, it returns loss and gradients.
        """
        W1, b1 = self.W1, self.b1
        W2, b2 = self.W2, self.b2
        N, D = X.shape

        # Computing the forward pass
        x1 = X@W1 + b1
        x_h = sigmoid(x1)
        y1 = x_h@W2 + b2
        y_hat = sigmoid(y1)

        if y is None: return y_hat
        
        loss = np.mean((y_hat - y)**2)/2
        # Computing the backward pass via chain rule
        dloss = (y_hat - y)
        dy_hat = dsigmoid(y1) * dloss
        dW2 = x1.T@dy_hat
        db2 = np.sum(dy_hat,axis=0)

        dx_h = dy_hat@(W2.T)
        dx1 = dx_h*dsigmoid(x1)
        dW1 = X.T@dx1
        db1 = np.sum(dx1,axis=0)

        grads = (dW1, db1, dW2, db2)
        return loss, grads

    def train(self, X, y, X_t, y_t):
        """
        Method for training model on given data.
        """
        num_train = X.shape[0]

        self.losses, self.train_accs, self.test_accs = [],[],[]
        
        for it in range(1,self.num_iters+1):
            
            loss, grads = self.forward(X, y=y)
            dW1, db1, dW2, db2 = grads
            self.W1 -= self.lr*dW1
            self.b1 -= self.lr*np.squeeze(db1)
            self.W2 -= self.lr*dW2
            self.b2 -= self.lr*np.squeeze(db2)

            train_acc = (self.predict(X) == np.argmax(y,axis=1)).mean()
            test_acc = (self.predict(X_t) == np.argmax(y_t,axis=1)).mean()

            if it % 100 == 0:
                print("Iteration %d:\t%f cost\t%.4f train accuracy and\t%.4f test accuracy" % (it, loss, train_acc, test_acc))
            
            self.losses.append(loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
    
    def predict(self, X):
        y = self.forward(X)
        return np.argmax(y,axis=1)

    def visualize(self):
        
        X, X_n, y = data_helper(self.path, visualize=True)
        mean, std = X.mean(axis = 0), X.std(axis = 0)

        plt.title('Scatter Plot with two features pre-normalisation')
        data_scatter_plot(X, y, "unnormalized", 4, 5)

        plt.title('Scatter Plot with two features post-normalisation')
        data_scatter_plot(X_n, y, "normalized", 4, 5)
        
        # Visualize Cost vs Epoch
        plt.plot(self.losses)
        plt.ylabel('Cost')
        plt.xlabel('Epoch')
        plt.title('Training Loss vs Epochs')
        plt.savefig("CostvEpoch.jpg")
        plt.show()
        
        # Visualize Training and Testing accuracy
        plt.plot(self.train_accs, label = 'Train Acc')
        plt.plot(self.test_accs, label = 'Test Acc')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Accuracy vs Epochs')
        plt.legend()
        plt.savefig("AccvEpoch.jpg")
        plt.show()


if __name__ == '__main__':
    config = {'in_size':9,
              'hid_size':8,
              'out_size':3,
              'batch_size':25,
              'epochs':500,
              'learning_rate':0.01,
              'path': './Snails.csv'}           # Change to data path accordingly

    X,y = data_helper(config['path'])
    # breakpoint()
    # print(X,y)
    split = int(0.75*len(X))
    model = NN(config)
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]
    model.train(X_train, y_train, X_test, y_test)
    model.visualize()
