# Author: Shikhar Mohan
# Roll No.: 18EC10054
# Date: 17/09/2021

import numpy as np
import pandas as pd

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.depth = None
        self.depth = None

        self.attribute = None
        self.threshold = None
        self.probs = None
        self.is_terminal = False


class DecisionTree:
    def __init__(self,train_path=None,test_path=None,max_depth=99,ccp_alpha=0):
        self.train_path = train_path
        self.test_path = test_path
        train_pd = pd.read_csv(self.train_path)
        test_pd = pd.read_csv(self.test_path)
        self.ccp_alpha = ccp_alpha
        
        self.X_train, self.X_test, self.y_train, self.y_test = self._np_from_pd(train_pd, test_pd)

        self.train_preds = None
        self.test_preds = None
        self.classes = range(3)
        self.Tree = None
        self.max_depth = max_depth
        self.leaves = 0
        self.complexity = 9999
        self.tree_depth = 0

    def calculate_entropy(self, probs):
        entropy = sum([-p*np.log(p) if p != 0 else 0 for p in probs])
        return entropy

    def nodeProbs(self, y):
        probs = []
        total = len(y)
        for _class in self.classes:
            matches = sum(y == _class)
            p = matches*1.0/total
            probs.append(p)

        return np.asarray(probs)

    def calc_info_gain(self, y, x_attr, c, node_entropy):
        y_right = y[x_attr > c]
        y_left = y[x_attr <= c]
        s_r = y_right.shape[0]
        s_l = y_left.shape[0]
        s = y.shape[0]

        if y_right.shape[0] == 0 or y_left.shape == 0:
            return 0

        left_probs = self.nodeProbs(y_left)
        right_probs = self.nodeProbs(y_right)
        left_entropy = self.calculate_entropy(left_probs)
        right_entropy = self.calculate_entropy(right_probs)

        infoGain = node_entropy
        infoGain -= (s_r*right_entropy)/s + (s_l*left_entropy)/s

        return infoGain

    def calculate_split(self, X, y):

        splitAttr = None
        threshold = None
        maxinfoGain = -9999

        node_probs = self.nodeProbs(y)
        node_entropy = self.calculate_entropy(node_probs)
        s = y.shape[0]

        for attr in range(len(X[0])):
            x_attr = X[:,attr]
            thresh_candidates = np.unique(x_attr)
            for c in thresh_candidates:
                infoGain = self.calc_info_gain(y, x_attr, c, node_entropy)
                if infoGain is None:
                    return None, None

                if infoGain > maxinfoGain:
                    maxinfoGain = infoGain
                    splitAttr = attr
                    threshold = c

        if maxinfoGain < 0.1:
            return None,None

        return splitAttr, threshold

    def termCriteria(self, node):
        if node.depth >= self.max_depth:
            return True

    def buildTree(self, X, y, node: Node):

        if self.termCriteria(node) == True:
            node.is_terminal = True
            return
        
        splitAttr, threshold = self.calculate_split(X,y)

        if splitAttr is None:
            node.is_terminal = True
            return

        # Split accordingly

        attr_col = X[:,splitAttr]
        l_idx = (attr_col < threshold)
        r_idx = (attr_col >= threshold)

        x_left, y_left = X[l_idx], y[l_idx]
        x_right, y_right = X[r_idx], y[r_idx]

        if x_left.shape[0] < 2 or x_right.shape[0] < 2:
            node.is_terminal = True
            return

        # Make node and its children according to splits

        node.attribute = splitAttr
        node.threshold = threshold

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.probs = self.nodeProbs(y_left)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.probs = self.nodeProbs(y_right)

        # Build further recursively

        self.buildTree(x_left, y_left, node.left)
        self.buildTree(x_right, y_right, node.right)

    def train(self, X_train=None ,y_train=None):
        
        if X_train is None and y_train is None:
            X_train, y_train = self.X_train, self.y_train

        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.probs = self.nodeProbs(y_train)

        self.buildTree(X_train,y_train,self.Tree)

    def predict_one(self, x, node=None):
        # Provide probability values for one sample

        if node is None:
            node = self.Tree
        if node.is_terminal:
            return node.probs

        if x[node.attribute] > node.threshold:
            probs = self.predict_one(x, node.right)
        else:
            probs = self.predict_one(x, node.left)
        
        return probs

    def predict(self, X_train=None, X_test=None):
        # Predict for all of training and test set

        if X_train is None and X_test is None:
            X_train, X_test = self.X_train, self.X_test

        preds = [[],[]]

        for idx,X in enumerate([X_train, X_test]):
            for x in X:
                probs = self.predict_one(x)
                pred = np.argmax(probs)
                preds[idx].append(pred)
        self.train_preds = np.asarray(preds[0])
        self.test_preds = np.asarray(preds[1])

    def calc_accuracy(self, type='test', X_train=None, X_test=None, y_train=None, y_test=None):
        # Calculate accuracy

        self.predict(X_train=X_train, X_test=X_test)

        if X_train is None:
            X_train, X_test = self.X_train, self.X_test
            y_train, y_test = self.y_train, self.y_test
        
        test_total,train_total = 0,0
        for idx in range(y_test.shape[0]):
            test_total += (y_test[idx] == self.test_preds[idx])
        for idx in range(y_train.shape[0]):
            train_total += (y_train[idx] == self.train_preds[idx])

        if type == 'test':
            return (1.0*test_total/y_test.shape[0])
        elif type == 'train':
            return (1.0*train_total/y_train.shape[0])
        else:
            return (1.0*(train_total+test_total)/(y_train.shape[0] + y_test.shape[0]))

    def recursive_pruning(self, node=None):
        
        if node is None: node = self.Tree
        if node.is_terminal is True: return
        node.is_terminal = True
        self.leaves = 0
        self.dfs_tree()
        current_complexity = 1 - self.calc_accuracy() + self.ccp_alpha*(self.leaves)
        
        if current_complexity <= self.complexity:
            self.complexity = current_complexity
            # print("PRUNED")
            return
        else:
            node.is_terminal = False
            self.recursive_pruning(node.left)
            self.recursive_pruning(node.right)

    def prune_tree(self):
        self.dfs_tree()
        self.complexity = 1 - self.calc_accuracy() + self.ccp_alpha*(self.leaves)
        self.recursive_pruning()
        return(str(self.calc_accuracy()))

    def _np_from_pd(self, train_pd, test_pd):
        train_np = train_pd.to_numpy()
        test_np = test_pd.to_numpy()
        np.random.shuffle(train_np)

        X_train, y_train = train_np[:,:4], train_np[:,4]
        X_test, y_test = test_np[:,:4], test_np[:,4]
        
        ylabel = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
        y_train = np.asarray([ylabel[x] for x in y_train])
        y_test = np.asarray([ylabel[x] for x in y_test])

        return X_train, X_test, y_train, y_test
    
    def calc_complexity(self, acc):
        complexity = 1 - acc + self.ccp_alpha*self.leaves
        return complexity

    def dfs_tree(self, node=None):
        # For debugging and analysing the decision tree
        if node == None:
            node = self.Tree

        # node_entropy = self.calculate_entropy(node.probs)
        # print(node.depth, node_entropy)
        self.tree_depth = max(node.depth, self.tree_depth)
        if node.is_terminal is True:
            self.leaves += 1
            return
        # print("left:\t")
        self.dfs_tree(node.left)
        # print("right:\t")
        self.dfs_tree(node.right)


if __name__ == '__main__':
    kwargs = {'train_path': './data/iris_train_data.csv',
              'test_path': './data/iris_test_data.csv',
              'max_depth': 3,
              'ccp_alpha': 0.25}
    tree = DecisionTree(**kwargs)
    tree.train()
    print("Training Accuracy: "+str(tree.calc_accuracy(type='train')))
    print("Testing Accuracy: "+str(tree.calc_accuracy()))
    print("Post-pruning Accuracy: "+str(tree.prune_tree()))
    print("Decision Tree Depth: "+str(tree.tree_depth))