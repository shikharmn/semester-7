Name: Shikhar Mohan
Roll No.: 18EC10054
Date: 17/09/2021

1. Run d_tree.py using Python 3. Code has been tested on Python 3.9.5, uses numpy 1.21.2 and pandas 1.3.3. The command line argument is:
    python3 d_tree.py

2. One can investigate the effects of hyperparameters on the written model by adding lines to the main section, but by default, it builds the class, trains the Decision Tree and prunes it.
    We obtain 96.67% training accuracy, 93.3% testing accuracy with a tree of max_depth 3.

3. We make a class Node using which we build the tree recursively, stopping when one of the halt criteria (refer pt. 4) are satisfied. The class DecisionTree builds, trains and predicts using the decision tree.
    We obtain the split by iteratively checking every element in the given attribute as a candidate for threshold and using the one which gives us the highest Entropy Gain. Entropy, entropy gain, probabilities etc, are evaluated using their respective helper functions.
    One forward pass through the tree predicts the output for one element, which we do iteratively to obtain predictions for the entire training set/testing set, where we use calc_accuracy function to obtain accuracy metrics.

4. Training Halt Criteria:
    We halt training at whichever of the following criteria happens first:
    If an attribute cannot be found to split on: We clearly cannot train the decision tree any better on this node, so we stop here.
    If the split of data on a node is less than 2: Splits too small in size clearly imply overfitting as the model is just memorizing the testing data.
    Maximum depth: Deeper trees with exponentially larger number of nodes tend to overfit more, so we cap the maximum depth at 3. Producing the smallest tree with the highest accuracy is the goal here.

5. Criteria for Pruning:
    We use a criteria named cost-complexity and perform Minimal Cost-Complexity Pruning. This criteria manages the tradeoff between size of tree and training accuracy. The formula is 1 - acc + alpha*leaves, where alpha is a hyperparameter.
    We set alpha = 0.1, which is slightly more aggressive than regular reduced error pruning. That is because in reduced error pruning we never give up training accuracy for making the tree smaller, but here we do in hopes that it would improve our test accuracy, and in many cases it does.
    It so happens that the training halt criteria give the smallest tree there is, and alpha = 0.275 is the first value where a node gets pruned, but the testing accuracy drops to 0.67 which we don't want.