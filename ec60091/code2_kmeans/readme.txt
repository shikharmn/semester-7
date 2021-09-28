Name: Shikhar Mohan
Roll No.: 18EC10054
Date: 28/09/2021

1. Run kmeans.py using Python 3. Code has been tested on Python 3.9.5, uses numpy 1.21.2 and pandas 1.3.3. The command line argument is:
    python3 kmeans.py
   The code creates two .png images, 'gt_scatter.png' and 'pred_scatter.png'. The former contains ground truth clusters and latter
   contains predicted clusters.

2. Implemented inside the KMeans class, one can observe that our model is quite robust to the choice of random seed, we pick one for
    reproducibility reasons.

3. The class has multiple helper methods, the functionality of which has been explained inside the function definition itself, but the
    important ones are:
    a. train(): This method trains the model on the entire dataset.
    b. evaluate(): This method finds out the correct permutation of class labels and k-means clusters and evaluates the jaccard distance.
    c. scatter_plots(): This method makes the desired scatter plots.