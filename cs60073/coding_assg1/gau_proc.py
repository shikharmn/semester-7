# Author: Shikhar Mohan
# Roll No.: 18EC10054
# Language: Python 3.9.7 (backwards compatible for Python > 3.5)
# Dependencies: numpy (1.21.2), pandas (1.3.3), matplotlib (3.4.3), scikit-learn (1.0), tqdm (4.62.2)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def kernel(a, b, sigma, l):
    dist = a**2 + (b**2).T - 2*(b.T)*a
    kr = sigma**2*np.exp(-dist/(2*l**2))
    return kr

def hparam_tune(x,y,maxrange=30):
    min_mse = 1e19
    best_l = 10
    for l in tqdm(range(1,maxrange)):
        kr = kernel(x.reshape(-1,1), x.reshape(-1,1), y.std(), l)
        k1 = kernel(x_miss.reshape(-1,1), x, y.std(), l)
        k2 = np.mat((kr + sigma_n*np.eye(kr.shape[0])))**(-1)
        k3 = np.matmul(k2,y)
        mu = np.matmul(k1,k3)
        mse = 0.0
        for idx,xi in enumerate(x_miss):
            mean_inter = (y[xi-1] + y[xi+1])/2
            mse += (mean_inter - mu[idx])**2
        if min_mse > mse:
            min_mse = mse
            best_l = l
            # print(best_l, mse, mu.mean())
    
    return best_l

if __name__ == '__main__':

    data = pd.read_csv('./vaccination.csv').to_numpy()
    mean_n,sigma_n = 0,0.1

    x,f_cum,f = data[:,0],data[:,1],data[:,2]
    x = x[:,np.newaxis]
    f = f[:,np.newaxis]

    # Find out the missing vaccination days
    x_miss = []
    for i in range(x.shape[0]-1):
        if x[i+1][0]-x[i][0] > 1:
            x_miss.append((x[i]+x[i+1])//2)
    x_miss = np.asarray(x_miss)

    # Fit to the mean function
    poly_features= PolynomialFeatures(degree=3)
    model = LinearRegression()
    x_poly = poly_features.fit_transform(x)
    model.fit(x_poly, f)
    f_poly_pred = model.predict(x_poly)
    x_poly_miss = poly_features.fit_transform(x_miss)
    f_poly_miss = model.predict(x_poly_miss)

    y = f_poly_pred - f

    # Find the mean and covariance
    l = hparam_tune(x,y)
    kr = kernel(x.reshape(-1,1), x.reshape(-1,1), y.std(), l)
    k1 = kernel(x_miss.reshape(-1,1), x, y.std(), l)
    k2 = np.linalg.inv(kr + (sigma_n**2)*np.eye(kr.shape[0]))
    k3 = np.matmul(k2,y)
    mu = np.matmul(k1,k3)

    k_miss = kernel(x_miss.reshape(-1,1), x_miss.reshape(-1,1), y.std(), l)
    k4 = kernel(x, x_miss.reshape(-1,1), y.std(), l)
    sigma_miss = k_miss + (sigma_n**2)*np.eye(k_miss.shape[0]) - np.dot(k1,k2).dot(k4)

    # Save plot
    plt.plot(x, f, label = "Vaccinations (training)")
    plt.plot(x, f_poly_pred, label = "Mean Function")
    plt.errorbar(x_miss, f_poly_miss - mu,yerr=np.sqrt(abs(sigma_miss.diagonal())), fmt='b.',capsize=4, label = "Vaccinations (testing)")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Vaccinations Plot')
    plt.legend()
    plt.savefig('Plots.jpg')