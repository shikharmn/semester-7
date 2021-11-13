from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
from argparse import RawTextHelpFormatter

def sampleSlice(i,mu,cov,x):
    mean = mu[i]
    var = cov[0][0]
    x[i] = mu[i] + np.random.standard_normal()*np.sqrt(var)
    return x

def createImage(x):
    img = np.zeros((30,30))
    for i in range(30):
        for j in range(30):
            img[i,j] = x[30*i+j]
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-inf", '--in_folder', default='images/', type=str,
                        help="input folder path for images.")
    parser.add_argument("-outf", "--out_folder", default="./output", type=str,
                        help="output folder path for figures")
    args = parser.parse_args()

    if args.out_folder != "./" and not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Defining the given image's mean and covariance
    mu = np.zeros(900)
    for i in range(30):
        for j in range(30):
            mu[30*i+j] = (i+j)/100
    cov = 0.1*np.eye(900)

    # Gibbs sampling algorithm
    imgs = []
    sample = 0.5*np.ones(900)
    for t in range(40):
        for i in range(900):
            sample = sampleSlice(i,mu,cov,sample)
        if (t+1)%10 == 0:
            img = createImage(sample)
            imgs.append(img)

    # Generate 4 plots for 40 iterations
    fig, axs = plt.subplots(1,5)
    axs[0].imshow(createImage(mu))
    axs[0].axis('off')
    for i in range(4):
        axs[i+1].imshow(imgs[i])
        axs[i+1].axis('off')
    out_path = os.path.join(args.out_folder, 'grad1.png')
    plt.savefig(out_path)

    # Observing sample at 10th iteration for 4 different images
    img_list = glob.glob(os.path.join(args.in_folder+"*.png"))      # Load 4 images in current directory
    diff_imgs = []
    gen_imgs = []
    for file in img_list:
        imgs = np.asarray(Image.open(file).resize((30,30)).convert('L'))/255
        diff_imgs.append(imgs)
        
        # Gibbs sampling algorithm
        sample = 0.5*np.ones(900)
        for t in range(10):
            for i in range(900):
                sample = sampleSlice(i,imgs.reshape(-1),cov,sample)
            if (t+1)%10 == 0:
                img = createImage(sample)
                gen_imgs.append(img)

    # Plot images and corresponding samples at 10th iteration
    plt.figure(figsize=(8, 1), dpi=40)
    fig, axs = plt.subplots(2,4)
    plt.subplots_adjust(wspace=0.1, 
                        hspace=-0.35)
    for i in range(4):
        axs[0,i].imshow(diff_imgs[i], cmap='gray')
        axs[0,i].axis('off')
    for i in range(4):
        axs[1,i].imshow(gen_imgs[i], cmap='gray')
        axs[1,i].axis('off')
    out_path = os.path.join(args.out_folder, 'comp2.png')
    plt.savefig(out_path)

    # Observing sample at 10th iteration for 2 different images and..
    # ..four different covariances
    img0 = mu.reshape((30,30))
    img1_path = os.path.join(args.in_folder, 'lenna.png')
    img1 = np.asarray(Image.open(img1_path).resize((30,30)).convert('L'))/255
    img_arr = [[img0],[img1]]

    lamdas = [0.1,0.01,0.0025,1e-4]
    for idx,imag in enumerate([img0,img1]):
        
        # Gibbs sampling algorithm
        for lamda in lamdas:
            nmu = imag.reshape(-1)
            ncov = lamda*np.eye(900)
            sample = 0.5*np.ones(900)
            for t in range(10):
                x = 0
                for i in range(900):
                    sample = sampleSlice(i,nmu,ncov,sample)
                if (t+1)%10 == 0:
                    img = createImage(sample)
                    img_arr[idx].append(img)

    plt.figure(figsize=(8, 1), dpi=40)
    fig, axs = plt.subplots(2,5)
    plt.subplots_adjust(wspace=0.1, 
                        hspace=-0.525)
    for i in range(5):
        axs[0,i].imshow(img_arr[0][i], cmap='gray')
        axs[0,i].axis('off')
    for i in range(5):
        axs[1,i].imshow(img_arr[1][i], cmap='gray')
        axs[1,i].axis('off')
    out_path = os.path.join(args.out_folder, 'comp3.png')
    plt.savefig(out_path)