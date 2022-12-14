#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
import shutil

def main():
    if len(sys.argv) != 2:
        print("USAGE: ./classifica file.csv")
        sys.exit()
    
    if not os.path.isfile(sys.argv[1]):
        print("File not found")
        sys.exit()

    if not os.path.isdir("output"):
        print("Output not found. First usage: ./processar.py")
        sys.exit()

    if not os.path.isfile("output/1.jpg"):
        print("No have files to process")
        sys.exit()
        

    data_set = pd.read_csv(sys.argv[1], delimiter = ',')

    X = data_set[['HU1','HU2','HU3']]

    X_normalized = normalize(X)

    kmeans = KMeans(n_clusters=7, init = 'random', random_state=0, max_iter = 600)
    kmeans.fit(X_normalized)
    y_kmeans = kmeans.predict(X_normalized)

    types = list(dict.fromkeys(y_kmeans))
    for t in types:
        if not os.path.isdir("output/{}".format(t)):
            os.mkdir("output/{}".format(t))


    data_set['kmeans']=y_kmeans

    data_set.to_csv("output/kmeans.csv", encoding='utf-8',index = False)
    print("Results in output/kmeans.csv")
    print silhouette_score(X_normalized, kmeans.labels_, metric = 'euclidean')

    imgs = data_set['segmento'].values.tolist()
    
    for i in range(len(y_kmeans)):
        shutil.move("output/{}".format(imgs[i]),"output/{}/".format(y_kmeans[i]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(X_normalized[:,0], X_normalized[:,1], X_normalized[:,2], c=y_kmeans, s=20, cmap='viridis')

    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1],centers[:, 2], c='red', s=150, alpha=0.5);

    ax.set_xlabel('Hu Moments 1')
    ax.set_ylabel('Hu Moments 2')
    ax.set_zlabel('Hu Moments 3')
    plt.title('7 Cluster K-Means')

    plt.colorbar(p)
    plt.show()

    # plt.scatter(X_normalized[:,0], X_normalized[:,1], c=y_kmeans, s=20, cmap='viridis')

    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, alpha=0.5);

    # plt.show()

if __name__== "__main__":
  main()
