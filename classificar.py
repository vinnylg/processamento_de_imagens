#!/usr/bin/env python

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
import shutil

def main():
    if len(sys.argv) != 2:
        print("USAGE: ./classifica file.csv")
        sys.exit()

    data_set = pd.read_csv(sys.argv[1], delimiter = ',')

    X = data_set[['HU1','HU2','HU3','HU4']]
    X_normalized = normalize(X)


    kmeans = KMeans(n_clusters=6, init = 'random', random_state=0, max_iter = 600)
    kmeans.fit(X_normalized)
    y_kmeans = kmeans.predict(X_normalized)

    types = list(dict.fromkeys(y_kmeans))
    for t in types:
        if not os.path.isdir("output/{}".format(t)):
            os.mkdir("output/{}".format(t))


    imgs = data_set['segmento'].values.tolist()
    
    for i in range(len(y_kmeans)):
        shutil.move("output/{}".format(imgs[i]),"output/{}/".format(y_kmeans[i]))

    print silhouette_score(X_normalized, kmeans.labels_, metric = 'euclidean')

    plt.scatter(X_normalized[:,0], X_normalized[:,1], c=y_kmeans, s=20, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, alpha=0.5);

    plt.show()

if __name__== "__main__":
  main()
