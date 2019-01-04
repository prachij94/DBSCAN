# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:51:15 2018

@author: IMART
"""

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN

# Initialize some documents
doc1 = {'Science':0.8, 'History':0.05, 'Politics':0.15, 'Sports':0.1}
doc2 = {'News':0.2, 'Art':0.8, 'Politics':0.1, 'Sports':0.1}
doc3 = {'Science':0.8, 'History':0.1, 'Politics':0.05, 'News':0.1}
doc4 = {'Science':0.1, 'Weather':0.2, 'Art':0.7, 'Sports':0.1}
doc5 = {'Science':0.2, 'Weather':0.7, 'Art':0.8, 'Sports':0.9}
doc6 = {'Science':0.2, 'Weather':0.8, 'Art':0.8, 'Sports':1.0}
collection = [doc1, doc2, doc3, doc4, doc5, doc6]
df = pd.DataFrame(collection)
# Fill missing values with zeros
df.fillna(0, inplace=True)
# Get Feature Vectors
feature_matrix = df.as_matrix()

# Fit DBSCAN
db = DBSCAN(min_samples=1).fit(feature_matrix)

'''

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
'''

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Plot result
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#  Perform dimensional reduction of the feature matrix with PCA
X = PCA(n_components=2).fit_transform(feature_matrix) 

# Select which points will be painted red
red_points = [1, 4]
for i in red_points:
    labels[i] = -2

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    if k == -2:
        # Red for selected points
        col = [1, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()