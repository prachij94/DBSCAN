# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:10:29 2018

@author: IMART
"""

''' Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.
'''

import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([[37.9358, -122.3478],[33.8312, -117.6053]])
db = DBSCAN(eps= 0.3,min_samples = 10).fit(X)
print(db)



###DBSCAN PARAMETERS

'''
algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
'''

print(db.algorithm)



'''
eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
'''
print(db.eps)


'''
leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
'''
print(db.leaf_size)

'''
metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.calculate_distance for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.
'''

print(db.metric)

'''
metric_params : dict, optional
        Additional keyword arguments for the metric function.
'''
print(db.metric_params)

'''
min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
'''

print(db.min_samples)

'''
n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
'''
print(db.n_jobs)

'''
p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
'''
print(db.p)


###DBSCAN ATTRIBUTES

'''
core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
'''
print(db.core_sample_indices_)

'''
components_ : array, shape = [n_core_samples, n_features]
        Copy of each core sample found by training.
'''
print(db.components_)

'''

    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
'''
print(db.labels_)
print(db.__doc__)
print(db._estimator_type)




