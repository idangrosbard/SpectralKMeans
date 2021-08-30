import spkmeans
import argparse
import sys
import numpy as np

# Given A Matrix and centroid returns a vector of length from each point to center
def calc_distance_single_centroid(matrix, centroid):
    temp_mat = matrix - centroid
    temp_mat = np.square(temp_mat)
    dist = np.sum(temp_mat, axis=1)
    return dist

def calc_centroids(T: np.array, k: int):
    np.random.seed(0)
    n, d = T.shape
    indices = []
    # dist [i,j] = l2 distance from point i to centroid j
    # every entry of the table is initialized to infinity (So min distance won't be affected)
    dists = np.ones((n, k)) * float('inf')

    # 4th Stage: Initialize first centroid
    centroid_index = np.random.choice(n)
    centroid = T[centroid_index, :]

    # 5th Stage: Main loop of algorithm: updates a new centroid, calculates distances and selects another
    for i in range(k):
        dists[:, i] = calc_distance_single_centroid(T, centroid)
        min_dist = np.min(dists, axis=1)
        total_dist = np.sum(min_dist)
        pr = min_dist / total_dist
        indices.append(centroid_index)
        centroid_index = np.random.choice(n, p=pr)
        centroid = T[centroid_index, :]

    init_centroids = np.zeros((k,k))
    for i in range(k):
        init_centroids[i, :] = T[indices[i], :]

    return init_centroids

if __name__=='__main__':
    k = int(sys.argv[1])
    goal = sys.argv[2]
    input_path = sys.argv[3]
    if goal in ['wam', 'ddg', 'lnorm', 'jacobi']:
        spkmeans.run(k, goal, input_path)
    if goal == 'spk':
        T = spkmeans.get_T(k, input_path)
        T_arr = np.array(T)
        k = T_arr.shape[1]
        centroids = calc_centroids(T_arr, k)
        spkmeans.kmeans(T, centroids.tolist())
    
