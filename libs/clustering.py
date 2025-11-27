import numpy as np
from scipy.cluster import hierarchy
from thirdparty import sklearn_dunn
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class clustering():

    def __init__(self, mtmc):
        self.clusters_frame = list()
        self.colors = mtmc.colors.list
        self.trajectories_f_w = list()


    def new_cluster(self):
        cluster = {
            'id': None,
            'xw': None,
            'yw': None,
            'det': list()}
        return cluster

    def new_detection(self):
        det = {
            'id_cam': None,
            'x': None,
            'y': None,
            'w': None,
            'h': None,
            'id': None,
            'id_global': None}
        return det

    def new_trajectory(self):
        det = {
            'id_cam': None,
            'id': None,
            'xw': None,
            'yy': None,
            'feature_descriptor': None}
        return det


    def compute_clusters(self, distance_matrix, association_matrix):
        """
        Robust computation of clusters:
         - sanitize distance_matrix (replace inf/nan with large finite values, set diagonal = 0)
         - try agglomerative clustering for k in [min_num_clusters, max_num_clusters]
         - compute Dunn index (sklearn_dunn.dunn), allow that dunn may return NaN
         - choose optimal number of clusters robustly:
             * if derivative of indices is usable, use argmax of derivative (np.nanargmax)
             * otherwise fallback to argmax of indices (best Dunn), or fallback to smallest cluster count
        Returns: idx_clusters (labels for each detection), optimal_clusters (int)
        """

        # Ensure numpy array float
        distance_matrix = np.array(distance_matrix, dtype=float)
        num_det_f = distance_matrix.shape[0]

        # Sanitize distance matrix: replace nan/inf with large finite values
        if not np.isfinite(distance_matrix).all():
            finite_mask = np.isfinite(distance_matrix)
            if np.any(finite_mask):
                max_finite = np.nanmax(distance_matrix[finite_mask])
                if not np.isfinite(max_finite):
                    max_finite = 1.0
            else:
                max_finite = 1.0
            # replace NaN/Inf with a large value (so clustering treats them as far apart)
            distance_matrix[~finite_mask] = max_finite * 10.0

        # ensure diagonal is zero (distance to self)
        np.fill_diagonal(distance_matrix, 0.0)

        # determine range of clusters to try
        max_num_clusters = max(1, num_det_f - 1)
        # association_matrix assumed to have shape (N,N), compute how many associations are 1
        assoc_ones = 0
        if association_matrix is not None:
            try:
                assoc_ones = np.sum(association_matrix == 1)
            except Exception:
                assoc_ones = 0
        min_num_clusters = max((num_det_f - assoc_ones + 1), 2)
        min_num_clusters = int(min(min_num_clusters, max_num_clusters))  # clamp

        # trivial cases
        if num_det_f <= 1:
            # nothing to cluster
            idx_clusters = np.zeros(num_det_f, dtype=int)
            optimal_clusters = num_det_f
            return idx_clusters, optimal_clusters

        if min_num_clusters > max_num_clusters:
            # fallback: every detection is its own cluster
            optimal_clusters = num_det_f
            idx_clusters = np.arange(num_det_f)
            return idx_clusters, optimal_clusters

        clusters_range = np.arange(min_num_clusters, max_num_clusters + 1)
        num_clusters = clusters_range.size

        labels = []
        indices = np.full(num_clusters, np.nan, dtype=float)

        for k_idx, k in enumerate(clusters_range):
            try:
                # AgglomerativeClustering with precomputed distances
                # sklearn API: metric / affinity change across versions; using metric='precomputed' as before
                clusterer = AgglomerativeClustering(n_clusters=int(k), metric='precomputed', linkage='complete')
                lab = clusterer.fit_predict(distance_matrix)
                labels.append(lab)
                # compute Dunn index; dunn may return NaN in degenerate cases
                indices[k_idx] = sklearn_dunn.dunn(lab, distance_matrix)
            except Exception as e:
                # on any failure, keep NaN and continue
                labels.append(np.zeros(num_det_f, dtype=int))
                indices[k_idx] = np.nan

        # If only a single tested cluster count, just return it
        if num_clusters == 1:
            optimal_clusters = int(clusters_range[0])
            idx_clusters = labels[0]
            return idx_clusters, optimal_clusters

        # compute derivative of indices and select best position robustly
        with np.errstate(invalid='ignore'):  # ignore warnings from nan operations
            derivative = np.diff(indices)  # length num_clusters-1, may contain NaN

        # choose pos based on derivative if possible, otherwise fallback to best Dunn index
        pos = None
        try:
            # if derivative contains valid numbers, use argmax ignoring NaNs
            if derivative.size > 0 and not np.all(np.isnan(derivative)):
                pos = int(np.nanargmax(derivative)) + 1
            else:
                # derivative invalid -> fallback to Dunn index argmax
                if not np.all(np.isnan(indices)):
                    pos = int(np.nanargmax(indices))
                else:
                    # everything NaN: fallback to smallest number of clusters
                    pos = 0
        except Exception:
            # safest fallback
            pos = 0

        # clamp pos
        pos = int(max(0, min(pos, num_clusters - 1)))

        optimal_clusters = int(clusters_range[pos])
        idx_clusters = labels[pos] if pos < len(labels) else labels[0]

        # ensure idx_clusters is numpy array of ints
        idx_clusters = np.asarray(idx_clusters, dtype=int)

        return idx_clusters, optimal_clusters

    def display_detections_cluster(self,sct_f, det_in_cluster,cl):

        for d  in range(det_in_cluster.__len__()):
            det = det_in_cluster[d]
            xw = sct_f['xw'][det]
            yw = sct_f['yw'][det]
            plt.plot(xw, yw, 'x', lineWidth = 1, markerSize = 10, color = self.colors[cl])

    def display_centroid_cluster(self, mean_xw, mean_yw, cl):

            plt.scatter(mean_xw, mean_yw, s = 80, facecolors='none', edgecolors=self.colors[cl])
            plt.text(mean_xw+0.000005, mean_yw+0.000005, str(cl), fontsize=15, color='black')
            plt.title('Tracks')