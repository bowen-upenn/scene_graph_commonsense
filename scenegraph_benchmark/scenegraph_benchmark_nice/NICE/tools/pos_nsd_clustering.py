import time
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import check_array, check_consistent_length, gen_batches
from scipy.spatial.distance import cdist
import math


class LDCOF(object):
    """
    Local density cluster-based outlier factor (LDCOF)
    """

    def __init__(self, cfg, alpha=0.8, n_subsets=2, n_clusters=8):
        super(LDCOF, self).__init__()
        self.alpha = alpha # % of entries for large clusters (LC)
        self.data = []
        self.distances = {}
        self.cluster_centers = []
        self.clusterer = KMeans(n_clusters=n_clusters, n_jobs=-1)
        self.n_subsets = n_subsets
        self.n_clusters = n_clusters
        self.LC = []
        self.SC = []
        self.head_rel_ids = cfg.HEAD_IDS
        self.body_rel_ids = cfg.BODY_IDS
        self.tail_rel_ids = cfg.TAIL_IDS

    def __clusters_separation(self):
        """ split clusters into large clusters (LC) and small clusters (SC) """
        D = len(self.data)
        cluster_sizes = []
        for cluster in range(self.n_clusters):
            cluster_sizes.append((cluster, self.data_clusters.tolist().count(cluster)))
        self.cluster_sizes = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)

        cumulative_size = 0
        threshold = None
        for i in range(self.n_clusters):
            cumulative_size += self.cluster_sizes[i][1]
            if cumulative_size > D * self.alpha:
                threshold = i
                break

        self.LC = [elem[0] for elem in self.cluster_sizes][:threshold]
        self.SC = [elem[0] for elem in self.cluster_sizes][threshold:]

    def __cluster_avg_distances(self):
        """ calculates mean distances within clusters """
        distances = {}
        for cluster in range(self.n_clusters):
            idx = np.where(self.data_clusters == cluster)[0]
            diff = self.data[idx] - self.cluster_centers[cluster]
            dists = [d for d in map(lambda t: math.sqrt(sum([pow(e, 2) for e in t])), diff)]
            if len(dists) > 0:
                distance = sum(dists) / len(dists)
            else:
                distance = 0.00000000001

            distances[cluster] = distance

        self.distances = distances

    def fit(self, features,  rel_labels, batch_max=500000, verbose=True):
        """ fit model on data """
        self.features = features
        self.rel_labels = rel_labels
        unique_rel_categories = set(self.rel_labels)
        all_clustered_labels = np.full(len(rel_labels), -1, dtype=np.intp)

        for cluster_idx, current_category in enumerate(unique_rel_categories):
            if verbose:
                t0 = time.time()

            dist_list = [i for i, label in enumerate(rel_labels) if label == current_category]

            for batch_range in gen_batches(len(dist_list), batch_size=batch_max):
                batch_dist_list = dist_list[batch_range]
                # print('batch len', len(batch_dist_list))
                # Load data subset
                subset_vectors = np.zeros((len(batch_dist_list), self.features.shape[1]), dtype=np.float32)
                for subset_idx, global_idx in enumerate(batch_dist_list):
                    subset_vectors[subset_idx, :] = self.features[global_idx, :]

            self.data = subset_vectors
            print('current category', current_category)
            if int(current_category) in self.head_rel_ids:
                self.n_clusters = 8
            elif int(current_category) in self.body_rel_ids:
                self.n_clusters = 4
            else:
                self.n_clusters = 2
            self.n_clusters = min(self.n_clusters, len(subset_vectors))
            print(self.n_clusters)
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(subset_vectors)
            self.clusterer = kmeans

            self.data_clusters = self.clusterer.predict(subset_vectors)
            self.cluster_centers = self.clusterer.cluster_centers_

            self.__clusters_separation()
            self.__cluster_avg_distances()

            lc_centers = self.cluster_centers[self.LC]
            clusters = self.clusterer.predict(subset_vectors)
            res = []
            for i, cluster in zip(range(len(clusters)), clusters):
                if cluster in self.LC:
                    _sum = [pow(elem[0] - elem[1], 2) for elem in zip(subset_vectors[i], self.cluster_centers[cluster])]
                    dist = math.sqrt(sum(_sum))
                    if self.distances[cluster] == 0:
                        res.append(dist / 0.00001)
                    else:
                        res.append(dist / self.distances[cluster])
                else:
                    entry = subset_vectors[i]
                    min_dist_to_cluster = 99999999999
                    for lc in self.LC:
                        center = self.cluster_centers[lc]
                        dist = math.sqrt(sum([e for e in map(lambda x: pow(x, 2), subset_vectors[i] - center)]))
                        if dist < min_dist_to_cluster:
                            min_dist_to_cluster = dist

                    if self.distances[cluster] == 0:
                        res.append(min_dist_to_cluster / 0.00001)
                    else:
                        res.append(min_dist_to_cluster / self.distances[cluster])

            res = np.array(res).reshape(-1)
            print('res', res.shape)
            model = KMeans(n_clusters=self.n_subsets)
            model.fit(res.reshape(len(res), 1))
            clusters = [res[np.where(model.labels_ == i)] for i in range(self.n_subsets)]
            n_clusters_made = len(set([k for j in clusters for k in j]))
            if n_clusters_made < self.n_subsets:
                continue

            cluster_maxs = [np.max(c) for c in clusters]
            bound = np.sort(np.array(cluster_maxs))
            other_bounds = range(1, self.n_subsets)
            for i in range(len(res)):

                if res[i] <= bound[0]:
                    all_clustered_labels[batch_dist_list[i]] = 0
                else:
                    for j in other_bounds:
                        # print(j)
                        if bound[j - 1] <= res[i] < bound[j]:
                            all_clustered_labels[batch_dist_list[i]] = j
            if verbose:
                print ("Clustering {} of {} categories into {} subsets ({:.2f} secs).".format(
                    cluster_idx + 1, len(unique_rel_categories), self.n_subsets, time.time() - t0))

        if (all_clustered_labels > 0).all():
            raise ValueError("A clustering error occurred: incomplete labels detected.")

        self.output_labels = all_clustered_labels
        return self



class PosNSDClustering(BaseEstimator, ClusterMixin):

    def __init__(self, cfg, n_subsets=3, method='default', head_density_t=0.5, body_density_t=0.5, tail_density_t=0.5, verbose=False,
                 dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
        self.n_subsets = n_subsets
        self.method = method
        self.head_density_t = head_density_t
        self.body_density_t = body_density_t
        self.tail_density_t = tail_density_t
        self.verbose = verbose
        self.output_labels = None
        self.random_state = random_state
        self.dim_reduce = dim_reduce
        self.batch_max = batch_max
        self.calc_auxiliary = calc_auxiliary
        self.cfg = cfg

    def fit(self, X, y):
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)
        self.output_labels, self.densities, _ = self.cluster_pos_nsd_subsets(X=X, y=y, **self.get_params())
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.output_labels

    def cluster_pos_nsd_subsets(self, X, y, cfg=None, n_subsets=3, method='default', head_density_t=0.5, body_density_t=0.5,
                                tail_density_t=0.5, verbose=False,
                                dim_reduce=256, batch_max=500000, random_state=None, calc_auxiliary=False):
        '''
        Pos-NSD clustering process based on local density
        Args:
            X: Feature of each triplet
            y: Relation labels
            n_subsets: Number of subsets
            method:  The algorithm to be used to calculate local density values
            density_t: Threshold for cutoff distance
            verbose: Whether to print progress messages to stdout
            dim_reduce: The dimensionality to reduce the feature vector to, prior to calculating distance
            batch_max: The maximum batch of feature vectors to process at one time (loaded into memory)
            random_state: Random seed, used to produce the same result
            calc_auxiliary: Provide auxiliary including delta centers and density centers

        Returns:
            all_clustered_labels : Clustered labels for each triplet, labels are integers ordered from most simple to most complex
            auxiliary_info : If calc_auxiliary is set to True, this list contains collected auxiliary information

        '''
        head_rel_ids = cfg.HEAD_IDS
        body_rel_ids = cfg.BODY_IDS
        tail_rel_ids = cfg.TAIL_IDS
        density_t = head_density_t
        if not density_t > 0.0:
            raise ValueError("density_thresh must be positive.")
        X = check_array(X, accept_sparse='csr')
        check_consistent_length(X, y)

        unique_categories = set(y)
        t0 = None
        pca = None
        auxiliary_info = []
        if X.shape[1] > dim_reduce:
            pca = PCA(n_components=dim_reduce, copy=False, random_state=random_state)

        all_clustered_labels = np.full(len(y), -1, dtype=np.intp)

        all_densities = np.full(len(y), -1, dtype=np.intp)

        for cluster_idx, current_category in enumerate(unique_categories):
            if int(current_category) in head_rel_ids:
                density_t = head_density_t
            elif int(current_category) in body_rel_ids:
                density_t = body_density_t
            else:
                density_t = tail_density_t
            print(current_category, density_t)
            if verbose:
                t0 = time.time()

            # Collect the "learning material" for this particular category
            dist_list = [i for i, label in enumerate(y) if label == current_category]

            for batch_range in gen_batches(len(dist_list), batch_size=batch_max):
                batch_dist_list = dist_list[batch_range]

                # Load data subset
                subset_vectors = np.zeros((len(batch_dist_list), X.shape[1]), dtype=np.float32)
                for subset_idx, global_idx in enumerate(batch_dist_list):
                    subset_vectors[subset_idx, :] = X[global_idx, :]

                # Calc distances
                if pca:
                    subset_vectors = pca.fit_transform(subset_vectors)
                m = np.dot(subset_vectors, np.transpose(subset_vectors))
                t = np.square(subset_vectors).sum(axis=1)
                distance = np.sqrt(np.abs(-2 * m + t + np.transpose(np.array([t]))))

                # Calc densities
                if method == 'gaussian':
                    densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                    distance = distance / np.max(distance)
                    for i in range(len(subset_vectors)):
                        densities[i] = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((-1) * np.power(distance[i], 2) / 2.0))
                else:
                    densities = np.zeros((len(subset_vectors)), dtype=np.float32)
                    flat_distance = distance.reshape(distance.shape[0] * distance.shape[1])
                    dist_cutoff = np.sort(flat_distance)[int(distance.shape[0] * distance.shape[1] * density_t) - 1]
                    for i in range(len(batch_dist_list)):
                        densities[i] = len(np.where(distance[i] < dist_cutoff)[0]) - 1  # remove itself
                if len(densities) < n_subsets:
                    # raise ValueError("Cannot cluster into {} subsets due to lack of density diversification,"
                    #                  " please try a smaller n_subset number.".format(n_subsets))
                    continue

                # Optionally, calc auxiliary info
                if calc_auxiliary:
                    # Calculate deltas
                    deltas = np.zeros((len(subset_vectors)), dtype=np.float32)
                    densities_sort_idx = np.argsort(densities)
                    for i in range(len(densities_sort_idx) - 1):
                        larger = densities_sort_idx[i + 1:]
                        larger = larger[np.where(larger != densities_sort_idx[i])]  # remove itself
                        deltas[i] = np.min(distance[densities_sort_idx[i], larger])

                    # Find the centers and package
                    center_id = np.argmax(densities)
                    center_delta = np.max(distance[np.argmax(densities)])
                    center_density = densities[center_id]
                    auxiliary_info.append((center_id, center_delta, center_density))

                model = KMeans(n_clusters=n_subsets, random_state=random_state)
                model.fit(densities.reshape(len(densities), 1))
                clusters = [densities[np.where(model.labels_ == i)] for i in range(n_subsets)]
                n_clusters_made = len(set([k for j in clusters for k in j]))
                if n_clusters_made < n_subsets:
                    # raise ValueError("Cannot cluster into {} subsets, please try a smaller n_subset number, such as {}.".
                    #                  format(n_subsets, n_clusters_made))
                    continue

                cluster_mins = [np.min(c) for c in clusters]
                bound = np.sort(np.array(cluster_mins))

                # Distribute into subsets, and package into global adjusted returnable array, optionally aux too
                other_bounds = range(n_subsets - 1)
                for i in range(len(densities)):

                    # Check if the most 'clean'
                    if densities[i] >= bound[n_subsets - 1]:
                        all_clustered_labels[batch_dist_list[i]] = 0
                        all_densities[batch_dist_list[i]] = densities[i]
                    # Else, check the others
                    else:
                        for j in other_bounds:
                            if bound[j] <= densities[i] < bound[j + 1]:
                                all_clustered_labels[batch_dist_list[i]] = len(bound) - j - 1
                                all_densities[batch_dist_list[i]] = densities[i]
                if int(current_category) in head_rel_ids:
                    print('head:', current_category, 'clean len:', sum(all_clustered_labels == 0), 'noisy len:',
                          sum(all_clustered_labels == n_subsets - 1),
                          sum(all_clustered_labels == 0) / (sum(all_clustered_labels == n_subsets - 1) + 1e-3))
                elif int(current_category) in body_rel_ids:
                    print('body:', current_category, 'clean len:', sum(all_clustered_labels == 0), 'noisy len:',
                          sum(all_clustered_labels == n_subsets - 1),
                          sum(all_clustered_labels == 0) / (sum(all_clustered_labels == n_subsets - 1) + 1e-3))
                else:
                    print('tail:', current_category, 'clean len:', sum(all_clustered_labels == 0), 'noisy len:',
                          sum(all_clustered_labels == n_subsets - 1),
                          sum(all_clustered_labels == 0) / (sum(all_clustered_labels == n_subsets - 1) + 1e-3))
            if verbose:
                print("Clustering {} of {} categories into {} subsets ({:.2f} secs).".format(
                    cluster_idx + 1, len(unique_categories), n_subsets, time.time() - t0))

        if (all_clustered_labels > 0).all():
            raise ValueError("A clustering error occurred: incomplete labels detected.")

        return all_clustered_labels, all_densities, auxiliary_info