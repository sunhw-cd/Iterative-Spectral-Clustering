# --- Standard library ---
import os
import time

# --- Scientific stack ---
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Machine learning / statistics (scikit-learn) ---
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import pairwise_kernels

# --- SciPy ---
from scipy.linalg import qr, svd
from scipy.sparse import csgraph

# --- Utilities ---
from tqdm import tqdm
import joblib

# --- Local / custom modules ---
import fpsample



# Self-Tuning Spectral Clustering

def local_scaled_affinity(x, y, k=3):
    """
    Computes a locally scaled affinity matrix using k-nearest neighbor distances.
    
    Parameters:
    x, y : numpy arrays representing data points
    k : Number of neighbors to use for scaling (default: 3). Since the feature dimension of the data is large, It means there is no significant different bewteen distances to first k nearest neighbor
    
    Returns:
    affinity : Computed affinity matrix
    """
    
    # Find k-nearest neighbors for input data points
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=-1).fit(y)
    
    # Get distances to k-th nearest neighbor for each point
    x_nn_distances, _ = nbrs.kneighbors(x)
    sigmas_x = x_nn_distances[:, k-1]
    
    y_nn_distances, _ = nbrs.kneighbors(y)
    sigmas_y = y_nn_distances[:, k-1]

    # Compute pairwise distances between all points
    pd = pairwise_distances(x, y, metric="euclidean", n_jobs=-1)

    # Compute affinity matrix using Gaussian kernel with locally-scaled sigmas
    affinity = np.exp(-1 * (pd)**2 / (np.outer(sigmas_x, sigmas_y)))

    # Set diagonal elements to zero (no self-affinity)
    if x.shape[0] == y.shape[0] and np.allclose(x, y):
        np.fill_diagonal(affinity, 0)
 
    return affinity

def Laplacian_matrix(affinity, mode="rw"):
    """
    Computes the random walk Laplacian matrix from the affinity matrix.
    
    Parameters:
    affinity : Affinity matrix
    mode : "rw" or "sym"
    
    Returns:
    Laplacian matrix
    """
     
    A_matrix = affinity  # Affinity matrix

    # Compute random walk Laplacian matrix
    if mode == "rw":
        D_inv = np.diag(1.0 / np.sum(A_matrix, axis=-1))
        return np.eye(affinity.shape[0]) - D_inv @ affinity
    
    # Compute sym Laplacian matrix
    elif mode == "sym":
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(affinity, axis=-1)))
        return np.eye(affinity.shape[0]) - D_inv_sqrt @ affinity @ D_inv_sqrt


def eigenthings(L_matrix):
    """
    Computes eigenvalues and eigenvectors of the Laplacian matrix.
    
    Parameters:
    L_matrix : Laplacian matrix
    
    Returns:
    eigenvalues : Sorted eigenvalues
    eigenvectors : Corresponding eigenvectors
    """
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_matrix)

    # Sort eigenvalues in ascending order
    sort_index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_index]
    eigenvectors = eigenvectors[:, sort_index]

    return eigenvalues, eigenvectors

def clustering(eigenvectors, n_components=2, n_clusters=2, assign_labels='kmeans'):
    """
    Performs clustering based on the spectral embedding using eigenvectors.

    Parameters:
    eigenvectors : Eigenvectors obtained from Laplacian matrix
    n_components : Number of eigenvectors to use for clustering
    n_clusters : Number of clusters
    assign_labels : Clustering method ('kmeans', 'cluster_qr', 'GMM')

    Returns:
    cluster_space : Spectral embedding space
    labels : Cluster assignments
    model : Clustering model used
    """
    
    # Select top eigenvectors for clustering
    cluster_space = eigenvectors[:, :n_components]

    # Apply chosen clustering method
    if assign_labels == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters).fit(cluster_space)
        labels = kmeans.labels_
        model = kmeans

    elif assign_labels == 'cluster_qr':
        labels = cluster_qr(cluster_space)  # Ensure this function is implemented

    elif assign_labels == 'GMM':
        gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(cluster_space)
        labels = gmm.predict(cluster_space)
        model = gmm

    return cluster_space, labels, model

def eigerngap(x):
    """
    Computes the eigengap heuristic to determine the optimal number of clusters.

    Parameters:
    x : Eigenvalues

    Returns:
    max_index + 1 : Suggested number of clusters
    """
    
    # Initialize eigengap array
    eigengap = np.zeros(np.shape(x)[0] - 1)

    # Compute eigengap values
    for i in range(1, np.shape(x)[0] - 1):
        eigengap[i] = (x[i+1] - np.mean(x[:i+1])) / (np.mean(x[:i+1]) - x[0] + 1e-9)

    # Find index of maximum eigengap
    max_index = np.argmax(eigengap)

    return max_index + 1



# cluster_qr from sklearn

def cluster_qr(vectors):

    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors.argmax(axis=1)



def Spectral_embeding(Pea_fragment, sigma_nn=3):
    """
    Performs spectral embedding on a given data fragment using local-scaled affinity.

    Parameters:
    Pea_fragment : 2D numpy array representing the data fragment
    sigma_nn : Number of nearest neighbors for computing the affinity matrix (default: 3)

    Returns:
    eigenvalues : Computed eigenvalues from the Laplacian matrix
    eigenvectors : Corresponding eigenvectors used for spectral embedding
    Laplacian_mat : Laplacian matrix for clustering analysis
    affinity_matrix : Affinity matrix representing similarity relationships
    """

    # Compute affinity matrix using local-scaled affinity with k-nearest neighbors
    affinity_matrix = local_scaled_affinity(x=Pea_fragment, y=Pea_fragment, k=sigma_nn)

    # Compute Laplacian matrix from the affinity matrix
    Laplacian_mat = Laplacian_matrix(affinity_matrix)

    # Compute eigenvalues and eigenvectors for spectral embedding
    eigenvalues, eigenvectors = eigenthings(Laplacian_mat)

    return eigenvalues, eigenvectors, Laplacian_mat, affinity_matrix


def process_cluster(model, cluster, cluster_idx,labels, save=False):
    '''
    processes a given cluster assignment and adds new subclusters to 'cluster'-list
    
    Parameters
    ----------
    cluster: list of dictionaries
        stores relevant data of clusters, e.g., assigned frames
    cluster_idx: int
        Index of 'Cluster'-list row-entry that will be processed
    corr_array: array
        pair correlation map
    cluster_assignment: array
        assignment of frames to cluster
    save: bool
        save new subclusters in "cluster"-list and delete current cluster from list
    
    Returns
    -------
    cluster: list of dicts
        updated "cluster"-list
    -------
    author: CK 2022
    '''
    
    length = len(cluster)
    
    #Get initial frames in cluster
    frames = cluster[cluster_idx]['Cluster_Frames']
    frames = np.reshape(frames,frames.shape[0])
    
    #Get nr of new subclusters
    nr = np.unique(labels)
    
    #Vary subclusters
    for ii in nr:
        print(f'Creating sub-cluster: {cluster_idx}-{ii}')
    
        #Get assignment
        tmp_assignment = np.argwhere(labels == ii)
        tmp_assignment = np.reshape(tmp_assignment,tmp_assignment.shape[0])
        
        #Get subcluster correlation array

        #Save new cluster and model
        if save == True:
            print(f'Saving subcluster {cluster_idx}-{ii} as new cluster {length + ii}')
            cluster.append({"Cluster_Nr": length + ii,"Cluster_Frames": frames[np.ix_(tmp_assignment)]})

            file_name =f"knn_{cluster_idx}.pkl"
            directory = f"D:/used/model"
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, file_name)
            joblib.dump(model, file_path, compress=1)
    
    #Del old cluster from 'cluster'-list
    if save == True:
        cluster[cluster_idx] = {}
                            
    return cluster

def reconstruct_correlation_map(sample_idx, feature_idx, matrix, order=1, metric="correlation"):
    """
    Reconstructs a correlation map using pairwise distance calculations.

    Parameters:
    sample_idx : List or array of indices for the sample data
    feature_idx : List or array of indices for the feature set
    matrix : 2D numpy array representing the correlation matrix
    order : Number of times to refine the correlation calculation (default: 1)
    metric : Distance metric to use for reconstruction (default: "correlation")

    Returns:
    test_corr_frag : Reconstructed correlation map for the sample data
    """

    # Extract submatrices from the main correlation matrix
    train_corr_frag = matrix[np.ix_(feature_idx, feature_idx)]  # Feature-feature correlation matrix
    test_corr_frag = matrix[np.ix_(sample_idx, feature_idx)]    # Sample-feature correlation matrix

    # Apply iterative refinement using pairwise distance calculations if order > 0
    if order != 0:
        for i in range(order):
            test_corr_frag = pairwise_distances(test_corr_frag, train_corr_frag, n_jobs=-1, metric=metric)
            train_corr_frag = pairwise_distances(train_corr_frag, train_corr_frag, n_jobs=-1, metric=metric)

    return test_corr_frag  # Return the refined correlation map



## Code for sampling

def estimate_local_density(features, k=5):
    # Compute average distance to the k nearest neighbors (excluding self)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
    distances, _ = nbrs.kneighbors(features)
    avg_dist = np.mean(distances[:, 1:], axis=1)  # exclude self-distance
    density = 1 / (avg_dist + 1e-8)  # avoid division by zero
    return density

def scale_features_by_density(features, density, alpha=0.5):
    # Normalize density to [0, 1]
    norm_density = (density - density.min()) / (density.max() - density.min() + 1e-8)
    # Invert and apply power to exaggerate density contrast
    scale = norm_density
    #scale = (1 / (norm_density + 1e-5)) ** alpha
    # Scale each feature vector by its local density weight
    return features * scale[:, np.newaxis]

class Sampler:
    def __init__(self, indices, feature_space=None, max_samples=800, min_samples=400):
        """
        Initializes the Sampler object.

        Parameters:
        indices : Array of indices to be sampled
        feature_space : Optional array of features for FPS
        max_samples : Upper limit for the number of training samples
        min_samples : Lower limit for the number of training samples
        """
        self.indices = np.array(indices)
        self.feature_space = feature_space
        self.max = max_samples
        self.min = min_samples
        self.total_len = len(indices)

    def _clip_count(self, proportion):
        """
        Ensures the number of samples falls between min and max bounds.
        """
        return int(np.minimum(np.clip(self.total_len * proportion, self.min, self.max),self.total_len))

    def _random_sampling(self, proportion):
        """
        Randomly selects a clipped number of indices from the dataset.
        """
        count = self._clip_count(proportion)
        chosen = np.random.permutation(self.total_len)[:count]
        return np.sort(chosen)

    def _temporal_random_sampling(self, proportion):
        """
        Samples indices by randomly selecting full frames and extra trailing indices.
        Intuitively balances samples across temporal regions, adapting to sequence structure.
        """
        fra_length = 50  # size of each temporal frame
        fra_num = self.total_len // fra_length
        max_por = self.max / self.total_len
        min_por = self.min / self.total_len
        porty = np.clip(proportion, min_por, max_por)
        chosen = []

        if fra_num > 0:
            # Reshape into frames and sample a subset of frame columns
            fra_indices = np.arange(fra_num * fra_length).reshape((fra_num, fra_length)).T
            frame_ids = np.random.permutation(fra_num)[:int(fra_num * porty)]
            sampled_frames = fra_indices[:, frame_ids].flatten()
            chosen.extend(sampled_frames)

        # Sample any tail elements beyond full frames
        extra = np.arange(fra_num * fra_length, self.total_len)
        extra_sample = np.random.permutation(extra)[:int(len(extra) * porty)]
        chosen.extend(extra_sample)

        return np.sort(np.unique(chosen))

    def _fps_sampling(self, proportion, subset_indices=None):
        """
        Applies Farthest Point Sampling over the entire or subset feature space.
        """
        count = self._clip_count(proportion)
        if subset_indices is not None:
            features = self.feature_space[subset_indices]
            selected = fpsample.fps_npdu_sampling(features, count)
            return np.sort(subset_indices[selected])
        else:
            chosen=np.sort(fpsample.fps_npdu_sampling(self.feature_space, count))
            return chosen

    def _LS_fps_sampling(self, proportion, subset_indices=None):
        """
        Performs Local Scaled FPS: scales feature space using inverse local density,
        then applies FPS on that space.
        """
        count = self._clip_count(proportion)
        features = self.feature_space
        density = estimate_local_density(features, k=10)
        features = scale_features_by_density(features, density, alpha=0.5)

        if subset_indices is not None:
            features = features[subset_indices]
            selected = fpsample.fps_npdu_sampling(features, count)
            return np.sort(subset_indices[selected])
        else:
            return np.sort(fpsample.fps_npdu_sampling(features, count))

    def _temporal_fps(self, proportion, subset_indices=None):
        """
        Combines temporal sampling and FPS by taking half from each method.
        Useful for blending sequence continuity and global diversity.
        """
        # Temporarily halve the limits
        original_max = self.max
        original_min = self.min
        self.max = original_max // 2
        self.min = original_min // 2

        chosen_0 = self._temporal_random_sampling(proportion / 2)
        chosen_1 = self._fps_sampling(proportion / 2, subset_indices=subset_indices)

        self.max = original_max
        self.min = original_min

        chosen = np.concatenate((chosen_0, chosen_1))
        return np.sort(np.unique(chosen))

    def shuffle(self, proportion=0.2, method="random"):
        """
        Main entry point for applying the selected sampling strategy.

        Returns:
        - train_idx: sampled training subset (original indices)
        - test_idx: remaining indices (original)
        - chosen: positions of training indices in dataset
        - rest: positions of test indices in dataset
        """
        if method == "random":
            chosen = self._random_sampling(proportion)
        elif method == "temporal_random":
            chosen = self._temporal_random_sampling(proportion)
        elif method == "FPS":
            chosen = self._fps_sampling(proportion)
        elif method == "LS_FPS":
            chosen = self._LS_fps_sampling(proportion)
        elif method == "temporal_FPS":
            chosen = self._temporal_fps(proportion)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        rest = np.setdiff1d(np.arange(self.total_len), chosen)
        train_idx = self.indices[chosen]
        test_idx = self.indices[rest]

        return train_idx, test_idx, chosen, rest
