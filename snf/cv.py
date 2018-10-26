# -*- coding: utf-8 -*-
"""
Code for implementing cross-validation of Similarity Network Fusion.
"""

import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.model_selection import KFold, ParameterGrid
from snf import compute, utils, metrics


def compute_SNF(inputs, *, K=20, mu=1, n_clusters=None,
                t=20, n_perms=1000, normalize=True):
    """
    Runs a full SNF on `inputs` and returns cluster affinity scores and labels

    Parameters
    ----------
    inputs : list-of-tuples
        Each tuple should be comprised of an N x M array_like object, where N
        is samples and M is features, and a distance metric string (e.g.,
        'euclidean', 'sqeuclidean')
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    mu : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5
    n_clusters : int or list-of-int, optional
        Number of clusters to find in combined data. Default: determined by
        eigengap (see `compute.get_n_clusters()`)
    t : int, optional
        Number of iterations to perform information swapping. Default: 20
    n_perms : int, optional
        Number of permutations for calculating z_affinity. Default: 1000
    normalize : bool, optional
        Whether to normalize (zscore) the data before constructing the affinity
        matrix. Each feature is separately normalized. Default: True

    Returns
    -------
    z_affinity : list-of-float
        Z-score of silhouette (affinity) score
    snf_labels : list of (N,) np.ndarray
        Cluster labels for subjects
    """

    # make affinity matrices for input datatypes
    all_aff = [compute.make_affinity(dtype, K=K, mu=mu,
                                     metric=metric,
                                     normalize=normalize)
               for (dtype, metric) in inputs]

    # run SNF, get number of est. clusters, and perform spectral clustering
    snf_aff = compute.SNF(all_aff, K=K, t=t)
    if n_clusters is None:
        n_clusters = [compute.get_n_clusters(snf_aff)[0]]
    elif isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    snf_labels = [spectral_clustering(snf_aff, clust) for clust in n_clusters]
    z_affinity = [metrics.affinity_zscore(snf_aff, label, n_perms=n_perms)
                  for label in snf_labels]

    return z_affinity, snf_labels


def SNF_gridsearch(data, *, mu=None, K=None, n_clusters=None, t=20, folds=3,
                   n_perms=1000, normalize=True, seed=None, verbose=False):
    """
    Performs grid search for SNF hyperparameters `mu`, `K`, and `n_clusters`

    Uses `folds`-fold CV to subsample `data` and performs grid search on `mu`,
    `K`, and `n_clusters` hyperparameters for SNF. There is no testing on the
    left-out sample for each CV fold---it is simply removed.

    Parameters
    ----------
    data : list-of-tuple
        Each tuple should contain (1) an (N x M) data array, where N is samples
        M is features, and (2) a string indicating the metric to use to compute
        a distance matrix for the given data. This MUST be one of the options
        available in ``scipy.spatial.distance.cdist``.
    mu : array_like, optional
        Array of `mu` values to search over. Default: np.arange(0.35, 1.05,
        0.05)
    K : array_like, optional
        Array of `K` values to search over. Default: np.arange(5, N // 2, 5)
    n_clusters : array_like, optional
        Array of cluster numbers to search over. Default: np.arange(2, N // 20)
    t : int, optional
        Number of iterations for SNF. Default: 20
    folds : int, optional
        Number of folds to use for cross-validation. Default: 3
    n_perms : int, optional
        Number of permutations for generating z-score of silhouette (affinity)
        to assess reliability of SNF clustering output. Default: 1000
    normalize : bool, optional
        Whether to normalize (zscore) the data before constructing the affinity
        matrix. Each feature is separately normalized. Default: True
    seed : int, optional
        Random seed. Default: None
    verbose : bool, optional
        Whether to print status updates. Default: False

    Returns
    -------
    grid_zaff : (F,) list of (S x K x C) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The entries in the individual arrays correspond to the
        z-scored silhouette (affinity).
    grid_labels : (F,) list of (S x K x C x N) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The `N` entries along the last dimension correspond to
        the cluster labels for the given parameter combination.
    """

    # get inputs
    if seed is not None:
        np.random.seed(seed)
    n_samples = len(data[0][0])
    if mu is None:
        mu = np.arange(0.35, 1.05, 0.05)
    if K is None:
        K = np.arange(5, n_samples // folds, 5, dtype='int')
    if n_clusters is None:
        n_clusters = np.arange(2, n_samples // 20, dtype='int')

    mu, K, n_clusters = np.asarray(mu), np.asarray(K), np.asarray(n_clusters)

    # empty matrices to hold outputs of SNF
    grid_zaff, grid_labels = [], []

    # iterate through folds
    kf = KFold(n_splits=folds)
    for n_fold, (train_index, _) in enumerate(kf.split(data[0][0])):
        # create parameter generator for current iteration of grid search
        param_grid = ParameterGrid(dict(K=K, mu=mu))
        # subset data arrays
        fold_data = [(np.asarray(d[0])[train_index], d[1]) for d in data]
        # create empty arrays to store outputs
        fold_zaff = np.empty(shape=(K.size, mu.size, n_clusters.size,))
        fold_labels = np.empty(shape=(K.size, mu.size, n_clusters.size,
                                      len(train_index)))
        # generate SNF metrics for all parameter combinations in current fold
        for n, curr_params in enumerate(param_grid):
            inds = np.unravel_index(n, fold_zaff.shape[:2])
            zaff, labels = compute_SNF(fold_data, n_clusters=n_clusters, t=t,
                                       n_perms=n_perms, normalize=normalize,
                                       **curr_params)
            fold_zaff[inds] = zaff
            fold_labels[inds] = labels

        # we want S x K x C[ x N] matrices
        # ParameterGrid iterates through params alphabetically, so transpose
        grid_zaff.append(fold_zaff.transpose(1, 0, 2))
        grid_labels.append(fold_labels.transpose(1, 0, 2, 3))

    return grid_zaff, grid_labels


def get_optimal_params(zaff, labels, neighbors='edges'):
    """
    Finds optimal parameters for SNF based on K-folds grid search

    Parameters
    ----------
    zaff : (F,) list of (S x K x C) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The entries in the individual arrays correspond to the
        z-scored silhouette (affinity).
    labels : (F,) list of (S x K x C x N) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The `N` entries along the last dimension correspond to
        the cluster labels for the given parameter combination.
    neighbors : str, optional
        How many neighbors to consider when calculating z-Rand kernel. Must be
        in ['edges', 'corners']. Default: 'edges'

    Returns
    -------
    mu : int
        Index along S indicating optimal mu parameter
    K : int
        Index along K indicating optimal K parameter
    """

    # extract S x K information from zaff
    shape = zaff[0].shape[:-1] + (-1,)
    # get max indices for z-affinity arrays
    indices = [utils.extract_max_inds(aff) for aff in zaff]
    # get labels corresponding to optimal parameters
    labgrid = [lab[inds] for inds, lab in zip(indices, labels)]
    # convolve zrand with label array to find stable parameter solutions
    zrand = [utils.zrand_convolve(lab, neighbors)[0] for lab in labgrid]
    # get indices of parameters
    mu, K = np.unravel_index(np.mean(zrand, axis=0).argmax(), shape[:-1])

    return mu, K
