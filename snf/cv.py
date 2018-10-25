# -*- coding: utf-8 -*-
"""
Code for implementing cross-validation of Similarity Network Fusion.
"""

import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.validation import check_random_state
from snf import compute, utils, metrics


def compute_SNF(*data, metric='sqeuclidean', K=20, mu=1, n_clusters=None,
                t=20, n_perms=1000, normalize=True, seed=None):
    """
    Runs a full SNF on `data` and returns cluster affinity scores and labels

    Parameters
    ----------
    *data : (N, M) array_like
        Raw data arrays, where `N` is samples and `M` is features.
    metric : str or list-of-str, optional
        Distance metrics to compute on `data`. Must be one of available metrics
        in ``scipy.spatial.distance.pdist``. If a list is provided for `data` a
        list of equal length may be supplied here. Default: 'sqeuclidean'
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    mu : (0, 1) float, optional
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

    rs = check_random_state(seed)

    # make affinity matrices for all inputs and run SNF
    all_aff = compute.make_affinity(*data, metric=metric, K=K, mu=mu,
                                    normalize=normalize)
    snf_aff = compute.SNF(*all_aff, K=K, t=t)

    # get estimated number of clusters (if not provided)
    if n_clusters is None:
        n_clusters = [compute.get_n_clusters(snf_aff)[0]]
    elif isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    # perform spectral clustering across all `n_clusters` and get z-affinity
    snf_labels = [spectral_clustering(snf_aff, clust) for clust in n_clusters]
    z_affinity = [metrics.affinity_zscore(snf_aff, label, n_perms, seed=rs)
                  for label in snf_labels]

    return z_affinity, snf_labels


def SNF_gridsearch(*data, metric='sqeuclidean', mu=None, K=None,
                   n_clusters=None, t=20, folds=3, n_perms=1000,
                   normalize=True, seed=None):
    """
    Performs grid search for SNF hyperparameters `mu`, `K`, and `n_clusters`

    Uses `folds`-fold CV to subsample `data` and performs grid search on `mu`,
    `K`, and `n_clusters` hyperparameters for SNF. There is no testing on the
    left-out sample for each CV fold---it is simply removed.

    Parameters
    ----------
    *data : (N, M) array_like
        Raw data arrays, where `N` is samples and `M` is features.
    metric : str or list-of-str, optional
        Distance metrics to compute on `data`. Must be one of available metrics
        in ``scipy.spatial.distance.pdist``. If a list is provided for `data` a
        list of equal length may be supplied here. Default: 'sqeuclidean'
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
        Whether to normalize (z-score) `data` arrrays before constructing
        affinity matrices. Each feature is separately normalized. Default: True
    seed : int, optional
        Random seed. Default: None

    Returns
    -------
    grid_zaff : (F,) list of (S, K, C) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The entries in the individual arrays correspond to the
        z-scored silhouette (affinity).
    grid_labels : (F,) list of (S, K, C, N) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The `N` entries along the last dimension correspond to
        the cluster labels for the given parameter combination.
    """

    # get inputs
    rs = check_random_state(seed)

    # check gridsearch parameter inputs
    n_samples = len(data[0])
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
    for n_fold, (train_index, _) in enumerate(kf.split(data[0])):
        # subset data arrays
        fold_data = [d[train_index] for d in data]

        # create empty arrays to store outputs
        fold_zaff = np.empty(shape=(K.size, mu.size, n_clusters.size,))
        fold_labels = np.empty(shape=(K.size, mu.size, n_clusters.size,
                                      len(train_index)))

        # generate SNF metrics for all parameter combinations in current fold
        for n, curr_params in enumerate(ParameterGrid(dict(K=K, mu=mu))):
            zaff, labels = compute_SNF(*fold_data, metric=metric,
                                       n_clusters=n_clusters, t=t,
                                       n_perms=n_perms, normalize=normalize,
                                       seed=rs, **curr_params)

            # get indices for current parameters and store zaff / labels
            inds = np.unravel_index(n, fold_zaff.shape[:2])
            fold_zaff[inds] = zaff
            fold_labels[inds] = labels

        # we want matrices of shape (mu, K, n_clusters[, folds])
        # ParameterGrid iterates through params alphabetically, so transpose
        grid_zaff.append(fold_zaff.transpose(1, 0, 2))
        grid_labels.append(fold_labels.transpose(1, 0, 2, 3))

    return grid_zaff, grid_labels


def get_optimal_params(zaff, labels, neighbors='edges'):
    """
    Finds optimal parameters for SNF based on K-folds grid search

    Parameters
    ----------
    zaff : (F,) list of (S, K, C) np.ndarray
        Where `S` is mu, `K` is K, `C` is n_clusters, and `F` is the number of
        folds for CV. The entries in the individual arrays correspond to the
        z-scored silhouette (affinity).
    labels : (F,) list of (S, K, C, N) np.ndarray
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
    labgrid = [lab[tuple(inds)] for inds, lab in zip(indices, labels)]
    # convolve zrand with label array to find stable parameter solutions
    zrand = [utils.zrand_convolve(lab, neighbors)[0] for lab in labgrid]
    # get indices of parameters
    mu, K = np.unravel_index(np.mean(zrand, axis=0).argmax(), shape[:-1])

    return mu, K
