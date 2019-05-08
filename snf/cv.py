# -*- coding: utf-8 -*-
"""
Code for implementing cross-validation of similarity network fusion. Useful for
determining the "optimal" number of clusters in a dataset within a
cross-validated, data-driven framework.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import spectral_clustering
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.extmath import cartesian
from sklearn.utils.validation import check_random_state
from . import compute, metrics

try:
    from numba import njit, prange
    use_numba = True
except ImportError:
    prange = range
    use_numba = False

try:
    from joblib import delayed, Parallel
    use_joblib = True
except ImportError:
    use_joblib = False


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
    snf_aff = compute.snf(*all_aff, K=K, t=t)

    # get estimated number of clusters (if not provided)
    if n_clusters is None:
        n_clusters = [compute.get_n_clusters(snf_aff)[0]]
    elif isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    # perform spectral clustering across all `n_clusters`
    snf_labels = [spectral_clustering(snf_aff, clust, random_state=rs)
                  for clust in n_clusters]

    # get z-affinity as desired
    if n_perms is not None and n_perms > 0:
        z_affinity = [metrics.affinity_zscore(snf_aff, label, n_perms, seed=rs)
                      for label in snf_labels]
        return z_affinity, snf_labels

    return snf_labels


def snf_gridsearch(*data, metric='sqeuclidean', mu=None, K=None,
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
    indices = [extract_max_inds(aff) for aff in zaff]
    # get labels corresponding to optimal parameters
    labgrid = [lab[tuple(inds)] for inds, lab in zip(indices, labels)]
    # convolve zrand with label array to find stable parameter solutions
    zrand = [zrand_convolve(lab, neighbors)[0] for lab in labgrid]
    # get indices of parameters
    mu, K = np.unravel_index(np.mean(zrand, axis=0).argmax(), shape[:-1])

    return mu, K


def get_neighbors(ijk, shape, neighbors='faces'):
    """
    Returns indices of neighbors to `ijk` in array of shape `shape`

    Parameters
    ----------
    ijk : array_like
        Indices of coordinates of interest
    shape : tuple
        Tuple indicating shape of array from which `ijk` is drawn
    neighbors : str, optional
        One of ['faces', 'edges', 'corners']. Default: 'faces'

    Returns
    -------
    inds : tuple of tuples
        Indices of neighbors to `ijk` (includes input coordinates)
    """

    neigh = ['faces', 'edges', 'corners']
    if neighbors not in neigh:
        raise ValueError('Provided neighbors {} not valid. Must be one of {}.'
                         .format(neighbors, neigh))

    ijk = np.asarray(ijk)
    if ijk.ndim != 2:
        ijk = ijk[np.newaxis]
    if ijk.shape[-1] != len(shape):
        raise ValueError('Provided coordinate {} needs to have same '
                         'dimensions as provided shape {}'.format(ijk, shape))

    dist = np.sqrt(neigh.index(neighbors) + 1)
    xyz = cartesian([range(i) for i in shape])
    inds = tuple(map(tuple, xyz[np.ravel(cdist(ijk, xyz) <= dist)].T))

    return inds


def extract_max_inds(grid, axis=-1):
    """
    Returns indices to extract max arguments from `grid` along `axis` dimension

    Parameters
    ----------
    grid : array_like
        Input array
    axis : int, optional
        Which axis to extract maximum arguments along. Default: -1

    Returns
    -------
    inds : list of np.ndarray
        Indices
    """

    # get shape of `grid` without `axis`
    shape = np.delete(np.asarray(grid.shape), axis)

    # get indices to index maximum values along `axis`
    iind = np.meshgrid(*(range(f) for f in shape[::-1]))
    if len(iind) > 1:
        iind = [iind[1], iind[0], *iind[2:]]
    imax = grid.argmax(axis=axis)

    if axis == -1:
        return iind + [imax]
    elif axis < -1:
        axis += 1

    iind.insert(axis, imax)
    return iind


def _dummyvar(labels):
    """
    Generates dummy-coded array from provided community assignment `labels`

    Parameters
    ----------
    labels : (N,) array_like
        Labels assigning `N` samples to `G` groups

    Returns
    -------
    ci : (N, G) numpy.ndarray
        Dummy-coded array where 1 indicates that a sample belongs to a group
    """

    comms = np.unique(labels)

    ci = np.zeros((len(labels), len(comms)))
    for n, grp in enumerate(comms):
        ci[:, n] = labels == grp

    return ci


def zrand(X, Y):
    """
    Calculates the z-Rand index of two community assignments

    Parameters
    ----------
    X, Y : (n, 1) array_like
        Community assignment vectors to compare

    Returns
    -------
    z_rand : float
        Z-rand index

    References
    ----------
    .. [1] `Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A.
       Porter. (2011). Comparing Community Structure to Characteristics in
       Online Collegiate Social Networks. SIAM Review, 53, 526-543
       <https://arxiv.org/abs/0809.0690>`_
    """

    if X.ndim > 1 or Y.ndim > 1:
        if X.shape[-1] > 1 or Y.shape[-1] > 1:
            raise ValueError('X and Y must have only one-dimension each. '
                             'Please check inputs.')

    Xf = X.flatten()
    Yf = Y.flatten()

    n = len(Xf)
    indx, indy = _dummyvar(Xf), _dummyvar(Yf)
    Xa = indx.dot(indx.T)
    Ya = indy.dot(indy.T)

    M = n * (n - 1) / 2
    M1 = Xa.nonzero()[0].size / 2
    M2 = Ya.nonzero()[0].size / 2

    wab = np.logical_and(Xa, Ya).nonzero()[0].size / 2

    mod = n * (n**2 - 3 * n - 2)
    C1 = mod - (8 * (n + 1) * M1) + (4 * np.power(indx.sum(0), 3).sum())
    C2 = mod - (8 * (n + 1) * M2) + (4 * np.power(indy.sum(0), 3).sum())

    a = M / 16
    b = ((4 * M1 - 2 * M)**2) * ((4 * M2 - 2 * M)**2) / (256 * (M**2))
    c = C1 * C2 / (16 * n * (n - 1) * (n - 2))
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M))
         * (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M))
         / (64 * n * (n - 1) * (n - 2) * (n - 3)))

    sigw2 = a - b + c + d
    # catch any negatives
    if sigw2 < 0:
        return 0
    z_rand = (wab - ((M1 * M2) / M)) / np.sqrt(sigw2)

    return z_rand


def zrand_partitions(communities):
    """
    Calculates average and std of z-Rand for all pairs of community assignments

    Iterates through every pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.
    Returns the mean and standard deviation of all z-Rand scores.

    Parameters
    ----------
    communities : (S, R) array_like
        Community assignments for `S` samples over `R` partitions

    Returns
    -------
    zrand_avg : float
        Average z-Rand score over pairs of community assignments
    zrand_std : float
        Standard deviation of z-Rand over pairs of community assignments
    """

    n_partitions = communities.shape[-1]
    all_zrand = np.zeros(int(n_partitions * (n_partitions - 1) / 2))

    for c1 in range(n_partitions):
        for c2 in range(c1 + 1, n_partitions):
            idx = int((c1 * n_partitions) + c2 - ((c1 + 1) * (c1 + 2) // 2))
            all_zrand[idx] = zrand(communities[:, c1], communities[:, c2])

    return np.nanmean(all_zrand), np.nanstd(all_zrand)


if use_numba:
    _dummyvar = njit(_dummyvar)
    zrand = njit(zrand)
    zrand_partitions = njit(zrand_partitions)


def zrand_convolve(labelgrid, neighbors='edges', return_std=False, n_proc=-1):
    """
    Calculates the avg and std z-Rand index using kernel over `labelgrid`

    Kernel is determined by `neighbors`, which can include all entries with
    touching edges (i.e., 4 neighbors) or corners (i.e., 8 neighbors).

    Parameters
    ----------
    grid : (S, K, N) array_like
        Array containing cluster labels for each `N` samples, where `S` is mu
        and `K` is K.
    neighbors : str, optional
        How many neighbors to consider when calculating Z-rand kernel. Must be
        in ['edges', 'corners']. Default: 'edges'
    return_std : bool, optional
        Whether to return `zrand_std` in addition to `zrand_avg`. Default: True

    Returns
    -------
    zrand_avg : (S, K) np.ndarray
        Array containing average of the z-Rand index calculated using provided
        neighbor kernel
    zrand_std : (S, K) np.ndarray
        Array containing standard deviation of the z-Rand index
    """

    def _get_zrand(ijk):
        ninds = get_neighbors(ijk, shape=shape, neighbors=neighbors)
        return zrand_partitions(labelgrid[ninds].T)

    shape = labelgrid.shape[:-1]
    inds = cartesian([range(i) for i in shape])

    if use_joblib:
        _zr = Parallel(n_jobs=n_proc)(delayed(_get_zrand)(ijk) for ijk in inds)
    else:
        _zr = [_get_zrand(ijk) for ijk in inds]

    zr = np.empty(shape=shape + (2,))
    for ijk, z in zip(inds, _zr):
        zr[tuple(ijk)] = z

    if return_std:
        return zr[..., 0], zr[..., 1]

    return zr[..., 0]
