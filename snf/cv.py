# -*- coding: utf-8 -*-
"""
Code for implementing Similarity Network Fusion.

.. testsetup::
    # change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname(os.path.realpath(__file__))
    >>> datadir = os.path.realpath(os.path.join(filepath, 'tests/data/sim'))
    >>> os.chdir(datadir)
"""

from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.extmath import cartesian
from snf import compute


def compute_SNF(inputs, *, K=20, mu=1, n_clusters=None,
                t=20, n_perms=1000, normalize=True):
    """
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

    snf_labels = [compute.spectral_labels(snf_aff, clust)
                  for clust in n_clusters]
    z_affinity = [compute.affinity_zscore(snf_aff, label, n_perms=n_perms)
                  for label in snf_labels]

    return z_affinity, snf_labels


def SNF_gridsearch(data, *, mu=None, K=None, n_clusters=None, t=20, folds=3,
                   n_perms=1000, normalize=True, seed=None, verbose=False):
    """
    Performs grid search for SNF hyperparameters `mu`, `K`, and `n_clusters`

    Uses `folds`-fold CV to subsample `data` and performs grid search on `mu`,
    `K`, and `n_clusters` hyperparameters for SNF. There is no testing on the
    left-out sample for each CV fold---it is simply removed to prevent
    overfitting.

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
        Where `S` is mu, `K` is K, and `C` is n_clusters that were searched and
        `F` is the number of folds for CV. The entries in the individual arrays
        correspond to the z-scored silhouette (affinity).
    grid_labels : (F,) list of (S x K x C x N) np.ndarray
        Where `S` is mu, `K` is K, and `C` is n_clusters that were searched and
        `F` is the number of folds for CV. The `N` entries along the last
        dimension correspond to the cluster labels for the given parameter
        combination.
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

    # parameters
    parameters = dict(K=K, mu=mu)
    # empty matrices to hold outputs of SNF
    grid_zaff, grid_labels = [], []
    # iterate through folds
    kf = KFold(n_splits=folds)
    for n_fold, (train_index, _) in enumerate(kf.split(data[0][0])):
        # create parameter generator for current iteration of grid search
        param_grid = ParameterGrid(parameters)
        # subset data arrays
        fold_data = [(d[0].iloc[train_index], d[1]) for d in data]
        # create empty arrays to store outputs
        fold_zaff = np.empty(shape=(K.size, mu.size, n_clusters.size,))
        fold_labels = np.empty(shape=(K.size, mu.size, n_clusters.size,
                                      len(train_index)))
        # generate SNF metrics for all parameter combinations in current fold
        for n, curr_params in enumerate(param_grid):
            inds = np.unravel_index(n, fold_zaff.shape)
            # if verbose:
            #     clear_output(True)
            #     print(f'\r++ Current fold: {n_fold+1:>4}/{folds}\n'
            #           f' + Iteration: {n+1:>7}/{len(param_grid)}', end='')
            #     sys.stdout.flush()
            zaff, labels = compute_SNF(fold_data, n_clusters=n_clusters, t=t,
                                       n_perms=n_perms, normalize=normalize,
                                       **curr_params)
            fold_zaff[inds] = zaff
            fold_labels[inds] = labels

        # we want S x K x C[ x N] matrices
        # ParameterGrid iterates through params alphabetically, so transpose
        grid_zaff.append(fold_zaff.transpose(1, 0, 2))
        grid_labels.append(fold_labels.transpose(1, 0, 2, 3))

    # if verbose:
    #     print('\t...DONE!')

    return grid_zaff, grid_labels


def edge_neighbors(x, y):
    """
    Returns indices of edge neighbors of `x` and `y`

    Parameters
    ----------
    x, y : int
        Indices of coordinates

    Returns
    -------
    inds : list
        X and y-indices of edge neighbors (includes input coordinates)
    """

    middle = [[0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]
    xinds, yinds = np.meshgrid(np.arange(x-1, x+2), np.arange(y-1, y+2))
    xinds, yinds = xinds[middle], yinds[middle]
    keep = np.logical_and(xinds >= 0, yinds >= 0)

    return xinds[keep], yinds[keep]


def corner_neighbors(x, y):
    """
    Returns indices of corner AND edge neighbors of `x` and `y`

    Parameters
    ----------
    x, y : int
        Indices of coordinates

    Returns
    -------
    inds : list
        x- and y-indices of edge/corner neighbors (includes input coordinates)
    """

    xinds, yinds = np.meshgrid(np.arange(x-1, x+2), np.arange(y-1, y+2))
    xinds, yinds = xinds.flatten(), yinds.flatten()
    keep = np.logical_and(xinds >= 0, yinds >= 0)

    return xinds[keep], yinds[keep]


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

    def dummyvar(i):
        return np.column_stack([i == grp for grp in np.unique(i)]).astype(int)

    # we need 2d arrays for this to work; shape (n,1)
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    n = X.shape[0]

    indx, indy = dummyvar(X), dummyvar(Y)
    Xa, Ya = indx @ indx.T, indy @ indy.T

    M = n * (n - 1) / 2
    M1 = Xa.nonzero()[0].size / 2
    M2 = Ya.nonzero()[0].size / 2

    wab = np.logical_and(Xa, Ya).nonzero()[0].size / 2
    muab = (M1 * M2) / M

    nx = indx.sum(0)
    ny = indy.sum(0)

    mod = n * (n**2 - 3 * n - 2)
    C1 = mod - (8 * (n + 1) * M1) + (4 * np.power(nx, 3).sum())
    C2 = mod - (8 * (n + 1) * M2) + (4 * np.power(ny, 3).sum())

    a = M / 16
    b = ((4 * M1 - 2 * M)**2) * ((4 * M2 - 2 * M)**2) / (256 * (M**2))
    c = C1 * C2 / (16 * n * (n - 1) * (n - 2))
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M)) *
         (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M)) /
         (64 * n * (n - 1) * (n - 2) * (n - 3)))

    sigw2 = a - b + c + d
    # catch any negatives
    if sigw2 < 0:
        return 0
    sigw = np.sqrt(sigw2)
    z_rand = (wab - muab) / sigw

    return z_rand


def zrand_partitions(communities):
    """
    Calculates average and std of z-Rand for all pairs of community assignments

    Iterates through every pair of community assignment vectors in
    `communities` and calculates the z-Rand score to assess their similarity.
    Returns the mean and standard deviation of all z-Rand scores.

    Parameters
    ----------
    communities : array_like
        Community assignments; shape (parcels x repeats)

    Returns
    -------
    zrand_avg : float
        Average z-Rand score over pairs of community assignments
    zrand_std : float
        Standard deviation of z-Rand over pairs of community assignments
    """

    all_zrand = [zrand(f[0][:, None], f[1][:, None]) for f in
                 combinations(communities.T, 2)]
    zrand_avg, zrand_std = np.nanmean(all_zrand), np.nanstd(all_zrand)

    return zrand_avg, zrand_std


def zrand_convolve(labelgrid, neighbors='edges'):
    """
    Calculates the avg and std z-Rand index using small kernel over `grid`

    Kernel is determined by `neighbors`, which can include all entries with
    touching edges (i.e., 4 neighbors) or corners (i.e., 8 neighbors).

    Parameters
    ----------
    grid : (S x K x N) array_like
        Array containing cluster labels for each N samples, where S is mu
        and K is K.
    neighbors : str, optional
        How many neighbors to consider when calculating Z-rand kernel. Must be
        in ['edges', 'corners']. Default: 'edges'

    Returns
    -------
    zrand_avg : (S x K) np.ndarray
        Array containing average of the z-Rand index calculated using provided
        neighbor kernel
    zrand_std : (S x K) np.ndarray
        Array containing standard deviation of the z-Rand index
    """

    inds = cartesian([range(labelgrid.shape[0]), range(labelgrid.shape[1])])
    if neighbors == 'edges':
        get_neighbors = edge_neighbors
    else:
        get_neighbors = corner_neighbors

    zrand = np.empty(shape=labelgrid.shape[:-1] + (2,))
    for x, y in inds:
        ninds = get_neighbors(x, y)
        zrand[x, y] = zrand_partitions(labelgrid[ninds].T)

    return zrand[..., 0], zrand[..., 1]


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
    iind = np.meshgrid(*(range(f) for f in shape))
    if len(iind) > 1:
        iind = [iind[1], iind[0], *iind[2:]]
    imax = grid.argmax(axis=axis)

    if axis == -1:
        return iind + [imax]
    elif axis < -1:
        axis += 1

    iind.insert(axis, imax)
    return iind


def get_optimal_params(zaff, labels, neighbors='edges'):
    """
    Finds optimal parameters for SNF based on K-folds grid search

    Parameters
    ----------
    zaff : (F,) list of (S x K x C) np.ndarray
        Where S is mu, K is K, and C is n_clusters that were searched and
        F is the number of folds for CV. The entries in the individual arrays
        correspond to the z-scored silhouette (affinity).
    labels : (F,) list of (S x K x C x N) np.ndarray
        Where S is mu, K is K, and C is n_clusters that were searched and
        F is the number of folds for CV. The N entries along the third
        dimension correspond to the cluster labels for the given parameter
        combination.
    neighbors : str, optional
        How many neighbors to consider when calculating Z-rand kernel. Must be
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
    labgrid = [lab[inds] for inds, lab in zip(indices, labels)]
    # convolve zrand with label array to find stable parameter solutions
    zrand = [zrand_convolve(lab, neighbors)[0] for lab in labgrid]
    # get indices of parameters
    mu, K = np.unravel_index(np.mean(zrand, axis=0).argmax(), shape[:-1])

    return mu, K
