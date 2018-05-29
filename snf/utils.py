# -*- coding: utf-8 -*-
"""
Utilities for implementing Similarity Network Fusion.

.. testsetup::
    # change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname(os.path.realpath(__file__))
    >>> datadir = os.path.realpath(os.path.join(filepath, 'tests/data/sim'))
    >>> os.chdir(datadir)
"""

from itertools import combinations
import numpy as np
from sklearn.utils.extmath import cartesian


def get_neighbors(x, y, neighbors='edges', shape=None):
    """
    Returns indices of corner AND edge neighbors of `x` and `y`

    Parameters
    ----------
    x, y : int
        Indices of coordinates
    neighbors : str, optional
        One of ['edges', 'corners']. Default: 'edges'
    shape : tuple, optional
        Tuple of array that neighbors will be indexing. Providing will ensure
        outputs do not induce out-of-bounds errors. Default: None

    Returns
    -------
    inds : list
        x- and y-indices of edge/corner neighbors (includes input coordinates)
    """

    # make mesh grid
    xinds, yinds = np.meshgrid(np.arange(x - 1, x + 2),
                               np.arange(y - 1, y + 2))

    # extract neighbors, as appropriate
    if neighbors == 'edges':
        middle = [[0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]
        xinds, yinds = xinds[middle], yinds[middle]
    elif neighbors == 'corners':
        xinds, yinds = xinds.flatten(), yinds.flatten()
    else:
        raise ValueError('Provided neighbors value "{}" not in '
                         '[\'edges\', \'corners\']'.format(neighbors))

    # ensure we won't have any out-of-bounds errors (<0, >shape if provided)
    keep = np.logical_and(xinds >= 0, yinds >= 0)
    if shape is not None:
        keep = np.logical_and(keep, np.logical_and(xinds < shape[0],
                                                   yinds < shape[1]))

    return xinds[keep], yinds[keep]


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
    if X.shape[0] < X.shape[1]:
        X = X.T
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    n = X.shape[0]

    indx, indy = dummyvar(X), dummyvar(Y)
    Xa, Ya = indx @ indx.T, indy @ indy.T

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
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M)) *
         (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M)) /
         (64 * n * (n - 1) * (n - 2) * (n - 3)))

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
    communities : (S x R) array_like
        Community assignments for `S` samples over `R` partitions

    Returns
    -------
    zrand_avg : float
        Average z-Rand score over pairs of community assignments
    zrand_std : float
        Standard deviation of z-Rand over pairs of community assignments
    """

    # TODO: parallelize this
    all_zrand = [zrand(f[0][:, None], f[1][:, None]) for f in
                 combinations(communities.T, 2)]
    zrand_avg, zrand_std = np.nanmean(all_zrand), np.nanstd(all_zrand)

    return zrand_avg, zrand_std


def zrand_convolve(labelgrid, neighbors='edges'):
    """
    Calculates the avg and std z-Rand index using kernel over `labelgrid`

    Kernel is determined by `neighbors`, which can include all entries with
    touching edges (i.e., 4 neighbors) or corners (i.e., 8 neighbors).

    Parameters
    ----------
    grid : (S x K x N) array_like
        Array containing cluster labels for each `N` samples, where `S` is mu
        and `K` is K.
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
    zrand = np.empty(shape=labelgrid.shape[:-1] + (2,))
    for x, y in inds:
        ninds = get_neighbors(x, y, neighbors=neighbors, shape=labelgrid.shape)
        zrand[x, y] = zrand_partitions(labelgrid[ninds].T)

    return zrand[..., 0], zrand[..., 1]
