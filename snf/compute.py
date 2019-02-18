# -*- coding: utf-8 -*-
"""
Contains the primary functions for conducting similarity network fusion
workflows.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse, stats
from sklearn.utils.validation import (check_array, check_symmetric,
                                      check_consistent_length)


def _flatten(messy):
    """
    Flattens a messy list of mixed iterables / not-iterables

    Parameters
    ----------
    messy : list of ???
        Combined list of iterables / non-iterables

    Yields
    ------
    data : ???
        Entries from `messy`

    Notes
    -----
    Thanks to https://stackoverflow.com/a/2158532 :chef-kissing-fingers-emoji:
    """

    for m in messy:
        if isinstance(m, (list, tuple)):
            yield from _flatten(m)
        else:
            yield m


def _check_data_metric(data, metric):
    """
    Confirms inputs to `make_affinity()` are appropriate

    Parameters
    ----------
    data : (F,) list of (M, N) array_like
        Input data arrays. All arrays should have same first dimension
    metric : str or (F,) list of str
        Input distance metrics. If provided as a list, should be the same
        length as `data`

    Yields
    ------
    data, metric : numpy.ndarray, str
        Tuples of an input data array and the corresponding distance metric
    """

    # make sure all inputs are the same length
    check_consistent_length(*list(_flatten(data)))

    # check provided metric -- if not a list, make it so
    if not isinstance(metric, (list, tuple)):
        metric = [metric] * len(data)

    # expand provided data arrays and metric so that it's 1:1
    for d, m in zip(data, metric):
        # if it's an iterable, recurse down
        if isinstance(d, (list, tuple)):
            yield from _check_data_metric(d, m)
        else:
            yield check_array(d, force_all_finite=False), m


def make_affinity(*data, metric='sqeuclidean', K=20, mu=0.5, normalize=True):
    r"""
    Constructs affinity (i.e., similarity) matrix from `data`

    Performs columnwise normalization on `data`, computes distance matrix based
    on provided `metric`, and then constructs affinity matrix. Uses a scaled
    exponential similarity kernel to determine the weight of each edge based on
    the distance matrix. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

    Parameters
    ----------
    *data : (N, M) array_like
        Raw data array, where `N` is samples and `M` is features. If multiple
        arrays are provided then affinity matrices will be generated for each.
    metric : str or list-of-str, optional
        Distance metric to compute. Must be one of available metrics in
        :py:func`scipy.spatial.distance.pdist`. If multiple arrays a provided
        an equal number of metrics may be supplied. Default: 'sqeuclidean'
    K : (0, N) int, optional
        Number of neighbors to consider when creating affinity matrix. See
        `Notes` of :py:func`snf.compute.affinity_matrix` for more details.
        Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel when constructing
        affinity matrix. See `Notes` of :py:func`snf.compute.affinity_matrix`
        for more details. Default: 0.5
    normalize : bool, optional
        Whether to normalize (i.e., zscore) `arr` before constructing the
        affinity matrix. Each feature (i.e., column) is normalized separately.
        Default: True

    Returns
    -------
    affinity : (N, N) numpy.ndarray or list of numpy.ndarray
        Affinity matrix (or matrices, if multiple inputs provided)

    Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

    Examples
    --------
    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()

    >>> from snf import compute
    >>> aff = compute.make_affinity(simdata.data[0], K=20, mu=0.5)
    >>> aff.shape
    (200, 200)
    """

    affinity = []
    for inp, met in _check_data_metric(data, metric):
        # normalize data, taking into account potentially missing data
        if normalize:
            mask = np.isnan(inp).all(axis=1)
            zarr = np.zeros_like(inp)
            zarr[mask] = np.nan
            zarr[~mask] = np.nan_to_num(stats.zscore(inp[~mask], ddof=1))
        else:
            zarr = inp

        # construct distance matrix using `metric` and make affinity matrix
        distance = cdist(zarr, zarr, metric=met)
        affinity += [affinity_matrix(distance, K=K, mu=mu)]

    # match input type (if only one array provided, return array not list)
    if len(data) == 1 and not isinstance(data[0], list):
        affinity = affinity[0]

    return affinity


def affinity_matrix(dist, *, K=20, mu=0.5):
    r"""
    Calculates affinity matrix given distance matrix `dist`

    Uses a scaled exponential similarity kernel to determine the weight of each
    edge based on `dist`. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

    You'd probably be best to use :py:func`snf.compute.make_affinity` instead
    of this, as that command also handles normalizing the inputs and creating
    the distance matrix.

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix
    K : (0, N) int, optional
        Number of neighbors to consider. Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel. Default: 0.5

    Returns
    -------
    W : (N, N) np.ndarray
        Affinity matrix

    Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

    Examples
    --------
    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()

    We need to construct a distance matrix before we can create a similarity
    matrix using :py:func:`snf.compute.affinity_matrix`:

    >>> from scipy.spatial.distance import cdist
    >>> dist = cdist(simdata.data[0], simdata.data[0])

    >>> from snf import compute
    >>> aff = compute.affinity_matrix(dist)
    >>> aff.shape
    (200, 200)
    """

    # check inputs
    dist = check_array(dist, force_all_finite=False)
    dist = check_symmetric(dist, raise_warning=False)

    # get mask for potential NaN values and set diagonals zero
    mask = np.isnan(dist)
    dist[np.diag_indices_from(dist)] = 0

    # sort array and get average distance to K nearest neighbors
    T = np.sort(dist, axis=1)
    TT = np.vstack(T[:, 1:K + 1].mean(axis=1) + np.spacing(1))

    # compute sigma (see equation in Notes)
    sigma = (TT + TT.T + dist) / 3
    msigma = np.ma.array(sigma, mask=mask)  # mask for NaN
    sigma = sigma * np.ma.greater(msigma, np.spacing(1)).data + np.spacing(1)

    # get probability density function with scale = mu*sigma and symmetrize
    scale = (mu * np.nan_to_num(sigma)) + mask
    W = stats.norm.pdf(np.nan_to_num(dist), loc=0, scale=scale)
    W[mask] = np.nan
    W = check_symmetric(W, raise_warning=False)

    return W


def _find_dominate_set(W, K=20):
    """
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) np.ndarray
        Thresholded version of `W`
    """

    # let's not modify W in place
    Wk = W.copy()

    # determine percentile cutoff that will keep only `K` edges for each sample
    # remove everything below this cutoff
    cutoff = 100 - (100 * (K / len(W)))
    Wk[Wk < np.percentile(Wk, cutoff, axis=1, keepdims=True)] = 0

    # normalize by strength of remaining edges
    Wk = Wk / np.nansum(Wk, axis=1, keepdims=True)

    return Wk


def _B0_normalized(W, alpha=1.0):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF
    alpha : (0, 1) float, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N, N) np.ndarray
        Normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    W = W + (alpha * np.eye(len(W)))
    W = check_symmetric(W, raise_warning=False)

    return W


def snf(*aff, K=20, t=20, alpha=1.0):
    r"""
    Performs Similarity Network Fusion on `aff` matrices

    Parameters
    ----------
    *aff : (N, N) array_like
        Input similarity arrays; all arrays should be square and of equal size.
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20
    t : int, optional
        Number of iterations to perform information swapping. Default: 20
    alpha : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N, N) np.ndarray
        Fused similarity network of input arrays

    Notes
    -----
    In order to fuse the supplied :math:`m` arrays, each must be normalized. A
    traditional normalization on an affinity matrix would suffer from numerical
    instabilities due to the self-similarity along the diagonal; thus, a
    modified normalization is used:

    .. math::

       \mathbf{P}(i,j) =
         \left\{\begin{array}{rr}
           \frac{\mathbf{W}_(i,j)}
                 {2 \sum_{k\neq i}^{} \mathbf{W}_(i,k)} ,& j \neq i \\
                                                       1/2 ,& j = i
         \end{array}\right.

    Under the assumption that local similarities are more important than
    distant ones, a more sparse weight matrix is calculated based on a KNN
    framework:

    .. math::

       \mathbf{S}(i,j) =
         \left\{\begin{array}{rr}
           \frac{\mathbf{W}_(i,j)}
                 {\sum_{k\in N_{i}}^{}\mathbf{W}_(i,k)} ,& j \in N_{i} \\
                                                         0 ,& \text{otherwise}
         \end{array}\right.

    The two weight matrices :math:`\mathbf{P}` and :math:`\mathbf{S}` thus
    provide information about a given patient's similarity to all other
    patients and the `K` most similar patients, respectively.

    These :math:`m` matrices are then iteratively fused. At each iteration, the
    matrices are made more similar to each other via:

    .. math::

       \mathbf{P}^{(v)} = \mathbf{S}^{(v)}
                          \times
                          \frac{\sum_{k\neq v}^{}\mathbf{P}^{(k)}}{m-1}
                          \times
                          (\mathbf{S}^{(v)})^{T},
                          v = 1, 2, ..., m

    After each iteration, the resultant matrices are normalized via the
    equation above. Fusion stops after `t` iterations, or when the matrices
    :math:`\mathbf{P}^{(v)}, v = 1, 2, ..., m` converge.

    The output fused matrix is full rank and can be subjected to clustering and
    classification.
    """

    aff = _check_SNF_inputs(aff)
    Wk = [0] * len(aff)
    Wsum = np.zeros(aff[0].shape)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff) - np.sum([np.isnan(a) for a in aff], axis=0)

    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        mat = mat / np.nansum(mat, axis=1, keepdims=True)
        aff[n] = check_symmetric(mat, raise_warning=False)
        # apply KNN threshold to normalized affinity matrix
        Wk[n] = _find_dominate_set(aff[n], int(K))

    # take sum of all normalized (not thresholded) affinity matrices
    Wsum = np.nansum(aff, axis=0)

    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = np.nan_to_num(Wk[n])
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = nzW @ (Wsum - aw) @ nzW.T / (n_aff - 1)  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = _B0_normalized(aff0, alpha=alpha)

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff, axis=0)

    # all entries of `aff` should be identical after the fusion procedure
    # dividing by len(aff) is hypothetically equivalent to selecting one
    # however, if fusion didn't converge then this is just average of `aff`
    W = Wsum / len(aff)

    # normalize fused matrix and update diagonal similarity
    W = W / np.nansum(W, axis=1, keepdims=True)  # TODO: / by NaN
    W = (W + W.T + np.eye(len(W))) / 2

    return W


def _check_SNF_inputs(aff):
    """
    Confirms inputs to SNF are appropriate

    Parameters
    ----------
    aff : `m`-list of (N x N) array_like
        Input similarity arrays. All arrays should be square and of equal size.
    """

    prep = []
    for a in _flatten(aff):
        ac = check_array(a, force_all_finite=True, copy=True)
        prep.append(check_symmetric(ac, raise_warning=False))
    check_consistent_length(*prep)

    # TODO: actually do this check for missing data
    nanaff = len(prep) - np.sum([np.isnan(a) for a in prep], axis=0)
    if np.any(nanaff == 0):
        pass

    return prep


def _label_prop(W, Y, *, t=1000):
    """
    Label propogation of labels in `Y` via similarity of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array generated by `SNF`
    Y : (N, G) array_like
        Dummy-coded array grouping N subjects in G groups. Some subjects should
        have no group indicated
    t : int, optional
        Number of iterations to perform label propogation. Default: 1000

    Returns
    -------
    Y : (N, G) array_like
        Psuedo-dummy-coded array grouping N subjects into G groups. Subjects
        with no group indicated now have continuous weights indicating
        likelihood of group membership
    """

    W_norm, Y_orig = _dnorm(W, 'ave'), Y.copy()
    train_index = Y.sum(axis=1) == 1

    for iteration in range(t):
        Y = W_norm @ Y
        # retain training labels every iteration
        Y[train_index, :] = Y_orig[train_index, :]

    return Y


def _dnorm(W, norm='ave'):
    """
    Normalizes a symmetric kernel `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array generated by `SNF`
    norm : str, optional
        Type of normalization to perform. Must be one of ['ave', 'gph'].
        Default: 'ave'

    Returns
    -------
    W_norm : (N, N) array_like
        Normalized `W`
    """

    if norm not in ['ave', 'gph']:
        raise ValueError('Provided `norm` {} not in [\'ave\', \'gph\'].'
                         .format(norm))

    D = W.sum(axis=1) + np.spacing(1)

    if norm == 'ave':
        W_norm = sparse.diags(1. / D) @ W
    else:
        D = sparse.diags(1. / np.sqrt(D))
        W_norm = D @ (W @ D)

    return W_norm


def group_predict(train, test, labels, *, K=20, mu=0.4, t=20):
    """
    Propogates `labels` from `train` data to `test` data via SNF

    Parameters
    ----------
    train : `m`-list of (S1, F) array_like
        Input subject x feature training data. Subjects in these data sets
        should have been previously labelled (see: `labels`).
    test : `m`-list of (S2, F) array_like
        Input subject x feature testing data. These should be similar to the
        data in `train` (though the first dimension can differ). Labels will be
        propogated to these subjects.
    labels : (S1,) array_like
        Cluster labels for `S1` subjects in `train` data sets. These could have
        been obtained from some ground-truth labelling or via a previous
        iteration of SNF with only the `train` data (e.g., the output of
        :py:func:`sklearn.cluster.spectral_clustering` would be appropriate).
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        `snf.affinity_matrix` for more details. Default: 20
    mu : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        `snf.affinity_matrix` for more details. Default: 0.5
    t : int, optional
        Number of iterations to perform information swapping during SNF.
        Default: 20

    Returns
    -------
    predicted_labels : (S2,) np.ndarray
        Cluster labels for subjects in `test` assigning to groups in `labels`
    """

    # check inputs are legit
    try:
        check_consistent_length(train, test)
    except ValueError:
        raise ValueError('Training and testing set must have same number of '
                         'data types.')
    if not all([len(labels) == len(t) for t in train]):
        raise ValueError('Training data must have the same number of subjects '
                         'as provided labels.')

    # generate affinity matrices for stacked train/test data sets
    affinities = []
    for (tr, te) in zip(train, test):
        try:
            check_consistent_length(tr.T, te.T)
        except ValueError:
            raise ValueError('Train and test data must have same number of '
                             'features for each data type. Make sure to '
                             'supply data types in the same order.')
        affinities += [make_affinity(np.row_stack([tr, te]), K=K, mu=mu)]

    # fuse with SNF
    fused_aff = snf(*affinities, K=K, t=t)

    # get unique groups in training data and generate array to hold all labels
    groups = np.unique(labels)
    all_labels = np.zeros((len(fused_aff), groups.size))
    # reassign training labels to all_labels array
    for i in range(groups.size):
        all_labels[np.where(labels == groups[i])[0], i] = 1

    # propogate labels from train data to test data using SNF fused array
    propogated_labels = _label_prop(fused_aff, all_labels, t=1000)
    predicted_labels = groups[propogated_labels[len(test[0]):].argmax(axis=1)]

    return predicted_labels


def get_n_clusters(arr, n_clusters=range(2, 6)):
    """
    Finds optimal number of clusters in `arr` via eigengap method

    Parameters
    ----------
    arr : (N, N) array_like
        Input array (e.g., the output of :py:func`snf.compute.snf`)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    opt_cluster : int
        Optimal number of clusters
    second_opt_cluster : int
        Second best number of clusters
    """

    # confirm inputs are appropriate
    n_clusters = check_array(n_clusters, ensure_2d=False)
    n_clusters = n_clusters[n_clusters > 1]

    # don't overwrite provided array!
    graph = arr.copy()

    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di

    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    eigengap = np.abs(np.diff(eigs))
    eigengap = eigengap * (1 - eigs[:-1]) / (1 - eigs[1:])
    n = eigengap[n_clusters - 1].argsort()[::-1]

    return n_clusters[n[:2]]
