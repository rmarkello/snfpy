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

import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score


def make_affinity(arr, *, K=20, mu=0.5, metric='sqeuclidean', normalize=True):
    """
    Constructs affinity (i.e., similarity) matrix given feature matrix `arr`

    Performs columnwise normalization on `arr`, computes distance matrix based
    on provided `metric`, and then constructs affinity matrix by calling
    `affinity_matrix()`

    Parameters
    ----------
    arr : (N x M) array_like
        Raw data array, where `N` is samples and `M` is features
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        ``snf.compute.affinity_matrix`` for more details. Default: 20
    mu : (0,1) float, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        ``snf.compute.affinity_matrix`` for more details. Default: 0.5
    metric : str, optional
        Distance metric to compute. Must be one of available metrics in
        ``scipy.spatial.distance.cdist``. Default: 'sqeuclidean'
    normalize : bool, optional
        Whether to normalize (i.e., zscore) `arr` before constructing the
        affinity matrix. Each feature (i.e., column) is normalized separately.
        Default: True

    Returns
    -------
    affinity : (N x N) np.ndarray
        Affinity matrix

    Examples
    --------
    >>> data = np.loadtxt('data1.csv')
    >>> data.shape
    (200, 2)
    >>> aff = make_affinity(data, K=20, mu=0.5, metric='sqeuclidean')
    >>> aff.shape
    (200, 200)
    """

    # normalize data using ddof=1 for stdev calculation and convert NaNs
    if normalize:
        arr = np.nan_to_num(scipy.stats.zscore(arr, ddof=1))

    # construct distance matrix using `metric` and make affinity matrix
    distance = cdist(arr, arr, metric=metric)
    affinity = affinity_matrix(distance, K=K, mu=mu)

    return affinity


def affinity_matrix(dist, *, K=20, mu=0.5):
    """
    Calculates affinity matrix given distance matrix `dist`

    Uses a scaled exponential similarity kernel to determine the weight of each
    edge based on `dist`. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

    You'd probably be best to use ``snf.make_affinity`` instead of this, as
    that command also handles normalizing the inputs and creating the distance
    matrix.

    Parameters
    ----------
    dist : (N x N) array_like
        Distance matrix
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20
    mu : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5

    Returns
    -------
    W : (N x N) np.ndarray
        Affinity matrix

    Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \\mathbf{W}(i, j) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}
                           \\ exp^{-\\frac{\\rho^2(x_{i},x_{j})}{2\\sigma^2}}

    where :math:`\\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \\sigma = \\mu\\ \\frac{\\overline{\\rho}(x_{i},N_{i}) +
                               \\overline{\\rho}(x_{j},N_{j}) +
                               \\rho(x_{i},x_{j})}
                              {3}

    where :math:`\\overline{\\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\\mu\\in(0, 1)\\subset\\mathbb{R}`.

    Examples
    --------
    >>> data = np.loadtxt('data1.csv')
    >>> dist = cdist(data, data, metric='euclidean')
    >>> dist.shape
    (200, 200)
    >>> aff = affinity_matrix(dist)
    >>> aff.shape
    (200, 200)
    """

    # if distance matrix is directed, symmetrize based on average of weights
    dist = np.array(dist)
    dist = (dist + dist.T) / 2
    dist[np.diag_indices_from(dist)] = 0

    # sort array and get average distance to K nearest neighbors
    T = np.sort(dist, axis=1)
    TT = np.vstack(T[:, 1:K + 1].mean(axis=1) + np.spacing(1))

    # compute sigma (see equation in Notes)
    sigma = (TT + TT.T + dist) / 3
    sigma = (sigma * (sigma > np.spacing(1))) + np.spacing(1)

    # get probability density function with scale mu*sigma and symmetrize
    W = scipy.stats.norm.pdf(dist, loc=0, scale=mu*sigma)
    W = (W + W.T) / 2

    return W


def _find_dominate_set(W, K=20):
    """
    Parameters
    ----------
    W : (N x N) array_like
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20

    Returns
    -------
    newW : (N x N) np.ndarray
    """

    m, n = W.shape
    IW1 = np.flip(W.argsort(axis=1), axis=1)
    newW = np.zeros(W.size)
    I1 = ((IW1[:, :K] * m) + np.vstack(np.arange(n))).flatten(order='F')
    newW[I1] = W.flatten(order='F')[I1]
    newW = newW.reshape(W.shape, order='F')
    newW = newW / newW.sum(axis=1)[:, np.newaxis]

    return newW


def _B0_normalized(W, alpha=1.0):
    """
    Normalizes `W` so that subjects are always most similar to themselves

    Parameters
    ----------
    W : (N x N) array_like
        Similarity array from SNF
    alpha : (0,1) float, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N x N) np.ndarray
        "Normalized" similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    W = W + (alpha * np.eye(len(W)))
    W = (W + W.T) / 2

    return W


def SNF(aff, *, K=20, t=20, alpha=1.0):
    """
    Performs Similarity Network Fusion on `aff` matrices

    Parameters
    ----------
    aff : `m`-list of (N x N) array_like
        Input similarity arrays. All arrays should be square and of equal size.
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20
    t : int, optional
        Number of iterations to perform information swapping. Default: 20
    alpha : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N x N) np.ndarray
        Fused similarity network of input arrays

    Notes
    -----
    In order to fuse the supplied :math:`m` arrays, each must be normalized. A
    traditional normalization on an affinity matrix would suffer from numerical
    instabilities due to the self-similarity along the diagonal; thus, a
    modified normalization is used:

    .. math::

       \\mathbf{P}(i,j) =
         \\left\{\\begin{array}{rr}
           \\frac{\\mathbf{W}_(i,j)}
                 {2 \\sum_{k\\neq i}^{} \\mathbf{W}_(i,k)} ,& j \\neq i \\\\
                                                       1/2 ,& j = i
         \\end{array}\\right.

    Under the assumption that local similarities are more important than
    distant ones, we also calculate a more sparse weight matrix based on a KNN
    framework:

    .. math::

       \\mathbf{S}(i,j) =
         \\left\{\\begin{array}{rr}
           \\frac{\\mathbf{W}_(i,j)}
                 {\\sum_{k\\in N_{i}}^{}\\mathbf{W}_(i,k)} ,& j \\in N_{i} \\\\
                                                         0 ,& \\text{otherwise}
         \\end{array}\\right.

    The two weight matrices :math:`\\mathbf{P}` and :math:`\\mathbf{S}` thus
    provide information about a given patient's similarity to all other
    patients and the `K` most similar patients, respectively.

    These :math:`m` matrices are then iteratively fused. At each iteration, the
    matrices are made more similar to each other via:

    .. math::

       \\mathbf{P}^{(v)} = \\mathbf{S}^{(v)}
                           \\times
                           \\frac{\\sum_{k\\neq v}^{}\\mathbf{P}^{(k)}}{m-1}
                           \\times
                           (\\mathbf{S}^{(v)})^{T},
                           v = 1, 2, ..., m

    After each iteration, the resultant matrices are normalized via the
    normalization equation above. Fusion stops after `t` iterations, or when
    the matrices :math:`\\mathbf{P}^{(v)}, v = 1, 2, ..., m` converge.

    The output fused matrix is full rank and can be subjected to clustering and
    classification.
    """

    m, n = aff[0].shape
    newW, aff0 = [0] * len(aff), [0] * len(aff)
    Wsum = np.zeros((m, n))

    for i in range(len(aff)):
        aff[i] = aff[i] / aff[i].sum(axis=1)[:, np.newaxis]
        aff[i] = (aff[i] + aff[i].T) / 2
        newW[i] = _find_dominate_set(aff[i], round(K))
    Wsum = np.sum(aff, axis=0)

    for iteration in range(t):
        for i in range(len(aff)):
            aff0[i] = newW[i] @ (Wsum - aff[i]) @ newW[i].T / (len(aff) - 1)
            aff[i] = _B0_normalized(aff0[i], alpha=alpha)
        Wsum = np.sum(aff, axis=0)

    W = Wsum / len(aff)
    W = W / W.sum(axis=1)[:, np.newaxis]
    W = (W + W.T + np.eye(n)) / 2

    return W


def _label_prop(W, Y, *, t=1000):
    """
    Label propogation of labels in `Y` via similarity of `W`

    Parameters
    ----------
    W : (N x N) array_like
        Similarity array generated by `SNF`
    Y : (N x G) array_like
        Dummy-coded array grouping N subjects in G groups. Some subjects should
        have no group indicated
    t : int, optional
        Number of iterations to perform label propogation. Default: 1000

    Returns
    -------
    Y : (N x G) array_like
        Psuedo-dummy-coded array grouping N subjects into G groups. Subjects
        with no group indicated now have continuous weights indicating
        likelihood of group membership
    """

    W_norm, Y_orig = _dnorm(W, 'ave'), Y.copy()
    train_index = np.sum(Y, axis=1) == 1

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
    W : (N x N) array_like
        Similarity array generated by `SNF`
    norm : str, optional
        Type of normalization to perform. Must be one of ['ave', 'gph'].
        Default: 'ave'

    Returns
    -------
    W_norm : (N x N) array_like
        Normalized `W`
    """

    from scipy.sparse import diags

    if norm not in ['ave', 'gph']:
        raise ValueError('`norm` must be in [\'ave\', \'gph\']. Provided '
                         'value is: {}'.format(norm))

    D = W.sum(axis=1) + np.spacing(1)

    if norm == 'ave':
        W_norm = diags(1. / D) @ W
    else:
        D = diags(1. / np.sqrt(D))
        W_norm = D @ (W @ D)

    return W_norm


def group_predict(train, test, labels, *, K=20, mu=0.4, t=20):
    """
    Propogates `labels` from `train` data to `test` data via SNF

    Parameters
    ----------
    train : `m`-list of (S1 x F) array_like
        Input subject x feature training data. Subjects in these data sets
        should have been previously labelled (see: `labels`).
    test : `m`-list of (S2 x F) array_like
        Input subject x feature testing data. These should be similar to the
        data in `train` (though the first dimension can differ). Labels will be
        propogated to these subjects.
    labels : (S1,) array_like
        Cluster labels for `S1` subjects in `train` data sets. These could have
        been obtained from some ground-truth labelling or via a previous
        iteration of SNF with only the `train` data (e.g., the output of
        ``snf.compute.spectral_labels`` would be appropriate here).
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        `snf.compute.affinity_matrix` for more details. Default: 20
    mu : (0,1) float, optional
        Hyperparameter normalization factor for scaling. See `Notes` of
        `snf.compute.affinity_matrix` for more details. Default: 0.5
    t : int, optional
        Number of iterations to perform information swapping during SNF.
        Default: 20

    Returns
    -------
    predicted_labels : (S2,) np.ndarray
        Cluster labels for subjects in `test` assigning to groups in `labels`
    """

    # check inputs are legit
    if len(train) != len(test):
        raise ValueError('Training and testing set must have same number of '
                         'data types.')
    if not all([len(labels) == len(t) for t in train]):
        raise ValueError('Training data must have the same number of subjects '
                         'as provided labels.')

    # generate affinity matrices for stacked train/test data sets
    affinities = []
    for (tr, te) in zip(train, test):
        if len(tr.T) != len(te.T):
            raise ValueError('Train and test data must have same number of '
                             'features for each data type. Make sure to '
                             'supply data types in the same order.')
        affinities += [make_affinity(np.row_stack([tr, te]), K=K, mu=mu)]

    # fuse with SNF
    fused_aff = SNF(affinities, K=K, t=t)

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


def nmi(labels):
    """
    Calculates normalized mutual information for all combinations of `labels`

    Uses the ``sklearn.metrics.normalized_mutual_info_score`` for calculation;
    refer to that codebase for information on algorithm.

    Parameters
    ----------
    labels : m-length list of (N,) array_like
        List of label arrays

    Returns
    -------
    nmi : (m x m) np.ndarray
        NMI score for all combinations of `labels`

    Examples
    --------
    >>> label1 = np.array([1,1,1,2,2,2])
    >>> label2 = np.array([1,1,2,2,2,2])
    >>> nmi([label1, label2])
    array([[1.        , 0.47913877],
           [0.47913877, 1.        ]])
    """

    # create empty array for output
    nmi = np.empty(shape=(len(labels), len(labels)))
    # get indices for all combinations of labels and calculate NMI
    for x, y in np.column_stack(np.triu_indices_from(nmi)):
        nmi[x, y] = normalized_mutual_info_score(labels[x], labels[y])
    # make output symmetric
    nmi = np.triu(nmi) + np.triu(nmi, k=1).T

    return nmi


def get_n_clusters(arr, n_clusters=range(2, 6)):
    """
    Finds optimal number of clusters in `arr` via eigengap method

    Parameters
    ----------
    arr : (N x N) array_like
        Input array (output from `snf.SNF()`)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    opt_cluster : int
        Optimal number of clusters
    second_opt_cluster : int
        Second best number of clusters

    Examples
    --------
    >>> np.random.seed(1234)
    >>> data = np.random.rand(100, 100)
    >>> get_n_clusters(data)
    (2, 4)
    """

    from sklearn.decomposition import PCA

    n_clusters = np.asarray(n_clusters)
    eigenvalue = PCA().fit(arr).singular_values_[:-1]
    eigengap = np.abs(np.diff(eigenvalue))
    eigengap = eigengap * (1 - eigenvalue[:-1]) / (1 - eigenvalue[1:])
    n = eigengap[n_clusters-1].argsort()[::-1]

    return n_clusters[n[0]], n_clusters[n[1]]


def rank_feature_by_nmi(inputs, W, *, K=20, mu=0.5, n_clusters=None):
    """
    Calculates NMI of each feature in `inputs` with `W`

    Parameters
    ----------
    inputs : list-of-tuple
        Each tuple should contain (1) an (N x M) data array, where N is samples
        M is features, and (2) a string indicating the metric to use to compute
        a distance matrix for the given data. This MUST be one of the options
        available in ``scipy.spatial.distance.cdist``.
    W : (N x N) array_like
        Similarity array generated by `SNF`
    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20
    mu : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5
    n_clusters : int, optional
        Number of desired clusters. Default: determined by eigengap (see
        `snf.get_n_clusters()`)

    Returns
    -------
    nmi : list of (M,) np.ndarray
        Normalized mutual information scores for each feature of input arrays
    """

    if n_clusters is None:
        n_clusters = get_n_clusters(W)[0]
    snf_labels = spectral_labels(W, n_clusters)
    nmi = [np.empty(shape=(d.shape[-1])) for d, m in inputs]
    for ndtype, (dtype, metric) in enumerate(inputs):
        for nfeature, feature in enumerate(np.asarray(dtype).T):
            aff = make_affinity(np.vstack(feature), K=K, mu=mu,
                                metric=metric)
            aff_labels = spectral_labels(aff, n_clusters)
            nmi[ndtype][nfeature] = normalized_mutual_info_score(snf_labels,
                                                                 aff_labels)

    return nmi


def spectral_labels(arr, n_clusters=3, affinity='precomputed'):
    """
    Performs spectral clustering on `arr` and returns assigned cluster labels

    Parameters
    ----------
    arr : {(N x N), (N x M)} array_like
        Array to be clustered. Default assumes that the array is an affinity
        matrix (N x N), where N is samples. If supplying an (N x M) samples by
        features array, you must also set `affinity`.
    n_clusters : int, optional
        Number of clusters in `arr`. Default: 3
    affinity : str, optional
        Affinity metric. If `arr` is (N x N), must be 'precomputed' (default).
        Otherwise, must be one of ['nearest_neighbors', 'rbf', 'sigmoid',
        'polynomial', 'poly', 'linear', 'cosine'].

    Returns
    -------
    labels : (N,) np.ndarray
        Cluster labels
    """

    sc = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    labels = sc.fit_predict(arr)

    return labels


def _silhouette_samples(arr, labels):
    """
    Calculates modified silhouette score from affinity matrix

    The Silhouette Coefficient is calculated using the mean intra-cluster
    affinity (`a`) and the mean nearest-cluster affinity (`b`) for each
    sample. The Silhouette Coefficient for a sample is `(b - a) / max(a,b)`.
    To clarify, `b` is the distance between a sample and the nearest cluster
    that the sample is not a part of. This corresponds to the cluster with the
    next *highest* affinity (opposite how this metric would be computed for a
    distance matrix).

    Parameters
    ----------
    arr : (N x N) array_like
        Array of pairwise affinities between samples
    labels : (N,) array_like
        Predicted labels for each sample

    Returns
    -------
    sil_samples : (N,) np.ndarray
        Modified (affinity) silhouette scores for each sample

    Notes
    -----
    Code is *lightly* modified from the `sklearn` implementation. See:
    `sklearn.metrics.silhouette_samples`

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis. Computational
       and Applied Mathematics, 20, 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    .. [3] `Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
       B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine
       learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
       <https://github.com/scikit-learn/>`_
    """

    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import check_X_y

    def check_number_of_labels(n_labels, n_samples):
        if not 1 < n_labels < n_samples:
            raise ValueError("Number of labels is %d. Valid values are 2 "
                             "to n_samples - 1 (inclusive)" % n_labels)

    arr, labels = check_X_y(arr, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), arr.shape[0])

    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    # For sample i, store the mean distance of the cluster to which
    # it belongs in intra_clust_dists[i]
    intra_clust_aff = np.zeros(arr.shape[0], dtype=arr.dtype)

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_aff = intra_clust_aff.copy()

    for curr_label in range(len(unique_labels)):

        # Find inter_clust_dist for all samples belonging to the same
        # label.
        mask = labels == curr_label
        current_distances = arr[mask]

        # Leave out current sample.
        n_samples_curr_lab = n_samples_per_label[curr_label] - 1
        if n_samples_curr_lab != 0:
            intra_clust_aff[mask] = np.sum(
                current_distances[:, mask], axis=1) / n_samples_curr_lab

        # Now iterate over all other labels, finding the mean
        # cluster distance that is closest to every sample.
        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = np.mean(
                    current_distances[:, other_mask], axis=1)
                inter_clust_aff[mask] = np.maximum(
                    inter_clust_aff[mask], other_distances)

    sil_samples = intra_clust_aff - inter_clust_aff
    sil_samples /= np.maximum(intra_clust_aff, inter_clust_aff)

    # score 0 for clusters of size 1, according to the paper
    sil_samples[n_samples_per_label.take(labels) == 1] = 0

    return sil_samples


def silhouette_score(arr, labels):
    """
    Calculates modified silhouette score from affinity matrix

    The Silhouette Coefficient is calculated using the mean intra-cluster
    affinity (`a`) and the mean nearest-cluster affinity (`b`) for each
    sample. The Silhouette Coefficient for a sample is `(b - a) / max(a,b)`.
    To clarify, `b` is the distance between a sample and the nearest cluster
    that the sample is not a part of. This corresponds to the cluster with the
    next *highest* affinity (opposite how this metric would be computed for a
    distance matrix).

    Parameters
    ----------
    arr : (N x N) array_like
        Array of pairwise affinities between samples
    labels : (N,) array_like
        Predicted labels for each sample

    Returns
    -------
    silhouette_score : float
        Modified (affinity) silhouette score

    Notes
    -----
    Code is *lightly* modified from the ``sklearn`` implementation. See:
    `sklearn.metrics.silhouette_score`
    """

    return np.mean(_silhouette_samples(arr, labels))


def affinity_zscore(arr, labels, n_perms=1000, seed=None):
    """
    Calculates z-score of silhouette (affinity) score by permutation

    Parameters
    ----------
    arr : (N x N) array_like
        Array of pairwise affinities between samples
    labels : (N,) array_like
        Predicted labels for each sample
    n_perms : int, optional
        Number of permutations. Default: 1000
    seed : int, optional
        Random seed. Default: None

    Returns
    -------
    z_aff : float
        Z-score of silhouette (affinity) score
    """

    if seed is not None:
        np.random.seed(seed)
    dist = np.empty(shape=(n_perms,))
    for perm in range(n_perms):
        new_labels = np.random.permutation(labels)
        dist[perm] = silhouette_score(arr, new_labels)
    true_aff_score = silhouette_score(arr, labels)
    z_aff = (true_aff_score - dist.mean()) / dist.std()

    return z_aff


def dist2(arr1, arr2=None):
    """
    Wrapper of `cdist` with squared euclidean as distance metric

    Available for `SNFtool` compatibility; you should probably just use
    `make_affinity()` instead.

    Parameters
    ----------
    arr1, arr2 : (N x M) array_like
        Input matrices. Can differ on N, but M *must* be the same.

    Returns
    -------
    dist : (N x N) np.ndarray
        Squared euclidean distance matrix
    """

    if arr2 is None:
        arr2 = np.array(arr1).copy()
    dist = cdist(arr1, arr2, metric='sqeuclidean')

    return dist


def chi_square_distance(arr1, arr2):
    """
    Computes chi-squared distance between `arr1` and `arr2`

    Parameters
    ----------
    arr1, arr2 : (N x M) array_like
        Input matrices. Can differ on `N` but `M` must be the same.

    Returns
    -------
    chisqd : (N x N) np.ndarray
        Chi-squared distance matrix
    """

    raise NotImplementedError
