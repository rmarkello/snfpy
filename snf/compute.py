# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats
from sklearn.metrics import normalized_mutual_info_score


def make_affinity(arr, K=20, sigma=0.5, metric='sqeuclidean', normalize=True):
    """
    Makes affinity matrix given ``arr``

    Parameters
    ----------
    arr : (N x M) array_like
        Data array where ``N`` is samples and ``M`` is features
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    sigma : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5
    metric : str, optional
        Distance metric to compute. Default: 'sqeuclidean'
    normalize : bool, optional
        Whether to normalize (zscore) the data before constructing the affinity
        matrix. Each feature is separately normalized. Default: True

    Returns
    -------
    affinity : (N x N) np.ndarray
        Affinity matrix
    """

    if normalize:
        arr = scipy.stats.zscore(arr, ddof=1)
    similarity = cdist(arr, arr, metric=metric)
    affinity = affinity_matrix(similarity, K=K, sigma=sigma)

    return affinity


def affinity_matrix(dist, K=20, sigma=0.5):
    """
    Calculates affinity matrix given a distance matrix

    Parameters
    ----------
    dist : (N x N) array_like
        Distance matrix
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    sigma : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5

    Returns
    -------
    (N x N) np.ndarray
        Affinity matrix

    References
    ----------
    .. [1] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Zu, T., Brudno, M.,
       Haibe-Kains, B., & Goldenberg, A. (2014). Similarity Network Fusion: a
       fast and effective method to aggregate multiple data types on a genome
       wide scale. Nature Methods, 11(3), 333-7.
    """

    dist = np.asarray(dist)
    dist = (dist + dist.T) / 2
    dist[np.diag_indices_from(dist)] = 0
    T = np.sort(dist, axis=1)
    TT = np.vstack(T[:, 1:K + 1].mean(axis=1) + np.spacing(1))
    sig = (TT + TT.T + dist) / 3
    sig = (sig * (sig > np.spacing(1))) + np.spacing(1)
    W = scipy.stats.norm.pdf(dist, 0, sigma * sig)
    W = (W + W.T) / 2

    return W


def find_dominate_set(W, K=20):
    """
    Parameters
    ----------
    W : (N x N) array_like
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20

    Returns
    -------
    (N x N) np.ndarray

    References
    ----------
    .. [1] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Zu, T., Brudno, M.,
       Haibe-Kains, B., & Goldenberg, A. (2014). Similarity Network Fusion: a
       fast and effective method to aggregate multiple data types on a genome
       wide scale. Nature Methods, 11(3), 333-7.
    """

    m, n = W.shape
    IW1 = np.flip(W.argsort(axis=1), axis=1)
    newW = np.zeros(W.size)
    I1 = ((IW1[:, :K] * m) + np.vstack(np.arange(n))).flatten(order='F')
    newW[I1] = W.flatten(order='F')[I1]
    newW = newW.reshape(W.shape, order='F')
    newW = newW / newW.sum(axis=1)[:, np.newaxis]

    return newW


def B0_normalized(W, alpha=1.0):
    """
    Normalizes ``W`` so that subjects are always most similar to themselves

    Parameters
    ----------
    W : (N x N) array_like
        Similarity array from SNF
    alpha : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    (N x N) np.ndarray
        Similarity array

    References
    ----------
    .. [1] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Zu, T., Brudno, M.,
       Haibe-Kains, B., & Goldenberg, A. (2014). Similarity Network Fusion: a
       fast and effective method to aggregate multiple data types on a genome
       wide scale. Nature Methods, 11(3), 333-7.
    """

    W = W + (alpha * np.eye(len(W)))
    W = (W + W.T) / 2

    return W


def SNF(aff, K=20, t=20, alpha=1.0):
    """
    Performs Similarity Network Fusion on ``aff`` matrices

    Parameters
    ----------
    aff : list of (N x N) array_like
        Input similarity arrays. All arrays should be square and of equal size.
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    t : int, optional
        Number of iterations to perform information swapping. Default: 20
    alpha : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    (N x N) np.ndarray
        Similarity network of fused input arrays

    References
    ----------
    .. [1] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Zu, T., Brudno, M.,
       Haibe-Kains, B., & Goldenberg, A. (2014). Similarity Network Fusion: a
       fast and effective method to aggregate multiple data types on a genome
       wide scale. Nature Methods, 11(3), 333-7.
    """

    m, n = aff[0].shape
    newW, aff0 = [0] * len(aff), [0] * len(aff)
    Wsum = np.zeros((m, n))

    for i in range(len(aff)):
        aff[i] = aff[i] / np.repeat(aff[i].sum(1)[:, np.newaxis], n, axis=1)
        aff[i] = (aff[i] + aff[i].T) / 2
        newW[i] = find_dominate_set(aff[i], round(K))
        Wsum = Wsum + aff[i]

    for iteration in range(t):
        for i in range(len(aff)):
            aff0[i] = newW[i] @ (Wsum - aff[i]) @ newW[i].T / (len(aff) - 1)
            aff[i] = B0_normalized(aff0[i], alpha)
        Wsum = np.zeros((m, n))
        for i in range(len(aff)):
            Wsum = Wsum + aff[i]

    W = Wsum / len(aff)
    W = W / np.repeat(W.sum(1)[:, np.newaxis], n, axis=1)
    W = (W + W.T + np.eye(n)) / 2

    return W


def snf_nmi(labels):
    """
    Calculates normalized mutual information for all combinations of ``labels``

    Parameters
    ----------
    labels : (N,) list
        List of label arrays

    Returns
    -------
    (N x N) np.ndarray
        NMI score for all combinations of ``labels``
    """

    nmi = np.empty(shape=(len(labels), len(labels)))
    for x, y in np.column_stack(np.triu_indices_from(nmi)):
        nmi[x, y] = normalized_mutual_info_score(labels[x], labels[y])
    nmi = np.triu(nmi) + np.triu(nmi, k=1).T

    return nmi


def get_n_clusters(arr, n_clusters=range(2, 6)):
    """
    Finds optimal number of clusters in ``arr`` via eigengap method

    Parameters
    ----------
    arr : (N x N) array_like
        Input array (output from ``snf.SNF()``)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    int
        Optimal number of clusters
    int
        Second best number of clusters

    References
    ----------
    .. [1] Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Zu, T., Brudno, M.,
       Haibe-Kains, B., & Goldenberg, A. (2014). Similarity Network Fusion: a
       fast and effective method to aggregate multiple data types on a genome
       wide scale. Nature Methods, 11(3), 333-7.
    """

    from sklearn.decomposition import PCA

    eigenvalue = PCA().fit(arr).singular_values_[:-1]
    eigengap = np.abs(np.diff(eigenvalue))
    eigengap = eigengap * (1 - eigenvalue[:-1]) / (1 - eigenvalue[1:])
    n = eigengap[n_clusters].argsort()[::-1]

    return n_clusters[n[0]], n_clusters[n[1]]


def rank_feature_by_nmi(inputs, W, K=20, sigma=0.5, n_clusters=None):
    """
    Calculates NMI of each feature in ``inputs`` with ``W``

    Parameters
    ----------
    inputs : list-of-tuples
        Each tuple should be comprised of an N x M array_like object, where N
        is samples and M is features, and a distance metric string (e.g.,
        'euclidean', 'sqeuclidean')
    W : (N x N) array_like
        Similarity array from SNF
    K : int, optional
        Number of neighbors to compare similarity against. Default: 20
    sigma : (0,1) float, optional
        Hyperparameter normalization factor for scaling. Default: 0.5
    n_clusters : int, optional
        Number of desired clusters. Default: determined by eigengap (see
        ``snf.get_n_clusters()``)

    Returns
    -------
    list of (M,) np.ndarray
        Normalized mutual information scores for each feature of input arrays
    """

    if n_clusters is None:
        n_clusters = get_n_clusters(W)[0]
    snf_labels = spectral_labels(W, n_clusters)
    nmi = [np.empty(shape=(d.shape[-1])) for d, m in inputs]
    for ndtype, (dtype, metric) in enumerate(inputs):
        for nfeature, feature in enumerate(np.asarray(dtype).T):
            aff = make_affinity(np.vstack(feature), K=K, sigma=sigma,
                                metric=metric)
            aff_labels = spectral_labels(aff, n_clusters)
            nmi[ndtype][nfeature] = normalized_mutual_info_score(snf_labels,
                                                                 aff_labels)

    return nmi


def spectral_labels(arr, n_clusters, affinity='precomputed'):
    """
    Performs spectral clustering on ``arr`` and returns assigned cluster labels

    Parameters
    ----------
    arr : {(N x N), (N x M)} array_like
        Array to be clustered. If N x M, must set ``affinity``.
    n_clusters : int
        Number of desired clusters.
    affinity : str, optional
        Affinity metric. If ``arr`` is N x N, must be 'precomputed' (default).
        Otherwise, must be one of ['nearest_neighbors', 'rbf', 'sigmoid',
        'polynomial', 'poly', 'linear', 'cosine'].

    Returns
    -------
    labels : (N,) np.ndarray
        Labels for clustering
    """

    from sklearn.cluster import SpectralClustering

    sc = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    labels = sc.fit_predict(arr)

    return labels


def silhouette_samples(arr, labels):
    """
    Calculates modified silhouette score from affinity matrix

    The Silhouette Coefficient is calculated using the mean intra-cluster
    affinity (``a``) and the mean nearest-cluster affinity (``b``) for each
    sample. The Silhouette Coefficient for a sample is ``(b - a) / max(a,b)``.
    To clarify, ``b`` is the distance between a sample and the nearest cluster
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
    (N,) np.ndarray
        Modified (affinity) silhouette scores for each sample

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    .. [3] `Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
       B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine
       learning in Python. Journal of Machine Learning Research, 12(Oct),
       2825-2830.
       <https://github.com/scikit-learn/>`_

    Note
    ----
    Code is *lightly* modified from the ``sklearn`` implementation. See:
    ``sklearn.metrics.silhouette_samples``
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
    affinity (``a``) and the mean nearest-cluster affinity (``b``) for each
    sample. The Silhouette Coefficient for a sample is ``(b - a) / max(a,b)``.
    To clarify, ``b`` is the distance between a sample and the nearest cluster
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
    float
        Modified (affinity) silhouette score

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    .. [3] `Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
       B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine
       learning in Python. Journal of Machine Learning Research, 12(Oct),
       2825-2830.
       <https://github.com/scikit-learn/>`_

    Note
    ----
    Code is *lightly* modified from the ``sklearn`` implementation. See:
    ``sklearn.metrics.silhouette_score``
    """

    return np.mean(silhouette_samples(arr, labels))


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


def dist2(arr1, arr2):
    """
    Wrapper of ``cdist`` with squared euclidean for ``SNFtool`` compatibility

    Parameters
    ----------
    arr1, arr2 : (N x M) array_like
        Input matrices. Can differ on ``N`` but ``M`` must be the same.

    Returns
    -------
    (N x N) np.ndarray
        Squared euclidean distance matrix
    """

    return cdist(arr1, arr2, metric='sqeuclidean')


def chi_square_distance(arr1, arr2):
    """
    Computes chi-squared distance between ``arr1`` and ``arr2``

    Parameters
    ----------
    arr1, arr2 : (N x M) array_like
        Input matrices. Can differ on ``N`` but ``M`` must be the same.

    Returns
    -------
    (N x N) np.ndarray
        Chi-squared distance matrix
    """

    raise NotImplementedError
