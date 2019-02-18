.. _usage:

.. testsetup::

    import numpy as np
    np.random.seed(1234)

User guide
==========

Brief example
-------------

A brief example for those who just want to get started:

.. doctest::

    # load raw data / labels for supplied dataset
    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()
    >>> simdata.keys()
    dict_keys(['data', 'labels'])

    # this dataset has two data arrays representing features from 200 samples
    >>> len(simdata.data)
    2
    >>> len(simdata.labels)
    200

    # convert raw data arrays into sample x sample affinity matrices
    >>> from snf import compute
    >>> affinities = compute.make_affinity(simdata.data, metric='euclidean')

    # fuse the similarity matrices with SNF
    >>> fused = compute.snf(affinities)

    # estimate the number of clusters present in the fused matrix, derived via
    # an "eigengap" method (i.e., largest difference in eigenvalues of the
    # laplacian of the graph). note this function returns the top two options;
    # we'll only use the first
    >>> first, second = compute.get_n_clusters(fused)
    >>> first, second
    (2, 5)

    # apply clustering procedure
    # you can use any clustering method here, but since SNF returns an affinity
    # matrix (i.e., all entries are positively-valued and indicate similarity)
    # spectral clustering makes a lot of sense
    >>> from sklearn import cluster
    >>> fused_labels = cluster.spectral_clustering(fused, n_clusters=first)

    # compute normalized mutual information for clustering solutions
    >>> from snf import metrics
    >>> labels = [simdata.labels, fused_labels]
    >>> for arr in affinities:
    ...     labels += [cluster.spectral_clustering(arr, n_clusters=first)]
    >>> nmi = metrics.nmi(labels)

    # compute silhouette score to assess goodness-of-fit for clustering
    >>> silhouette = metrics.silhouette_score(fused, fused_labels)

In-depth example
----------------

Using SNF is pretty straightforward. There are only a handful of commands that
you'll need, and the output (a subject x subject array) can easily be carried
forward to any number of analysis pipelines.

Nonetheless, for a standard scenario, this package comes bundled with two
datasets provided by the original authors of SNF which can be quite
illustrative.

First, we'll load in the data; data arrays should be (N x M), where `N` is
samples and `M` are features.

.. doctest::

    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()
    >>> simdata.keys()
    dict_keys(['data', 'labels'])

The loaded object ``simdata`` is a dictionary with two keys containing our data
arrays and the corresponding labels:

.. doctest::

    >>> n_dtypes = len(simdata.data)
    >>> n_samp = len(simdata.labels)
    >>> print('Simdata has {} datatypes with {} samples each.'.format(n_dtypes, n_samp))
    Simdata has 2 datatypes with 200 samples each.

Once we have our data arrays loaded we need to create affinity matrices. Unlike
distance matrices, a higher number in an affinity matrix indicates increased
similarity. Thus, the highest numbers should always be along the diagonal,
since subjects are always most similar to themselves!

To construct our affinity matrix, we'll use ``snf.make_affinity``, which
first constructs a distance matrix (using a provided distance metric) and then
converts this into an affinity matrix based on a given subject's similarity to
their ``K`` nearest neighbors. As such, we need to provide a few
hyperparameters: ``K`` and ``mu``. ``K`` determines the number of nearest
neighbors to consider when constructing the affinity matrix; ``mu`` is a
scaling factor that weights the affinity matrix. While the appropriate numbers
for these varies based on scenario, a good rule is that ``K`` should be around
``N // 10``, and ``mu`` should be in the range (0.2 - 0.8).

.. doctest::

    >>> from snf import compute
    >>> affinities = compute.make_affinity(simdata.data, metric='euclidean', K=20, mu=0.5)

Note that we specified ``metric='euclidean'``, specifying that we wanted to use
euclidean distance in the generation of the initial distance array before
constructing the affinity matrix.

Once we have our affinity arrays, we can run them through the SNF algorithm. We
need to carry forward our ``K`` hyperparameter to this algorithm, as well.

.. doctest::

    >>> fused = compute.snf(affinities, K=20)

The array output by SNF is a fused affinity matrix; that is, it represents
data from all the inputs. It's designed to be full rank, and can thus be
subjected to clustering and classification. We'll do the former, now, by
estimating the number of clusters in the data via the "eigengap" method:

.. doctest::

    >>> first, second = compute.get_n_clusters(fused)
    >>> first, second
    (2, 5)

By default, ``compute.get_n_clusters`` returns two values. We'll use the first
for our clustering:

    >>> from sklearn import cluster
    >>> fused_labels = cluster.spectral_clustering(fused, n_clusters=first)

Now we can compare the clustering of our fused matrix to what would happen if
we had used the data from either of the original matrices, individually. To do
this we need to generate cluster labels from the individual affinity matrices:

.. doctest::

    >>> labels = [simdata.labels, fused_labels]
    >>> for arr in affinities:
    ...     labels += [cluster.spectral_clustering(arr, n_clusters=first)]

Then, we can calculate the normalized mutual information score (NMI) between
the labels generated by SNF and the ones we just obtained:

.. doctest::

    >>> from snf import metrics
    >>> nmi = metrics.nmi(labels)
    >>> print(nmi)
    [[1.         1.         0.25266274 0.07818002]
     [1.         1.         0.25266274 0.07818002]
     [0.25266274 0.25266274 1.         0.0355961 ]
     [0.07818002 0.07818002 0.0355961  1.        ]]


The output array is symmetric and the values range from 0 to 1, where 0
indicates no overlap and 1 indicates a perfect correspondence between the two
sets of labels.

The entry in (0, 1) indicates that the fused array generated by SNF has perfect
overlap with the "true" labels from the datasets. The entries in (0, 2) and
(0, 3) indicate the shared information from the individual (unfused) data
arrays (``simdata.data``) with the true labels.

While this example has the true labels to compare against, in unsupervised
clustering we would not have such information. In these instances, the NMI
cannot tell us that the fused array is **superior** to the individual data
arrays. Rather, it can only help distinguish how much data from each of the
individual arrays is contributing to the fused network.

We can also assess how well the clusters are defined using the silhouette
score. These values range from -1 to 1, where -1 indicates a poor clustering
solution and 1 indicates a fantastic solution. We set the diagonal of the
fused network to zero before construction because it was artifically inflated
during the fusion process; thus, this returns a *conservative* estimate of the
cluster goodness-of-fit.

.. doctest::

    >>> import numpy as np
    >>> np.fill_diagonal(fused, 0)
    >>> silhouette = metrics.silhouette_score(fused, fused_labels)
    >>> print(f"Silhouette score for the fused matrix is: {silhouette:.2f}")
    Silhouette score for the fused matrix is: 0.28

This indicates that the clustering solution for the data is not too bad! We
could try playing around with the hyperparameters to see if we can improve our
fit (being careful to do so in a way that won't overfit to the data). It's
worth noting that the silhouette score here is slightly modified to deal with
the fact that we're working with affinity matrices instead of distance
matrices. See the :ref:`API reference <ref_api>` for more information.
