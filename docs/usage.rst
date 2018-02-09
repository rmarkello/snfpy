.. _usage_ref:

Usage
=====
This package comes bundled with two datasets provided by the original authors
of SNF, which can serve as a wonderful illustrative example. ::

    import numpy as np
    import snf

    # load in the data
    data = [np.loadtxt(fname) for fname in ['data1.csv', 'data2.csv']]

    # set our parameters
    #   K determines the number of nearest neighbors to consider
    #   mu will determine the scaling of the affinity matrices
    params = dict(K=20, mu=0.5)
    aff = [snf.make_affinity(arr, metric='euclidean', **params) for arr in data]

    # feed the data into the SNF algorithm
    fused_aff = snf.SNF(aff, **params)

    # determine the optimal number of clusters in the data via eigengap
    # by default `snf.get_n_clusters` returns the top TWO ideal cluster numbers
    # we'll use the first
    n_clusters = snf.get_n_clusters(fused_aff)[0]

    # cluster the data and return the labels
    labels = snf.spectral_labels(fused_aff, n_clusters=n_clusters)

    
