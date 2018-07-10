# -*- coding: utf-8 -*-
"""
Code for clustering in support of Similarity Network Fusion.

.. testsetup::
    # change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname(os.path.realpath(__file__))
    >>> datadir = os.path.realpath(os.path.join(filepath, 'tests/data/sim'))
    >>> os.chdir(datadir)
"""

from multiprocessing import cpu_count
import bct
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from snf import utils


def consensus_modularity(adjacency, *, gamma=1, obj='modularity', repeats=250,
                         null_func=np.mean, n_procs=None):
    """
    Generates consensus-derived community assignments for `adjacency`

    Parameters
    ----------
    adjacency : (N x N) array_like
        Non-negative adjacency matrix to be clustered
    gamma : float, optional
        Weighting parameters used in modularity maximization. Default: 1
    obj : {'modularity', 'potts', 'negative_sym', 'negative_asym'}, optional
        Objective function for modularity maximization. Default: 'modularity'
    repeats : int, optional
        Number of times to repeat community detection. Generated community
        assignments will be combined into a consensus matrix. Default: 250
    null_func : function, optional
        Function that can accept an array and return a single number. Used to
        generated null threshold for consensus community assignment generation.
        Default: numpy.mean
    n_procs : int, optional
        Number of processes to parallelize community assignment algorithm.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    consensus : np.ndarray
        Consensus community assignments
    mod_avg : float
        Average modularity of generated community assignment partitions
    zrand_avg : float
        Average z-Rand of generated community assignment partitions
    zrand_std : float
        Standard deviation z-Rand of generated community assignment partitions

    References
    ----------
    .. [1] Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T.,
       Carlson, J. M., & Mucha, P. J. (2013). Robust detection of dynamic
       community structure in networks. Chaos: An Interdisciplinary Journal of
       Nonlinear Science, 23(1), 013142.
    """

    if n_procs is None:
        n_procs = cpu_count()

    # generate community partitions `repeat` times
    partitions = Parallel(n_jobs=n_procs)(
        delayed(bct.community_louvain)(adjacency, gamma=gamma, B=obj)
        for i in range(repeats)
    )

    # get community labels / modularity estimates and generate null labels
    comm_true, modularity = zip(*partitions)
    comm_true = np.column_stack(comm_true)
    comm_null = np.random.permutation(comm_true)

    # get agreement matrices and generate consensus clustering assignments
    agree_true = bct.clustering.agreement(comm_true) / repeats
    agree_null = bct.clustering.agreement(comm_null) / repeats
    consensus = bct.clustering.consensus_und(agree_true,
                                             tau=null_func(agree_null),
                                             reps=100)

    # get zrand statistics for partition similarity
    zrand_avg, zrand_std = utils.zrand_partitions(comm_true, n_procs=n_procs)

    return consensus, np.mean(modularity), zrand_avg, zrand_std
