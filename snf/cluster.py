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
from scipy.stats import zscore
from sklearn.externals.joblib import Parallel, delayed
from snf import utils


def consensus_modularity(adjacency, *, gamma=1, obj='modularity', repeats=250,
                         null_func=np.mean, n_procs=1):
    """
    Generates consensus-derived community assignments for `adjacency`

    Parameters
    ----------
    adjacency : (N x N) array_like
        Non-negative adjacency matrix
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
        Default: np.mean
    n_procs : int, optional
        Number of processes to parallelize community assignment algorithm.
        Default: 1

    Returns
    -------
    consensus : np.ndarray
        Consensus community assignments
    mod_avg : float
        Average modularity of generated community assignments
    zrand_avg : float
        Average z-Rand of generated community assignments
    zrand_std : float
        Standard deviation of z-Rand of generated community assignments

    References
    ----------
    .. [1] Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T.,
       Carlson, J. M., & Mucha, P. J. (2013). Robust detection of dynamic
       community structure in networks. Chaos: An Interdisciplinary Journal of
       Nonlinear Science, 23(1), 013142.
    """

    # generate community partitions `repeat` times
    partitions = Parallel(n_jobs=n_procs)(
        delayed(bct.community_louvain)(adjacency, gamma=gamma, B=obj)
        for i in range(repeats)
    )

    # get community labels / modularity estimates and generate null labels
    communities, modularity = zip(*partitions)
    comm_true = np.column_stack(communities)
    comm_null = np.random.permutation(comm_true)

    # get agreement matrices and generate consensus clustering assignments
    agree_true = bct.clustering.agreement(comm_true) / repeats
    agree_null = bct.clustering.agreement(comm_null) / repeats
    consensus = bct.clustering.consensus_und(agree_true,
                                             tau=null_func(agree_null),
                                             reps=100)

    # get zrand statistics for partition similarity
    zrand_avg, zrand_std = utils.zrand_partitions(comm_true)

    return consensus, np.mean(modularity), zrand_avg, zrand_std


def gamma_search(adjacency, *, gamma=None, obj='modularity', repeats=250,
                 null_func=np.mean, n_procs=None):
    """
    Parallelizes consensus community assignments of `adjacency` over `gamma`

    Parameters
    ----------
    adjacency : (N x N) array_like
        Non-negative adjacency matrix
    gamma : list, optional
        List of weighting parameters to be used in modularity maximization.
        Default: np.linspace(0.25, 2.5, 10)
    obj : {'modularity', 'potts', 'negative_sym', 'negative_asym'}, optional
        Objective function for modularity maximization. Default: 'modularity'
    repeats : int, optional
        Number of times to repeat community detection for each gamma. Generated
        community assignments will be combined into a consensus matrix.
        Default: 250
    null_func : function, optional
        Function that can accept an array and return a single number. Used to
        generated null threshold for consensus community assignment generation.
        Default: numpy.mean
    n_procs : int, optional
        Number of processes to parallelize community assignment algorithm.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    consensus : list-of-np.ndarray
        Consensus community assignments
    modularity : list-of-float
        Average modularity of generated community assignment partitions
    zrand_index : list-of-float
        Normalized z-Rand index of generated partitions. Average z-Rand /
        standard deviation of z-Rand, z-scored across all gammas
    """

    if n_procs is None:
        n_procs = cpu_count()
    if gamma is None:
        gamma = np.linspace(0.25, 2.5, 10)

    modules = Parallel(n_jobs=n_procs)(
        delayed(consensus_modularity)(adjacency, gamma=gam, obj=obj,
                                      repeats=repeats, null_func=null_func,
                                      n_procs=1)
        for gam in gamma
    )

    consensus, modularity, zrand_avg, zrand_std = zip(*modules)
    zrand_index = zscore(np.array(zrand_avg) / np.array(zrand_std))

    return consensus, modularity, zrand_index
