# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import numpy as np
from snf import cv

rs = np.random.RandomState(1234)
test_dir = pkg_resources.resource_filename('snf', 'tests/data/sim')
data1 = np.loadtxt(op.join(test_dir, 'data1.csv'))
data2 = np.loadtxt(op.join(test_dir, 'data2.csv'))
inputs = [(data1, 'euclidean'), (data2, 'euclidean')]


def test_compute_SNF():
    # don't define cluster number (find using compute.get_n_clusters)
    zaff, labels = cv.compute_SNF(inputs, n_perms=100)
    # define cluster number
    zaff, labels = cv.compute_SNF(inputs, n_clusters=3, n_perms=100)
    assert np.unique(labels).size == 3
    # provide list of cluster numbers
    zaff, labels = cv.compute_SNF(inputs, n_clusters=[3, 4], n_perms=100)
    assert isinstance(labels, list)
    for n, f in enumerate(labels, 3):
        assert np.unique(f).size == n


def test_SNF_gridsearch():
    # only a few parameters to test
    zaff, labels = cv.SNF_gridsearch(inputs, mu=[0.35, 0.85], K=[10, 20],
                                     n_clusters=[2, 3], n_perms=100, seed=1234)
    # get optimal parameters based on diff corners
    for neighbors in ['edges', 'corners']:
        mu, K = cv.get_optimal_params(zaff, labels, neighbors=neighbors)
