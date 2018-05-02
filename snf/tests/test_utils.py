# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import numpy as np
import pytest
from snf import utils

rs = np.random.RandomState(1234)
test_dir = pkg_resources.resource_filename('snf', 'tests/data/sim')
data1 = np.loadtxt(op.join(test_dir, 'data1.csv'))
data2 = np.loadtxt(op.join(test_dir, 'data2.csv'))
label = np.loadtxt(op.join(test_dir, 'label.csv'), dtype=int)


neighbors = {
    (0, 0): dict(edge=np.array([0, 4, 1]),
                 corners=np.array([0, 4, 1, 5])),
    (1, 2): dict(edge=np.array([5, 2, 6, 10, 7]),
                 corners=np.array([1, 5, 9, 2, 6, 10, 3, 7, 11])),
    (2, 1): dict(edge=np.array([8, 5, 9, 13, 10]),
                 corners=np.array([4, 8, 12, 5, 9, 13, 6, 10, 14])),
    (3, 3): dict(edge=np.array([14, 11, 15]),
                 corners=np.array([10, 14, 11, 15]))
}


def test_neighbors():
    X = np.arange(16).reshape(4, 4)
    for (x, y), out in neighbors.items():
        assert np.allclose(out['edge'],
                           X[utils.get_neighbors(x, y, 'edges', X.shape)])
        assert np.allclose(out['corners'],
                           X[utils.get_neighbors(x, y, 'corners', X.shape)])


def test_extract_max_inds():
    pass


def test_zrand():
    pass


def test_zrand_partitions():
    pass


def test_zrand_convolve():
    pass
