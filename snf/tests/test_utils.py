# -*- coding: utf-8 -*-

import numpy as np
import pytest
from snf import utils

rs = np.random.RandomState(1234)
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
    X = np.arange(4**2).reshape(4, 4)

    for (x, y), out in neighbors.items():
        assert np.allclose(out['edge'],
                           X[utils.get_neighbors(x, y, 'edges', X.shape)])
        assert np.allclose(out['corners'],
                           X[utils.get_neighbors(x, y, 'corners', X.shape)])

    with pytest.raises(ValueError):
        utils.get_neighbors(x, y, 'badneighbors')


def test_extract_max_inds():
    X2d, X3d = np.arange(4**2).reshape(4, 4), np.arange(4**3).reshape(4, 4, 4)

    assert np.allclose([np.array([0, 1, 2, 3]), np.array([3, 3, 3, 3])],
                       utils.extract_max_inds(X2d))
    assert np.allclose(utils.extract_max_inds(X2d, axis=0),
                       utils.extract_max_inds(X2d, axis=-2))
    assert np.allclose([np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2],
                                  [3, 3, 3, 3]]),
                        np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                                  [0, 1, 2, 3]]),
                        np.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3],
                                  [3, 3, 3, 3]])],
                       utils.extract_max_inds(X3d))


def test_zrand():
    # make the same two-group community assignments (with different labels)
    label = np.ones((100, 1))
    X, Y = np.vstack((label, label*2)), np.vstack((label*2, label))
    # compare
    assert utils.zrand(X, Y) == utils.zrand(X, Y[::-1])
    assert utils.zrand(X, Y) > utils.zrand(X, rs.choice([0, 1], size=X.shape))
    assert utils.zrand(X, Y) == utils.zrand(X[:, 0], Y[:, 0])


def test_zrand_partitions():
    # make random communities
    comm = rs.choice(range(6), size=(10, 100))
    all_diff = utils.zrand_partitions(comm)
    all_same = utils.zrand_partitions(np.repeat(comm[:, [0]], 10, axis=1))

    # partition of labels that are all the same should have higher average
    # zrand and lower stdev zrand
    assert all_same[0] > all_diff[0]
    assert all_same[1] < all_diff[1]


def test_zrand_convolve():
    # random grid of community labels, testing against diff neighbor cases
    grid = rs.choice(range(6), size=(10, 10, 100))
    for neighbors in ['edges', 'corners']:
        utils.zrand_convolve(grid,  neighbors=neighbors)
