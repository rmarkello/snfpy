# -*- coding: utf-8 -*-

import numpy as np
import pytest
from snf import compute

rs = np.random.RandomState(1234)


def test_check_data_metric(simdata, digits):
    # vanilla tests to check outputs match
    d, m = zip(*list(compute._check_data_metric(simdata.data, 'euclidean')))
    assert len(d) == len(m) == 2
    d, m = zip(*list(compute._check_data_metric(digits.data, 'euclidean')))
    assert len(d) == len(m) == 4

    # check that single value for metric is propagated through to all arrays
    d, m = zip(*list(compute._check_data_metric([simdata.data, simdata.data],
                                                'euclidean')))
    assert len(d) == len(m) == 4
    assert all(f == 'euclidean' for f in m)

    # check that metrics will be expanded to accompany data argument sublists
    d, m = zip(*list(compute._check_data_metric([simdata.data, simdata.data],
                                                ['euclidean', 'cityblock'])))
    assert len(d) == len(m) == 4
    assert m == tuple(['euclidean'] * 2 + ['cityblock'] * 2)


@pytest.fixture(scope='module')
def test_affinity(simdata):

    # this should handle a lot of different argument definitions
    aff = compute.make_affinity(simdata.data)
    aff_copy = compute.make_affinity(*simdata.data)
    assert all(np.allclose(a1, a2) for (a1, a2) in zip(aff, aff_copy))

    # outputs are square with shape (samples, samples)
    assert all(a.shape == (len(d), len(d)) for a, d in zip(aff, simdata.data))

    # all outputs are entirely positive (i.e., similarity / affinity)
    assert all(np.all(a > 0) for a in aff)

    return aff


def test_find_dominate_set(test_affinity):
    compute._find_dominate_set(test_affinity[0])


def test_B0_normalized(test_affinity):
    compute._B0_normalized(test_affinity[0])


def test_snf(test_affinity):
    out = compute.snf(*test_affinity)
    assert out.shape == test_affinity[0].shape
    assert out.shape == test_affinity[1].shape
    # both individual arrays and list should work as input
    assert np.allclose(out, compute.snf(test_affinity))


def test_get_n_clusters(test_affinity):
    out = compute.snf(*test_affinity)
    compute.get_n_clusters(out)


def test_dist2(simdata):
    for data in simdata.data:
        assert compute.dist2(data).shape == (len(data), len(data))


def test_group_predict(simdata):
    # split train/test and labels
    train = [d[::2] for d in simdata.data]
    test = [d[1::2] for d in simdata.data]
    train_lab = simdata.labels[::2]

    compute.group_predict(train, test, train_lab)
    with pytest.raises(ValueError):
        tr = [d[:, 0] for d in train]
        compute.group_predict(tr, test, train_lab)
    with pytest.raises(ValueError):
        compute.group_predict([train[0]], test, train_lab)
    with pytest.raises(ValueError):
        compute.group_predict(train, test, train_lab[:10])


def test_dnorm():
    compute._dnorm(np.random.rand(100, 100), norm='gph')
    with pytest.raises(ValueError):
        compute._dnorm(np.random.rand(100, 100), norm='bad')
