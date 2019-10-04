# -*- coding: utf-8 -*-

import numpy as np
import pytest
from snf import compute

rs = np.random.RandomState(1234)


def test_check_data_metric(simdata, digits):
    # vanilla tests to check outputs
    d, m = zip(*list(compute._check_data_metric(simdata.data, 'euclidean')))
    assert len(d) == len(m) == 2
    d, m = zip(*list(compute._check_data_metric(digits.data, 'euclidean')))
    assert len(d) == len(m) == 4

    # single value for `metric` is propagated through to all arrays
    d, m = zip(*list(compute._check_data_metric([simdata.data, simdata.data],
                                                'euclidean')))
    assert len(d) == len(m) == 4
    assert all(f == 'euclidean' for f in m)

    # `metric` will be expanded separately for sublists of `data`
    d, m = zip(*list(compute._check_data_metric([simdata.data, simdata.data],
                                                ['euclidean', 'cityblock'])))
    assert len(d) == len(m) == 4
    assert m == tuple(['euclidean'] * 2 + ['cityblock'] * 2)


@pytest.fixture(scope='module')
def affinity(simdata):
    # this should handle a lot of different argument definitions
    # just test two (single list, individual arguments) to compare outputs
    aff = compute.make_affinity(simdata.data)
    aff_copy = compute.make_affinity(*simdata.data)

    # generated affinity matrices are identical regardless of how args provided
    assert all(np.allclose(a1, a2) for (a1, a2) in zip(aff, aff_copy))
    # outputs are square with shape (samples, samples)
    assert all(a.shape == (len(d), len(d)) for a, d in zip(aff, simdata.data))
    # all outputs are entirely positive (i.e., similarity / affinity)
    assert all(np.all(a > 0) for a in aff)

    return aff


@pytest.mark.parametrize('K', [10, 50, 100, 200])
def test_find_dominate_set(affinity, K):
    out = compute._find_dominate_set(affinity[0], K=K)
    # ensure only K non-zero entries remain for each row
    assert np.all(np.count_nonzero(out, axis=1) == K)
    # resulting array is NOT symmetrical
    assert not np.allclose(out, out.T)


@pytest.mark.parametrize('alpha', [0, 0.5, 1, 2])
def test_B0_normalized(affinity, alpha):
    out = compute._B0_normalized(affinity[0], alpha=alpha)
    # amounts to adding alpha to the diagonal (and symmetrizing)
    assert np.allclose(np.diag(out), np.diag(affinity[0]) + alpha)
    # resulting array IS symmetrical
    assert np.allclose(out, out.T)


def test_snf(affinity):
    out = compute.snf(*affinity)
    # output shape should be the same as input shape
    assert out.shape == affinity[0].shape == affinity[1].shape
    # both individual arrays and list should work as input
    assert np.allclose(out, compute.snf(affinity))


def test_get_n_clusters(affinity):
    fused = compute.snf(*affinity)
    n1, n2 = compute.get_n_clusters(fused)
    # all outputs are integers
    assert all(isinstance(n, np.integer) for n in [n1, n2])
    # providing alternative cluster numbers to search is valid
    n1, n2 = compute.get_n_clusters(fused, range(2, 10))
    # cannot provide single integer -- literally doesn't make sense
    with pytest.raises(TypeError):
        compute.get_n_clusters(fused, 5)


def test_group_predict(simdata):
    # split train/test and labels and run group prediction
    train = [d[:150] for d in simdata.data]
    test = [d[150:] for d in simdata.data]
    train_lab = simdata.labels[:150]
    test_lab = compute.group_predict(train, test, train_lab)

    # test labels same length as test data
    assert len(test_lab) == len(test[0])
    # test labels come from subset of label in train labels
    assert set(test_lab).issubset(train_lab)

    with pytest.raises(ValueError):
        tr = [d[:, 0] for d in train]
        compute.group_predict(tr, test, train_lab)
    with pytest.raises(ValueError):
        compute.group_predict([train[0]], test, train_lab)
    with pytest.raises(ValueError):
        compute.group_predict(train, test, train_lab[:10])


@pytest.mark.parametrize('norm', ['gph', 'ave'])
def test_dnorm(norm):
    compute._dnorm(rs.rand(100, 100), norm=norm)
    with pytest.raises(ValueError):
        compute._dnorm(rs.rand(100, 100), norm='bad')
