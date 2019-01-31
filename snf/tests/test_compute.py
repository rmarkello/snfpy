# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import numpy as np
import pytest
from snf import compute

rs = np.random.RandomState(1234)
test_dir = pkg_resources.resource_filename('snf', 'tests/data/sim')
data1 = np.loadtxt(op.join(test_dir, 'data1.csv'))
data2 = np.loadtxt(op.join(test_dir, 'data2.csv'))
label = np.loadtxt(op.join(test_dir, 'label.csv'), dtype=int)


def test_make_affinity():
    aff = compute.make_affinity(data1)
    assert aff.shape == (len(data1), len(data1))


def test_find_dominate_set():
    aff = compute.make_affinity(data1)
    compute._find_dominate_set(aff)


def test_B0_normalized():
    aff = compute.make_affinity(data1)
    compute._B0_normalized(aff)


def test_snf():
    aff = compute.make_affinity(data1, data2)
    out = compute.snf(*aff)
    assert out.shape == aff[0].shape
    assert out.shape == aff[1].shape
    # both individual arrays and list should work as input
    assert np.allclose(out, compute.snf(aff))


def test_get_n_clusters():
    aff = compute.make_affinity(data1, data2)
    out = compute.snf(*aff)
    compute.get_n_clusters(out)


def test_dist2():
    dist = compute.dist2(data1)
    assert dist.shape == (len(data1), len(data1))


def test_group_predict():
    train = [d[::2] for d in [data1, data2]]
    test = [d[1::2] for d in [data1, data2]]
    train_lab = label[::2]
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
