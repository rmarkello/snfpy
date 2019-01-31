# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import numpy as np
import pytest
from snf import compute, metrics

rs = np.random.RandomState(1234)
test_dir = pkg_resources.resource_filename('snf', 'tests/data/sim')
data1 = np.loadtxt(op.join(test_dir, 'data1.csv'))
data2 = np.loadtxt(op.join(test_dir, 'data2.csv'))
label = np.loadtxt(op.join(test_dir, 'label.csv'), dtype=int)


def test_nmi():
    metrics.nmi([label, label, label])


def test_rank_feature_by_nmi():
    aff = compute.make_affinity(data1, data2)
    out = compute.snf(*aff)
    inp = [(data, 'sqeuclidean') for data in [data1, data2]]
    metrics.rank_feature_by_nmi(inp, out)


def test_silhouette_samples():
    aff = compute.make_affinity(data1)
    out = metrics._silhouette_samples(aff, label)
    assert out.shape == label.shape
    with pytest.raises(ValueError):
        metrics._silhouette_samples(aff, np.ones(len(aff)))


def test_silhouette_score():
    aff = compute.make_affinity(data1)
    out = metrics.silhouette_score(aff, label)
    assert isinstance(out, float)


def test_affinity_zscore():
    aff = compute.make_affinity(data1)
    out = metrics.affinity_zscore(aff, label, seed=1234)
    assert isinstance(out, float)
