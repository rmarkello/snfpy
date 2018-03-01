# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest
from snf import compute

rs = np.random.RandomState(1234)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
data1 = np.loadtxt(os.path.join(TEST_DIR, 'data/data1.csv'))
data2 = np.loadtxt(os.path.join(TEST_DIR, 'data/data2.csv'))
label = np.loadtxt(os.path.join(TEST_DIR, 'data/label.csv'), dtype=int)


def test_make_affinity():
    aff = compute.make_affinity(data1)
    assert aff.shape == (len(data1), len(data1))


def test_find_dominate_set():
    aff = compute.make_affinity(data1)
    compute._find_dominate_set(aff)


def test_B0_normalized():
    aff = compute.make_affinity(data1)
    compute._B0_normalized(aff)


def test_SNF():
    aff = [compute.make_affinity(data) for data in [data1, data2]]
    out = compute.SNF(aff)
    assert out.shape == aff[0].shape
    assert out.shape == aff[1].shape


def test_nmi():
    compute.nmi([label, label, label])


def test_get_n_clusters():
    aff = [compute.make_affinity(data) for data in [data1, data2]]
    out = compute.SNF(aff)
    compute.get_n_clusters(out)


def test_rank_feature_by_nmi():
    aff = [compute.make_affinity(data) for data in [data1, data2]]
    inp = [(data, 'sqeuclidean') for data in [data1, data2]]
    out = compute.SNF(aff)
    compute.rank_feature_by_nmi(inp, out)


def test_spectral_labels():
    aff = compute.make_affinity(data1)
    compute.spectral_labels(aff, 2)


def test_silhouette_samples():
    aff = compute.make_affinity(data1)
    out = compute._silhouette_samples(aff, label)
    assert out.shape == label.shape
    with pytest.raises(ValueError):
        compute._silhouette_samples(aff, np.ones(len(aff)))


def test_silhouette_score():
    aff = compute.make_affinity(data1)
    out = compute.silhouette_score(aff, label)
    assert isinstance(out, float)


def test_affinity_zscore():
    aff = compute.make_affinity(data1)
    out = compute.affinity_zscore(aff, label, seed=1234)
    assert isinstance(out, float)


def test_dist2():
    dist = compute.dist2(data1, data2)
    assert dist.shape == (len(data1), len(data1))


def test_chi_square_distance():
    with pytest.raises(NotImplementedError):
        compute.chi_square_distance(data1, data2)
