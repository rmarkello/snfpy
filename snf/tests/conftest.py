# -*- coding: utf-8 -*-

import pytest
from snf import datasets


@pytest.fixture(scope='session')
def simdata():
    data = datasets.load_simdata()
    assert all(k in data for k in ['data', 'labels'])
    assert len(data.data) == 2

    return data


@pytest.fixture(scope='session')
def digits():
    data = datasets.load_digits()
    assert all(k in data for k in ['data', 'labels'])
    assert len(data.data) == 4

    return data
