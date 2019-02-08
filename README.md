# SNFpy

This package provides a Python implementation of similarity network fusion (SNF), a technique for combining multiple data sources into a single graph representing sample relationships.

[![Build Status](https://travis-ci.org/rmarkello/snfpy.svg?branch=master)](https://travis-ci.org/rmarkello/snfpy)
[![Codecov](https://codecov.io/gh/rmarkello/snfpy/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/snfpy)
[![Documentation Status](https://readthedocs.org/projects/snfpy/badge/?version=latest)](https://snfpy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)

## Table of Contents

If you know where you're going, feel free to jump ahead:

* [Installation](#requirements-and-installation)
* [Purpose](#purpose)
* [Example usage](#usage)
* [How to get involved](#how-to-get-involved)
* [Acknowlegments](#acknowledgments)
* [License information](#license-information)

## Requirements and installation

This package requires Python version 3.5 or greater.
Assuming you have the correct version of Python, you can install this package by opening a command terminal and running the following:

```bash
git clone https://github.com/rmarkello/snfpy.git
cd snfpy
python setup.py install
```

You can install the latest release from PyPi, with:

```bash
pip install snfpy
```

## Purpose

Similarity network fusion is a technique originally proposed by [Wang et al., 2014, Nature Methods](https://www.ncbi.nlm.nih.gov/pubmed/24464287) to combine data from different sources for a shared group of samples.
The procedure works by constructing networks of these samples for each data source that represent how *similar* each sample is to all the others, and then fusing the networks together.
This figure from the original paper, which applied the method to genetics data, provides a nice demonstration:

![Similarity network fusion](https://media.nature.com/lw926/nature-assets/nmeth/journal/v11/n3/images/nmeth.2810-F1.jpg)

The similarity network generation and fusion process use a [K-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) procedure to down-weight weaker relationships between samples.
However, weak relationships that are consistent across data sources are retained via the fusion process.
For more information on the math behind SNF you can read the [original paper](https://www.ncbi.nlm.nih.gov/pubmed/24464287) or take a look at the [reference documentation](https://snfpy.readthedocs.io/en/latest/api.html).

## Usage

There are ~three functions from which you can construct _most_ SNF workflows.
To demonstrate, we can use one of the example datasets provided with the `snfpy` distribution:

```python
>>> from snf import datasets
>>> digits = datasets.load_digits()
>>> digits.keys()
dict_keys(['data', 'labels'])
```

The `digits` dataset is comprised of 600 samples profiled across four data types which have 76, 240, 216, and 47 features each; these data can be accessed via `digits.data`:

```python
>>> for arr in digits.data:
...     print(arr.shape)
(600, 76)
(600, 240)
(600, 216)
(600, 47)
```

The dataset also comes with a set of labels grouping the samples into one of three categories (0, 1, or 2); these labels can be accessed via `digits.labels`.
We can see that there are 200 samples per group:

```python
import numpy as np
>>> groups, samples = np.unique(digits.labels, return_counts=True)
>>> for grp, count in zip(groups, samples):
...     print('Group {:.0f}: {} samples'.format(grp, count))
Group 0: 200 samples
Group 1: 200 samples
Group 2: 200 samples
```

The first step in SNF is converting these data arrays into similarity (or "affinity") networks.
To do so, we provide the data arrays to the `snf.make_affinity()` function:

```python
>>> import snf
>>> affinity_networks = snf.make_affinity(digits.data, metric='euclidean', K=20, mu=0.5)
```

The arguments provided to this function (`metric`, `K`, and `mu`) all play a role in how the similarity network is constructed.
The `metric` argument controls the distance function used when constructing the network.
Distance networks and similarity networks are inverses of one another, however, so we convert from one to the another.
To do so, we apply a `mu`-weighted `K`-nearest neighbors kernel to the distance array (for more math, see the [`snf.make_affinity()` docstring](https://snfpy.readthedocs.io/en/latest/generated/snf.compute.make_affinity.html)).

Once we have our similarity networks we can fuse them together with SNF:

```python
>>> fused_network = snf.snf(affinity_networks, K=20)
```

The fused network is then ready for other analyses (like clustering!).
If we want to cluster the network we can try and determine the "optimal" number of clusters to form using `snf.get_n_clusters()`.
This functions returns the top _two_ "optimal" number of clusters (estimated via an eigengap approach):

```python
>>> best, second = snf.get_n_clusters(fused_network)
>>> best, second
(3, 4)
```

Then, we can cluster the network using, for example, spectral clustering and compare the resulting labels to the "true" labels provided with the test dataset:

```python
>>> from sklearn.cluster import spectral_clustering
>>> from sklearn.metrics import v_measure_score

>>> labels = spectral_clustering(fused_network, n_clusters=best)
>>> v_measure_score(labels, digits.labels)
0.9734300455833589
```

The metric (`v_measure_score`) used here ranges from 0 to 1, where 1 indicates perfect overlap between the derived and true labels and 0 indicates no overlap.
We see that 0.97 indicates that the SNF fused network clustering results were highly accurate!

Of course, your mileage may (will) vary for other datasets, but this should be sufficient to get you started!
For more detailed examples and alternative uses check out our [documentation](https://snfpy.readthedocs.io).

## How to get involved

We're thrilled to welcome new contributors!
If you're interesting in getting involved, you should start by reading our [contributing guidelines](https://github.com/rmarkello/snfpy/blob/master/CONTRIBUTING.md) and [code of conduct](https://github.com/rmarkello/snfpy/blob/master/CODE_OF_CONDUCT.md).
Once you're done with that, take a look at our [issues](https://github.com/rmarkello/snfpy/issues) to see if there is anything you might like to work on.
Alternatively, if you've found a bug, are experiencing a problem, or have a question, create a new issue with some information about it!

## Acknowledgments

This code is a translation of the original similarity network fusion code implemented in [R](http://compbio.cs.toronto.edu/SNF/SNF/Software.html) and [Matlab](http://compbio.cs.toronto.edu/SNF/SNF/Software.html).
As such, if you use this code please (1) provide a link back to the `snfpy` GitHub repository with the version of the code used, and (2) cite the original similarity network fusion paper:

    Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Tu, Z., Brudno, M., Haibe-Kains, B., &
    Goldenberg, A. (2014). Similarity network fusion for aggregating data types on a genomic scale.
    Nature Methods, 11(3), 333.

## License information

This code is partially translated from the [original SNF code](http://compbio.cs.toronto.edu/SNF/SNF/Software.html) and is therefore released as a derivative under the [LGPLv3](https://github.com/rmarkello/snfpy/blob/master/LICENSE).
The full license may be found in the [LICENSE](https://github.com/rmarkello/snfpy/blob/master/LICENSE) file in the `snfpy` distribution.
