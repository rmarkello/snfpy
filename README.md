## SNFpy

`snfpy` is a Python implementation of [`SNFtool`](https://github.com/maxconway/SNFtool), a toolbox for performing [Similarity Network Fusion](http://compbio.cs.toronto.edu/SNF/SNF/Software.html).

This code is a translation of the original source code and is released as a derivative under the [LGPLv3](https://github.com/rmarkello/snfpy/blob/master/LICENSE). You can learn more about the original code by reading the original paper ([Wang et al., 2014, Nature Methods](https://www.ncbi.nlm.nih.gov/pubmed/24464287)).

## Status
[![Build Status](https://travis-ci.org/rmarkello/snfpy.svg?branch=master)](https://travis-ci.org/rmarkello/snfpy)
[![codecov](https://codecov.io/gh/rmarkello/snfpy/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/snfpy)
[![Documentation Status](https://readthedocs.org/projects/snfpy/badge/?version=latest)](http://snfpy.readthedocs.io/en/latest/?badge=latest)

## Requirements
Python 3.5 or higher

See [`requirements.txt`](https://github.com/rmarkello/snfpy/blob/master/requirements.txt) for more info on required modules.

## Installation
Using `git clone` and `python setup.py install` should do the trick.

## Usage
```python
>>> import snf
>>> data = [array1, array2]
>>> affinity_matrices = [snf.make_affinity(arr) for arr in data]
>>> snf = snf.SNF(affinity_matrices)
```

R- and Matlab-compatible function names are accessible via ``snf.R`` and ``snf.matlab``, respectively.

## Copyright
Implements Python-based version of [Similarity Network Fusion](http://compbio.cs.toronto.edu/SNF/SNF/Software.html).
Copyright (C) 2017 Ross Markello

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
