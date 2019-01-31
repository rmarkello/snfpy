# SNFpy

This package provides a Python implementation of similarity network fusion (SNF), a technique for combining multiple data sources into a single graph representing the strength of relationships between samples.

[![Build Status](https://travis-ci.org/rmarkello/snfpy.svg?branch=master)](https://travis-ci.org/rmarkello/snfpy)
[![Codecov](https://codecov.io/gh/rmarkello/snfpy/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/snfpy)
[![Documentation Status](https://readthedocs.org/projects/snfpy/badge/?version=latest)](http://snfpy.readthedocs.io/en/latest/?badge=latest)
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
The procedur works by constructing networks of these samples for each data source, representing how *similar* each sample is to the other, and then fusing the networks together.
This figure from the original paper, which applied the method to genetics data, provides a nice demonstration:

![Similarity network fusion](https://media.nature.com/lw926/nature-assets/nmeth/journal/v11/n3/images/nmeth.2810-F1.jpg)


## Usage

For detailed examples check out our [documentation](https://snfpy.readthedocs.io)!

## How to get involved

We're thrilled to welcome new contributors!
If you're interesting in getting involved, you should start by reading our [contributing guidelines](https://github.com/rmarkello/snfpy/blob/master/CONTRIBUTING.md) and [code of conduct](https://github.com/rmarkello/snfpy/blob/master/CODE_OF_CONDUCT.md).
Once you're done with that, take a look at our [issues](https://github.com/rmarkello/snfpy/issues) to see if there is anything you might like to work on.
Alternatively, if you've found a bug, are experiencing a problem, or have a question, create a new issue with some information about it!

## Acknowledgments

This code is a translation of the original similarity network fusion code implemented in [R](http://compbio.cs.toronto.edu/SNF/SNF/Software.html) and [Matlab](http://compbio.cs.toronto.edu/SNF/SNF/Software.html).
As such, if you use this code please (1) provide a link back to the `snfpy` GitHub repository with the version of the code used, and (2) cite the original similarity network fusion paper:

    Wang, B., Mezlini, A. M., Demir, F., Fiume, M., Tu, Z., Brudno, M., Haibe-Kains, B., & Goldenberg, A. (2014). Similarity network fusion for aggregating data types on a genomic scale. Nature Methods, 11(3), 333.

## License information

This code is partially translated from the [original SNF code]((http://compbio.cs.toronto.edu/SNF/SNF/Software.html)) and is therefore released as a derivative under the [LGPLv3](https://github.com/rmarkello/snfpy/blob/master/LICENSE).
The full license may be found in the [LICENSE](https://github.com/rmarkello/snfpy/blob/master/LICENSE) file in the `snfpy` distribution.
