# -*- coding: utf-8 -*-

__version__ = '0.1'

NAME = 'snfpy'
MAINTAINER = 'Ross Markello'
EMAIL = 'rossmarkello@gmail.com'
VERSION = __version__
LICENSE = 'LGPLv3'
DESCRIPTION = """\
A toolbox for Similarity Network Fusion (SNF)\
"""
LONG_DESCRIPTION = 'README.md'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/rmarkello/{name}'.format(name=NAME)
DOWNLOAD_URL = ('http://github.com/rmarkello/{name}/archive/{ver}.tar.gz'
                .format(name=NAME, ver=__version__))

INSTALL_REQUIRES = [
    'numpy',
    'scikit-learn',
    'scipy',
]

TESTS_REQUIRE = [
    'codecov'
    'pytest',
    'pytest-cov'
]

EXTRAS_REQUIRE = {
    'doc': [
        'numpydoc',
        'sphinx>=1.2',
        'sphinx_rtd_theme',
    ],
    'numba': [
        'numba',
    ],
    'tests': TESTS_REQUIRE
}

EXTRAS_REQUIRE['all'] = list(set([
    v for deps in EXTRAS_REQUIRE.values() for v in deps
]))

PACKAGE_DATA = {
    'snfpy': ['tests/data/*']
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
