# -*- coding: utf-8 -*-

__author__ = 'snfpy developers'
__copyright__ = 'Copyright 2018, snfpy developers'
__credits__ = ['Ross Markello']
__license__ = 'LGPGv3'
__maintainer__ = 'Ross Markello'
__email__ = 'rossmarkello@gmail.com'
__status__ = 'Prototype'
__url__ = 'http://github.com/rmarkello/snfpy'
__packagename__ = 'snfpy'
__description__ = ('snfpy is a Python toolbox for performing similarity '
                   'network fusion.')
__longdesc__ = 'README.md'
__longdesctype__ = 'text/markdown'

INSTALL_REQUIRES = [
    'numpy>=1.14',
    'scikit-learn',
    'scipy',
]

TESTS_REQUIRE = [
    'pytest>=3.6',
    'pytest-cov'
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx>=1.2',
        'sphinx_rtd_theme',
    ],
    'tests': TESTS_REQUIRE
}

EXTRAS_REQUIRE['all'] = list(set([
    v for deps in EXTRAS_REQUIRE.values() for v in deps
]))

PACKAGE_DATA = {
    'snf': [
        'tests/data/digits/*csv',
        'tests/data/sim/*csv'
    ]
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
