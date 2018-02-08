__version__ = '0.0.1'

NAME = 'snf'
MAINTAINER = 'Ross Markello'
VERSION = __version__
LICENSE = 'GLPLv3'
DESCRIPTION = 'A toolbox for Similarity Network Fusion (SNF)'
DOWNLOAD_URL = 'http://github.com/rmarkello/snfpy'

INSTALL_REQUIRES = [
    'numpy',
    'scikit-learn',
    'scipy',
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

PACKAGE_DATA = {
     'snf': ['tests/data/*']
}
