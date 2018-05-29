__version__ = '0.1'

NAME = 'snfpy'
MAINTAINER = 'Ross Markello'
EMAIL = 'rossmarkello@gmail.com'
VERSION = __version__
LICENSE = 'GLPLv3'
DESCRIPTION = 'A toolbox for Similarity Network Fusion (SNF)'
URL = 'http://github.com/rmarkello/snfpy'

DOWNLOAD_URL = (
    'https://github.com/rmarkello/snfpy/archive/{ver}.tar.gz'.format(
        ver=__version__))

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
    'snfpy': ['tests/data/*']
}
