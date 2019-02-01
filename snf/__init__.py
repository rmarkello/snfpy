__all__ = [
    '__author__', '__description__', '__email__', '__license__',
    '__maintainer__', '__packagename__', '__url__', '__version__',
    'cv', 'metrics', 'make_affinity', 'snf', 'get_n_clusters'
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .info import (
    __author__,
    __description__,
    __email__,
    __license__,
    __maintainer__,
    __packagename__,
    __url__,
)

from . import (cv, metrics)
from .compute import (make_affinity, snf, get_n_clusters)
