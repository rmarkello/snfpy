__all__ = [
    '__version__', 'cv', 'metrics', 'make_affinity', 'snf', 'get_n_clusters'
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import (cv, metrics)
from .compute import (make_affinity, snf, get_n_clusters)
