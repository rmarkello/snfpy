__all__ = ['__version__', 'matlab', 'R', 'make_affinity', 'SNF',
           'get_n_clusters', 'spectral_clustering', 'nmi', 'silhouette_score',
           'affinity_zscore', 'SNF_gridsearch', 'get_optimal_params']

from snf.info import (__version__)
from snf import (matlab, R)
from snf.compute import (make_affinity, SNF, get_n_clusters)
from snf.metrics import (nmi, silhouette_score, affinity_zscore)
from snf.cv import (SNF_gridsearch, get_optimal_params)
from sklearn.cluster import spectral_clustering
