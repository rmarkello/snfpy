__all__ = ['compute', 'plotting', 'matlab', 'R']

from snf.info import (__version__)
from snf import (matlab, R)
from snf.compute import (make_affinity, SNF, nmi,
                         get_n_clusters, spectral_labels,
                         silhouette_score, affinity_zscore)
from snf.cv import (SNF_gridsearch, get_optimal_params)
