# -*- coding: utf-8 -*-
"""
Aliases for easier compatibility with the SNF toolbox written in R
"""

from . import compute as _compute, metrics as _metrics

affinityMatrix = _compute.affinity_matrix
calNMI = _metrics.v_measure_score
concordanceNetworkMNI = _metrics.nmi
estimateNumberOfClustersGivenGraph = _compute.get_n_clusters
rankFeaturesByNMI = _metrics.rank_feature_by_nmi
SNF = _compute.snf
standardNormalization = _compute.stats.zscore
