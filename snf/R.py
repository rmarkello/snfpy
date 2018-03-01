# -*- coding: utf-8 -*-

from snf import compute as _compute
from snf import plotting as _plotting

affinityMatrix = _compute.affinity_matrix
calNMI = _compute.normalized_mutual_info_score
concordanceNetworkMNI = _compute.nmi
displayClustersWithHeatmap = _plotting.mod_heatmap
dist2 = _compute.dist2
estimateNumberOfClustersGivenGraph = _compute.get_n_clusters
rankFeaturesByNMI = _compute.rank_feature_by_nmi
SNF = _compute.SNF
spectralClustering = _compute.spectral_labels
standardNormalization = _compute.scipy.stats.zscore
