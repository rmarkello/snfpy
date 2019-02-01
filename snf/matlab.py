# -*- coding: utf-8 -*-
"""
Aliases for easier compatibility with the SNF toolbox written in Matlab
"""

from . import compute as _compute, metrics as _metrics

affinityMatrix = _compute.affinity_matrix
B0normalized = _compute._B0_normalized
Cal_NMI = _metrics.normalized_mutual_info_score
Concordance_Network_NMI = _metrics.nmi
Estimate_Number_of_Clusters_given_graph = _compute.get_n_clusters
FindDominateSet = _compute._find_dominate_set
SNF = _compute.snf
Standard_Normalization = _compute.scipy.stats.zscore
