# -*- coding: utf-8 -*-

from snf import compute, plotting, metrics
from sklearn.cluster import spectral_clustering

affinityMatrix = compute.affinity_matrix
B0normalized = compute._B0_normalized
Cal_NMI = metrics.normalized_mutual_info_score
Concordance_Network_NMI = metrics.nmi
displayClusters = plotting.mod_heatmap
dist2 = compute.dist2
Estimate_Number_of_Clusters_given_graph = compute.get_n_clusters
FindDominateSet = compute._find_dominate_set
SNF = compute.SNF
SpectralClustering = spectral_clustering
Standard_Normalization = compute.scipy.stats.zscore
