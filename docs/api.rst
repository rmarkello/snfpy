.. _ref_api:

Reference API
=============

This is the primary reference of ``snfpy``. Please refer to the :ref:`user
guide <usage>` for more information on how to best implement these functions in
your own workflows.

.. contents:: **List of modules**
   :local:

.. _ref_compute:

:mod:`snf.compute` - Primary SNF functionality
----------------------------------------------

.. automodule:: snf.compute
   :no-members:
   :no-inherited-members:

.. currentmodule:: snf.compute

.. autosummary::
   :template: function.rst
   :toctree:  _generated/

   make_affinity
   get_n_clusters
   snf
   group_predict

.. _ref_metrics:

:mod:`snf.metrics` - Evaluation metrics
---------------------------------------

.. automodule:: snf.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: snf.metrics

.. autosummary::
   :template: function.rst
   :toctree:  _generated/

   nmi
   rank_feature_by_nmi
   silhouette_score
   affinity_zscore

.. _ref_cv:

:mod:`snf.cv` - Cross-validation functions
------------------------------------------

.. automodule:: snf.cv
   :no-members:
   :no-inherited-members:

.. currentmodule:: snf.cv

.. autosummary::
   :template: function.rst
   :toctree:  _generated/

   snf_gridsearch
   get_optimal_params
