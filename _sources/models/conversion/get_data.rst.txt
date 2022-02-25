ERA5 to HadUK-Grid - get the training data
==========================================

Get the :doc:`HadUK-Grid data as described in the documentation for that dataset <../haduk-grid/get_data>`.

Get the :doc:`ERA5 data as described in the documentation for that ERA5 T2m <../ERA5-T2m/get_data>`, except that you will want daily Tmax instead of daily T2m. (It will still work with T2m, but not as well).


Utility script for ERA5 data access (imported in most other scripts using this data):

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_HKUG_Tmax/ERA5_load.py

Utility script for HadUK-Grid data access (imported in most other scripts using this data):

.. literalinclude:: ../../../data/prepare_training_tensors_HUKG_Tmax/HUKG_load_tmax.py

