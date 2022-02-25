HadUK-Grid Tmax - get the training data
=======================================

Ideally I'd put in here a script to download the training data to use. This time, however, I cheated and used pre-prepared data provided by a colleague at the Met Office (thanks Mark). So you are on your own for this one.

You can get the data I used from `CEDA <https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb>`_ but I don't have a pre-prepared script for it. (Download the daily Tmax, and the monthly Tmax climatology). 

Utility script for data access (imported in most other scripts using this data):

.. literalinclude:: ../../../data/prepare_training_tensors_HUKG_Tmax/HUKG_load_tmax.py


