ERA5 T2m - convert the data into tf.Tensors
===========================================

Script to make a tensor from a single day's data:

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/make_training_tensor.py

Script to make a tensor for every day in a 40-year period (runs the above script many times):

.. literalinclude:: ../../../data/prepare_training_tensors_ERA5_T2m/makeall.py
