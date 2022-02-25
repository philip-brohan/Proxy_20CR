Mean-sea-level pressure - convert the data into tf.Tensors
==========================================================

Script to make a tensor from a single hour's data:

.. literalinclude:: ../../../data/prepare_training_tensors/make_training_tensor.py

Script to make a tensor for every day in a 40-year period (runs the above script many times):

.. literalinclude:: ../../../data/prepare_training_tensors/makeall.py
