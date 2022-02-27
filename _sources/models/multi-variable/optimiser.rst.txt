Pressure from wind - DA by optimisation in latent space
=======================================================

.. figure:: ../../figures/multi-variable/fit_U+V.jpg
   :width: 85%
   :align: center
   :figwidth: 85%

   Assimilation validation - fields of mslp: Red contours are from the input dataset, blue contours show generator output - for each of 25 samples with a different starting point in latent space. Black dots mark the locations of (wind) observations assimilated.

Script to find the optimal latent-space location and make the validation figure (options allow us to choose which variables to assimilate):

.. literalinclude:: ../../../models/DCVAE_single_PUV/fit_to_pseudo_obs/fit_multi.py
