Mean-sea-level pressure - DA by optimisation in latent space
============================================================

.. figure:: ../../figures/mslp/fit_multi.jpg
   :width: 85%
   :align: center
   :figwidth: 85%

   Assimilation validation: Red contours are from the input dataset, blue contours show generator output - for each of 25 samples with a different starting point in latent space. Black dots mark the locations of observations assimilated.

Script to find the optimal latent-space location and make the validation figure:

.. literalinclude:: ../../../models/DCVAE_single_PRMSL/fit_to_pseudo_obs/fit_multi.py
