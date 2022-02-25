ERA5 T2m - validate the trained VAE
===================================

.. figure:: ../../figures/DCVAE_validation.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   VAE validation: top left - original field, top right - generator output, bottom left - difference, bottom right - scatter original::output. (Note that a substantially better result could be produced with more model-building effort and a larger latent space, but this is good enough for present purposes).

Script to make the validation figure

.. literalinclude:: ../../../models/DCVAE_single_ERA5_T2m/validation/validate.py

Utility functions used in the plot

.. literalinclude:: ../../../models/DCVAE_single_ERA5_T2m/validation/plot_ERA5_comparison.py



