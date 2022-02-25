HadUK-Grid Tmax - validate the trained VAE
==========================================

.. figure:: ../../figures/haduk-grid/comparison.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   VAE validation: top left - original field, top right - generator output, bottom left - difference, bottom right - scatter original::output. The point anomalies in the original, at the locations of some of the stations used, are an artefact of the simple spatial covariance model used in the dataset gridding process. That the generator does not reproduce them `might` be an advantage.

Script to make the validation figure

.. literalinclude:: ../../../models/DCVAE_single_HUKG_Tmax/validation/validate.py

Utility functions used in the plot

.. literalinclude:: ../../../models/DCVAE_single_HUKG_Tmax/validation/plot_HUKG_comparison.py



