ERA5 to HadUK-Grid - validate the trained VAE
=============================================

.. figure:: ../../figures/conversion/comparison.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   VAE validation: top left - original field, top centre - ERA5 source field, top right - generator output, bottom right - difference original::generator, bottom centre - difference ERA::generator bottom left - scatter plots. 

Script to make the validation figure

.. literalinclude:: ../../../models/DCVAE_single_ERA5_to_HUKG_Tmax/validation/validate.py

Utility functions used in the plot

.. literalinclude:: ../../../models/DCVAE_single_ERA5_to_HUKG_Tmax/validation/plot_HUKG_comparison.py



