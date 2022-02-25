HadUK-Grid Tmax - DA by optimisation in latent space
====================================================

.. figure:: ../../figures/haduk-grid/fit_full.jpg
   :width: 65%
   :align: center
   :figwidth: 65%

   Assimilation validation: top left - original field, top right - generator output at the latent-space location that maximises fit to pseudo-observations at the station locations, bottom left - difference, bottom right - scatter original::output. This uses the locations of the 310 stations used in making the original field.

Script to find the optimal latent-space location and make the validation figure:

.. literalinclude:: ../../../models/DCVAE_single_HUKG_Tmax/fit_to_obs/fit.py
