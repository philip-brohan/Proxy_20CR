Pressure from wind - validate the trained VAE
=============================================

.. figure:: ../../figures/multi-variable/comparison.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   VAE validation: Red contours are from the input dataset, blue contours show generator output. The three rows are the variables: mslp - top, u10m - middle, v10m - bottom. The left hand column shows comparisons of a case from the training dataset, the middle column a case from the test dataset, and the right hand column an example of generator output. 


Script to make the validation figure

.. literalinclude:: ../../../models/DCVAE_single_PUV/validation/validate.py

Utility functions used in the plot

.. literalinclude:: ../../../models/DCVAE_single_PUV/validation/plot_prmsl_comparison.py



